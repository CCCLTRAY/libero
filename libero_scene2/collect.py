#!/usr/bin/env python3
"""Generate recoverable-intervention LIBERO Scene-2 data in immutable shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from libero_scene2.runtime import observation_proprio


TASKS = {
    "alphabet": {
        "bddl": "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl",
        "instruction": "pick up the alphabet soup and put it in the basket",
        "offset": [0.008916445736341407, -0.0047663230010204505, 0.010760073656931712],
    },
    "tomato": {
        "bddl": "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl",
        "instruction": "pick up the tomato sauce and put it in the basket",
        "offset": [0.006471640009783043, -0.0030734822166479103, 0.005569792705104187],
    },
    "butter": {
        "bddl": "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl",
        "instruction": "pick up the butter and put it in the basket",
        "offset": [0.005033897444638877, 0.00017858589607018527, 0.0015829072442838377],
    },
    "milk": {
        "bddl": "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.bddl",
        "instruction": "pick up the milk and put it in the basket",
        "offset": [0.0034351570921239485, -0.00031244976678798664, 0.029967486182561953],
    },
    "orange": {
        "bddl": "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket.bddl",
        "instruction": "pick up the orange juice and put it in the basket",
        "offset": [-0.005842761091536029, -0.0011439102609022006, 0.03123146007477029],
    },
}
TASK_IDS = {name: index for index, name in enumerate(TASKS)}
PHASE_IDS = {
    "hover_object": 0,
    "descend": 1,
    "close": 2,
    "lift": 3,
    "transport": 4,
    "lower": 5,
    "release": 6,
    "retreat": 7,
}
PHASE_GROUPS = {
    "approach": {"hover_object"},
    "grasp": {"descend", "close"},
    "lift": {"lift"},
    "transport": {"transport", "lower"},
}
PHASE_GROUP_NAMES = tuple(PHASE_GROUPS)
PHASE_GROUP_PROBS = np.asarray([0.20, 0.25, 0.25, 0.30], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=620)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument(
        "--image-compression", choices=("lzf", "gzip"), default="lzf"
    )
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=1)
    return parser.parse_args()


def sample_burst_length(rng: np.random.Generator) -> int:
    if rng.random() < 0.05:
        return 10
    return int(np.clip(1 + np.floor(10 * rng.beta(1.0, 1.5)), 1, 10))


def sample_episode_schedule(rng: np.random.Generator) -> list[dict]:
    count = int(rng.choice([0, 1, 2], p=[0.30, 0.50, 0.20]))
    if count == 0:
        return []

    # Multiple interventions must follow task progress. Independent sampling can
    # request an earlier phase after a later one (for example lift then grasp),
    # making the second scheduled burst impossible to trigger.
    groups = rng.choice(
        PHASE_GROUP_NAMES,
        size=count,
        replace=False,
        p=PHASE_GROUP_PROBS,
    )
    order = {name: index for index, name in enumerate(PHASE_GROUP_NAMES)}
    groups = sorted((str(group) for group in groups), key=order.__getitem__)
    return [
        {
            "burst_id": burst_id,
            "phase_group": phase_group,
            "length": sample_burst_length(rng),
        }
        for burst_id, phase_group in enumerate(groups)
    ]


class ShardWriter:
    def __init__(
        self, path: Path, *, image_compression: str, gzip_level: int
    ) -> None:
        self.final = path
        self.temporary = path.with_suffix(path.suffix + ".partial")
        self.image_compression = image_compression
        self.gzip_level = gzip_level
        if self.final.exists() or self.temporary.exists():
            raise FileExistsError(self.final)
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.temporary, "w", libver="latest")
        self.frame_ptr = 0
        self.episodes = 0

    def _create(self, columns: dict[str, np.ndarray]) -> None:
        for key, values in columns.items():
            kwargs = {}
            if key in {"pixels", "wrist_pixels"}:
                kwargs = {
                    "compression": self.image_compression,
                    "chunks": (1, *values.shape[1:]),
                }
                if self.image_compression == "gzip":
                    kwargs.update(
                        compression_opts=self.gzip_level,
                        shuffle=True,
                    )
            else:
                rows = min(512, max(1, len(values)))
                kwargs = {"chunks": (rows, *values.shape[1:])}
            self.handle.create_dataset(
                key,
                shape=(0, *values.shape[1:]),
                maxshape=(None, *values.shape[1:]),
                dtype=values.dtype,
                **kwargs,
            )
        self.handle.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self.handle.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

    def write(self, columns: dict[str, np.ndarray], metadata: dict) -> None:
        length = len(next(iter(columns.values())))
        if self.episodes == 0:
            self._create(columns)
            self.handle.create_group("episode_metadata")
        for key, values in columns.items():
            dataset = self.handle[key]
            dataset.resize(self.frame_ptr + length, axis=0)
            dataset[self.frame_ptr : self.frame_ptr + length] = values
        for key, value in (("ep_len", length), ("ep_offset", self.frame_ptr)):
            dataset = self.handle[key]
            dataset.resize(self.episodes + 1, axis=0)
            dataset[self.episodes] = value
        self.handle["episode_metadata"].attrs[str(self.episodes)] = json.dumps(metadata)
        self.frame_ptr += length
        self.episodes += 1
        self.handle.flush()

    def close(self, attrs: dict) -> tuple[int, str]:
        for key, value in attrs.items():
            self.handle.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        self.handle.attrs["complete"] = True
        self.handle.close()
        self.temporary.replace(self.final)
        digest = hashlib.sha256()
        with self.final.open("rb") as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
        return self.final.stat().st_size, digest.hexdigest()


def run_episode(env, oracle_cls, task_objects, task: str, seed: int, max_steps: int):
    rng = np.random.default_rng(seed + 932_771)
    env.seed(seed)
    observation = env.reset()
    inner = env.env
    object_name, site_name = task_objects(inner)
    oracle = oracle_cls(
        inner=inner,
        object_name=object_name,
        site_name=site_name,
        grasp_offset=np.asarray(TASKS[task]["offset"], dtype=np.float64),
        position_gain=16.0,
        max_action=0.55,
        hover_height=0.14,
        basket_clearance=0.16,
        reference_eef_quat=observation["robot0_eef_quat"],
    )
    schedule = sample_episode_schedule(rng)
    next_burst = 0
    active = None
    active_step = 0
    cooldown = 0
    noise_sigma = 0.0 if rng.random() < 0.20 else float(0.10 * rng.beta(2.0, 5.0))
    noise_state = np.zeros(6, dtype=np.float32)
    rows: dict[str, list] = {
        key: []
        for key in (
            "pixels", "wrist_pixels", "action", "oracle_action", "proprio",
            "reward", "done", "task_id", "episode_seed", "action_mode",
            "phase_id", "burst_id", "burst_length", "burst_step",
            "noise_sigma", "privileged_state",
        )
    }
    success = False
    for step in range(max_steps):
        oracle_action, info = oracle.act(observation)
        if active is None and next_burst < len(schedule) and cooldown <= 0:
            candidate = schedule[next_burst]
            if info["phase"] in PHASE_GROUPS[candidate["phase_group"]]:
                active = candidate
                active_step = 0
                next_burst += 1
        if active is not None:
            executed = rng.uniform(-1.0, 1.0, 7).astype(np.float32)
            mode = 2
            burst_id = active["burst_id"]
            burst_length = active["length"]
            burst_step = active_step
            active_step += 1
            if active_step >= active["length"]:
                active = None
                cooldown = 40
        else:
            noise_state = (0.8 * noise_state + np.sqrt(1.0 - 0.8**2) * rng.normal(
                0.0, noise_sigma, 6
            )).astype(np.float32)
            executed = oracle_action.copy()
            executed[:6] = np.clip(executed[:6] + noise_state, -1.0, 1.0)
            mode = 0 if noise_sigma == 0.0 else 1
            burst_id = -1
            burst_length = 0
            burst_step = 0
            cooldown -= 1
        obj = np.asarray(info["object"], dtype=np.float32)
        eef = np.asarray(info["eef"], dtype=np.float32)
        site = np.asarray(info["site"], dtype=np.float32)
        privileged = np.concatenate(
            [
                eef, obj, site,
                np.asarray(
                    [info["left_contact"], info["right_contact"], info["contained"]],
                    dtype=np.float32,
                ),
            ]
        )
        rows["pixels"].append(np.asarray(observation["agentview_image"], dtype=np.uint8))
        rows["wrist_pixels"].append(
            np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8)
        )
        rows["action"].append(executed)
        rows["oracle_action"].append(oracle_action.astype(np.float32))
        rows["proprio"].append(observation_proprio(observation))
        rows["task_id"].append(TASK_IDS[task])
        rows["episode_seed"].append(seed)
        rows["action_mode"].append(mode)
        rows["phase_id"].append(PHASE_IDS[info["phase"]])
        rows["burst_id"].append(burst_id)
        rows["burst_length"].append(burst_length)
        rows["burst_step"].append(burst_step)
        rows["noise_sigma"].append(noise_sigma)
        rows["privileged_state"].append(privileged)
        observation, reward, done, _ = env.step(executed)
        success = bool(env.check_success())
        rows["reward"].append(float(reward))
        rows["done"].append(bool(done or success))
        if success and not info["grasped"] and oracle.phase in {"release", "retreat"}:
            break
    dtypes = {
        "pixels": np.uint8, "wrist_pixels": np.uint8, "action": np.float32,
        "oracle_action": np.float32, "proprio": np.float32, "reward": np.float32,
        "done": np.uint8, "task_id": np.int16, "episode_seed": np.int32,
        "action_mode": np.uint8, "phase_id": np.uint8, "burst_id": np.int16,
        "burst_length": np.uint8, "burst_step": np.uint8,
        "noise_sigma": np.float32, "privileged_state": np.float32,
    }
    columns = {key: np.asarray(value, dtype=dtypes[key]) for key, value in rows.items()}
    metadata = {
        "seed": seed,
        "task": task,
        "success": success,
        "length": len(columns["action"]),
        "noise_sigma": noise_sigma,
        "burst_schedule": schedule,
        "bursts_started": next_burst,
    }
    return columns, metadata


def main() -> None:
    args = parse_args()
    from libero_scene2.oracle import BasketOracle, task_objects
    from libero_scene2.runtime import (
        make_env,
        observation_proprio,
        resolve_bddl,
        runtime_versions,
    )

    bddl_file = resolve_bddl(TASKS[args.task]["bddl"])
    env = make_env(bddl_file=bddl_file, camera_size=args.camera_size)
    writer = ShardWriter(
        args.out,
        image_compression=args.image_compression,
        gzip_level=args.gzip_level,
    )
    metadata = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            columns, episode = run_episode(
                env, BasketOracle, task_objects, args.task, seed, args.max_steps
            )
            writer.write(columns, episode)
            metadata.append(episode)
            print(json.dumps(episode, sort_keys=True), flush=True)
    finally:
        env.close()
    attrs = {
        "protocol": "scene2_oracle_beta_burst_v1",
        "task": args.task,
        "task_id": TASK_IDS[args.task],
        "instruction": TASKS[args.task]["instruction"],
        "split": args.split,
        "seed_start": args.seed_start,
        "episodes": args.episodes,
        "shard_id": args.shard_id,
        "image_compression": args.image_compression,
        "runtime": runtime_versions(),
    }
    if args.image_compression == "gzip":
        attrs["gzip_level"] = args.gzip_level
    size, digest = writer.close(attrs)
    manifest = {
        **attrs,
        "path": args.out.name,
        "bytes": size,
        "sha256": digest,
        "successes": sum(row["success"] for row in metadata),
        "transitions": sum(row["length"] for row in metadata),
        "episodes_metadata": metadata,
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ("episodes", "successes", "transitions", "bytes", "sha256")}, indent=2))


if __name__ == "__main__":
    main()

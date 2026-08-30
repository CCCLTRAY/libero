#!/usr/bin/env python3
"""Audit immutable Scene-2 shards and build zero-copy HDF5 VDS indices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


SPLITS = ("train", "validation", "test")
TASKS = ("alphabet", "tomato", "butter", "milk", "orange")
STEP_KEYS = (
    "pixels",
    "wrist_pixels",
    "action",
    "oracle_action",
    "proprio",
    "reward",
    "done",
    "task_id",
    "episode_seed",
    "action_mode",
    "phase_id",
    "burst_id",
    "burst_length",
    "burst_step",
    "noise_sigma",
    "privileged_state",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_shard(path: Path, verify_sha: bool) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(sidecar)
    manifest = json.loads(sidecar.read_text())
    if verify_sha and digest(path) != manifest["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")

    with h5py.File(path, "r", swmr=True) as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"Incomplete shard: {path}")
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        expected_episodes = int(manifest["episodes"])
        if len(lengths) != expected_episodes:
            raise ValueError(
                f"Expected {expected_episodes} episodes in {path}, got {len(lengths)}"
            )
        expected_offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1])]
        )
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(f"Invalid episode offsets: {path}")
        transitions = int(lengths.sum())
        schema = {}
        for key in STEP_KEYS:
            if key not in handle:
                raise KeyError(f"{key} missing from {path}")
            dataset = handle[key]
            if len(dataset) != transitions:
                raise ValueError(
                    f"{path}:{key} has {len(dataset)} rows, expected {transitions}"
                )
            schema[key] = {
                "shape": tuple(dataset.shape[1:]),
                "dtype": np.dtype(dataset.dtype),
            }
        if schema["pixels"]["shape"] != (128, 128, 3):
            raise ValueError(f"Unexpected agent camera shape: {path}")
        if schema["wrist_pixels"]["shape"] != (128, 128, 3):
            raise ValueError(f"Unexpected wrist camera shape: {path}")
        if schema["action"]["shape"] != (7,):
            raise ValueError(f"Unexpected action shape: {path}")
        if schema["proprio"]["shape"] != (9,):
            raise ValueError(f"Unexpected proprio shape: {path}")

    metadata = manifest["episodes_metadata"]
    if len(metadata) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} metadata records in {sidecar}"
        )
    scheduled = sum(len(row["burst_schedule"]) for row in metadata)
    started = sum(int(row["bursts_started"]) for row in metadata)
    if started > scheduled:
        raise ValueError(
            f"More bursts started than scheduled in {path}: "
            f"scheduled={scheduled}, started={started}"
        )
    seeds = [int(row["seed"]) for row in metadata]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate episode seeds inside {path}")
    return {
        "path": path.resolve(),
        "manifest": manifest,
        "lengths": lengths,
        "transitions": transitions,
        "schema": schema,
        "metadata": metadata,
        "scheduled_bursts": scheduled,
        "started_bursts": started,
        "successes": sum(bool(row["success"]) for row in metadata),
        "seeds": seeds,
    }


def require_matching_schema(reference: dict, current: dict) -> None:
    for key in STEP_KEYS:
        left = reference[key]
        right = current[key]
        if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
            raise ValueError(f"Schema mismatch for {key}: {left} vs {right}")


def create_vds(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    total = sum(row["transitions"] for row in records)
    reference = records[0]["schema"]

    with h5py.File(temporary, "w", libver="latest") as destination:
        for key in STEP_KEYS:
            shape = reference[key]["shape"]
            dtype = reference[key]["dtype"]
            layout = h5py.VirtualLayout(shape=(total, *shape), dtype=dtype)
            cursor = 0
            for row in records:
                source_shape = (row["transitions"], *shape)
                # Relative source paths keep the VDS portable when the whole
                # dataset directory is moved to another machine.
                source_path = os.path.relpath(row["path"], output.parent)
                source = h5py.VirtualSource(source_path, key, shape=source_shape)
                layout[cursor : cursor + row["transitions"]] = source
                cursor += row["transitions"]
            destination.create_virtual_dataset(key, layout)

        lengths = np.concatenate([row["lengths"] for row in records]).astype(
            np.int32
        )
        offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(lengths[:-1], dtype=np.int64),
            ]
        )
        destination.create_dataset("ep_len", data=lengths)
        destination.create_dataset("ep_offset", data=offsets)
        destination.attrs["complete"] = True
        destination.attrs["virtual"] = True
        destination.attrs["source_shards"] = len(records)
        destination.attrs["episodes"] = len(lengths)
        destination.attrs["transitions"] = total
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Shard root; repeat to build one zero-copy VDS over multiple roots.",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--verify-sha", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Build only the listed splits; defaults to train/validation/test.",
    )
    parser.add_argument(
        "--expected-episodes",
        nargs="*",
        default=(),
        metavar="SPLIT=COUNT",
        help="Optional split cardinality checks, e.g. train=32000 test=2000.",
    )
    parser.add_argument("--require-task-balance", action="store_true")
    args = parser.parse_args()

    expected_episodes = {}
    for item in args.expected_episodes:
        split, value = item.split("=", 1)
        if split not in SPLITS:
            raise ValueError(f"Unknown split in --expected-episodes: {split}")
        expected_episodes[split] = int(value)

    all_paths = []
    for data_root in args.data_root:
        all_paths.extend(
            path
            for path in data_root.glob("*/*/*.hdf5")
            if path.relative_to(data_root).parts[0] in args.splits
        )
    all_paths = sorted(all_paths)
    if not all_paths:
        raise ValueError(f"No HDF5 shards found under {args.data_root}")

    records = [load_shard(path, args.verify_sha) for path in all_paths]
    reference = records[0]["schema"]
    for row in records[1:]:
        require_matching_schema(reference, row["schema"])

    seen_seeds: set[int] = set()
    summary = {
        "protocol": "scene2_oracle_beta_burst_v1",
        "verify_sha": args.verify_sha,
        "splits": {},
    }
    for split in args.splits:
        split_records = [
            row for row in records if row["manifest"]["split"] == split
        ]
        if not split_records:
            raise ValueError(f"No shards found for requested split: {split}")
        # Interleave tasks at shard granularity. Besides avoiding a long
        # single-task prefix, this makes bounded validation deterministic and
        # task-covering without materializing a second shuffled dataset.
        task_order = {task: index for index, task in enumerate(TASKS)}
        split_records.sort(
            key=lambda row: (
                int(row["manifest"]["shard_id"]),
                task_order[row["manifest"]["task"]],
            )
        )
        episodes = sum(len(row["lengths"]) for row in split_records)
        if split in expected_episodes and episodes != expected_episodes[split]:
            raise ValueError(
                f"{split}: expected {expected_episodes[split]} episodes, got {episodes}"
            )

        per_task = {}
        for task in TASKS:
            task_rows = [
                row for row in split_records if row["manifest"]["task"] == task
            ]
            count = sum(len(row["lengths"]) for row in task_rows)
            per_task[task] = count
        if args.require_task_balance and len(set(per_task.values())) != 1:
            raise ValueError(f"{split} is not task balanced: {per_task}")

        split_seeds = {
            seed for row in split_records for seed in row["seeds"]
        }
        if len(split_seeds) != episodes:
            raise ValueError(f"Duplicate seeds in {split}")
        if seen_seeds.intersection(split_seeds):
            raise ValueError(f"Episode seed leakage into {split}")
        seen_seeds.update(split_seeds)

        output = args.out_root / f"{split}.h5"
        create_vds(split_records, output)
        split_map = []
        episode_index = 0
        for row in split_records:
            for local_episode, metadata in enumerate(row["metadata"]):
                split_map.append(
                    {
                        "episode_index": episode_index,
                        "task": row["manifest"]["task"],
                        "task_id": int(row["manifest"]["task_id"]),
                        "seed": int(metadata["seed"]),
                        "length": int(row["lengths"][local_episode]),
                        "success": bool(metadata["success"]),
                        "source_shard": os.path.relpath(
                            row["path"], args.out_root
                        ),
                        "source_episode": local_episode,
                    }
                )
                episode_index += 1
        map_path = args.out_root / f"{split}_episodes.json"
        map_path.write_text(json.dumps(split_map, indent=2))
        successes = sum(row["successes"] for row in split_records)
        scheduled = sum(row["scheduled_bursts"] for row in split_records)
        started = sum(row["started_bursts"] for row in split_records)
        summary["splits"][split] = {
            "shards": len(split_records),
            "episodes": episodes,
            "transitions": sum(row["transitions"] for row in split_records),
            "successes": successes,
            "success_rate": successes / episodes,
            "scheduled_bursts": scheduled,
            "started_bursts": started,
            "all_bursts_triggered": scheduled == started,
            "untriggered_bursts": scheduled - started,
            "per_task_episodes": per_task,
            "vds": output.name,
            "vds_bytes": output.stat().st_size,
            "episode_map": map_path.name,
        }

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

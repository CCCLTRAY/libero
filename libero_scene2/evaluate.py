#!/usr/bin/env python3
"""Replay-certified evaluation for LIBERO Scene-2 controllers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np

from libero_scene2.collect import TASKS
from libero_scene2.runtime import (
    make_env,
    observation_proprio,
    resolve_bddl,
    validate_runtime,
)


TASK_NAMES = tuple(TASKS)


@dataclass(frozen=True)
class ControllerContext:
    task: str
    instruction: str
    seed: int
    horizon: int
    goal_agent: np.ndarray
    goal_wrist: np.ndarray


class Controller(Protocol):
    def reset(self, context: ControllerContext) -> None: ...

    def act(self, observation: dict[str, Any]) -> np.ndarray: ...


class ReplayController:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        self.index = 0

    def reset(self, context: ControllerContext) -> None:
        self.index = 0

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if self.index >= len(self.actions):
            raise IndexError("Replay controller exhausted its recorded actions")
        action = self.actions[self.index]
        self.index += 1
        return action.copy()


def parse_indices(spec: str | None, total: int, limit: int | None) -> list[int]:
    if spec:
        indices = [int(value) for value in spec.split(",")]
    else:
        stop = total if limit is None else min(total, limit)
        indices = list(range(stop))
    if any(index < 0 or index >= total for index in indices):
        raise IndexError(f"Episode indices outside [0, {total}): {indices}")
    return indices


def episode_slice(handle: h5py.File, index: int) -> slice:
    start = int(handle["ep_offset"][index])
    length = int(handle["ep_len"][index])
    return slice(start, start + length)


def observation_errors(
    observation: dict[str, Any], handle: h5py.File, row: int
) -> dict[str, float]:
    agent = np.asarray(observation["agentview_image"], dtype=np.int16)
    wrist = np.asarray(
        observation["robot0_eye_in_hand_image"], dtype=np.int16
    )
    reference_agent = np.asarray(handle["pixels"][row], dtype=np.int16)
    reference_wrist = np.asarray(handle["wrist_pixels"][row], dtype=np.int16)
    proprio = observation_proprio(observation)
    reference_proprio = np.asarray(handle["proprio"][row], dtype=np.float32)
    return {
        "agent_pixel_mae": float(np.abs(agent - reference_agent).mean()),
        "agent_pixel_max": float(np.abs(agent - reference_agent).max()),
        "wrist_pixel_mae": float(np.abs(wrist - reference_wrist).mean()),
        "wrist_pixel_max": float(np.abs(wrist - reference_wrist).max()),
        "proprio_max": float(np.abs(proprio - reference_proprio).max()),
    }


def load_controller(spec: str, context: ControllerContext) -> Controller:
    if ":" not in spec:
        raise ValueError(
            "External controller must be MODULE:FACTORY; "
            "the factory receives ControllerContext"
        )
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    controller = factory(context)
    if not hasattr(controller, "reset") or not hasattr(controller, "act"):
        raise TypeError("Controller must implement reset(context) and act(observation)")
    return controller


def replay_certificate(
    *,
    handle: h5py.File,
    episode: slice,
    task: str,
    seed: int,
    stride: int,
    prefix_steps: int,
    pixel_mae_tolerance: float,
    proprio_tolerance: float,
) -> dict[str, Any]:
    env = make_env(bddl_file=resolve_bddl(TASKS[task]["bddl"]))
    env.seed(seed)
    observation = env.reset()
    rollout_maxima = {
        "agent_pixel_mae": 0.0,
        "agent_pixel_max": 0.0,
        "wrist_pixel_mae": 0.0,
        "wrist_pixel_max": 0.0,
        "proprio_max": 0.0,
    }
    prefix_maxima = dict(rollout_maxima)
    success = False
    try:
        for local_step, row in enumerate(range(episode.start, episode.stop)):
            should_compare = (
                local_step < prefix_steps
                or local_step % stride == 0
                or row == episode.stop - 1
            )
            if should_compare:
                errors = observation_errors(observation, handle, row)
                rollout_maxima = {
                    key: max(rollout_maxima[key], value)
                    for key, value in errors.items()
                }
                if local_step < prefix_steps:
                    prefix_maxima = {
                        key: max(prefix_maxima[key], value)
                        for key, value in errors.items()
                    }
            action = np.asarray(handle["action"][row], dtype=np.float32)
            observation, _, _, _ = env.step(action)
            success = bool(env.check_success())
            if success:
                break
    finally:
        env.close()
    reference_success = bool(np.asarray(handle["done"][episode]).any())
    passed = (
        success == reference_success
        and prefix_maxima["agent_pixel_mae"] <= pixel_mae_tolerance
        and prefix_maxima["wrist_pixel_mae"] <= pixel_mae_tolerance
        and prefix_maxima["proprio_max"] <= proprio_tolerance
    )
    return {
        "passed": passed,
        "success": success,
        "reference_success": reference_success,
        "prefix_steps": min(prefix_steps, episode.stop - episode.start),
        "prefix": prefix_maxima,
        "full_replay_drift": rollout_maxima,
    }


def run_controller(
    *,
    controller: Controller,
    context: ControllerContext,
    task: str,
    max_steps: int,
) -> dict[str, Any]:
    env = make_env(bddl_file=resolve_bddl(TASKS[task]["bddl"]))
    env.seed(context.seed)
    observation = env.reset()
    controller.reset(context)
    success = False
    steps = 0
    try:
        for steps in range(1, max_steps + 1):
            action = np.asarray(controller.act(observation), dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"Controller returned invalid action: {action}")
            observation, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
            success = bool(env.check_success())
            if success:
                break
    finally:
        env.close()
    return {"success": success, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--controller",
        default="replay",
        help="replay or MODULE:FACTORY",
    )
    parser.add_argument("--episodes", help="Comma-separated episode indices")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--certificate-stride", type=int, default=10)
    parser.add_argument("--certificate-prefix-steps", type=int, default=25)
    parser.add_argument("--pixel-mae-tolerance", type=float, default=1.0)
    parser.add_argument("--proprio-tolerance", type=float, default=1e-5)
    parser.add_argument("--allow-runtime-mismatch", action="store_true")
    parser.add_argument("--skip-certificate", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    versions = validate_runtime(strict=not args.allow_runtime_mismatch)
    results = []
    with h5py.File(args.data, "r", swmr=True) as handle:
        total = len(handle["ep_len"])
        indices = parse_indices(args.episodes, total, args.limit)
        for index in indices:
            episode = episode_slice(handle, index)
            task_id = int(handle["task_id"][episode.start])
            task = TASK_NAMES[task_id]
            seed = int(handle["episode_seed"][episode.start])
            context = ControllerContext(
                task=task,
                instruction=TASKS[task]["instruction"],
                seed=seed,
                horizon=episode.stop - episode.start,
                goal_agent=np.asarray(handle["pixels"][episode.stop - 1]),
                goal_wrist=np.asarray(handle["wrist_pixels"][episode.stop - 1]),
            )
            certificate = None
            if not args.skip_certificate:
                certificate = replay_certificate(
                    handle=handle,
                    episode=episode,
                    task=task,
                    seed=seed,
                    stride=args.certificate_stride,
                    prefix_steps=args.certificate_prefix_steps,
                    pixel_mae_tolerance=args.pixel_mae_tolerance,
                    proprio_tolerance=args.proprio_tolerance,
                )
                if not certificate["passed"]:
                    raise RuntimeError(
                        f"Replay certificate failed for episode {index}: {certificate}"
                    )
            actions = np.asarray(handle["action"][episode], dtype=np.float32)
            controller = (
                ReplayController(actions)
                if args.controller == "replay"
                else load_controller(args.controller, context)
            )
            outcome = run_controller(
                controller=controller,
                context=context,
                task=task,
                max_steps=args.max_steps or len(actions),
            )
            result = {
                "episode": index,
                "task": task,
                "seed": seed,
                "reference_horizon": len(actions),
                "certificate": certificate,
                **outcome,
            }
            results.append(result)
            print(json.dumps(result), flush=True)

    summary = {
        "data": str(args.data),
        "controller": args.controller,
        "runtime": versions,
        "episodes": len(results),
        "successes": sum(row["success"] for row in results),
        "success_rate": float(np.mean([row["success"] for row in results])),
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("episodes", "successes", "success_rate")}, indent=2))


if __name__ == "__main__":
    main()

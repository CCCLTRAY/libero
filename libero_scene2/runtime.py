"""Pinned LIBERO runtime helpers shared by collection and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import platform
from typing import Any

import numpy as np


EXPECTED_RUNTIME = {
    "python": "3.10",
    "mujoco": "2.3.7",
    "robosuite": "1.4.0",
}


class TaskEnv:
    """Small adapter over a directly constructed LIBERO task environment."""

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def sim(self):
        return self.env.sim

    def seed(self, seed: int) -> None:
        self.env.seed(int(seed))

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray):
        return self.env.step(np.asarray(action, dtype=np.float32))

    def check_success(self) -> bool:
        return bool(self.env._check_success())

    def close(self) -> None:
        self.env.close()


def resolve_bddl(filename: str) -> Path:
    candidates = []
    if os.environ.get("LIBERO_BDDL_ROOT"):
        candidates.append(Path(os.environ["LIBERO_BDDL_ROOT"]))

    import libero.libero as libero_package

    candidates.append(Path(libero_package.__file__).resolve().parent / "bddl_files")
    try:
        from libero.libero import get_libero_path

        candidates.append(Path(get_libero_path("bddl_files")))
    except (OSError, TypeError, ValueError):
        pass

    for root in candidates:
        if root.is_dir():
            path = next(root.rglob(filename), None)
            if path is not None:
                return path
    raise FileNotFoundError(
        f"LIBERO BDDL not found: {filename}; searched {candidates}"
    )


def make_env(*, bddl_file: Path, camera_size: int = 128) -> TaskEnv:
    from libero.libero.envs import TASK_MAPPING

    problem_name = "libero_living_room_tabletop_manipulation"
    if problem_name not in TASK_MAPPING:
        raise KeyError(f"LIBERO task mapping lacks {problem_name!r}")
    kwargs = {
        "robots": ["Panda"],
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
        "bddl_file_name": str(bddl_file),
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": True,
        "camera_depths": False,
        "camera_names": ["robot0_eye_in_hand", "agentview"],
        "reward_shaping": True,
        "control_freq": 20,
        "camera_heights": int(camera_size),
        "camera_widths": int(camera_size),
        "camera_segmentations": None,
    }
    return TaskEnv(TASK_MAPPING[problem_name](**kwargs))


def runtime_versions() -> dict[str, str]:
    import mujoco
    import robosuite

    return {
        "python": platform.python_version(),
        "mujoco": str(mujoco.__version__),
        "robosuite": str(robosuite.__version__),
        "numpy": str(np.__version__),
    }


def validate_runtime(*, strict: bool = True) -> dict[str, str]:
    versions = runtime_versions()
    mismatches = {
        key: {"expected": expected, "actual": versions.get(key)}
        for key, expected in EXPECTED_RUNTIME.items()
        if not str(versions.get(key, "")).startswith(expected)
    }
    if strict and mismatches:
        raise RuntimeError(f"Runtime mismatch: {mismatches}")
    return versions


def observation_proprio(observation: dict) -> np.ndarray:
    values = []
    for key in ("robot0_joint_pos", "robot0_gripper_qpos"):
        if key in observation:
            values.append(
                np.asarray(observation[key], dtype=np.float32).reshape(-1)
            )
    if not values:
        return np.zeros(9, dtype=np.float32)
    output = np.concatenate(values)
    if len(output) < 9:
        output = np.pad(output, (0, 9 - len(output)))
    return output[:9].astype(np.float32)

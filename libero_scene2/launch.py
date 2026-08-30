#!/usr/bin/env python3
"""Launch balanced Scene-2 collection as independent resumable shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py

from libero_scene2.collect import TASKS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_shard(path: Path, episodes: int, verify_sha: bool) -> bool:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        manifest = json.loads(sidecar.read_text())
        if int(manifest["episodes"]) != episodes:
            return False
        with h5py.File(path, "r", swmr=True) as handle:
            if not bool(handle.attrs.get("complete", False)):
                return False
            if len(handle["ep_len"]) != episodes:
                return False
        return not verify_sha or sha256(path) == manifest["sha256"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_job(job: dict, gpu: str, verify_existing_sha: bool) -> dict:
    output = Path(job["output"])
    if valid_shard(output, job["episodes"], verify_existing_sha):
        return {**job, "status": "skipped"}
    if output.exists() or output.with_suffix(".json").exists():
        raise RuntimeError(
            f"Refusing to overwrite invalid existing shard: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / "logs" / f"{output.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "libero_scene2.collect",
        "--task",
        job["task"],
        "--split",
        job["split"],
        "--seed-start",
        str(job["seed_start"]),
        "--episodes",
        str(job["episodes"]),
        "--shard-id",
        str(job["shard_id"]),
        "--out",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(
        MUJOCO_GL="egl",
        CUDA_VISIBLE_DEVICES=gpu,
        MUJOCO_EGL_DEVICE_ID=gpu,
    )
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not valid_shard(
        output, job["episodes"], verify_sha=False
    ):
        raise RuntimeError(f"Collection failed; inspect {log}")
    return {**job, "status": "completed", "log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--episodes-per-shard", type=int, default=50)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--verify-existing-sha", action="store_true")
    args = parser.parse_args()

    if args.episodes_per_task <= 0 or args.episodes_per_shard <= 0:
        raise ValueError("Episode counts must be positive")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    workers = args.workers or len(gpus)
    if workers <= 0:
        raise ValueError("--workers must be positive")

    jobs = []
    for task_id, task in enumerate(TASKS):
        remaining = args.episodes_per_task
        shard_id = 0
        task_seed_base = args.seed_base + task_id * 1_000_000
        while remaining:
            episodes = min(args.episodes_per_shard, remaining)
            seed_start = task_seed_base + shard_id * args.episodes_per_shard
            filename = (
                f"{task}_{args.split}_shard{shard_id:04d}_"
                f"seed{seed_start}_n{episodes}.hdf5"
            )
            jobs.append(
                {
                    "task": task,
                    "split": args.split,
                    "shard_id": shard_id,
                    "seed_start": seed_start,
                    "episodes": episodes,
                    "output": str(args.out / args.split / task / filename),
                }
            )
            remaining -= episodes
            shard_id += 1

    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol": "scene2_oracle_beta_burst_v1",
        "split": args.split,
        "episodes_per_task": args.episodes_per_task,
        "episodes_per_shard": args.episodes_per_shard,
        "seed_base": args.seed_base,
        "gpus": gpus,
        "workers": workers,
        "jobs": jobs,
    }
    (args.out / f"{args.split}_collection_plan.json").write_text(
        json.dumps(plan, indent=2)
    )

    counts = {"completed": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_job,
                job,
                gpus[index % len(gpus)],
                args.verify_existing_sha,
            ): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            result = future.result()
            counts[result["status"]] += 1
            print(json.dumps(result), flush=True)
    print(json.dumps({"jobs": len(jobs), **counts}, indent=2))


if __name__ == "__main__":
    main()

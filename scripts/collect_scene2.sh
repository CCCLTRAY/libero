#!/usr/bin/env bash
set -euo pipefail

task="${1:-butter}"
split="${2:-train}"
seed_start="${3:-100000}"
episodes="${4:-50}"
shard_id="${5:-0}"
out_root="${6:-data}"

mkdir -p "${out_root}/${split}/${task}"
MUJOCO_GL="${MUJOCO_GL:-egl}" python -m libero_scene2.collect \
  --task "${task}" \
  --split "${split}" \
  --seed-start "${seed_start}" \
  --episodes "${episodes}" \
  --shard-id "${shard_id}" \
  --out "${out_root}/${split}/${task}/shard_${shard_id}.hdf5"

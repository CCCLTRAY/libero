# LIBERO Scene-2 World-Model Infrastructure

Minimal, model-agnostic infrastructure for collecting and evaluating
recoverable-intervention trajectories in five LIBERO Living Room Scene 2
tasks. The repository contains no model checkpoints and no training code.

**[Dataset download and usage](DATA.md)** | **[Official LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)**

## What is included

- A privileged closed-loop `Grasp -> PlaceIn` oracle.
- A unified behavior distribution with correlated action noise and occasional
  random-action bursts. The oracle replans from the disturbed state.
- Immutable HDF5 shards with per-shard SHA256 manifests.
- A zero-copy, relocatable HDF5 virtual-dataset builder.
- Replay-certified official-success evaluation with a small controller API.

The five tasks place alphabet soup, tomato sauce, butter, milk, or orange
juice into the Scene-2 basket. Each step stores agent and wrist RGB, the
executed and oracle actions, proprioception, official reward/success, and
diagnostic intervention metadata.

## Installation

The released data use Python 3.10, MuJoCo 2.3.7, robosuite 1.4.0, and a pinned
official LIBERO revision. The relevant Scene-2 source, BDDL, and assets were
hash-audited against the data-generation runtime.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

git clone https://github.com/CCCLTRAY/libero.git
cd libero
conda env create -f environment.yml
conda activate libero-scene2
pip install -e ../LIBERO
python scripts/check_runtime.py
```

Headless execution uses EGL:

```bash
export MUJOCO_GL=egl
```

## Collect data

One immutable shard contains an arbitrary contiguous range of seeds:

```bash
bash scripts/collect_scene2.sh butter train 100000 50 0 data
```

Equivalent explicit command:

```bash
python -m libero_scene2.collect \
  --task butter --split train --seed-start 100000 \
  --episodes 50 --shard-id 0 \
  --out data/train/butter/shard_0.hdf5
```

Collection writes `shard_0.hdf5` only after all episodes finish, then writes
`shard_0.json` with its schema, episode metadata, runtime, and SHA256 digest.
Different workers must receive disjoint seed ranges and shard ids.

For balanced multi-GPU collection, one command creates the deterministic plan,
assigns independent shards to workers, and resumes already valid shards:

```bash
python -m libero_scene2.launch \
  --out data --split train --episodes-per-task 1600 \
  --episodes-per-shard 50 --seed-base 1000000 \
  --gpus 0,1,2,3
```

Use disjoint `--seed-base` values for train, validation, and test. The emitted
`<split>_collection_plan.json` is the reproducibility manifest.

## Build a portable index

The index is an HDF5 virtual dataset: it duplicates metadata, not images.
Source paths are relative, so the whole data directory can move intact.

```bash
python -m libero_scene2.build_index \
  --data-root data \
  --out-root data/index \
  --splits train validation test \
  --verify-sha --require-task-balance
```

For the full release, add cardinality checks such as
`--expected-episodes train=8000 validation=1000 test=1000`.

## Verify a machine

Before evaluating a controller, replay one recorded episode. The evaluator
reconstructs the environment from its task and seed, replays the stored raw
actions, checks the official predicate, and compares RGB/proprio observations
against the dataset. It never restores a serialized MuJoCo state.

```bash
python -m libero_scene2.evaluate \
  --data smoke/butter_seed4910000.hdf5 \
  --controller replay --episodes 0
```

The certificate requires exact startup-domain agreement over the first 25
steps and matching full-replay official success. It also reports full-episode
RGB/proprio drift. Contact-rich MuJoCo trajectories can gradually diverge
across CPU models despite identical source and initial observations, so this
post-contact drift is diagnostic rather than a hard gate. A startup-domain or
official-success failure means the runtime does not match and model results
should not be compared.

## Evaluate a controller

Expose a factory as `MODULE:FUNCTION`. The factory receives task, instruction,
seed, horizon, and terminal goal images. The controller implements
`reset(context)` and `act(observation) -> float32[7]`.

```bash
python -m libero_scene2.evaluate \
  --data data/index/test.h5 \
  --controller examples.controller:make_controller \
  --limit 50 --out results/controller.json
```

By default, every selected episode first passes the replay certificate, then
the controller starts from the same fresh reset. Success is the official
LIBERO task predicate with early stopping. See `examples/controller.py` for
the complete adapter.

## Data release

The released dataset contains 8,000 training, 1,000 validation, and 1,000
test trajectories. See [DATA.md](DATA.md) for package names, integrity checks,
extraction, indexing, and the replay certificate. Large files are hosted
separately from GitHub.

Large shards are distributed separately from Git. A data release should keep
this layout unchanged:

```text
data/
  train/<task>/shard_*.{hdf5,json}
  validation/<task>/shard_*.{hdf5,json}
  test/<task>/shard_*.{hdf5,json}
  index/{train,validation,test}.h5
  index/{train,validation,test}_episodes.json
  index/audit_summary.json
```

The full training set can be regenerated from the public collector and seed
manifest. For collaboration, sharing the sealed test shards plus a small
smoke shard is sufficient to reproduce evaluation without distributing model
weights.

This repository includes one 9.46 MB successful butter trajectory solely for
the runtime certificate. It is not a benchmark result or a training set.

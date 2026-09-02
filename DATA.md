# LIBERO Scene-2 Dataset

This dataset contains recoverable-intervention trajectories for five LIBERO
Living Room Scene 2 tasks: placing alphabet soup, tomato sauce, butter, milk,
or orange juice in the basket.

## Dataset split

| Split | Episodes |
|---|---:|
| Train | 8,000 |
| Validation | 1,000 |
| Test | 1,000 |

Every episode stores agent-view and wrist-view RGB, executed and oracle
actions, robot proprioception, official reward/success, and intervention
metadata. Training and model selection must use only the train and validation
splits. The test split must not be used to select checkpoints, planner
parameters, CEM parameters, or thresholds.

## Download

Large files are hosted on Quark Drive. The public share URL will be added here.
The files currently reside in these release folders:

```text
/libero-worldmodel-20260831/
/cc18-upload/
```

For environment and evaluation reproduction, download:

| File | Size | Purpose |
|---|---:|---|
| `scene2-collab-eval-250-v1.tar` | 2.49 GB | Sealed evaluation set: 50 episodes per task |
| `collab-eval-manifest.json` | 1.4 KB | Episode counts and SHA256 manifest |
| `scene2_oracle_8k1k1k_indices.tar.gz` | 376 KB | Relocatable HDF5 virtual indices |

For training, additionally download:

| File | Size | Purpose |
|---|---:|---|
| `libero-scene2-8k-v1.tar` | 98.66 GB | Complete 8k/1k/1k dataset |
| `libero-scene2-8k-v1.tar.sha256` | 133 B | SHA256 checksum |

The older `libero-lewm-scene2-repro-v1.*` files are historical archives and
should not be used as the default release.

## Verify and extract

Verify the complete archive before extraction:

```bash
sha256sum libero-scene2-8k-v1.tar
```

Expected SHA256:

```text
c3d24ed8cd1bed75f59e98ece2af337f609b7d356845f2f564cd87c8b9b8fa95
```

Extract the complete dataset into this repository:

```bash
mkdir -p data
tar -xf libero-scene2-8k-v1.tar -C data
```

For the smaller sealed evaluation package:

```bash
mkdir -p data
tar -xf scene2-collab-eval-250-v1.tar -C data
```

The resulting layout must remain:

```text
data/
  train/<task>/shard_*.{hdf5,json}
  validation/<task>/shard_*.{hdf5,json}
  test/<task>/shard_*.{hdf5,json}
```

## Install the runtime

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

conda env create -f environment.yml
conda activate libero-scene2
pip install -e ../LIBERO
export MUJOCO_GL=egl
python scripts/check_runtime.py
```

The generation runtime uses Python 3.10, MuJoCo 2.3.7, and robosuite 1.4.0.

## Run the replay certificate

Do not evaluate a controller before the replay certificate passes:

```bash
python -m libero_scene2.evaluate \
  --data smoke/butter_seed4910000.hdf5 \
  --controller replay \
  --episodes 0
```

The evaluator performs a fresh reset, replays raw actions, checks RGB and
proprioception agreement, and verifies the official LIBERO success predicate.
It never restores a serialized MuJoCo state across machines.

## Build the portable index

```bash
python -m libero_scene2.build_index \
  --data-root data \
  --out-root data/index \
  --splits train validation test \
  --verify-sha \
  --require-task-balance \
  --expected-episodes train=8000 validation=1000 test=1000
```

The generated virtual datasets reference the source shards without copying
their images.

## Evaluate a controller

Expose a controller factory as `MODULE:FUNCTION`. The returned controller must
implement `reset(context)` and `act(observation) -> float32[7]`.

```bash
python -m libero_scene2.evaluate \
  --data data/index/test.h5 \
  --controller examples.controller:make_controller \
  --limit 50 \
  --out results/controller.json
```

Success is measured by the official task predicate with early stopping.

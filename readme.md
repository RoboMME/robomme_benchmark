## RoboMME: A Robotic Benchmark for Memory-Augmented Manipulation

![Robomme bench](assets/robomme_bench.jpg)

- [Announcements](#announcements)
- [Installation](#installation)
- [Running Examples](#running-examples)
- [Tasks](#tasks)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Announcements

[02/2026] We release RoboMME! It's a cognitive-motivated large-scale robotic benchmark for memory-augmented manipulation, spanning 4 task suites with a total of 16 carefully designed tasks.

## Installation

After cloning the repo, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
uv pip install -e .
```

## Running Examples

Run an environment with fixed predefined setups:

```bash
uv run scripts/run_example.py --action-space-type joint_angle --dataset test
```

- **Train dataset:** 100 episodes per env.
- **Val/test dataset:** 50 episodes per env.

We provide four action types: `joint_action`, `ee_pose`, `keypoint`, and `multi_choice`, e.g. predict continuous actions with `joint_action` or `ee_pose`, discrete waypoint actions with `keypoint`, or use `multi_choice` for VideoQA-style data.

## Datasets

### Training data

Training data can be downloaded [here](). There are 1,600 demonstrations in total (100 per task). The HDF5 format is described in [doc/h5_data_format.md](doc/h5_data_format.md).

After downloading, replay the dataset for a sanity check:

```bash
uv run scripts/dataset_replay.py --h5-data-dir=<your_downloaded_data_dir>
```

You can also re-generate your own HDF5 data (see scripts in `scripts/dev/xxxx`).

## Model Training

The [MME-VLA-Suite](https://github.com/RoboMME/MME-VLA-Suite) repo provides VLA model training and evaluation. Please check it out.

> **Note:** Currently, environment spawning is set up for imitation learning. We are working on extending it to support more general parallel environments for reinforcement learning.

## Tasks

We have four task suites, each with 4 tasks:

| Suite      | Focus             | Tasks                                                                 |
| ---------- | ----------------- | --------------------------------------------------------------------- |
| Counting   | Temporal memory   | BinFill, PickXtimes, SwingXtimes, StopCube                            |
| Permanence | Spatial memory    | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap         |
| Reference  | Object memory     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder         |
| Imitation  | Procedural memory | MoveCube, InsertPeg, PatternLock, RouteStick                          |

All tasks are defined in `src/robomme/robomme_env`.

## Troubleshooting

**Q1: RuntimeError: Create window failed: Renderer does not support display.**

A1: Use a physical display or set up a virtual display for GUI rendering (e.g. install a VNC server and set the `DISPLAY` variable correctly).

**Q2: Failure related to Vulkan installation.**

A2: We recommend reinstalling the NVIDIA driver and Vulkan packages. We use NVIDIA driver 570.211.01 and Vulkan 1.3.275.


## Acknowledgements


## Citation
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

Start an environment with a specificed setup.

```bash
uv run scripts/run_example.py --action-space-type joint_angle --dataset test --env-id PickXtimes --episode-idx 0
```
It'll generate a video in `sample_run_videos` directory for the rollout.

We provide four action types: `joint_action`, `ee_pose`, `keypoint`, and `multi_choice`.   
`joint_action` and `ee_pose` can be used to predict continuous actions. `keypoint` is used to predict discrete waypoint actions. `multi_choice` is designed for VideoQA problem.

> **Note:** Currenetly, only `joint_action` is verified. please use it rather than other types.

## Datasets

### Training data

Training data can be downloaded [here](https://huggingface.co/Yinpei/data_0214). There are 1,600 demonstrations in total (100 per task). The HDF5 format is described in [doc/h5_data_format.md](doc/h5_data_format.md).

> **Note:** Currenetly, the training data is not finalized, and has difference from the doc.

After downloading, replay the dataset for a sanity check:

```bash
uv run scripts/dataset_replay.py --h5-data-dir <your_downloaded_data_dir> --action-space-type joint_angle
```

### Data Generation
You can also re-generate your own HDF5 data with 
```
uv run scripts/dev/xxxx 
```
explain sth about parrellel generation (@hongze)

### Play with Online Demo
Start gradio GUI to play (@hongze)
```
```


## Model Training

The [MME-VLA-Suite](https://github.com/RoboMME/MME-VLA-Suite) repo provides VLA model training and evaluation. Please check it out.

> **Note:** Currently, environment spawning is set up for imitation learning. We are working on extending it to support more general parallel environments for reinforcement learning.

## Tasks

We have four task suites, each with 4 tasks:

| Suite      | Focus             | Task ID                                                                 |
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

A2: We recommend reinstalling the NVIDIA driver and Vulkan packages. We use NVIDIA driver 570.211.01 and Vulkan 1.3.275.  If it still does not work, you can switch to CPU rendering:
```
os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
os.environ['MUJOCO_GL'] = 'osmesa'
```



## Acknowledgements
This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085. We would also like to thank the wonderful [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## Citation

```
...
```
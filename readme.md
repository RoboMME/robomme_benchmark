# RoboMME: A Robotic Benchmark for Memory-Augmented Manipulation

![Robomme bench](assets/robomme_bench.jpg)

## 📢 Announcements

[03/2026] We release RoboMME! It's a cognitive-motivated large-scale robotic benchmark for memory-augmented manipulation, spanning 4 task suites with a total of 16 carefully designed tasks.

## 📦 Installation

After cloning the repo, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
uv pip install -e .
```

## 🚀 Quick Start

Start an environment with a specified setup:

```bash
uv run scripts/run_example.py --action-space-type joint_angle --task-id PickXtimes
```

This generates a rollout video in the `sample_run_videos` directory.

We provide four action types: `joint_action`, `ee_pose`, `waypoint`, and `multi_choice`, e.g. predict continuous actions with `joint_action` or `ee_pose`, discrete waypoint actions with `waypoint`, or use `multi_choice` for VideoQA-style problems.

## 📁 Benchmark

### 🤖 Tasks

We have four task suites, each with 4 tasks:

| Suite      | Focus             | Task ID                                                                 |
| ---------- | ----------------- | --------------------------------------------------------------------- |
| Counting   | Temporal memory   | BinFill, PickXtimes, SwingXtimes, StopCube                            |
| Permanence | Spatial memory    | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap         |
| Reference  | Object memory     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder         |
| Imitation  | Procedural memory | MoveCube, InsertPeg, PatternLock, RouteStick                          |

All tasks are defined in `src/robomme/robomme_env`. Detailed description can be found in our paper appendix. 

### 📥 Training Data

Training data can be downloaded [here](https://huggingface.co/datasets/Yinpei/robomme_data). There are 1,600 demonstrations in total (100 per task). The HDF5 format is described in [doc/h5_data_format.md](doc/h5_data_format.md).

After downloading, replay the dataset for a sanity check:

```bash
uv run scripts/dataset_replay.py --h5-data-dir <your_downloaded_data_dir>
```

### 📊 Evaluation

To evaluate on test set, set the `dataset` argument of `BenchmarkEnvBuilder`:

```python
task_id = "PickXtimes"
episode_idx = 0
env_builder = BenchmarkEnvBuilder(
    env_id=task_id,
    dataset="test",
    ...
)

env = env_builder.make_env_for_episode(episode_idx)
obs, info = env.reset() # inital step
...
obs, _, terminated, truncated, info = env.step(action) # each step
```
Train split has 100 episodes. Val/test split has 50 episodes. All seeds are fixed for benchmarking.

The environment input/output format is provided in [doc/env_format.md](doc/env_format.md)

### 🔧 Data Generation

You can also re-generate your own HDF5 data via parallel processing using
@hongze
```bash
uv run scripts/dev/xxxx
```


## 🧠 Model Training

### MME-VLA-Suite

The [MME Policy Learning](https://github.com/RoboMME/robomme_policy_learning) repo provides MME-VLA model training and evaluation used in our paper. It contains a family of  memory-aguemnetd VLA models built on [pi05](https://github.com/Physical-Intelligence/openpi) backbone and our implementation of [MemER](https://jen-pan.github.io/memer/). 

### Prior Methods

**MemER**: The [MME Policy Learning](https://github.com/RoboMME/robomme_policy_learning) repo also provides our implementation of the [MemER](https://jen-pan.github.io/memer/), using the same GroundSG policy model as in MME-VLA.

**SAM2Act+**: The [RoboMME_SAM2Act](https://github.com/RoboMME/SAM2Act) repo provides our implementation adapt from the [SAM2Act](https://github.com/sam2act/sam2act) repo.

**MemoryVLA**: The [RoboMME_MemoryVLA](https://github.com/RoboMME/MemoryVLA) repo provides our implementation adapt from the [MemoryVLA](https://github.com/shihao1895/MemoryVLA) repo.
 
**Diffusion Policy**: The [RoboMME_DP](https://github.com/RoboMME/DiffusionPolicy) repo provides our implementation adapt from the [diffusion_policy](https://github.com/real-stanford/diffusion_policy) repo.


<!-- ## TODO List
[] Release data generation scripts.  
[] Release point clound reconstruciton script.  
[] Currently, environment spawning is set up only for imitation learning. We are working on extending it to support more general parallel environments for reinforcement learning in the future. -->

## 🔧 Troubleshooting

**Q1: RuntimeError: Create window failed: Renderer does not support display.**

A1: Use a physical display or set up a virtual display for GUI rendering (e.g. install a VNC server and set the `DISPLAY` variable correctly).

**Q2: Failure related to Vulkan installation.**

A2: We recommend reinstalling the NVIDIA driver and Vulkan packages. We use NVIDIA driver 570.211.01 and Vulkan 1.3.275. If it still does not work, you can switch to CPU rendering:

```python
os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
os.environ['MUJOCO_GL'] = 'osmesa'
```


## 🙏 Acknowledgements

This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085. We would also like to thank the wonderful [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## 📄 Citation

```
...
```
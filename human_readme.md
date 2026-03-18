# RoboMME: A Robotic Benchmark for Memory-Augmented Manipulation

![Robomme bench](assets/robomme_bench.jpg)

## 📢 Announcements

[03/2026] We are thrilled to release RoboMME, the first large-scale robotic benchmark dedicated to memory-augmented manipulation! Spanning 4 cognitively motivated task suites with 16 carefully designed tasks, RoboMME pushes robots to remember, reason, and act.

## 📦 Installation

After cloning the repo, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
uv pip install -e .
```

## Gradio App and ZeroGPU

The Gradio app now targets native Hugging Face Gradio Spaces instead of Docker Spaces.

```bash
uv sync
uv run python app.py
```

Local runs default to CPU fallback rendering. In Hugging Face ZeroGPU environments, the Space entrypoint remains root `app.py`, GPU is preserved at runtime, and GPU is requested only around the heavy environment functions via `@spaces.GPU`.

Optional metadata override:

```bash
ROBOMME_METADATA_ROOT=src/robomme/env_metadata/train uv run python app.py
```

Notes:
- The Spaces entrypoint is root `app.py`.
- Hugging Face Gradio Spaces install Python dependencies from root `requirements.txt`.
- Hugging Face Gradio Spaces install Debian/system dependencies from root `packages.txt`.
- Existing `uv` workflow for training/testing remains unchanged.
- Space metadata is configured via root `README.md` with `sdk: gradio`.
- `Dockerfile` and `.dockerignore` are not part of the native Gradio Space startup path.

## 🚀 Quick Start

Start an environment with a specified setup:

```bash
uv run scripts/run_example.py
```

This generates a rollout video in the `sample_run_videos` directory.

We provide four action types: `joint_action`, `ee_pose`, `waypoint`, and `multi_choice`, e.g., predict continuous actions with `joint_action` or `ee_pose`, discrete waypoint actions with `waypoint`, or use `multi_choice` for VideoQA-style problems.

## 📁 Benchmark

### 🤖 Tasks

We have four task suites, each with 4 tasks:

| Suite      | Focus             | Task ID                                                                 |
| ---------- | ----------------- | --------------------------------------------------------------------- |
| Counting   | Temporal memory   | BinFill, PickXtimes, SwingXtimes, StopCube                            |
| Permanence | Spatial memory    | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap         |
| Reference  | Object memory     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder         |
| Imitation  | Procedural memory | MoveCube, InsertPeg, PatternLock, RouteStick                          |

All tasks are defined in `src/robomme/robomme_env`. A detailed description can be found in our paper appendix.

### 📥 Training Data

Training data can be downloaded [here](https://huggingface.co/datasets/Yinpei/robomme_data). There are 1,600 demonstrations in total (100 per task). The HDF5 format is described in [doc/h5_data_format.md](doc/h5_data_format.md).

After downloading, replay the dataset for a sanity check:

```bash
uv run scripts/dataset_replay.py --h5-data-dir <your_downloaded_data_dir>
```

### 📊 Evaluation

To evaluate on the test set, set the `dataset` argument of `BenchmarkEnvBuilder`:

```python
task_id = "PickXtimes"
episode_idx = 0
env_builder = BenchmarkEnvBuilder(
    env_id=task_id,
    dataset="test",
    ...
)

env = env_builder.make_env_for_episode(episode_idx)
obs, info = env.reset() # initial step
...
obs, _, terminated, truncated, info = env.step(action) # each step
```
The train split has 100 episodes. The val/test splits each have 50 episodes. All seeds are fixed for benchmarking.

The environment input/output format is described in [doc/env_format.md](doc/env_format.md).

> Currently, environment spawning is set up only for imitation learning. We are working on extending it to support more general parallel environments for reinforcement learning in the future.

### 🔧 Data Generation

You can also re-generate your own HDF5 data via parallel processing using
@hongze
```bash
uv run scripts/dev/xxxx
```


## 🧠 Model Training

### 🌟 MME-VLA-Suite

The [MME Policy Learning](https://github.com/RoboMME/robomme_policy_learning) repo provides MME-VLA model training and evaluation used in our paper. It contains a family of memory-augmented VLA models built on [pi05](https://github.com/Physical-Intelligence/openpi) backbone and our implementation of [MemER](https://jen-pan.github.io/memer/). 

### 📚 Prior Methods

**MemER**: The [MME Policy Learning](https://github.com/RoboMME/robomme_policy_learning) repo also provides our implementation of the [MemER](https://jen-pan.github.io/memer/), using the same GroundSG policy model as in MME-VLA.

**SAM2Act+**: The [RoboMME_SAM2Act](https://github.com/RoboMME/SAM2Act) repo provides our implementation adapted from the [SAM2Act](https://github.com/sam2act/sam2act) repo.

**MemoryVLA**: The [RoboMME_MemoryVLA](https://github.com/RoboMME/MemoryVLA) repo provides our implementation adapted from the [MemoryVLA](https://github.com/shihao1895/MemoryVLA) repo.
 
**Diffusion Policy**: The [RoboMME_DP](https://github.com/RoboMME/DP) repo provides our implementation adapted from the [diffusion_policy](https://github.com/real-stanford/diffusion_policy) repo.



## 🏆 Submit Your Models
Want to add your model? Download the [dataset](https://huggingface.co/datasets/Yinpei/robomme_data) from Hugging Face, run evaluation using our [eval scripts](scripts/evaluation.py), then submit a PR with your results by adding `<your_model>.md` to the `doc/submission/` [directory](https://github.com/RoboMME/robomme_benchmark/tree/main/doc/submission). We will review it and update our leaderboard.


## 🔧 Troubleshooting

**Q1: RuntimeError: Create window failed: Renderer does not support display.**

A1: Use a physical display or set up a virtual display for GUI rendering (e.g. install a VNC server and set the `DISPLAY` variable correctly).

**Q2: Failure related to ManiSkill/SAPIEN rendering initialization.**

A2: For local development, keep the process on CPU-only defaults unless you are explicitly debugging a GPU path. In Hugging Face ZeroGPU Spaces, do not force `CUDA_VISIBLE_DEVICES=-1`; the Space runtime should preserve GPU visibility and let `@spaces.GPU` allocate hardware only for decorated functions.

Recommended local CPU fallback:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
os.environ.setdefault('ROBOMME_RENDER_BACKEND', 'cpu')
os.environ.pop('SAPIEN_RENDER_DEVICE', None)
os.environ.pop('NVIDIA_DRIVER_CAPABILITIES', None)
os.environ.pop('MUJOCO_GL', None)
os.environ.setdefault('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
```


## 🙏 Acknowledgements

This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085. We would also like to thank the wonderful [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## 📄 Citation

```
...
```

# Embodied Astra on RoboMME-Plus

Uses the upstream waypoint policy with the benchmark motion planner.
One GPU runs the simulator; inference uses the Astra API. The default is
zero-shot transfer (ZS); the one-shot variant uses AfD with no training rollouts.

## Setup

From the repository root:

```bash
git submodule update --init scripts/plus_eval/agentic_methods/astra/embodied/embodied-astra-robomme
uv sync --locked --no-dev --python 3.11
export SIM_PYTHON="$PWD/.venv/bin/python"
uv pip install --python "$SIM_PYTHON" -r scripts/plus_eval/agentic_methods/astra/requirements.txt
```

Run this Python directly afterward: `uv sync` can undo the dependency overrides.
Set `OPENAI_API_KEY` in your environment, then run:

```bash
CUDA_VISIBLE_DEVICES=0 "$SIM_PYTHON" scripts/plus_eval/agentic_methods/astra/embodied/eval_plus.py   --tasks BinFill --episodes 0 --prompt-variant lr_note   --output-dir runs/embodied-astra-smoke
```

Omit the task/episode filters for all 800 episodes. See [shared options](../../../../../doc/robomme_plus/robomme_plus_baseline_results.md#running-baselines)
for resume, rerun and recorded outputs. The model defaults to `gpt-6-astra` and
frame stride to 3. `lr_note` supplies the left/right convention used in our runs.

## Image and step budgets

The prompt receives `max_steps` from the environment. The initial request is
capped at 750 images; longer demonstrations are sampled uniformly with endpoints
preserved. The conversation is capped at 1,500 cumulative images, including
retained history. Reaching that limit ends the episode as a scored model stop.
The upstream 150-call and three-no-op limits also apply. Sampling indices and
stop reasons are recorded per episode.

## One-shot

Download the [H5 dataset](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5):

```bash
hf download Yinpei/robomme_plus_data_h5 --repo-type dataset --local-dir data/plus_train
CUDA_VISIBLE_DEVICES=0 "$SIM_PYTHON" scripts/plus_eval/agentic_methods/astra/embodied/eval_plus_fs.py   --tasks BinFill --episodes 0 --training-root data/plus_train   --training-context images --output-dir runs/embodied-astra-one-shot
```

Each test episode receives the same task's medium training example; InsertPeg
uses episode 1. MoveCube receives no training example. Its current episode already includes a video
demonstration showing how to move the cube using the required method.
Training frames contain front and wrist views
side by side, with demonstration/execution phases labeled. The default prompt
uses images and the training goal; recorded actions and subgoal annotations are excluded.
`--training-context state` additionally includes observed robot/camera state.

The first request permits 1,100 images, allocating up to 550 to training by
default. All retained images count toward the 1,500-image conversation limit.
`--training-max-images` and `--training-difficulty` control example selection.
Use a separate output directory from zero-shot or another context setting.
Validate the training files without GPU/API calls with:

```bash
"$SIM_PYTHON" scripts/plus_eval/agentic_methods/astra/embodied/eval_plus_fs.py   --training-root data/plus_train --validate-training   --output-dir runs/one-shot-data-check
```

[Link finalized rollouts](../../../results/README.md) into the ignored `runs/` directory; videos and logs remain outside Git.

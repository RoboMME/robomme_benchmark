# PonderPounce on RoboMME-Plus

Calls upstream [`PonderPounceRoboMMEServer`](https://github.com/worv-ai/PonderPounce/blob/723df35762bb641e1d520e4fa9359b98644adc21/ponderpounce/eval/robomme_server.py)
callbacks directly. Ponder (Qwen3.5-9B), Pounce (Pi0.5) and the simulator share one
GPU; allow roughly 35 GB of free GPU memory. No Docker or separate server is needed.
See the [shared protocol](../README.md) for the 800-episode run options, outputs
and metrics.

## Setup

From the **RoboMME repository root**, create a dedicated Python 3.11 environment:

```bash
export ROBOMME_ROOT="$PWD"
export POLICY_REPO="$ROBOMME_ROOT/scripts/plus_eval/ponderpounce/PonderPounce"
export PP_PYTHON="$ROBOMME_ROOT/scripts/plus_eval/ponderpounce/.venv/bin/python"
git submodule update --init scripts/plus_eval/ponderpounce/PonderPounce
uv venv scripts/plus_eval/ponderpounce/.venv --python 3.11
uv pip install --python "$PP_PYTHON" -e "$POLICY_REPO" -e "$ROBOMME_ROOT" \
  torch==2.9.1 torchvision==0.24.1 transformers==5.13.1 numpy==1.26.4 \
  'mani-skill @ git+https://github.com/YinpeiDai/ManiSkill.git@07be6fbc66350ddca200abfb0a11b692f078f7fd' \
  sapien==3.0.3 opencv-python==4.11.0.86 'imageio[ffmpeg]'
```

Run `"$PP_PYTHON"`
directly; upstream `uv sync` selects CUDA 13. Optional Qwen fast kernels are not
required; upstream's PyTorch fallback preserves inference behavior at lower throughput.

## Checkpoints

Download the released checkpoint, which includes both systems and normalization statistics:

```bash
export PONDERPOUNCE_CHECKPOINT="$ROBOMME_ROOT/runs/ckpts/ponderpounce-9b-robomme"
"$(dirname "$PP_PYTHON")/hf" download worv-ai/ponderpounce-9b-robomme \
  config.json model.safetensors norm_stats.json \
  --revision 5691e05614e8e6581d6358959e60561acfa73534 \
  --local-dir "$PONDERPOUNCE_CHECKPOINT"
```

`--checkpoint` overrides `PONDERPOUNCE_CHECKPOINT`. The loader also caches
configuration/tokenizer/processor files for `Qwen/Qwen3.5-9B` and
`google/paligemma-3b-pt-224`; their backbone weights are not needed.

## Run

Run one BinFill episode:

```bash
cd "$ROBOMME_ROOT"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$PP_PYTHON" scripts/plus_eval/ponderpounce/eval_plus.py \
  --tasks BinFill --episodes 0 --model-seed 0 \
  --output-dir runs/ponderpounce-plus-smoke
```

## Adapter notes

- Preserves upstream prompts, demonstration sampling, normalization and scheduling:
  a 20 Hz control clock, 1-second system periods, 300 ms cognition delay, initial
  hold, 20-action chunks, BF16 and 10 denoising steps.
- Corrects the initial hold's gripper units: measured finger positions in metres
  are converted using the Panda controller's physical limits to its normalized
  action range. An open gripper stays open while waiting for the first prediction.
  Learned action chunks retain their original gripper commands; the stick tasks
  keep their unused gripper channel. The correction and limits are recorded in
  `config.json` under `policy_configuration`.
- Each episode gets a fresh session. The historical `ood_<Task>_ep<N>` ID is
  retained because upstream derives action noise from it, the episode counter
  and `--model-seed`; renaming it would change predictions.
- Ponder reserves space for the next response before appending an observation.
  When its context would exceed the checkpoint/cache limit (16,384 tokens for
  the released model), it discards the oldest complete execution queries and
  rebuilds both the full-attention and recurrent caches. It retains the task,
  recent observations and latest generated plan when space permits. Oversized
  demonstrations are uniformly downsampled as complete images. A rebuild frees
  roughly a quarter of the window to avoid rebuilding on every query.
  `--ponder-context-tokens` optionally sets a smaller limit for validation.
  The policy/cap are recorded in `config.json`; each Ponder output includes the
  truncation count and trace of retained queries and demonstration frame indices.
- Ponder records images/demo input, appended token segments/cache rollbacks,
  readable prompts, generated subgoals/reasoning, gate scores and cognition.
  Previous call IDs link persistent context. Pounce records inputs, cognition age,
  prompt IDs, noise and action chunks.

Omit task/episode filters for all 800 episodes. See [shared run options](../README.md) for resume, rerun and outputs.

[Link finalized rollouts](../results/README.md) into the ignored `runs/` directory; videos and logs remain outside Git.

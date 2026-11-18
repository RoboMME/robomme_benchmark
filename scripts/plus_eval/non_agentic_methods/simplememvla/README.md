# SimpleMemVLA on RoboMME-Plus

Uses upstream [`build_policy()` and batched inference](https://github.com/YinpeiDai/SimpleMemVLA/blob/c564c17d276d7294200122b286c21901a3bfb99f/robomme_sim/eval_success.py),
including history sampling, prompts, subtask generation and action prediction.
The model and simulator share one GPU/process. See the [shared run instructions](../../../../doc/robomme_plus/robomme_plus_baseline_results.md#running-baselines)
for the 800-episode run options, outputs and metrics.

## Setup

From the repository root, initialize the policy submodule:

```bash
export ROBOMME_ROOT="$PWD"
git submodule update --init scripts/plus_eval/non_agentic_methods/simplememvla/SimpleMemVLA
```

For a fresh environment, use Python 3.10 and upstream Torch 2.4.1:

```bash
conda create -n simplememvla-plus python=3.10 -y
conda activate simplememvla-plus
cd "$ROBOMME_ROOT/scripts/plus_eval/non_agentic_methods/simplememvla/SimpleMemVLA"
pip install -r requirements.txt
pip install transformers==5.13.1
bash scripts/install/install_robomme_sim.sh
pip install sapien==3.0.3 numpy==1.26.4 opencv-python==4.11.0.86
conda install -c conda-forge libvulkan-loader -y
export SIMPLEMEMVLA_PYTHON="$(command -v python)"
```

The adapter adds RoboMME's `src/` to its import path. Installing the root project
into this environment would replace upstream dependencies. The example uses
`sdpa`; for FlashAttention, run upstream `scripts/install/install_fast_path.sh`
with a matching CUDA toolkit.

## Checkpoints

Set the self-contained RoboMME checkpoint location:

```bash
export SIMPLEMEMVLA_CHECKPOINT="$HOME/SimpleMemVLA/checkpoints/simplememvla_robomme"
```

For another machine, download `yinchenghust/simplememvla_robomme` following the
[upstream instructions](https://github.com/YinpeiDai/SimpleMemVLA/blob/c564c17d276d7294200122b286c21901a3bfb99f/README.md#-data--checkpoints). Keep weights,
configuration, processor/tokenizer files and `stats.json` together.
`--checkpoint` overrides `SIMPLEMEMVLA_CHECKPOINT`.

## Run

Run one BinFill episode with SDPA:

```bash
cd "$ROBOMME_ROOT"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$SIMPLEMEMVLA_PYTHON" scripts/plus_eval/non_agentic_methods/simplememvla/eval_plus.py \
  --tasks BinFill --episodes 0 --attn-implementation sdpa \
  --output-dir runs/simplememvla-plus-smoke
```

Omit `--attn-implementation sdpa` to use the default `flash_attention_2` if installed.

## Adapter notes

- Preserves non-pipelined inference with seed 0, BF16, 10 denoising steps,
  temperature 1.0, at most 64 subtask tokens and 16 executed actions per prediction.
  Reset demonstrations and execution frames enter the original history buffers.
- Records the **predicted subgoal**, sampled images, decoded prompt, normalized
  state, processed tensors/patch keys and action chunk. Policy `input.json` links
  to the preceding preprocessing `output.json`, avoiding duplicate tensor copies.

Omit task/episode filters for all 800 episodes. See [shared run options](../../../../doc/robomme_plus/robomme_plus_baseline_results.md#running-baselines) for resume, rerun and outputs.

[Link finalized rollouts](../../results/README.md) into the ignored `runs/` directory; videos and logs remain outside Git.

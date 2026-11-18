# 3-tier Astra on RoboMME-Plus

Uses [Astra-on-RoboMME](Astra-on-RoboMME)'s Astra API planner, local System 1.5
Qwen completion monitor and GroundSG Pi0.5 policy server. The checkout includes
`mme_vla_suite`; a separate `robomme_policy_learning` checkout is unnecessary.
See the [shared run instructions](../../../../../doc/robomme_plus/robomme_plus_baseline_results.md#running-baselines) for the 800-episode run options, outputs,
metrics and API pricing.

## Setup

Requires two NVIDIA GPUs, CUDA/Vulkan, Git LFS and `uv`; see the
[upstream system dependencies](https://github.com/YinpeiDai/Astra-on-RoboMME/blob/4c3fd6a8667e7a219e547fe1a533a4b9726fc6db/docs/evaluation.md).
From the **RoboMME repository root**, keep the VLA and monitor environments separate:

```bash
export ROBOMME_ROOT="$PWD"
export ASTRA_REPO="$ROBOMME_ROOT/scripts/plus_eval/agentic_methods/astra/three_tier/Astra-on-RoboMME"
git submodule update --init scripts/plus_eval/agentic_methods/astra/three_tier/Astra-on-RoboMME

# JAX/VLA environment; pytest is imported by upstream inference code.
(cd "$ASTRA_REPO" && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --python 3.11)
export VLA_PYTHON="$ASTRA_REPO/.venv/bin/python"
uv pip install --python "$VLA_PYTHON" pytest==8.4.2

# Simulator and completion monitor.
uv sync --no-dev --python 3.11
export SIM_PYTHON="$ROBOMME_ROOT/.venv/bin/python"
uv pip install --python "$SIM_PYTHON" -e "$ASTRA_REPO/packages/openpi-client" \
  -r "$ASTRA_REPO/examples/champ/requirements-monitor.txt"
uv pip install --python "$SIM_PYTHON" packaging ninja
uv pip install --python "$SIM_PYTHON" flash-attn==2.8.3 --no-build-isolation
```

FlashAttention needs a matching CUDA toolkit. Use `"$SIM_PYTHON"` directly afterward;
`uv sync` / `uv run` can undo the monitor dependency overrides.

## Checkpoints

Download the revisions pinned in upstream
[`weights.json`](https://github.com/YinpeiDai/Astra-on-RoboMME/blob/4c3fd6a8667e7a219e547fe1a533a4b9726fc6db/examples/champ/weights.json):

```bash
export CKPTS="$ASTRA_REPO/checkpoints"
uv pip install --python "$SIM_PYTHON" huggingface-hub==0.36.2
"$ROBOMME_ROOT/.venv/bin/hf" download bingaochen/Astra-on-RoboMME-Monitor \
  --revision 18ac28316a8b22c758945546c95f3910fc5dd27b \
  --local-dir "$CKPTS/Astra-on-RoboMME-Monitor"
"$ROBOMME_ROOT/.venv/bin/hf" download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir "$CKPTS/Qwen3-VL-4B-Instruct"
"$ROBOMME_ROOT/.venv/bin/hf" download Yinpei/mme_vla_suite \
  symbolic-grounded-subgoal/79999.zip symbolic-grounded-subgoal/history_config.txt \
  --revision 5db4d53ddb98c7f80cab08792dd53d985d712ab1 \
  --local-dir "$CKPTS/mme_vla_suite"
"$VLA_PYTHON" "$ASTRA_REPO/scripts/unzip_ckpt.py" "$CKPTS/mme_vla_suite" --processes 1

export VLA_CHECKPOINT="$CKPTS/mme_vla_suite/symbolic-grounded-subgoal/79999"
export MONITOR_BASE="$CKPTS/Qwen3-VL-4B-Instruct"
export MONITOR_ADAPTER="$CKPTS/Astra-on-RoboMME-Monitor"
(cd "$MONITOR_ADAPTER" && sha256sum -c SHA256SUMS)
"$VLA_PYTHON" -c 'from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()'
```

Keep `history_config.txt` beside `79999/`, which must contain `params/` and
`assets/`. Paths can also be supplied with `--vla-checkpoint`, `--monitor-base`
and `--monitor-adapter`.

## Run

In the same shell, start the VLA server on GPU 0:

```bash
mkdir -p "$ROBOMME_ROOT/runs"
(cd "$ASTRA_REPO" && exec env -u OPENAI_API_KEY CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$ASTRA_REPO/src" \
  "$VLA_PYTHON" scripts/serve_policy.py --port=18762 --seed=42 \
  policy:checkpoint --policy.config=mme_vla_suite --policy.dir="$VLA_CHECKPOINT") \
  > >("$SIM_PYTHON" "$ROBOMME_ROOT/scripts/plus_eval/privacy.py" filter \
  > "$ROBOMME_ROOT/runs/astra-plus-vla.log") 2>&1 &
VLA_PID=$!
```

Supply an API key with access to `gpt-6-astra` (paid calls), then run one BinFill
episode with the simulator/monitor on GPU 1. The client waits for the server:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
CUDA_VISIBLE_DEVICES=1 "$SIM_PYTHON" "$ROBOMME_ROOT/scripts/plus_eval/agentic_methods/astra/three_tier/eval_plus.py" \
  --tasks BinFill --episodes 0 --model-seed 42 \
  --output-dir "$ROBOMME_ROOT/runs/astra-plus-smoke"
```

Keep server/client ports and seeds consistent: `--model-seed` records the server
seed; it does not change it. Stop the server with `kill "$VLA_PID"` after evaluation.

## Adapter notes

- The first planner request has a 750-image budget, including its current and
  execution-start images. Demonstrations exceeding the remaining capacity are
  sampled uniformly across the entire video with both endpoints retained.
  Each demonstration contact sheet is one API image containing up to 16 frames.
  Execution history keeps upstream sampling. Every planner and button-review
  request is checked against 1,500 API images; exceeding that limit stops the
  episode as a scored failure. Requests are independent and do not carry an
  accumulating `previous_response_id` conversation. Selected frame indices and
  image-budget outcomes are saved with each rollout.
- Preserves upstream prompts, monitor triggering, 16-action chunks and the
  24-call planner limit. API requests retain the 20-second minimum interval and
  up to nine attempts; each attempt is recorded.
- Records System 1.5 images, prompt, raw boolean and replan decision; Astra
  prompts/image sheets, responses and parsed subgoals; Pi0.5 inputs/action chunks.
  Sanitized copies of planner artifacts also appear under `native/astra/` in each
  episode; credentials and private paths are redacted.

Omit task/episode filters for all 800 episodes. See [shared run options](../../../../../doc/robomme_plus/robomme_plus_baseline_results.md#running-baselines) for resume, rerun and outputs.

[Link finalized rollouts](../../../results/README.md) into the ignored `runs/` directory; videos and logs remain outside Git.

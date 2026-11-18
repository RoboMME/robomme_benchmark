# FrameSamp+Modul on RoboMME-Plus

Evaluates the upstream **perceptual-framesamp-modul** policy through its WebSocket
server, using the original history buffer, frame sampling, modulation and
16-action chunks. See the [shared run instructions](../../README.md) for the 800-episode run
options, outputs and metrics. Inference is local; no API or subgoal model is needed.

## Setup

Use separate Python 3.11 environments for the VLA server and simulator.
From the **RoboMME repository root**:

```bash
export ROBOMME_ROOT="$PWD"
export POLICY_REPO="$ROBOMME_ROOT/scripts/plus_eval/non_agentic_methods/framesamp_modul/robomme_policy_learning"
git submodule update --init scripts/plus_eval/non_agentic_methods/framesamp_modul/robomme_policy_learning
(cd "$POLICY_REPO" && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --python 3.11)
export VLA_PYTHON="$POLICY_REPO/.venv/bin/python"
uv pip install --python "$VLA_PYTHON" pytest==8.4.2

uv sync --no-dev --python 3.11
export SIM_PYTHON="$ROBOMME_ROOT/.venv/bin/python"
uv pip install --python "$SIM_PYTHON" -e "$POLICY_REPO/packages/openpi-client" 'imageio[ffmpeg]'
```

`pytest` is imported by upstream inference code.
The pinned commit in `RoboMME/robomme_policy_learning` includes the merged
long-history fix. It preserves positional encodings for executions exceeding
the original cache; no local patch is needed.

## Checkpoints

Set the checkpoint location:

```bash
export VLA_CHECKPOINT="$HOME/MME-VLA-Suite/runs/ckpts/mme_vla_suite_v0/perceptual-framesamp-modul/79999"
```

For another machine, follow the [upstream download instructions](https://github.com/RoboMME/robomme_policy_learning/blob/ac846a9a4aec841e483b1dc5d620dfe426bde853/README.md#-download-fine-tuned-vlavlm-checkpoints-optional).
Keep `79999/params`, `79999/assets` and the adjacent `history_config.txt` together.
The latter must contain exactly `perceptual-framesamp-modul.yaml`, **without a
trailing newline**; the upstream loader reads it verbatim.

## Run

In the same shell, start the server on GPU 0 and save its diagnostics separately:

```bash
mkdir -p "$ROBOMME_ROOT/runs"
(cd "$POLICY_REPO" && exec env CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$POLICY_REPO/src:$POLICY_REPO/packages/openpi-client/src" \
  "$VLA_PYTHON" scripts/serve_policy.py --seed=7 --port=18763 \
  policy:checkpoint --policy.config=mme_vla_suite --policy.dir="$VLA_CHECKPOINT") \
  > >("$SIM_PYTHON" "$ROBOMME_ROOT/scripts/plus_eval/privacy.py" filter \
       > "$ROBOMME_ROOT/runs/framesamp-modul-server.log") 2>&1 &
VLA_PID=$!
```

Run one BinFill episode on GPU 1; the client waits for the server:

```bash
CUDA_VISIBLE_DEVICES=1 "$SIM_PYTHON" "$ROBOMME_ROOT/scripts/plus_eval/non_agentic_methods/framesamp_modul/eval_plus.py" \
  --tasks BinFill --episodes 0 --port 18763 --model-seed 7 \
  --output-dir "$ROBOMME_ROOT/runs/framesamp-modul-plus-smoke"
```

`--checkpoint` (or `VLA_CHECKPOINT`) and `--model-seed` record server settings;
they do not reconfigure it. Keep both commands consistent. First inference
includes JAX compilation. Stop the server with `kill "$VLA_PID"` after evaluation.

## Adapter notes

- FrameSamp+modul alone caps demonstrations over 40 seconds to 800 uniformly
  spaced frames at the evaluation video's 20 FPS, keeping both endpoints and
  aligned camera/state observations. Execution observations retain their cadence.
- Long executions can exceed the original 4096-frame positional cache; overflow
  embeddings use the same positional formula without clipping history indices.
- Forwards reset demonstrations and new execution observations through the
  upstream buffer protocol: RGB, joint/gripper state and the task instruction.
- Records uploaded image/state history and demo boundaries, policy inputs,
  action chunks and returned server timings. WebSocket latency includes transport;
  internal server features are not exposed by the interface.

Omit task/episode filters for all 800 episodes. See [shared run options](../../README.md) for resume, rerun and outputs.

[Link finalized rollouts](../../results/README.md) into the ignored `runs/` directory; videos and logs remain outside Git.

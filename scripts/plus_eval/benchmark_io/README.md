# Shared agentic evaluation

Shared simulator I/O, budget ledger, frozen evaluation, queue and API accounting
for [ASPIRE](../agentic_methods/aspire/README.md) and [RAX](../agentic_methods/rax/README.md).
Read the [experimental protocol](../../../doc/plus_agentic_evaluation.md) for
settings and rules. Final agent-facing prompts are in [`prompts/current`](prompts/current).

## Setup

From the repository root, using the simulator's Python 3.11 environment:

```bash
git submodule update --init scripts/plus_eval/agentic_methods/aspire/ASPIRE scripts/plus_eval/agentic_methods/rax/rax
source .venv/bin/activate
uv pip install -r scripts/plus_eval/agentic_methods/requirements.txt
export PYTHONPATH="$PWD/src:$PWD/scripts/plus_eval"
export ROBOMME_AGENTIC_REPO="$PWD"
python -m benchmark_io.runtime_source --output runs/agentic-tools
RAX_ARTIFACT_ROOT="$PWD/runs/agentic-tools/rax/runtime" \
  python runs/agentic-tools/rax/scripts/build_agent_image.py --tag robomme-agentic:latest
```

Runtime assembly verifies the submodule pins, exports committed source, applies
integration patches and overlays adapters. It records hashes and never edits the
submodules. Use new output directories for changed code. Original licenses remain
with the upstream source.

Install the host Claude CLI (`ROBOMME_CLAUDE_CLI` overrides `PATH`), configure AWS
Bedrock credentials, and provision ASPIRE's providers using its README. The relay
uses Opus 5.5, high reasoning effort, and `AWS_REGION` (default `us-east-1`).
Credentials remain outside agent containers. Neither method uses our CaP code.

## Develop and evaluate

Download the [final training H5 data](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5),
revision `63cb9b5b62f5413993845c7d89b82b3bd58dc2fa`, then:

```bash
python -m benchmark_io.prepare_unified prepare \
  --root "<RUN_ROOT>" --demos "<DATA_ROOT>/h5"
python -m benchmark_io.watch_unified --root "<RUN_ROOT>"
```

Preparation snapshots the benchmark, 50 recordings, protocol, runtime and image
digest without API calls or simulator starts. It creates 18 configurations:
2 methods × (AfD R0; AfI R1/R3/R5/R10; AfD+I R1/R3/R5/R10).

The queue shares six development and sixteen evaluation slots across two GPUs.
Evaluation starts only after that configuration freezes. To hold evaluation
until all development finishes, create `<RUN_ROOT>/schedule_policy.json`:

```json
{
  "development_workers_total": 6,
  "development_workers_per_gpu": 3,
  "evaluation_workers_total": 16,
  "evaluation_workers_by_gpu": {"0": 8, "1": 8},
  "evaluation_mode": "after_all_development"
}
```

Modes: `asynchronous`, `after_all_development`, `held`. Optional
`evaluation_episode_indices: [0, 1]` selects a sanity subset; remove it to complete
the frozen run. Create `STOP` to stop dispatching. The monitor adopts owned workers
rather than replaying trials. Limits are in [`runtime_policy.py`](runtime_policy.py).
For reruns using published policies, use each method's `eval_plus.py`.

## Artifacts and checks

Under `METHOD/SETTING/Rn/`: `study/` contains metadata/attempts;
`development/` contains native commands, transcripts, code and training rollouts;
`development_api/` contains usage/cost records; `evaluation/` contains frozen
policies, test results and logs. R0/R10 save full test media; R1/R3/R5 save logs.
All development history is retained. `progress.json` aggregates the study.

Costs use provider-reported tokens and the recorded rate card; missing usage is
not zero cost. Frozen evaluation has zero frontier-model API cost in this setup.
Keep `runtime/private/` and credentials out of public archives.
[Link external runs](../results/README.md) to browse logs from the method directories.

```bash
PYTHONPATH="$PWD/src:$PWD/runs/agentic-tools:$PWD/runs/agentic-tools/rax:$PWD/runs/agentic-tools/aspire/ASPIRE/src" \
  python -m pytest scripts/plus_eval/benchmark_io/tests scripts/plus_eval/agentic_methods/aspire/tests
```

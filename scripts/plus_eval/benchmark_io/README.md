# Agentic development and evaluation

Shared simulator I/O, budget ledger, frozen evaluation, queue and API accounting
for agentic methods, including [ASPIRE](../agentic_methods/aspire/README.md) and [RAX](../agentic_methods/rax/README.md).
Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol);
[baseline settings and results](../../../doc/robomme_plus/robomme_plus_baseline_results.md) describe
the finalized runs. [ASPIRE](../agentic_methods/aspire/README.md#adaptation) and
[RAX](../agentic_methods/rax/README.md#adaptation) document their adaptation settings
and runtime limits. Saved agent-facing prompts are in [`prompts/current`](prompts/current).

## Integrate your own method

Use the [development interface](../../../doc/robomme_plus/agentic_development.md)
for a minimal preparation command, seed-selectable API, logging requirements and
audits. Keep your own harness; run the simulator service outside the agent workspace.
The [audit walkthrough](../../../doc/robomme_plus/auditing.md) provides prompts
for the harness review and completed-development review.

## Setup

From the repository root, using the simulator's Python 3.11 environment:

```bash
git submodule update --init scripts/plus_eval/agentic_methods/aspire/ASPIRE
python3 scripts/plus_eval/agentic_methods/rax/restore_core.py
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
pin the downloaded revision in your run manifest, then:

```bash
python -m benchmark_io.prepare_unified prepare \
  --root "<RUN_ROOT>" --demos "<DATA_ROOT>/h5"
python -m benchmark_io.watch_unified --root "<RUN_ROOT>"
```

Preparation pins the benchmark, 50 recordings, protocol, runtime and image
digest without API calls or simulator starts. New studies create **6 configurations**:
2 methods × (`demo-only` at 0; each interaction setting at 10).
Each configuration evaluates 800 episodes: 4,800 tests across both methods.
Existing studies retain their pinned configuration list, including historical budgets.

Use these setting names in commands and directories:

| Setting | Main budget |
| --- | --- |
| `demo-only` | 0 |
| `interaction-only` | 10 |
| `demo&interaction` | 10 |

In shell commands, write `--setting 'demo&interaction'` with quotes.

Worker allocation is configurable. Evaluation requires frozen policies.
Example `<RUN_ROOT>/schedule_policy.json` for two GPUs:

```json
{
  "development_workers_total": 6,
  "development_workers_per_gpu": 3,
  "evaluation_workers_total": 16,
  "evaluation_workers_by_gpu": {"0": 8, "1": 8},
  "evaluation_mode": "after_all_development"
}
```

The default episode selection covers all 800 tests. Create `STOP` to stop
dispatching; running workers remain owned by the monitor. Limits are in
[`runtime_policy.py`](runtime_policy.py). To evaluate published frozen policies,
use each method's `eval_plus.py`.

## Artifacts and checks

Under `METHOD/SETTING/Rn/`: `study/` contains metadata/attempts;
`development/` contains native commands, transcripts, code and training rollouts;
`development_api/` contains usage/cost records; `evaluation/` contains frozen
policies, test results and logs. R0/R10 save full test media; R5 saves logs.
R1/R3/R5 results remain optional, with their original budgets and assignments.
All development history is retained. `progress.json` aggregates the study.

Public development archives may omit raw trajectories, images and videos while
retaining text logs, all code revisions, attempt records and payload references.
See the [publication requirements](../../../doc/robomme_plus/agentic_development.md#preserve-evidence).

Report demo use, actual attempts and simulator steps per task, and separate
development/evaluation API costs. Costs use provider tokens and the recorded rate
card; missing usage is not zero cost. Frozen evaluation has zero frontier-model API cost in this setup.
Keep `runtime/private/` and credentials out of public archives.
[Link external runs](../results/README.md) to browse logs from the method directories.

```bash
PYTHONPATH="$PWD/src:$PWD/runs/agentic-tools:$PWD/runs/agentic-tools/rax:$PWD/runs/agentic-tools/aspire/ASPIRE/src" \
  python -m pytest scripts/plus_eval/benchmark_io/tests scripts/plus_eval/agentic_methods/aspire/tests
```

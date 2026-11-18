# RAX on RoboMME-Plus

Core: [`rax/`](rax), the unchanged
[robot-agent-exploration/rax](https://github.com/robot-agent-exploration/rax)
submodule at `d49c8ad605008bb5a87e909d0473cc3f5e3600a6`.
Adapters, patches, solutions and this README live outside the submodule.

## Adaptation

We retain LIBERO's per-task exploration → bundle registration → exact-bundle
rehearsal → submission workflow. Rehearsals consume the same R budget. AfD allows
explicit offline submission because it forbids simulator access. We add no CaP
code, task solver or policy-selection algorithm.

Runtime integration patches cover eight upstream Python files in [`patches/robomme-plus.patch`](patches/robomme-plus.patch), applied
only to a separate runtime copy:

| Upstream area | Integration change |
| --- | --- |
| `services/benchmarks.py` | Register public RoboMME-Plus backend/tools |
| `controller/config.py`, `frontend.py` | AfD zero-interaction submission, reasoning effort and budget gates |
| `controller/supervisor.py`, `sandbox.py` | Bounded execution and safe continuation of unstarted tests |
| `services/artifact_io.py` | Stream large artifacts |
| `harness/{claude,codex}_cli/auto_research.py` | Public protocol, deadline and read-only wrap-up |

`adapter/` contains the simulator backend and native skill references.
`robomme_full.py` drives development; `eval_plus.py` runs frozen test policies.
`solutions/` contains all 144 evaluated task bundles under settings, R and task,
with hashes, portable runner settings and policy provenance.
`upstream.json` pins source and patches.

The [shared adaptation notes](../../../../doc/robomme_plus/robomme_plus_baseline_results.md#aspire-and-rax-adaptation-notes)
record the model, input export, 512-timestamp cap, runtime limits and budget rules.
RAX-specific changes:

- **Deadline:** 150 minutes from the recorded task-process start, including
  provisioning, API calls, offline work and rollouts. Resumes preserve the clock.
  Expiry stops the sandbox/simulator, then allows at most 120 seconds for read-only
  reporting or submission of an existing eligible bundle; no edits or rehearsals.
- **Submission:** interaction settings retain the clean rehearsal of the exact
  submitted bundle, charged to the same R. AfD adds offline submission with zero
  simulator starts. Test outcomes never return to development.
- **Providers and transport:** optional Gemini is disabled; native coding-model
  vision and RGB-D geometry tools remain. Large artifacts are streamed, initial
  videos sampled, and terminal/timeout handling uses the public simulator result.

Published `solutions/` contain the evaluated policies. Each task's `manifest.json`
records its source selection, evaluated hashes and interface-adjustment provenance.

See the [benchmark protocol](../../../../src/robomme_plus/README.md#evaluation-protocol).

## Run and inspect

Follow [shared setup/development](../../benchmark_io/README.md), then evaluate a
published configuration on all 800 test episodes:

```bash
python scripts/plus_eval/agentic_methods/rax/eval_plus.py \
  --setting AfI --budget 3 \
  --gpu 0 --output-dir "<RUN_ROOT>/rax-AfI-R3"
```

`--workers` sets concurrent processes on that GPU; `--artifacts logs` omits media. The same output directory resumes
unstarted tests; use a new directory for a fresh run.

[Link external artifacts](../../results/README.md) to inspect
`runs/SETTING/Rn/{development,evaluation}`. Public packet examples are in
[`RAX.md`](../../benchmark_io/prompts/current/RAX.md).

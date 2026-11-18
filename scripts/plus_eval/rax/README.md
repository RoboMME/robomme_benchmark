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

The working integration changed eight upstream Python files. Those edits are
preserved in [`patches/robomme-plus.patch`](patches/robomme-plus.patch), applied
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
with hashes, portable runner settings and recorded interface corrections.
`upstream.json` pins source and patches.

AfD/AfD+I receive released RGB recordings; interaction uses front/wrist RGB-D,
calibration, robot state and goal. No hidden object state or task source is exposed.
Opus 5.5 uses Bedrock with high reasoning effort. Optional Gemini is disabled;
frozen test execution makes no frontier-model API calls. R1/R3/R5/R10 develop
independently. See the [protocol](../../../doc/plus_agentic_evaluation.md).

## Run and inspect

Follow [shared setup/development](../benchmark_io/README.md), then rerun a
published solution without development:

```bash
python scripts/plus_eval/rax/eval_plus.py \
  --setting AfI --budget 3 --tasks StopCube --episodes 0 1 \
  --gpu 0 --output-dir "<RUN_ROOT>/rax-rerun"
```

Omit task/episode filters for all 800 tests. `--workers` sets concurrent processes
on that GPU; `--artifacts logs` omits media. The same output directory resumes
unstarted tests; use a new directory for a fresh run.

[Link external artifacts](../results/README.md) to inspect
`runs/SETTING/Rn/{development,evaluation}`. Public packet examples are in
[`RAX.md`](../benchmark_io/prompts/current/RAX.md).

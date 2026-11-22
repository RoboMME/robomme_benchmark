# RAX on RoboMME-Plus

Core: [`rax/`](rax), the unchanged
[robot-agent-exploration/rax](https://github.com/robot-agent-exploration/rax)
submodule at `d49c8ad605008bb5a87e909d0473cc3f5e3600a6`.
Adapters, patches, solutions and this README live outside the submodule.

The pin originates from upstream's `libero-pro/container` branch. A complete
[`upstream.bundle`](upstream.bundle), checksum-pinned in [`upstream.json`](upstream.json),
preserves the same commit if that branch is rewritten. From the repository root,
run `python3 scripts/plus_eval/agentic_methods/rax/restore_core.py` to initialize
`rax/` offline. Recursive clones skip this submodule's network fetch; use the
restore command during setup. Existing checkouts are left unchanged; evaluated code is identical.

## Adaptation

The released policies cover `Interaction-only` and `Hybrid`, each with at most
10 development rollouts per task.

We retain LIBERO's per-task exploration → bundle registration → exact-bundle
rehearsal → submission workflow. Rehearsals consume the same R budget. `Demo-only` allows
explicit offline submission because it forbids simulator access. We add no CaP
code, task solver or policy-selection algorithm.

Runtime integration patches cover nine upstream Python files in [`patches/`](patches), applied
only to a separate runtime copy:

| Upstream area | Integration change |
| --- | --- |
| `services/benchmarks.py` | Register public RoboMME-Plus backend/tools |
| `controller/config.py`, `frontend.py` | `Demo-only` zero-interaction submission, reasoning effort and budget gates |
| `controller/supervisor.py`, `sandbox.py` | Bounded execution and safe continuation of unstarted tests |
| `services/artifact_io.py` | Stream large artifacts |
| `harness/{claude,codex}_cli/auto_research.py` | Public protocol, deadline and read-only wrap-up |
| `harness/codex_cli/workspace_storage.py` | Local directory workspaces without loop-device mounts |

`adapter/` contains the simulator backend and native skill references.
`robomme_full.py` drives development; `eval_plus.py` runs frozen test policies.
`solutions/` contains 32 current R10 task bundles under `interaction-only/R10` and `hybrid/R10`,
with hashes, portable runner settings and policy provenance.
`upstream.json` pins source and patches.

These settings describe this baseline under the
[benchmark protocol](../../../../src/robomme_plus/README.md#evaluation-protocol).
The model, time limits and input sampling are implementation choices.

| Area | Configuration |
| --- | --- |
| Development model | Opus 5.5 via AWS Bedrock, high reasoning effort; development/evaluation usage and cost logged separately. |
| Budgets | `Demo-only`: 0 starts. Interaction settings: at most 10 starts per task. All starts, including failed setup and verification, share the budget. |
| Training demos | `Demo-only`/`Hybrid` receive front/wrist RGB recordings, goals, phases and timestamps for offline analysis. `Interaction-only` receives none. Simulator replay counts toward the development budget. |
| Live interface | Public RGB-D, calibration, robot state, actions, conditioning videos and terminal/error feedback; task source and hidden object state remain outside agent workspaces. |
| Video transport | Long conditioning videos are uniformly sampled to 512 timestamps, retaining endpoints, both cameras and a separate initial live frame. Training recordings are unchanged. |
| Frozen evaluation | Selected Python policies and local providers; no frontier-model API calls, test-driven revisions or cross-episode adaptive memory. |
| Reset recovery | One setup attempt per reserved training start; up to three identical test setup attempts. Policy execution is not retried automatically. |

- **Deadline:** 150 minutes from the recorded task-process start, including
  provisioning, API calls, offline work and rollouts. Resumes preserve the clock.
  Expiry stops the sandbox/simulator, then allows at most 120 seconds for read-only
  reporting or submission of an existing eligible bundle; no edits or rehearsals.
- **Submission:** interaction settings retain the clean rehearsal of the exact
  submitted bundle, charged to the same R. `Demo-only` adds offline submission with zero
  simulator starts. Test outcomes never return to development.
- **Providers and transport:** optional Gemini is disabled; native coding-model
  vision and RGB-D geometry tools remain. Large artifacts are streamed, initial
  videos sampled, and terminal/timeout handling uses the public simulator result.

Published `solutions/` contain the evaluated policies. Each task's `manifest.json`
records the source run, exact evaluated file hashes and benchmark fingerprint.

### Runtime limits

The [runtime policy](../../benchmark_io/runtime_policy.py) is pinned in
[`solutions/runtime_policy.json`](solutions/runtime_policy.json).

| Limit | Value |
| --- | --- |
| Controller wall time per rollout | 20 minutes |
| Setup / initial wait for a control step | 10 minutes each |
| No control-step progress after execution starts | 3 minutes |
| Provider request / simulator RPC | 60 seconds / 180 seconds |
| Outer episode worker | 35 minutes, including startup and cleanup |

These limits apply to development and testing without extending simulator step
limits. Timeouts count as unsuccessful tests. The queue's 48-hour job cap and
30-minute idle watchdog do not extend the per-task development deadline.
Clocks, commands, transcripts and API usage are retained in run artifacts.
Tool hooks also save host-owned command/results and code snapshots in each task's
`audit/history/`; simulator attempts record the workspace revision. Frozen
evaluation verifies the pinned harness/protocol and policy files. Workspace size
is checked at launch/resume; directory storage has no block-device quota.

## Run and inspect

Follow [shared setup/development](../../benchmark_io/README.md), then evaluate a
published configuration on all 800 test episodes:

```bash
python scripts/plus_eval/agentic_methods/rax/eval_plus.py \
  --setting interaction-only --budget 10 \
  --gpu 0 --output-dir "<RUN_ROOT>/rax-interaction-only-R10"
```

Use `--setting hybrid` for Hybrid. `--workers` sets concurrent processes on that GPU; `--artifacts logs` omits media.
The same output directory resumes unstarted tests; use a new directory for a fresh run.

[Link external artifacts](../../results/README.md) to inspect
`runs/SETTING/Rn/{development,evaluation}`. Public packet examples are in
[`RAX.md`](../../benchmark_io/prompts/current/RAX.md).

Development uses the shared [interaction accounting interface](../../../../doc/robomme_plus/agentic_development.md).
See the [submission and audit results](<../../../../doc/robomme_plus/submission/rax-plus (eval_by_RoboMME_team).md>).

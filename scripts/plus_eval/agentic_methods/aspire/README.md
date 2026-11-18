# ASPIRE on RoboMME-Plus

Core: [`ASPIRE/`](ASPIRE), the unchanged [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE)
submodule at `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`.
The upstream `cap_harness` package belongs to ASPIRE; our separate CaP method is
not used.

## Adaptation

We retain the LIBERO coordinator, task workers, verifier, shared skill library,
robot API and SAM3/Contact-GraspNet/PyRoki/cuRobo providers. Each setting/budget has
an independent multi-task campaign. Programs and libraries freeze before testing.

- `adapter/`: public RoboMME-Plus RGB-D, calibration, robot-state, goal and action bridge.
- `patches/`: robot/tool registration, bounded provider requests, local SAM3
  checkpoint support, and timeout outcome recording. Applied only to a runtime copy.
- `campaign.py` and companion modules: development isolation, budget/deadline
  enforcement, transport and logging.
- `eval_plus.py`: frozen test execution through the public benchmark interface.
- `solutions/`: all 144 evaluated task policies, organized by setting, R and task.
- `runs/`: optional ignored link to external development logs and test rollouts.

These settings describe this baseline under the
[benchmark protocol](../../../../src/robomme_plus/README.md#evaluation-protocol).
The model, time limits and input sampling are implementation choices.

| Area | Configuration |
| --- | --- |
| Development model | Opus 5.5 via AWS Bedrock, high reasoning effort; adaptation/evaluation usage and cost logged separately. |
| Budgets | Nine independent runs: AfD R0, AfI and AfD+I at R1/R3/R5/R10. Lower budgets are not R10 checkpoints. All training starts, including failed setup and verification, share R. |
| Training demos | AfD/AfD+I receive front/wrist RGB recordings, goals, phases and timestamps, with unlimited offline reuse. AfI receives none. |
| Live interface | Public RGB-D, calibration, robot state, actions, conditioning videos and terminal/error feedback; task source and hidden object state remain outside agent workspaces. |
| Video transport | Long conditioning videos are uniformly sampled to 512 timestamps, retaining endpoints, both cameras and a separate initial live frame. Training recordings are unchanged. |
| Frozen evaluation | Selected Python policies and local providers; no frontier-model API calls, test-driven revisions or cross-episode adaptive memory. |
| Reset recovery | One setup attempt per reserved training start; up to three identical test setup attempts. Policy execution is not retried automatically. |
| SAM3 recovery | At most three HTTP attempts per request for transient service/connection errors, with 1-second then 2-second backoff. Each attempt retains the 60-second timeout; episode watchdogs still apply. No retries for empty detections or invalid prompts. |

- **Deadline:** 150 minutes from a task's first worker assignment, including API
  calls, offline work and rollouts. Resumes preserve its clock. Expiry stops the
  campaign container, snapshots that task and remounts its directory read-only;
  the coordinator reports its limitations and continues other unfinished tasks.
- **Verification:** all worker/verifier starts share the task's R. Keep the native
  verification criterion; insufficient released seeds or budget leave a skill
  unverified. AfD has no simulator verification. Held-out evaluation and revisits
  using test feedback are disabled until the campaign is frozen.
- **Providers and isolation:** retain SAM3, Contact-GraspNet, PyRoki and cuRobo;
  add RoboMME robot/tool registration, local SAM3 loading and bounded requests.
  Agent containers access public APIs through the operator bridge.
  SAM3 attempts are recorded in `sam3_requests.jsonl`. Exhausted service errors
  mark the episode as infrastructure-affected even if the policy catches them.
  HTTP retries do not reset or step the simulator and consume no extra rollout.
  Infrastructure reruns preserve the frozen policy and test seed; retain the
  previous artifacts and replace selected results regardless of success/failure.

Published `solutions/` contain the evaluated policies. Each task's `manifest.json`
records its source selection, evaluated hashes and interface-adjustment provenance.

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

## Setup and run

Follow [shared setup/development](../../benchmark_io/README.md). Provision ASPIRE's
native providers using its `scripts/bootstrap_providers.sh` with
`--providers sam3,contact_graspnet,pyroki,curobo` and your hardware profile.
Set `ROBOMME_ASPIRE_PROVIDER_ROOT="<PROVIDER_ROOT>"` and
`ROBOMME_SAM3_CHECKPOINT="<MODEL_ROOT>/sam3.pt"`. The provider root supplies its
`.venv-{sam3,graspnet,pyroki,curobo}` environments and `third_party` dependencies.
Weights and environments are not included here.

Evaluate a published configuration on all 800 test episodes:

```bash
python scripts/plus_eval/agentic_methods/aspire/eval_plus.py \
  --setting AfI --budget 3 \
  --gpu 0 --output-dir "<RUN_ROOT>/aspire-AfI-R3"
```

`--workers` sets concurrent processes on that GPU; `--artifacts logs` omits media.
The same output directory resumes unstarted tests; use a new directory for a fresh run. Solution manifests preserve
file hashes and policy provenance.

[Link logs and rollouts](../../results/README.md) to access
`runs/SETTING/Rn/{development,evaluation}`. Exact public data formats and synthetic
examples are in [`ASPIRE.md`](../../benchmark_io/prompts/current/ASPIRE.md).

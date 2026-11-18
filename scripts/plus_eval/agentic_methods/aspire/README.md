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

The [shared adaptation notes](../../../../doc/robomme_plus/robomme_plus_baseline_results.md#aspire-and-rax-adaptation-notes)
record the model, input export, 512-timestamp cap, runtime limits and budget rules.
ASPIRE-specific changes:

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

Published `solutions/` contain the evaluated policies. Each task's `manifest.json`
records its source selection, evaluated hashes and interface-adjustment provenance.

See the [benchmark protocol](../../../../src/robomme_plus/README.md#evaluation-protocol).

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

`--workers` sets concurrent processes on that GPU; `--artifacts logs` omits media. The same output directory resumes
unstarted tests; use a new directory for a fresh run. Solution manifests preserve
file hashes and policy provenance.

[Link logs and rollouts](../../results/README.md) to access
`runs/SETTING/Rn/{development,evaluation}`. Exact public data formats and synthetic
examples are in [`ASPIRE.md`](../../benchmark_io/prompts/current/ASPIRE.md).

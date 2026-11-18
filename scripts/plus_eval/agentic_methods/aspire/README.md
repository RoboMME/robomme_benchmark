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
  checkpoint support, and correct timeout recording. Applied only to a runtime copy.
- `campaign.py` and companion modules: development isolation, budget/deadline
  enforcement, transport and logging.
- `eval_plus.py`: frozen test execution through the public benchmark interface.
- `solutions/`: all 144 evaluated task policies, organized by setting, R and task.
- `runs/`: optional ignored link to external development logs and test rollouts.

AfD/AfD+I receive released RGB recordings; live interaction uses calibrated RGB-D.
Every simulator start, including verifier runs, consumes R. AfD has none.
Opus 5.5 runs through Bedrock with high reasoning effort. Test execution makes no
frontier-model calls. See the [evaluation protocol](../../../../src/robomme_plus/README.md#evaluation-protocol)
and [baseline settings/results](../../../../doc/robomme_plus/robomme_plus_baseline_results.md).

## Setup and run

Follow [shared setup/development](../../benchmark_io/README.md). Provision ASPIRE's
native providers using its `scripts/bootstrap_providers.sh` with
`--providers sam3,contact_graspnet,pyroki,curobo` and your hardware profile.
Set `ROBOMME_ASPIRE_PROVIDER_ROOT="<PROVIDER_ROOT>"` and
`ROBOMME_SAM3_CHECKPOINT="<MODEL_ROOT>/sam3.pt"`. The provider root supplies its
`.venv-{sam3,graspnet,pyroki,curobo}` environments and `third_party` dependencies.
Weights and environments are not included here.

Rerun a published solution without development:

```bash
python scripts/plus_eval/agentic_methods/aspire/eval_plus.py \
  --setting AfI --budget 3 --tasks StopCube --episodes 0 1 \
  --gpu 0 --output-dir "<RUN_ROOT>/aspire-rerun"
```

Omit task/episode filters for all 800 tests. `--workers` sets concurrent processes
on that GPU; `--artifacts logs` omits media. The same output directory resumes
unstarted tests; use a new directory for a fresh run. Solution manifests preserve
file hashes and any recorded interface correction.

[Link logs and rollouts](../../results/README.md) to access
`runs/SETTING/Rn/{development,evaluation}`. Exact public data formats and synthetic
examples are in [`ASPIRE.md`](../../benchmark_io/prompts/current/ASPIRE.md).

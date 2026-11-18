# Overall results

- [Result tables and interpretation](../../../doc/robomme_plus/robomme_plus_baseline_results.md)
- [Model baseline summaries](model_baselines.json)
- [ASPIRE/RAX counts by setting and task](agentic_20261118.json)
- Frozen programs: [ASPIRE](../agentic_methods/aspire/solutions), [RAX](../agentic_methods/rax/solutions)

Connect an existing study without copying its videos or logs:

```bash
PYTHONPATH=scripts/plus_eval python -m benchmark_io.artifacts --study "<RUN_ROOT>"
```

This creates ignored `agentic_methods/aspire/runs`, `agentic_methods/rax/runs` and `results/agentic-runs` links.
Within each method, development logs and test rollouts live under
`runs/SETTING/Rn/{development,evaluation}`. Use `<RESULTS_URL>` for your own hosted
viewer. Local paths, credentials, logs and videos are not committed to the repo.

For the six model baselines, extract the published
[RoboMME-Plus rollouts](https://huggingface.co/datasets/Yinpei/RoboMME-plus-model-rollout)
(revision `ebef7bcd1abe4cc1748c3257cd787434489265fc`), then link that directory:

```bash
PYTHONPATH=scripts/plus_eval python -m benchmark_io.artifacts --model-rollouts "<ROLLOUT_ROOT>"
```

This creates `METHOD/runs/final` links and `results/model-runs`. Embodied Astra
uses `runs/zero-shot` and `runs/one-shot` within its shared method directory.

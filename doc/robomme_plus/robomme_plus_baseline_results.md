# RoboMME-Plus baseline results

Final results on the same **800 test episodes** (16 tasks × 50). Of these, 514
have no added variation, 182 have a perceptual variation and 104 a mechanism
variation. Only a simulator success counts; failures, timeouts and errors stay in
the denominator.

## Evaluation settings

Compare both tracks under the same setting and per-task rollout budget.

| Setting | Released Plus demonstrations | Maximum Plus rollouts per task |
| --- | --- | --- |
| `Zero-shot` | No | 0 |
| `Demo-only` | Yes | 0 |
| `Interaction-only` | No | 10 |
| `Hybrid` | Yes | 10 |

See the [evaluation protocol](../../src/robomme_plus/README.md#evaluation-protocol)
for permitted inputs, rollout accounting, audits and reporting.


## `Zero-shot`

These baselines use their pretrained models with no RoboMME-Plus demonstrations
or simulator rollouts.

| Method | Successes | Success rate |
|---|---:|---:|
| [Embodied Astra](<submission/embodied-astra-plus (eval_by_RoboMME_team).md>) | 194/800 | 24.25% |
| [FrameSamp+Modul](<submission/framesamp-modul-plus (eval_by_RoboMME_team).md>) | 19/800 | 2.38% |
| [SimpleMemVLA](<submission/simplememvla-plus (eval_by_RoboMME_team).md>) | 14/800 | 1.75% |
| [3-tier Astra](<submission/3-tier-astra-plus (eval_by_RoboMME_team).md>) | 70/800 | 8.75% |
| [PonderPounce](<submission/ponderpounce-plus (eval_by_RoboMME_team).md>) | 20/800 | 2.50% |

## `Demo-only`

One-shot Astra sees one training example per task (the medium one where labelled;
episode 1 for InsertPeg) and none for MoveCube. ASPIRE and RAX develop from all 50
released demonstrations, with no simulator rollouts.

| Method | Training input | Successes | Success rate |
|---|---|---:|---:|
| [One-shot Embodied Astra](<submission/one-shot-embodied-astra-plus (eval_by_RoboMME_team).md>) | One selected example per task | 281/800 | 35.13% |
| [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md>) | All 50 demonstrations | 140/800 | 17.50% |
| [RAX](<submission/rax-plus (eval_by_RoboMME_team).md>) | All 50 demonstrations | 139/800 | 17.38% |

## `Interaction-only`

| Rollouts per task | ASPIRE | RAX |
|---|---:|---:|
| 10 rollouts | 302/800 (37.75%) | 372/800 (46.50%) |

## `Hybrid`

| Rollouts per task | ASPIRE | RAX |
|---|---:|---:|
| 10 rollouts | 342/800 (42.75%) | 327/800 (40.88%) |

## Agentic development audit status

All six ASPIRE and RAX runs (`Demo-only`, `Interaction-only` and `Hybrid`) pass all
three audit checks: no forbidden information, within the rollout budget, and no
test-time optimization.
The full reports are in the [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md#agentic-development-audit-record>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md#agentic-development-audit-record>)
submissions. See the [audit procedure](agentic_development.md#step-7-audit).

## Per-task success rates

Each cell uses 50 episodes.

<details>
<summary><code>Zero-shot</code></summary>

| Task | Embodied Astra | FrameSamp+Modul | SimpleMemVLA | 3-tier Astra | PonderPounce |
|---|---:|---:|---:|---:|---:|
| BinFill | 26.0% | 0.0% | 0.0% | 6.0% | 2.0% |
| ButtonUnmask | 34.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ButtonUnmaskSwap | 12.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| InsertPeg | 20.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MoveCube | 22.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PatternLock | 14.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PickHighlight | 34.0% | 0.0% | 0.0% | 2.0% | 0.0% |
| PickXtimes | 54.0% | 0.0% | 0.0% | 6.0% | 0.0% |
| RouteStick | 12.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| StopCube | 22.0% | 4.0% | 0.0% | 2.0% | 2.0% |
| SwingXtimes | 32.0% | 0.0% | 8.0% | 10.0% | 8.0% |
| VideoPlaceButton | 36.0% | 16.0% | 14.0% | 56.0% | 18.0% |
| VideoPlaceOrder | 26.0% | 14.0% | 4.0% | 50.0% | 2.0% |
| VideoRepick | 22.0% | 4.0% | 2.0% | 4.0% | 8.0% |
| VideoUnmask | 20.0% | 0.0% | 0.0% | 4.0% | 0.0% |
| VideoUnmaskSwap | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% |

</details>

<details>
<summary><code>Demo-only</code></summary>

| Task | One-shot Astra | ASPIRE | RAX |
|---|---:|---:|---:|
| BinFill | 26.0% | 36.0% | 30.0% |
| ButtonUnmask | 78.0% | 46.0% | 52.0% |
| ButtonUnmaskSwap | 26.0% | 4.0% | 4.0% |
| InsertPeg | 28.0% | 0.0% | 0.0% |
| MoveCube | 18.0% | 2.0% | 4.0% |
| PatternLock | 18.0% | 0.0% | 20.0% |
| PickHighlight | 42.0% | 0.0% | 2.0% |
| PickXtimes | 80.0% | 0.0% | 0.0% |
| RouteStick | 8.0% | 20.0% | 6.0% |
| StopCube | 38.0% | 34.0% | 46.0% |
| SwingXtimes | 98.0% | 34.0% | 6.0% |
| VideoPlaceButton | 36.0% | 22.0% | 22.0% |
| VideoPlaceOrder | 22.0% | 20.0% | 24.0% |
| VideoRepick | 16.0% | 0.0% | 62.0% |
| VideoUnmask | 28.0% | 62.0% | 0.0% |
| VideoUnmaskSwap | 0.0% | 0.0% | 0.0% |
</details>

<details>
<summary><code>Interaction-only</code> · R10</summary>

| Task | ASPIRE | RAX |
| --- | ---: | ---: |
| BinFill | 50.0% | 62.0% |
| ButtonUnmask | 70.0% | 64.0% |
| ButtonUnmaskSwap | 8.0% | 30.0% |
| InsertPeg | 24.0% | 32.0% |
| MoveCube | 6.0% | 0.0% |
| PatternLock | 6.0% | 20.0% |
| PickHighlight | 34.0% | 36.0% |
| PickXtimes | 48.0% | 70.0% |
| RouteStick | 20.0% | 20.0% |
| StopCube | 80.0% | 52.0% |
| SwingXtimes | 54.0% | 66.0% |
| VideoPlaceButton | 18.0% | 62.0% |
| VideoPlaceOrder | 30.0% | 32.0% |
| VideoRepick | 48.0% | 66.0% |
| VideoUnmask | 54.0% | 56.0% |
| VideoUnmaskSwap | 54.0% | 76.0% |

</details>

<details>
<summary><code>Hybrid</code> · R10</summary>

| Task | ASPIRE | RAX |
| --- | ---: | ---: |
| BinFill | 82.0% | 50.0% |
| ButtonUnmask | 56.0% | 68.0% |
| ButtonUnmaskSwap | 28.0% | 20.0% |
| InsertPeg | 8.0% | 0.0% |
| MoveCube | 2.0% | 2.0% |
| PatternLock | 20.0% | 20.0% |
| PickHighlight | 22.0% | 38.0% |
| PickXtimes | 22.0% | 40.0% |
| RouteStick | 46.0% | 24.0% |
| StopCube | 58.0% | 66.0% |
| SwingXtimes | 66.0% | 66.0% |
| VideoPlaceButton | 18.0% | 38.0% |
| VideoPlaceOrder | 40.0% | 36.0% |
| VideoRepick | 72.0% | 64.0% |
| VideoUnmask | 68.0% | 58.0% |
| VideoUnmaskSwap | 76.0% | 64.0% |

</details>

## Running baselines

Use the [method READMEs](../../scripts/plus_eval/README.md) for installation and launch
commands. Model runners accept `--output-dir runs/METHOD` and default to all
800 test episodes. ASPIRE/RAX have separate frozen-policy
runners; use the [shared harness](../../scripts/plus_eval/benchmark_io/README.md) to
develop new policies.

### Model runner options

For Astra, FrameSamp+Modul, SimpleMemVLA and PonderPounce, `--resume` continues
missing/interrupted episodes and retries saved errors; `--fail-fast` stops on an
episode error. Keep inference settings fixed and use one writer per output
directory. Use a new directory for a different model, checkpoint or prompt;
`--help` lists all supported options.

Runs save configuration, summaries and logs; model episode folders also contain
results, metadata, videos, actions and policy-call artifacts. Rebuild summaries
without inference:

```bash
.venv/bin/python scripts/plus_eval/evaluation.py summarize runs/METHOD
```

### Reporting and artifacts

Report successes out of 800 and per task out of 50, the setting, the
demonstrations used, the rollout budget, the rollouts and simulator steps actually
used per task, and development and evaluation API costs in USD, using the
[submission template](submission/model_example_plus.md). Our cost estimates come
from saved provider token counts and price lists; missing usage records are
estimated, not counted as zero.

ASPIRE and RAX keep their development code, commands, transcripts and training
rollouts. The 10-rollout runs also keep full test videos. Frozen manifests record
the hash and origin of each evaluated policy. The sources below hold the final
evaluation results.

## Sources

- [Model baseline summaries](../../scripts/plus_eval/results/model_baselines.json)
- [ASPIRE/RAX overall and per-task counts](../../scripts/plus_eval/results/agentic_current.json)
- [ASPIRE/RAX development logs, audits and evaluation results](https://huggingface.co/datasets/Yinpei/RoboMME-plus-agentic-development_logs)
- Frozen solutions: [ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md), [RAX](../../scripts/plus_eval/agentic_methods/rax/solutions/README.md)
- [Logs and rollout locations](../../scripts/plus_eval/results/README.md)
- [Training data format and replay](plus_training_data_format.md)
- Model rollout dataset: [Yinpei/RoboMME-plus-model-rollout](https://huggingface.co/datasets/Yinpei/RoboMME-plus-model-rollout)

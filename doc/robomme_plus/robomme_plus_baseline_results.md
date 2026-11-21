# RoboMME-Plus baseline results

Finalized results on the same **800 test episodes**: 16 tasks × 50 episodes.
The split contains 514 episodes without added variation, 182 with perceptual
variation and 104 with mechanism variation. Only simulator success counts;
failures, timeouts and errors remain in the denominator.

## Evaluation settings

The same settings apply to Non-Agentic policies and Agentic systems. Compare methods
at the same setting and rollout budget, which caps simulator attempts per task on training seeds.

| Setting | Released Plus demonstrations | Maximum Plus rollouts per task |
| --- | --- | --- |
| Zero-shot | No | 0 |
| Demo-only | Yes | 0 |
| Interaction-only | No | 10 |
| Demo+interaction | Yes | 10 |

Original RoboMME's dataset and simulator may be used freely in every setting,
outside the Plus rollout budget. Plus demos are offline recordings; every Plus
simulator start counts, including replay, debugging, failures and verification.
Keep the complete development history and freeze the system before testing.
The [evaluation protocol](../../src/robomme_plus/README.md#evaluation-protocol)
defines permitted resources, audits, reporting and execution limits.


## Zero-shot

These baselines use their pretrained models without RoboMME-Plus training demos
or simulator adaptation.

| Method | Successes | Success rate |
|---|---:|---:|
| [Embodied Astra](<submission/embodied-astra-plus (eval_by_RoboMME_team).md>) | 194/800 | 24.25% |
| [FrameSamp+Modul](<submission/framesamp-modul-plus (eval_by_RoboMME_team).md>) | 19/800 | 2.38% |
| [SimpleMemVLA](<submission/simplememvla-plus (eval_by_RoboMME_team).md>) | 14/800 | 1.75% |
| [3-tier Astra](<submission/3-tier-astra-plus (eval_by_RoboMME_team).md>) | 70/800 | 8.75% |
| [PonderPounce](<submission/ponderpounce-plus (eval_by_RoboMME_team).md>) | 20/800 | 2.50% |

## Demo-only

ASPIRE and RAX can inspect the released 50 demonstrations without simulator
interaction. One-shot Astra receives one selected training example for the task
(medium when labelled; episode 1 for InsertPeg), and none for MoveCube.

| Method | Training input | Successes | Success rate |
|---|---|---:|---:|
| [One-shot Embodied Astra](<submission/one-shot-embodied-astra-plus (eval_by_RoboMME_team).md>) | One selected example per task | 281/800 | 35.13% |
| [ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/README.md) | All 50 demos available | 136/800 | 17.00% |
| [RAX](../../scripts/plus_eval/agentic_methods/rax/README.md) | All 50 demos available | 152/800 | 19.00% |

## Interaction-only

| Maximum attempts per task | ASPIRE | RAX |
|---|---:|---:|
| R10 | 341/800 (42.63%) | 312/800 (39.00%) |

## Demo+interaction

| Maximum attempts per task | ASPIRE | RAX |
|---|---:|---:|
| R10 | 279/800 (34.88%) | 337/800 (42.13%) |

## Optional interaction budgets

R1, R3 and R5 are supplementary results. The leaderboard shows only R10 for
interaction settings. Each historical budget was developed independently.

| Setting | Budget | ASPIRE | RAX |
| --- | --- | ---: | ---: |
| Interaction-only | R1 | 117/800 (14.63%) | 26/800 (3.25%) |
| Interaction-only | R3 | 134/800 (16.75%) | 351/800 (43.88%) |
| Interaction-only | R5 | 215/800 (26.88%) | 265/800 (33.13%) |
| Demo+interaction | R1 | 155/800 (19.38%) | 156/800 (19.50%) |
| Demo+interaction | R3 | 261/800 (32.63%) | 315/800 (39.38%) |
| Demo+interaction | R5 | 252/800 (31.50%) | 283/800 (35.38%) |

## Agentic development audit status

The [ASPIRE submission](<submission/aspire-plus (eval_by_RoboMME_team).md#development-audit>)
and [RAX submission](<submission/rax-plus (eval_by_RoboMME_team).md#development-audit>)
report **PASS WITHIN AUDITED SCOPE** for the historical main configurations'
harness and development records, with disclosed log gaps and interface corrections.
All 1,099 starts across main and optional studies reconcile with native records;
no assigned per-task allowance was exceeded. The main configurations' 11,384 saved
API requests were also inspected. Optional R1/R3/R5 received accounting and
frozen-artifact checks, not a complete API-input audit.

These runs used a pinned training snapshot. Ten recordings and nine execution
limits differ from current v2, so **current-v2 equivalence remains unverified**.
The submissions identify affected tasks and the scope of redevelopment needed
for that claim. Renaming settings alone does not require new runs.

See the [machine-readable evidence](audits/agentic_development_20261121.json) and
[audit record format](submission/model_example_plus.md#agentic-development-audit-record).
The [audit walkthrough](auditing.md) provides the procedure and prompts.
These audits concern development; held-out evaluation remains the fixed test protocol.

## Per-task success rates

Each cell uses 50 episodes. R5 columns are optional supplementary results.
The method submissions also include per-task R1/R3 results.

<details>
<summary>Zero-shot</summary>

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
<summary>Demo-only</summary>

| Task | One-shot Astra | ASPIRE | RAX |
|---|---:|---:|---:|
| BinFill | 26.0% | 66.0% | 20.0% |
| ButtonUnmask | 78.0% | 4.0% | 0.0% |
| ButtonUnmaskSwap | 26.0% | 18.0% | 0.0% |
| InsertPeg | 28.0% | 0.0% | 0.0% |
| MoveCube | 18.0% | 4.0% | 0.0% |
| PatternLock | 18.0% | 0.0% | 0.0% |
| PickHighlight | 42.0% | 0.0% | 0.0% |
| PickXtimes | 80.0% | 20.0% | 8.0% |
| RouteStick | 8.0% | 20.0% | 14.0% |
| StopCube | 38.0% | 0.0% | 54.0% |
| SwingXtimes | 98.0% | 72.0% | 20.0% |
| VideoPlaceButton | 36.0% | 22.0% | 30.0% |
| VideoPlaceOrder | 22.0% | 30.0% | 38.0% |
| VideoRepick | 16.0% | 16.0% | 58.0% |
| VideoUnmask | 28.0% | 0.0% | 62.0% |
| VideoUnmaskSwap | 0.0% | 0.0% | 0.0% |

</details>

<details>
<summary>Interaction-only</summary>

| Task | ASPIRE R5 | ASPIRE R10 | RAX R5 | RAX R10 |
| --- | ---: | ---: | ---: | ---: |
| BinFill | 20.0% | 50.0% | 70.0% | 50.0% |
| ButtonUnmask | 16.0% | 58.0% | 74.0% | 58.0% |
| ButtonUnmaskSwap | 6.0% | 20.0% | 20.0% | 20.0% |
| InsertPeg | 8.0% | 0.0% | 0.0% | 4.0% |
| MoveCube | 0.0% | 12.0% | 4.0% | 4.0% |
| PatternLock | 20.0% | 20.0% | 18.0% | 20.0% |
| PickHighlight | 14.0% | 48.0% | 72.0% | 66.0% |
| PickXtimes | 34.0% | 52.0% | 8.0% | 54.0% |
| RouteStick | 20.0% | 20.0% | 20.0% | 20.0% |
| StopCube | 70.0% | 58.0% | 52.0% | 62.0% |
| SwingXtimes | 38.0% | 64.0% | 6.0% | 12.0% |
| VideoPlaceButton | 36.0% | 58.0% | 0.0% | 40.0% |
| VideoPlaceOrder | 34.0% | 20.0% | 4.0% | 32.0% |
| VideoRepick | 28.0% | 70.0% | 56.0% | 56.0% |
| VideoUnmask | 66.0% | 56.0% | 56.0% | 64.0% |
| VideoUnmaskSwap | 20.0% | 76.0% | 70.0% | 62.0% |

</details>

<details>
<summary>Demo+interaction</summary>

| Task | ASPIRE R5 | ASPIRE R10 | RAX R5 | RAX R10 |
| --- | ---: | ---: | ---: | ---: |
| BinFill | 52.0% | 48.0% | 74.0% | 74.0% |
| ButtonUnmask | 56.0% | 52.0% | 56.0% | 54.0% |
| ButtonUnmaskSwap | 16.0% | 16.0% | 24.0% | 12.0% |
| InsertPeg | 8.0% | 0.0% | 18.0% | 20.0% |
| MoveCube | 2.0% | 2.0% | 2.0% | 6.0% |
| PatternLock | 20.0% | 20.0% | 16.0% | 12.0% |
| PickHighlight | 6.0% | 22.0% | 56.0% | 20.0% |
| PickXtimes | 24.0% | 54.0% | 54.0% | 58.0% |
| RouteStick | 20.0% | 20.0% | 20.0% | 20.0% |
| StopCube | 44.0% | 72.0% | 66.0% | 66.0% |
| SwingXtimes | 36.0% | 70.0% | 30.0% | 74.0% |
| VideoPlaceButton | 22.0% | 20.0% | 24.0% | 34.0% |
| VideoPlaceOrder | 46.0% | 36.0% | 12.0% | 32.0% |
| VideoRepick | 60.0% | 56.0% | 58.0% | 58.0% |
| VideoUnmask | 60.0% | 60.0% | 56.0% | 60.0% |
| VideoUnmaskSwap | 32.0% | 10.0% | 0.0% | 74.0% |

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

Report successes/800, per-task successes/50, setting, demo use, R, actual attempts
and simulator steps per task, and separate development/evaluation API costs in USD.
Use the [submission template](submission/model_example_plus.md). Cost estimates
use saved provider usage and rate cards; missing usage is not zero cost.

ASPIRE/RAX retain development code, commands, transcripts and training rollouts.
R0/R10 retain full test media; R1/R3/R5 retain results and logs. Frozen manifests
record the evaluated source hashes and policy provenance. The summaries below
contain the final evaluation results.

## Sources

- [Model baseline summaries](../../scripts/plus_eval/results/model_baselines.json)
- [ASPIRE/RAX overall and per-task counts](../../scripts/plus_eval/results/agentic_20261118.json)
- Frozen solutions: [ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md), [RAX](../../scripts/plus_eval/agentic_methods/rax/solutions/README.md)
- [Logs and rollout locations](../../scripts/plus_eval/results/README.md)
- [Training data format and replay](plus_training_data_format.md)
- Model rollout dataset: [Yinpei/RoboMME-plus-model-rollout](https://huggingface.co/datasets/Yinpei/RoboMME-plus-model-rollout)

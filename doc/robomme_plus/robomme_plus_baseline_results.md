# RoboMME-Plus baseline results

Finalized results on the same **800 test episodes**: 16 tasks × 50 episodes.
The split contains 514 episodes without added variation, 182 with perceptual
variation and 104 with mechanism variation. Only simulator success counts;
failures, timeouts and errors remain in the denominator.

## Evaluation settings

The same settings apply to Non-Agentic policies and Agentic systems. Compare methods
at the same setting and rollout budget, which caps simulator attempts per task on training seeds.

| Setting | Released demonstrations | Rollout budget per task |
| --- | --- | --- |
| Zero-shot | No | 0 |
| Few-shot demonstrations | Yes | 0 |
| Few-shot interactions | No | 5 or 10 |
| Few-shot demonstrations+interactions | Yes | 5 or 10 |

- Use only released training demos and seeds for the chosen setting. Offline
  demo reuse is unlimited. Every development simulator start counts toward the rollout budget,
  including failed setup, timeouts, early stops, retries and frozen-policy rehearsal.
  Prompt/harness tuning, debugging, verification and selection share this budget.
- Use public observations and feedback; task-source-derived solutions, oracle
  queries and privileged simulator information are forbidden. A task's conditioning video is a normal
  observation in every setting, distinct from an expert training recording.
  Task-specific prompts and strategies must come from permitted experience.
- Freeze weights, programs, prompts, tools and libraries before testing; reset
  adaptive memory between episodes. Test feedback cannot revise or select policies.
  Report test-time optimization separately.

The [evaluation guide](../../src/robomme_plus/README.md#evaluation-protocol) defines
the shared interface, default variations and execution limits.


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

## Few-shot demonstrations

ASPIRE and RAX can inspect the released 50 demonstrations without simulator
interaction. One-shot Astra receives one selected training example for the task
(medium when labelled; episode 1 for InsertPeg), and none for MoveCube.

| Method | Training input | Successes | Success rate |
|---|---|---:|---:|
| [One-shot Embodied Astra](<submission/one-shot-embodied-astra-plus (eval_by_RoboMME_team).md>) | One selected example per task | 281/800 | 35.13% |
| [ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/README.md) | All 50 demos available | 136/800 | 17.00% |
| [RAX](../../scripts/plus_eval/agentic_methods/rax/README.md) | All 50 demos available | 152/800 | 19.00% |

## Few-shot interactions

| Maximum attempts per task | ASPIRE | RAX |
|---|---:|---:|
| R5 | 215/800 (26.88%) | 265/800 (33.13%) |
| R10 | 341/800 (42.63%) | 312/800 (39.00%) |

## Few-shot demonstrations+interactions

| Maximum attempts per task | ASPIRE | RAX |
|---|---:|---:|
| R5 | 252/800 (31.50%) | 283/800 (35.38%) |
| R10 | 279/800 (34.88%) | 337/800 (42.13%) |

Current interaction results use R5 and R10 (5 or 10 attempts per task).
Archived R1/R3 results and policies remain in the linked JSON files and solution
directories; their budgets are not relabeled.

## Per-task success rates

Each cell uses 50 episodes. The tables below use the same setting and budget
as the overall results above.

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
<summary>Few-shot demonstrations</summary>

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
<summary>Few-shot interactions</summary>

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
<summary>Few-shot demonstrations+interactions</summary>

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

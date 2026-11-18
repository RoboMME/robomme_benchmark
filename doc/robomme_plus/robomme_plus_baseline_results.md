# RoboMME-Plus baseline results

Finalized results on the same **800 test episodes**: 16 tasks × 50 episodes.
The split contains 514 episodes without added variation, 182 with perceptual
variation and 104 with mechanism variation. Only simulator success counts;
failures, timeouts and errors remain in the denominator.

## Evaluation settings

The same settings apply to Standard policies and Agentic systems. Compare methods
at the same setting and R; R caps simulator attempts per task on training seeds.

| Setting | Released demonstrations | Rollout budget per task |
| --- | --- | --- |
| Zero-shot transfer (ZS) | No | R0 |
| Adapt from demos (AfD) | Yes; use listed below | R0 |
| Adapt from interactions (AfI) | No | R1, R3, R5 or R10 |
| Adapt from demos+interactions (AfD+I) | Yes | R1, R3, R5 or R10 |

- Use only released training demos and seeds for the chosen setting. Offline
  demo reuse is unlimited. Every development simulator start counts toward R,
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

### ASPIRE and RAX adaptation notes

Both retain their LIBERO development workflows. The choices below describe
these baseline runs; time limits, model/provider choices and input sampling are
implementation settings, not additional requirements for every benchmark method.
Core submodules stay unchanged; integration patches apply to runtime copies.

| Area | RoboMME-Plus adaptation |
| --- | --- |
| Development model | Opus 5.5 through AWS Bedrock, high reasoning effort; usage/cost logged separately from evaluation. |
| Settings | Nine independent runs per method: AfD R0, AfI and AfD+I at R1/R3/R5/R10. Lower budgets are not R10 checkpoints. |
| Inputs | AfD/AfD+I receive front/wrist RGB training recordings with goals, phases and timestamps. Live inputs include RGB-D, calibration and robot state. AfI receives no released expert recordings. |
| Interface | Public robot actions, observations, conditioning videos and terminal/error feedback replace LIBERO access; task source and hidden object state stay outside agent workspaces. |
| Development deadline | **150 minutes per task per run**, including model calls, offline work and simulator attempts. Clocks persist across resumes. At expiry, stop development and preserve existing code for reporting/submission. |
| Rollout accounting | All training starts share R, including verification, failed setup and frozen-bundle rehearsals. AfD permits no simulator rehearsal. |
| Video transport | Uniformly sample long conditioning videos to **512 timestamps**, retaining endpoints, both cameras and a separate initial live frame. Released training recordings remain available offline. |
| Frozen testing | Run selected Python policies and local providers, with no frontier-model API calls, test-driven revision or cross-episode adaptive memory. |
| Reset recovery | Development uses one setup attempt per reserved start. Test setup can retry three times with identical settings; policy execution is not retried automatically. |

The shared [runtime policy](../../scripts/plus_eval/benchmark_io/runtime_policy.py)
is copied into each method's `solutions/runtime_policy.json`:

| Runtime limit | Value |
| --- | --- |
| Controller wall time per rollout | 20 minutes |
| Setup / initial wait for a control step | 10 minutes each |
| No control-step progress after execution starts | 3 minutes |
| Provider request / simulator RPC | 60 seconds / 180 seconds |
| Outer episode worker | 35 minutes, including startup and cleanup |

These watchdogs also apply to development rollouts and do not extend native
simulator step limits. Timeouts remain unsuccessful test outcomes. The queue's
48-hour job cap and 30-minute idle watchdog are outer safeguards, not extra
per-task development time. Clocks and expiry records are saved under
`study/development_clocks/`; full commands, transcripts and API usage remain in
run artifacts.

Method-specific workflows, deadline handling and policy provenance are documented in [ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/README.md#adaptation)
and [RAX](../../scripts/plus_eval/agentic_methods/rax/README.md#adaptation).

## Zero-shot transfer (ZS)

These baselines use their pretrained models without RoboMME-Plus training demos
or simulator adaptation.

| Method | Successes | Success rate |
|---|---:|---:|
| [Embodied Astra](<submission/embodied-astra-plus (eval_by_RoboMME_team).md>) | 194/800 | 24.25% |
| [FrameSamp+Modul](<submission/framesamp-modul-plus (eval_by_RoboMME_team).md>) | 19/800 | 2.38% |
| [SimpleMemVLA](<submission/simplememvla-plus (eval_by_RoboMME_team).md>) | 14/800 | 1.75% |
| [3-tier Astra](<submission/3-tier-astra-plus (eval_by_RoboMME_team).md>) | 70/800 | 8.75% |
| [PonderPounce](<submission/ponderpounce-plus (eval_by_RoboMME_team).md>) | 20/800 | 2.50% |

## Adapt from demos (AfD, R0)

ASPIRE and RAX can inspect the released 50 demonstrations without simulator
interaction. One-shot Astra receives one selected training example for the task
(medium when labelled; episode 1 for InsertPeg), and none for MoveCube.

| Method | Training input | Successes | Success rate |
|---|---|---:|---:|
| [One-shot Embodied Astra](<submission/one-shot-embodied-astra-plus (eval_by_RoboMME_team).md>) | One selected example per task | 281/800 | 35.13% |
| [ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/README.md) | All 50 demos available | 135/800 | 16.88% |
| [RAX](../../scripts/plus_eval/agentic_methods/rax/README.md) | All 50 demos available | 152/800 | 19.00% |

## Adapt from interactions (AfI)

| Maximum attempts per task | ASPIRE | RAX |
|---|---:|---:|
| R1 | 117/800 (14.63%) | 26/800 (3.25%) |
| R3 | 134/800 (16.75%) | 351/800 (43.88%) |
| R5 | 213/800 (26.63%) | 265/800 (33.13%) |
| R10 | 340/800 (42.50%) | 312/800 (39.00%) |

## Adapt from demos+interactions (AfD+I)

| Maximum attempts per task | ASPIRE | RAX |
|---|---:|---:|
| R1 | 153/800 (19.13%) | 156/800 (19.50%) |
| R3 | 260/800 (32.50%) | 315/800 (39.38%) |
| R5 | 250/800 (31.25%) | 283/800 (35.38%) |
| R10 | 273/800 (34.13%) | 337/800 (42.13%) |

## Per-task success rates

Each cell uses 50 episodes. The tables below use the same setting and budget
as the overall results above.

<details>
<summary>Zero-shot transfer (ZS)</summary>

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
<summary>Adapt from demos (AfD, R0)</summary>

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
| SwingXtimes | 98.0% | 70.0% | 20.0% |
| VideoPlaceButton | 36.0% | 22.0% | 30.0% |
| VideoPlaceOrder | 22.0% | 30.0% | 38.0% |
| VideoRepick | 16.0% | 16.0% | 58.0% |
| VideoUnmask | 28.0% | 0.0% | 62.0% |
| VideoUnmaskSwap | 0.0% | 0.0% | 0.0% |

</details>

<details>
<summary>Adapt from interactions (AfI)</summary>

| Task | ASPIRE R1 | ASPIRE R3 | ASPIRE R5 | ASPIRE R10 | RAX R1 | RAX R3 | RAX R5 | RAX R10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BinFill | 2.0% | 10.0% | 20.0% | 50.0% | 12.0% | 74.0% | 70.0% | 50.0% |
| ButtonUnmask | 6.0% | 10.0% | 16.0% | 58.0% | 12.0% | 24.0% | 74.0% | 58.0% |
| ButtonUnmaskSwap | 2.0% | 10.0% | 6.0% | 20.0% | 0.0% | 22.0% | 20.0% | 20.0% |
| InsertPeg | 0.0% | 0.0% | 8.0% | 0.0% | 0.0% | 32.0% | 0.0% | 4.0% |
| MoveCube | 0.0% | 8.0% | 0.0% | 12.0% | 0.0% | 2.0% | 4.0% | 4.0% |
| PatternLock | 20.0% | 20.0% | 20.0% | 20.0% | 0.0% | 20.0% | 18.0% | 20.0% |
| PickHighlight | 0.0% | 6.0% | 14.0% | 48.0% | 0.0% | 42.0% | 72.0% | 66.0% |
| PickXtimes | 14.0% | 14.0% | 30.0% | 52.0% | 0.0% | 54.0% | 8.0% | 54.0% |
| RouteStick | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% |
| StopCube | 66.0% | 88.0% | 70.0% | 58.0% | 2.0% | 64.0% | 52.0% | 62.0% |
| SwingXtimes | 20.0% | 14.0% | 38.0% | 62.0% | 0.0% | 58.0% | 6.0% | 12.0% |
| VideoPlaceButton | 14.0% | 10.0% | 36.0% | 58.0% | 0.0% | 70.0% | 0.0% | 40.0% |
| VideoPlaceOrder | 18.0% | 26.0% | 34.0% | 20.0% | 2.0% | 48.0% | 4.0% | 32.0% |
| VideoRepick | 32.0% | 30.0% | 28.0% | 70.0% | 0.0% | 48.0% | 56.0% | 56.0% |
| VideoUnmask | 4.0% | 2.0% | 66.0% | 56.0% | 4.0% | 66.0% | 56.0% | 64.0% |
| VideoUnmaskSwap | 16.0% | 0.0% | 20.0% | 76.0% | 0.0% | 58.0% | 70.0% | 62.0% |

</details>

<details>
<summary>Adapt from demos+interactions (AfD+I)</summary>

| Task | ASPIRE R1 | ASPIRE R3 | ASPIRE R5 | ASPIRE R10 | RAX R1 | RAX R3 | RAX R5 | RAX R10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BinFill | 26.0% | 84.0% | 52.0% | 48.0% | 72.0% | 54.0% | 74.0% | 74.0% |
| ButtonUnmask | 46.0% | 36.0% | 56.0% | 48.0% | 0.0% | 58.0% | 56.0% | 54.0% |
| ButtonUnmaskSwap | 0.0% | 4.0% | 16.0% | 16.0% | 2.0% | 16.0% | 24.0% | 12.0% |
| InsertPeg | 6.0% | 2.0% | 8.0% | 0.0% | 2.0% | 42.0% | 18.0% | 20.0% |
| MoveCube | 0.0% | 6.0% | 2.0% | 2.0% | 4.0% | 6.0% | 2.0% | 6.0% |
| PatternLock | 0.0% | 20.0% | 20.0% | 20.0% | 20.0% | 20.0% | 16.0% | 12.0% |
| PickHighlight | 16.0% | 20.0% | 6.0% | 22.0% | 14.0% | 32.0% | 56.0% | 20.0% |
| PickXtimes | 26.0% | 62.0% | 24.0% | 54.0% | 20.0% | 48.0% | 54.0% | 58.0% |
| RouteStick | 0.0% | 20.0% | 20.0% | 20.0% | 14.0% | 16.0% | 20.0% | 20.0% |
| StopCube | 26.0% | 58.0% | 44.0% | 64.0% | 30.0% | 62.0% | 66.0% | 66.0% |
| SwingXtimes | 4.0% | 38.0% | 32.0% | 70.0% | 20.0% | 76.0% | 30.0% | 74.0% |
| VideoPlaceButton | 10.0% | 20.0% | 22.0% | 20.0% | 26.0% | 20.0% | 24.0% | 34.0% |
| VideoPlaceOrder | 32.0% | 34.0% | 46.0% | 36.0% | 54.0% | 22.0% | 12.0% | 32.0% |
| VideoRepick | 74.0% | 60.0% | 60.0% | 56.0% | 34.0% | 38.0% | 58.0% | 58.0% |
| VideoUnmask | 38.0% | 10.0% | 60.0% | 60.0% | 0.0% | 52.0% | 56.0% | 60.0% |
| VideoUnmaskSwap | 2.0% | 46.0% | 32.0% | 10.0% | 0.0% | 68.0% | 0.0% | 74.0% |

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
and simulator steps per task, and separate adaptation/evaluation API costs in USD.
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

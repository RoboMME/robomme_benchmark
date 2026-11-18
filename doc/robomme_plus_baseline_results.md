# RoboMME-Plus baseline results

Finalized results on the same **800 test episodes**: 16 tasks × 50 episodes.
The split contains 514 episodes without added variation, 182 with perceptual
variation and 104 with mechanism variation. Only simulator success counts;
failures, timeouts and errors remain in the denominator.

## Evaluation settings

| Setting | Released training demonstrations | Simulator starts per task during adaptation |
|---|---|---|
| ZS: direct transfer | None | 0 |
| AfD: demonstrations only | Yes; usage listed below | 0 (R0) |
| AfI: interaction only | None | At most R1, R3, R5 or R10 |
| AfD+I: demonstrations + interaction | All 50 available | At most R1, R3, R5 or R10 |

R counts simulator starts on **training seeds**, including verification,
debugging and rehearsal. ASPIRE/RAX develop each budget independently; R1/R3/R5
are not checkpoints of R10. Programs and libraries freeze before testing.
A task's conditioning video is part of its test observation in every setting;
it is distinct from a released expert training demonstration.
See the [adaptation protocol](plus_agentic_evaluation.md).

## ZS: direct transfer

These baselines use their pretrained models without RoboMME-Plus training demos
or simulator adaptation.

| Method | Successes | Success rate |
|---|---:|---:|
| [Embodied Astra](<submission/embodied-astra-plus (eval_by_RoboMME_team).md>) | 194/800 | 24.25% |
| [FrameSamp+Modul](<submission/framesamp-modul-plus (eval_by_RoboMME_team).md>) | 19/800 | 2.38% |
| [SimpleMemVLA](<submission/simplememvla-plus (eval_by_RoboMME_team).md>) | 14/800 | 1.75% |
| [3-tier Astra](<submission/3-tier-astra-plus (eval_by_RoboMME_team).md>) | 70/800 | 8.75% |
| [PonderPounce](<submission/ponderpounce-plus (eval_by_RoboMME_team).md>) | 20/800 | 2.50% |

## AfD: demonstrations only (R0)

ASPIRE and RAX can inspect the released 50 demonstrations without simulator
interaction. One-shot Astra receives one selected training example for the task
(medium when labelled; episode 1 for InsertPeg), and none for MoveCube.

| Method | Training input | Successes | Success rate |
|---|---|---:|---:|
| [One-shot Embodied Astra](<submission/one-shot-embodied-astra-plus (eval_by_RoboMME_team).md>) | One selected example per task | 281/800 | 35.13% |
| [ASPIRE](../scripts/plus_eval/agentic_methods/aspire/README.md) | All 50 demos available | 135/800 | 16.88% |
| [RAX](../scripts/plus_eval/agentic_methods/rax/README.md) | All 50 demos available | 152/800 | 19.00% |

## AfI: interaction only

| Maximum starts per task | ASPIRE | RAX |
|---|---:|---:|
| R1 | 117/800 (14.63%) | 26/800 (3.25%) |
| R3 | 134/800 (16.75%) | 351/800 (43.88%) |
| R5 | 213/800 (26.63%) | 265/800 (33.13%) |
| R10 | 340/800 (42.50%) | 312/800 (39.00%) |

## AfD+I: demonstrations + interaction

| Maximum starts per task | ASPIRE | RAX |
|---|---:|---:|
| R1 | 153/800 (19.13%) | 156/800 (19.50%) |
| R3 | 260/800 (32.50%) | 315/800 (39.38%) |
| R5 | 250/800 (31.25%) | 283/800 (35.38%) |
| R10 | 273/800 (34.13%) | 337/800 (42.13%) |

## Per-task success rates

Each cell uses 50 episodes. The tables below use the same setting and budget
as the overall results above.

<details>
<summary>ZS: direct transfer</summary>

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
<summary>AfD: demonstrations only (R0)</summary>

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
<summary>AfI: interaction only</summary>

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
<summary>AfD+I: demonstrations + interaction</summary>

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

## Sources and reproduction

The 18 ASPIRE/RAX configurations use the finalized full evaluations from
`full_20261116_protocol_v2`. Their manifests record the frozen source hashes and
any operator interface corrections. Frozen testing uses local providers without
frontier-model API calls. Separate reproducibility sanity runs do not replace
these full-evaluation totals.

- [Model baseline summaries](../scripts/plus_eval/results/model_baselines.json)
- [ASPIRE/RAX overall and per-task counts](../scripts/plus_eval/results/agentic_20261118.json)
- [Evaluation entry points and run options](../scripts/plus_eval/README.md)
- Frozen solutions: [ASPIRE](../scripts/plus_eval/agentic_methods/aspire/solutions/README.md), [RAX](../scripts/plus_eval/agentic_methods/rax/solutions/README.md)
- [Logs and rollout locations](../scripts/plus_eval/results/README.md)
- Model rollout dataset: [Yinpei/RoboMME-plus-model-rollout](https://huggingface.co/datasets/Yinpei/RoboMME-plus-model-rollout)

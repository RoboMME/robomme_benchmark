# RoboMME-Plus baseline results

V2 results on 800 test episodes per baseline: 16 tasks with 50 episodes each. Only simulator success counts; failures and timeouts remain in the denominator.

| Model | Successes | Success rate |
|---|---:|---:|
| [Embodied Astra](<submission/embodied-astra-plus (eval_by_RoboMME_team).md>) | 194/800 | 24.25% |
| [One-shot Embodied Astra](<submission/one-shot-embodied-astra-plus (eval_by_RoboMME_team).md>) | 281/800 | 35.13% |
| [FrameSamp+Modul](<submission/framesamp-modul-plus (eval_by_RoboMME_team).md>) | 19/800 | 2.38% |
| [SimpleMemVLA](<submission/simplememvla-plus (eval_by_RoboMME_team).md>) | 14/800 | 1.75% |
| [3-tier Astra](<submission/3-tier-astra-plus (eval_by_RoboMME_team).md>) | 70/800 | 8.75% |
| [PonderPounce](<submission/ponderpounce-plus (eval_by_RoboMME_team).md>) | 20/800 | 2.50% |

## Per-task success rates

| Task | Embodied Astra | One-shot Embodied Astra | FrameSamp+Modul | SimpleMemVLA | 3-tier Astra | PonderPounce |
|---|---:|---:|---:|---:|---:|---:|
| BinFill | 26.0% | 26.0% | 0.0% | 0.0% | 6.0% | 2.0% |
| ButtonUnmask | 34.0% | 78.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| ButtonUnmaskSwap | 12.0% | 26.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| InsertPeg | 20.0% | 28.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MoveCube | 22.0% | 18.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PatternLock | 14.0% | 18.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| PickHighlight | 34.0% | 42.0% | 0.0% | 0.0% | 2.0% | 0.0% |
| PickXtimes | 54.0% | 80.0% | 0.0% | 0.0% | 6.0% | 0.0% |
| RouteStick | 12.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| StopCube | 22.0% | 38.0% | 4.0% | 0.0% | 2.0% | 2.0% |
| SwingXtimes | 32.0% | 98.0% | 0.0% | 8.0% | 10.0% | 8.0% |
| VideoPlaceButton | 36.0% | 36.0% | 16.0% | 14.0% | 56.0% | 18.0% |
| VideoPlaceOrder | 26.0% | 22.0% | 14.0% | 4.0% | 50.0% | 2.0% |
| VideoRepick | 22.0% | 16.0% | 4.0% | 2.0% | 4.0% | 8.0% |
| VideoUnmask | 20.0% | 28.0% | 0.0% | 0.0% | 4.0% | 0.0% |
| VideoUnmaskSwap | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

V2 includes more held-out perceptual and mechanism variations to make testing harder.

Rollout dataset: [Yinpei/RoboMME-plus-model-rollout](https://huggingface.co/datasets/Yinpei/RoboMME-plus-model-rollout).

## Agentic adaptation

Saved full evaluation from `full_20261116_protocol_v2`; each row uses 800 episodes
with both variation types enabled. Interaction budgets were developed independently.

| Setting | R | ASPIRE | RAX |
| --- | ---: | ---: | ---: |
| AfD | 0 | 135/800 (16.88%) | 152/800 (19.00%) |
| AfI | 1 | 117/800 (14.62%) | 26/800 (3.25%) |
| AfI | 3 | 134/800 (16.75%) | 351/800 (43.88%) |
| AfI | 5 | 213/800 (26.62%) | 265/800 (33.12%) |
| AfI | 10 | 340/800 (42.50%) | 312/800 (39.00%) |
| AfD+I | 1 | 153/800 (19.12%) | 156/800 (19.50%) |
| AfD+I | 3 | 260/800 (32.50%) | 315/800 (39.38%) |
| AfD+I | 5 | 250/800 (31.25%) | 283/800 (35.38%) |
| AfD+I | 10 | 273/800 (34.12%) | 337/800 (42.12%) |

Three RAX RPC infrastructure errors remain counted as failures (AfI R1: two;
AfD+I R3: one). Controller timeouts also remain failures. The published frozen
solution manifests identify any operator interface corrections used in evaluation.
See [machine-readable per-task counts](../scripts/plus_eval/results/agentic_20261118.json)
and the [protocol](plus_agentic_evaluation.md).

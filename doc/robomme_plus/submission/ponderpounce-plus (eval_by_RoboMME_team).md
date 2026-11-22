# PonderPounce

### [Website](https://worv-ai.github.io/ponderpounce/) | [Paper](https://arxiv.org/abs/2608.24115) | [Code](https://github.com/worv-ai/PonderPounce)

## Introduction

PonderPounce combines a Qwen3.5-9B episode-context model (Ponder) with a Pi0.5 action controller (Pounce). Ponder maintains visual history and produces a cognition representation that conditions Pounce. We evaluate the released RoboMME checkpoint on RoboMME-Plus.

## Results

**Track:** Non-Agentic Track.

**Total Params:** **13.04B** (13,039,166,464), including Ponder, Pounce and their adapters.
Frozen parameters are included; shared weights are counted once.

**Evaluated by:** RoboMME Team. **Evaluation:** Final RoboMME-Plus test split (800 episodes).

**Setting:** `Zero-shot`.

**Demo use:** No released RoboMME-Plus training demonstrations.

**Rollout budget:** 0; 0 training-seed attempts.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`joint_angle`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

Success rates are percentages on the RoboMME-Plus V2 test split. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 1/50 | 2.00 |
| Counting | PickXtimes | 0/50 | 0.00 |
| Counting | SwingXtimes | 4/50 | 8.00 |
| Counting | StopCube | 1/50 | 2.00 |
| Permanence | VideoUnmask | 0/50 | 0.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 0/50 | 0.00 |
| Permanence | ButtonUnmaskSwap | 0/50 | 0.00 |
| Reference | PickHighlight | 0/50 | 0.00 |
| Reference | VideoRepick | 4/50 | 8.00 |
| Reference | VideoPlaceButton | 9/50 | 18.00 |
| Reference | VideoPlaceOrder | 1/50 | 2.00 |
| Imitation | MoveCube | 0/50 | 0.00 |
| Imitation | InsertPeg | 0/50 | 0.00 |
| Imitation | PatternLock | 0/50 | 0.00 |
| Imitation | RouteStick | 0/50 | 0.00 |
| **Overall** | **All tasks** | **20/800** | **2.50** |

**Overall:** 20/800 successes (**2.50%**); 603 failures and 177 timeouts.

[Baseline results](../robomme_plus_baseline_results.md).

### Training Details

- **RoboMME-Plus adaptation:** none. The released 9B RoboMME checkpoint is shared across all tasks; model sampling seed 0.
- **Inference:** BF16, 10 denoising steps, 20-action chunks at 20 Hz, one-second Ponder/Pounce periods, and a 300 ms cognition delay.
- **RoboMME-Plus execution:** the initial hold converts gripper positions into normalized actions. Ponder uses a 16,384-token context limit and rebuilds its caches after trimming old execution queries.

### Reproduce

[Evaluation setup and commands](../../../scripts/plus_eval/non_agentic_methods/ponderpounce/README.md).

### Released Checkpoints

The evaluated checkpoint is [worv-ai/ponderpounce-9b-robomme](https://huggingface.co/worv-ai/ponderpounce-9b-robomme), revision `5691e05614e8e6581d6358959e60561acfa73534`. It contains both Ponder and Pounce; no RoboMME-Plus fine-tuning was performed.

### Citations

- [PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control](https://arxiv.org/abs/2608.24115).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

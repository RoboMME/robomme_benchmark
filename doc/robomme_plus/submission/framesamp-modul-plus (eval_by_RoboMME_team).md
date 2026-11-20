# FrameSamp+Modul

### [Website](https://robomme.github.io/) | [Paper](https://arxiv.org/abs/2603.04639) | [Code](https://github.com/RoboMME/robomme_policy_learning)

## Introduction

FrameSamp+Modul is the RoboMME perceptual-memory baseline. It samples historical visual frames and uses their features to modulate a Pi0.5 action policy. We evaluate the released RoboMME checkpoint directly on RoboMME-Plus.

## Results

**Track:** Non-Agentic Track.

**Total Params:** **3.44B** (3,441,899,024), including frozen parameters.

**Evaluated by:** RoboMME Team. **Evaluation:** Final RoboMME-Plus test split (800 episodes).

**Setting:** Zero-shot.

**Demo use:** No released RoboMME-Plus training demonstrations.

**Rollout budget:** 0; 0 training-seed attempts.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`joint_angle`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

Success rates are percentages on the RoboMME-Plus V2 test split. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 0/50 | 0.00 |
| Counting | PickXtimes | 0/50 | 0.00 |
| Counting | SwingXtimes | 0/50 | 0.00 |
| Counting | StopCube | 2/50 | 4.00 |
| Permanence | VideoUnmask | 0/50 | 0.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 0/50 | 0.00 |
| Permanence | ButtonUnmaskSwap | 0/50 | 0.00 |
| Reference | PickHighlight | 0/50 | 0.00 |
| Reference | VideoRepick | 2/50 | 4.00 |
| Reference | VideoPlaceButton | 8/50 | 16.00 |
| Reference | VideoPlaceOrder | 7/50 | 14.00 |
| Imitation | MoveCube | 0/50 | 0.00 |
| Imitation | InsertPeg | 0/50 | 0.00 |
| Imitation | PatternLock | 0/50 | 0.00 |
| Imitation | RouteStick | 0/50 | 0.00 |
| **Overall** | **All tasks** | **19/800** | **2.38** |

**Overall:** 19/800 successes (**2.38%**); 645 failures and 136 timeouts.

[Baseline results](../robomme_plus_baseline_results.md).

### Training Details

- **RoboMME-Plus adaptation:** none. The original RoboMME-trained checkpoint is shared across all 16 tasks.
- **Checkpoint:** `perceptual-framesamp-modul/79999`; model sampling seed 7.
- **Inference:** upstream history sampling and modulation, with 16-action chunks. Demonstrations longer than 40 seconds are uniformly sampled to 800 frames; execution history retains its original cadence.

### Reproduce

[Evaluation setup and commands](../../../scripts/plus_eval/non_agentic_methods/framesamp_modul/README.md).

### Released Checkpoints

The original checkpoint is available in [Yinpei/mme_vla_suite](https://huggingface.co/Yinpei/mme_vla_suite/tree/main/perceptual-framesamp-modul). This entry uses the original RoboMME checkpoint; it is not the separately evaluated RoboMME-Plus-fine-tuned variant.

### Citations

- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

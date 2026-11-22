# 3-tier Astra

### [Website](https://bingaochen.github.io/Astra-on-RoboMME/) | [Code](https://github.com/bingaochen/Astra-on-RoboMME)

## Introduction

3-tier Astra separates planning, monitoring, and action. GPT-6 Astra proposes grounded subtasks, a Qwen3-VL-4B completion monitor decides when to replan, and a GroundSG Pi0.5 policy executes joint actions. We transfer the released system to RoboMME-Plus.

## Results

**Track:** Agentic Track.

**Final evaluation API cost:** Not aggregated.

**Evaluated by:** RoboMME Team. **Evaluation:** Final RoboMME-Plus test split (800 episodes).

**Setting:** `Zero-shot`.

**Demo use:** No released RoboMME-Plus training demonstrations.

**Rollout budget:** 0; 0 training-seed attempts.

> Multi-task evaluation on the RoboMME-Plus test split: **16 tasks × 50 episodes = 800 episodes**, using **`joint_angle`** actions. Each episode uses its supplied execution budget, `ceil(1.5 * oracle_policy_step) + 200` control steps.

Success rates are percentages on the RoboMME-Plus V2 test split. Only simulator success counts as success. All 800 episodes remain in the denominator, including failures, timeouts, model stops and errors.

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 3/50 | 6.00 |
| Counting | PickXtimes | 3/50 | 6.00 |
| Counting | SwingXtimes | 5/50 | 10.00 |
| Counting | StopCube | 1/50 | 2.00 |
| Permanence | VideoUnmask | 2/50 | 4.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 0/50 | 0.00 |
| Permanence | ButtonUnmaskSwap | 0/50 | 0.00 |
| Reference | PickHighlight | 1/50 | 2.00 |
| Reference | VideoRepick | 2/50 | 4.00 |
| Reference | VideoPlaceButton | 28/50 | 56.00 |
| Reference | VideoPlaceOrder | 25/50 | 50.00 |
| Imitation | MoveCube | 0/50 | 0.00 |
| Imitation | InsertPeg | 0/50 | 0.00 |
| Imitation | PatternLock | 0/50 | 0.00 |
| Imitation | RouteStick | 0/50 | 0.00 |
| **Overall** | **All tasks** | **70/800** | **8.75** |

**Overall:** 70/800 successes (**8.75%**); 466 failures and 264 timeouts.

[Baseline results](../robomme_plus_baseline_results.md).

### Training Details

- **RoboMME-Plus adaptation:** none. The RoboMME-trained GroundSG policy and completion monitor are reused without RoboMME-Plus fine-tuning or training examples.
- **Components:** the Astra API planner, Qwen3-VL-4B with the released monitor adapter, and `symbolic-grounded-subgoal/79999`; model sampling seed 42.
- **Planner input:** visual history uses contact sheets with up to 16 frames each. Initial planner inputs are capped at 750 API images, and independent requests at 1,500 images. Oversized requests count as unsuccessful episodes.

### Reproduce

[Evaluation setup and commands](../../../scripts/plus_eval/agentic_methods/astra/three_tier/README.md).

### Released Checkpoints

The local components are [Yinpei/mme_vla_suite: symbolic-grounded-subgoal](https://huggingface.co/Yinpei/mme_vla_suite/tree/main/symbolic-grounded-subgoal), [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and [bingaochen/Astra-on-RoboMME-Monitor](https://huggingface.co/bingaochen/Astra-on-RoboMME-Monitor). Astra remains an API model.

### Citations

- [Astra on RoboMME](https://bingaochen.github.io/Astra-on-RoboMME/).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

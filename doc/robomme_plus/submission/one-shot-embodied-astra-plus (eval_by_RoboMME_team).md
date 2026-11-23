# One-shot Embodied-Astra

### [Code](https://github.com/jackbjed/embodied-astra-robomme) | [RoboMME-Plus training data](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5)

## Introduction

One-shot Embodied-Astra uses the same `gpt-6-astra` waypoint policy as Embodied-Astra, with a successful training episode prepended as context. The example supplies the task goal and sampled RGB frames, allowing adaptation within the prompt without updating model weights.

## Results

**Track:** Agentic Track.

**Final evaluation API cost:** ≈$2,100 for all 800 test episodes.

**Evaluated by:** RoboMME Team. **Evaluation:** Final RoboMME-Plus test split (800 episodes).

**Setting:** `Demo-only`, using a released training example as context.

**Demo use:** One selected training episode per task, except MoveCube; RGB images and task goals.

**Rollout budget:** 0; 0 training-seed attempts.

> Evaluated on all 800 test episodes with **`waypoint`** actions.

### Per-task success

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | 13/50 | 26.00 |
| Counting | PickXtimes | 40/50 | 80.00 |
| Counting | SwingXtimes | 49/50 | 98.00 |
| Counting | StopCube | 19/50 | 38.00 |
| Permanence | VideoUnmask | 14/50 | 28.00 |
| Permanence | VideoUnmaskSwap | 0/50 | 0.00 |
| Permanence | ButtonUnmask | 39/50 | 78.00 |
| Permanence | ButtonUnmaskSwap | 13/50 | 26.00 |
| Reference | PickHighlight | 21/50 | 42.00 |
| Reference | VideoRepick | 8/50 | 16.00 |
| Reference | VideoPlaceButton | 18/50 | 36.00 |
| Reference | VideoPlaceOrder | 11/50 | 22.00 |
| Imitation | MoveCube | 9/50 | 18.00 |
| Imitation | InsertPeg | 14/50 | 28.00 |
| Imitation | PatternLock | 9/50 | 18.00 |
| Imitation | RouteStick | 4/50 | 8.00 |
| **Overall** | **All tasks** | **281/800** | **35.13** |

**Overall:** 281/800 successes (**35.13%**); 481 failures and 38 timeouts.

### Training Details

- **Adaptation:** one medium training example for each of 15 tasks. InsertPeg uses training episode 1; MoveCube receives no training example.
- **Example format:** front and wrist images are paired side by side, with demonstration/execution phase labels and the training goal. The context includes RGB images and the goal; recorded actions and subgoal annotations are excluded.
- **Current episode:** the same front-camera history, current wrist view, robot state, camera calibration, and `lr_note` prompt as the `Zero-shot` policy.
- **Image budget:** 1,100 images in the initial request, with up to 550 allocated to the example, and 1,500 images across the conversation. No gradient updates or RoboMME-Plus checkpoint training are performed.

### Reproduce

[Evaluation setup and commands](../../../scripts/plus_eval/agentic_methods/astra/embodied/README.md#one-shot).

### Released Checkpoints

The underlying Astra model is accessed through its API. Adaptation uses the [RoboMME-Plus training set](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5); this entry does not produce a fine-tuned checkpoint.

### Citations

- [Embodied Astra on RoboMME](https://github.com/jackbjed/embodied-astra-robomme).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

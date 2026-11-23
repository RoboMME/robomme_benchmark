# Embodied-Astra on RoboMME-Plus (eval_by_RoboMME_team)

Embodied-Astra uses the `gpt-6-astra` API model directly as a robot policy. Given the task goal, camera observations, and robot state, it predicts end-effector poses and gripper commands. The benchmark motion planner executes these waypoints, while the conversation retains the observed episode history.

- **Track:** Agentic.
- **Code:** [Code](https://github.com/jackbjed/embodied-astra-robomme) · [Run instructions](../../../scripts/plus_eval/agentic_methods/astra/embodied/README.md).
- **Final evaluation solutions:** N/A (no agentic development).
- **Development logs:** N/A (no agentic development).
- **Paper / website / blog:** [Paper](https://arxiv.org/abs/2610.17350) · [Website](https://embodied-astra.github.io/)
- **Model / checkpoint:** Astra is accessed through its API; no local model checkpoint is released with this entry. The original policy, prompts, and waypoint interface are available in the [Embodied-Astra repository](https://github.com/jackbjed/embodied-astra-robomme).
- **Parameters:** N/A (API model)
- **Evaluated by:** RoboMME Team.

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Zero-shot` | 0 | 194/800 (24.25%) | 0 | N/A | ≈$760 |

Outcomes: 194 successes, 527 failures and 79 timeouts.

### Per-task success

| Suite | Task | `Zero-shot` |
| --- | --- | ---: |
| Counting | BinFill | 13/50 (26.00%) |
| Counting | PickXtimes | 27/50 (54.00%) |
| Counting | SwingXtimes | 16/50 (32.00%) |
| Counting | StopCube | 11/50 (22.00%) |
| Permanence | VideoUnmask | 10/50 (20.00%) |
| Permanence | VideoUnmaskSwap | 1/50 (2.00%) |
| Permanence | ButtonUnmask | 17/50 (34.00%) |
| Permanence | ButtonUnmaskSwap | 6/50 (12.00%) |
| Reference | PickHighlight | 17/50 (34.00%) |
| Reference | VideoRepick | 11/50 (22.00%) |
| Reference | VideoPlaceButton | 18/50 (36.00%) |
| Reference | VideoPlaceOrder | 13/50 (26.00%) |
| Imitation | MoveCube | 11/50 (22.00%) |
| Imitation | InsertPeg | 10/50 (20.00%) |
| Imitation | PatternLock | 7/50 (14.00%) |
| Imitation | RouteStick | 6/50 (12.00%) |
| **Overall** | **All tasks** | **194/800 (24.25%)** |

## Method details

- **Action space:** `waypoint`.
- **Demo use:** No released RoboMME-Plus training demonstrations.
- **Training and adaptation:** no RoboMME or RoboMME-Plus fine-tuning, and no RoboMME-Plus training example in the prompt.
- **Inputs:** front-camera demonstration/execution history sampled every three control steps, the current wrist-camera image, robot state, and camera calibration. The prompt uses `lr_note` for the robot's left/right convention.
- **Image budget:** at most 750 images in the initial request and 1,500 images across the conversation. Episodes that reach this limit count as unsuccessful.

## Agentic development audit record

Not applicable: no agentic development.

## Citations

- [Embodied Astra on RoboMME](https://github.com/jackbjed/embodied-astra-robomme).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

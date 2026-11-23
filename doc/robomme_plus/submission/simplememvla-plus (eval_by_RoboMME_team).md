# SimpleMemVLA on RoboMME-Plus (eval_by_RoboMME_team)

SimpleMemVLA supplies sampled visual history as native video context to its vision-language backbone. Generated subtask representations condition its action head. We evaluate the released RoboMME checkpoint without RoboMME-Plus fine-tuning.

- **Track:** Non-Agentic.
- **Code:** [Run instructions](../../../scripts/plus_eval/non_agentic_methods/simplememvla/README.md).
- **Final evaluation solutions:** N/A (no agentic development).
- **Development logs:** N/A (no agentic development).
- **Paper / website / blog:** [Paper](https://arxiv.org/abs/2609.05533)
- **Model / checkpoint:** The evaluated checkpoint is [yinchenghust/simplememvla_robomme](https://huggingface.co/yinchenghust/simplememvla_robomme), including its model configuration, processor, and normalization statistics. No additional checkpoint was trained for this RoboMME-Plus evaluation.
- **Parameters:** 5.71B (5,708,513,800), including the backbone and action head. Frozen parameters are included; shared weights are counted once.
- **Evaluated by:** RoboMME Team.

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Zero-shot` | 0 | 14/800 (1.75%) | 0 | N/A | N/A |

Outcomes: 14 successes, 657 failures and 129 timeouts.

### Per-task success

| Suite | Task | `Zero-shot` |
| --- | --- | ---: |
| Counting | BinFill | 0/50 (0.00%) |
| Counting | PickXtimes | 0/50 (0.00%) |
| Counting | SwingXtimes | 4/50 (8.00%) |
| Counting | StopCube | 0/50 (0.00%) |
| Permanence | VideoUnmask | 0/50 (0.00%) |
| Permanence | VideoUnmaskSwap | 0/50 (0.00%) |
| Permanence | ButtonUnmask | 0/50 (0.00%) |
| Permanence | ButtonUnmaskSwap | 0/50 (0.00%) |
| Reference | PickHighlight | 0/50 (0.00%) |
| Reference | VideoRepick | 1/50 (2.00%) |
| Reference | VideoPlaceButton | 7/50 (14.00%) |
| Reference | VideoPlaceOrder | 2/50 (4.00%) |
| Imitation | MoveCube | 0/50 (0.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) |
| Imitation | PatternLock | 0/50 (0.00%) |
| Imitation | RouteStick | 0/50 (0.00%) |
| **Overall** | **All tasks** | **14/800 (1.75%)** |

## Method details

- **Action space:** `joint_angle`.
- **Demo use:** No released RoboMME-Plus training demonstrations.
- **RoboMME-Plus adaptation:** none. One released RoboMME checkpoint is used for all 16 tasks.
- **Inference:** upstream history sampling, subtask prediction, and action generation; non-pipelined inference with BF16 and SDPA attention.
- **Sampling:** seed 0, 10 denoising steps, temperature 1.0, at most 64 subtask tokens, and 16 executed actions per prediction.

## Agentic development audit record

Not applicable: no agentic development.

## Citations

- [SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models](https://arxiv.org/abs/2609.05533).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

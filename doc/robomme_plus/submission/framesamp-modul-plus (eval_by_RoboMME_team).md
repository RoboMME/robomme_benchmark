# FrameSamp+Modul on RoboMME-Plus (eval_by_RoboMME_team)

FrameSamp+Modul is the RoboMME perceptual-memory baseline. It samples historical visual frames and uses their features to modulate a Pi0.5 action policy. We evaluate the released RoboMME checkpoint directly on RoboMME-Plus.

- **Track:** Non-Agentic.
- **Code:** [Code](https://github.com/RoboMME/robomme_policy_learning) · [Run instructions](../../../scripts/plus_eval/non_agentic_methods/framesamp_modul/README.md).
- **Final evaluation solutions:** N/A (no agentic development).
- **Development logs:** N/A (no agentic development).
- **Paper / website / blog:** [Paper](https://arxiv.org/abs/2603.04639) · [Website](https://robomme.github.io/)
- **Model / checkpoint:** The original checkpoint is available in [Yinpei/mme_vla_suite](https://huggingface.co/Yinpei/mme_vla_suite/tree/main/perceptual-framesamp-modul). This entry uses the original RoboMME checkpoint; it is not the separately evaluated RoboMME-Plus-fine-tuned variant.
- **Parameters:** 3.44B (3,441,899,024), including frozen parameters.
- **Evaluated by:** RoboMME Team.

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Zero-shot` | 0 | 19/800 (2.38%) | 0 | N/A | N/A |

Outcomes: 19 successes, 645 failures and 136 timeouts.

### Per-task success

| Suite | Task | `Zero-shot` |
| --- | --- | ---: |
| Counting | BinFill | 0/50 (0.00%) |
| Counting | PickXtimes | 0/50 (0.00%) |
| Counting | SwingXtimes | 0/50 (0.00%) |
| Counting | StopCube | 2/50 (4.00%) |
| Permanence | VideoUnmask | 0/50 (0.00%) |
| Permanence | VideoUnmaskSwap | 0/50 (0.00%) |
| Permanence | ButtonUnmask | 0/50 (0.00%) |
| Permanence | ButtonUnmaskSwap | 0/50 (0.00%) |
| Reference | PickHighlight | 0/50 (0.00%) |
| Reference | VideoRepick | 2/50 (4.00%) |
| Reference | VideoPlaceButton | 8/50 (16.00%) |
| Reference | VideoPlaceOrder | 7/50 (14.00%) |
| Imitation | MoveCube | 0/50 (0.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) |
| Imitation | PatternLock | 0/50 (0.00%) |
| Imitation | RouteStick | 0/50 (0.00%) |
| **Overall** | **All tasks** | **19/800 (2.38%)** |

## Method details

- **Action space:** `joint_angle`.
- **Demo use:** No released RoboMME-Plus training demonstrations.
- **RoboMME-Plus adaptation:** none. The original RoboMME-trained checkpoint is shared across all 16 tasks.
- **Checkpoint:** `perceptual-framesamp-modul/79999`; model sampling seed 7.
- **Inference:** upstream history sampling and modulation, with 16-action chunks. Demonstrations longer than 40 seconds are uniformly sampled to 800 frames; execution history retains its original cadence.

## Agentic development audit record

Not applicable: no agentic development.

## Citations

- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

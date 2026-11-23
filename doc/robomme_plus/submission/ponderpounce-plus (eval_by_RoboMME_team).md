# PonderPounce on RoboMME-Plus (eval_by_RoboMME_team)

PonderPounce combines a Qwen3.5-9B episode-context model (Ponder) with a Pi0.5 action controller (Pounce). Ponder maintains visual history and produces a cognition representation that conditions Pounce. We evaluate the released RoboMME checkpoint on RoboMME-Plus.

- **Track:** Non-Agentic.
- **Code:** [Code](https://github.com/worv-ai/PonderPounce) · [Run instructions](../../../scripts/plus_eval/non_agentic_methods/ponderpounce/README.md).
- **Final evaluation solutions:** N/A (no agentic development).
- **Development logs:** N/A (no agentic development).
- **Paper / website / blog:** [Paper](https://arxiv.org/abs/2608.24115) · [Website](https://worv-ai.github.io/ponderpounce/)
- **Model / checkpoint:** The evaluated checkpoint is [worv-ai/ponderpounce-9b-robomme](https://huggingface.co/worv-ai/ponderpounce-9b-robomme), revision `5691e05614e8e6581d6358959e60561acfa73534`. It contains both Ponder and Pounce; no RoboMME-Plus fine-tuning was performed.
- **Parameters:** 13.04B (13,039,166,464), including Ponder, Pounce and their adapters. Frozen parameters are included; shared weights are counted once.
- **Evaluated by:** RoboMME Team.

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Zero-shot` | 0 | 20/800 (2.50%) | 0 | N/A | N/A |

Outcomes: 20 successes, 603 failures and 177 timeouts.

### Per-task success

| Suite | Task | `Zero-shot` |
| --- | --- | ---: |
| Counting | BinFill | 1/50 (2.00%) |
| Counting | PickXtimes | 0/50 (0.00%) |
| Counting | SwingXtimes | 4/50 (8.00%) |
| Counting | StopCube | 1/50 (2.00%) |
| Permanence | VideoUnmask | 0/50 (0.00%) |
| Permanence | VideoUnmaskSwap | 0/50 (0.00%) |
| Permanence | ButtonUnmask | 0/50 (0.00%) |
| Permanence | ButtonUnmaskSwap | 0/50 (0.00%) |
| Reference | PickHighlight | 0/50 (0.00%) |
| Reference | VideoRepick | 4/50 (8.00%) |
| Reference | VideoPlaceButton | 9/50 (18.00%) |
| Reference | VideoPlaceOrder | 1/50 (2.00%) |
| Imitation | MoveCube | 0/50 (0.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) |
| Imitation | PatternLock | 0/50 (0.00%) |
| Imitation | RouteStick | 0/50 (0.00%) |
| **Overall** | **All tasks** | **20/800 (2.50%)** |

## Method details

- **Action space:** `joint_angle`.
- **Demo use:** No released RoboMME-Plus training demonstrations.
- **RoboMME-Plus adaptation:** none. The released 9B RoboMME checkpoint is shared across all tasks; model sampling seed 0.
- **Inference:** BF16, 10 denoising steps, 20-action chunks at 20 Hz, one-second Ponder/Pounce periods, and a 300 ms cognition delay.
- **RoboMME-Plus execution:** the initial hold converts gripper positions into normalized actions. Ponder uses a 16,384-token context limit and rebuilds its caches after trimming old execution queries.

## Agentic development audit record

Not applicable: no agentic development.

## Citations

- [PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control](https://arxiv.org/abs/2608.24115).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

# GR-SAC

### [Technical Report](https://github.com/zhudongwork/GR-SAC/blob/main/release/GR-SAC-Technical-Report.pdf) | [Code](https://github.com/zhudongwork/GR-SAC) | [Residual Checkpoint](https://github.com/zhudongwork/GR-SAC/blob/main/release/residual_sac/actor_00035000.pt)

**Authors:** Dong Zhu, Yue Sun, Jinglun Zhu, Ya Wang  
**Affiliation:** Zoellent Technology

## Introduction

GR-SAC (Gated Residual Soft Actor-Critic) learns a visual residual controller on top of a frozen SimpleMemVLA policy. The actor observes front and wrist images, proprioception, the remaining base action sequence, and the current correction offset. An observable gate activates corrections according to target variation, tracking error, and gripper-command stability. Seven-dimensional joint residuals accumulate into bounded, decaying offsets; the gripper follows the base policy.

The deployed residual actor contains 232,110 parameters, adding 0.0041% to the frozen base model. Training uses SAC with balanced demonstration/online replay, a zero-action prior, successful-trajectory imitation, and an offset penalty.

## Results

**Track:**

- [x] Standard Track
- [ ] Agentic Track

**Total Params — Standard Track:** 5.708745910 B unique parameters at inference (5,708,513,800 frozen base parameters + 232,110 residual actor parameters). Shared embedding/output weights are counted once; training-only critics are excluded.

A single frozen base checkpoint and a single residual actor are shared across all 16 tasks. Evaluation uses the **test** split, **`joint_angle`** actions, **`max_steps=1300`**, and all **50 episodes per task** with the benchmark's fixed environment seeds.

We report a **single-seed evaluation**, achieving **711/800 successes (88.875%)**. Per-task results are listed below.

### Table

| Suite | Task | Success rate (%) | Successes / episodes |
| :--- | :--- | ---: | ---: |
| Counting | BinFill | 76.0 | 38 / 50 |
| Counting | PickXtimes | 100.0 | 50 / 50 |
| Counting | SwingXtimes | 100.0 | 50 / 50 |
| Counting | StopCube | 90.0 | 45 / 50 |
| Permanence | VideoUnmask | 100.0 | 50 / 50 |
| Permanence | VideoUnmaskSwap | 92.0 | 46 / 50 |
| Permanence | ButtonUnmask | 98.0 | 49 / 50 |
| Permanence | ButtonUnmaskSwap | 94.0 | 47 / 50 |
| Reference | PickHighlight | 90.0 | 45 / 50 |
| Reference | VideoRepick | 64.0 | 32 / 50 |
| Reference | VideoPlaceButton | 88.0 | 44 / 50 |
| Reference | VideoPlaceOrder | 86.0 | 43 / 50 |
| Imitation | MoveCube | 100.0 | 50 / 50 |
| Imitation | InsertPeg | 52.0 | 26 / 50 |
| Imitation | PatternLock | 94.0 | 47 / 50 |
| Imitation | RouteStick | 98.0 | 49 / 50 |
| **Overall** | **All 16 tasks** | **88.875** | **711 / 800** |

### Training Details

- **Base policy:** the released [SimpleMemVLA RoboMME checkpoint](https://huggingface.co/yinchenghust/simplememvla_robomme), kept frozen throughout residual training.
- **Residual learning data:** successful zero-residual rollouts initialize demonstration replay; residual-policy interaction supplies online replay. Both use the train split, with equal demonstration and online contributions to each critic batch. No additional human corrective demonstrations are used.
- **Optimization:** four critics, batch size 256, Adam learning rate 3e-4, discount 0.999; 35,000 SAC updates, continuing the full learner/replay state saved at 20,000 updates.
- **Control:** base execution horizon 16; offset increment 0.001 rad, decay 0.95, and per-joint bound 0.02 rad.
- **Checkpoint selection:** the reported 35k actor was selected from a test comparison of the 20k, 25k, 30k, 35k, and 40k checkpoints, as documented in the technical report.
- **Hardware:** eight NVIDIA H200 GPUs, with one learner and seven asynchronous collectors during training.
- Full settings: [training configuration](https://github.com/zhudongwork/GR-SAC/blob/main/configs/residual_rl/robomme_sac.json).

### Reproduction

Use release commit `fc4d610a1d372ab3efaa2d49261bad01a18bf0a5` and follow the [installation instructions](https://github.com/zhudongwork/GR-SAC#installation). Download the complete original SimpleMemVLA checkpoint, including tokenizer, processor, and normalization metadata. The residual actor is included in the repository.

```bash
git clone https://github.com/zhudongwork/GR-SAC.git
cd GR-SAC
git checkout fc4d610a1d372ab3efaa2d49261bad01a18bf0a5

CHECKPOINT=/path/to/simplememvla_robomme \
RESIDUAL_CHECKPOINT=release/residual_sac/actor_00035000.pt \
GPUS=0,1,2,3,4,5,6,7 GROUP_SIZE=10 MODE=residual \
  bash scripts/eval_residual_rl.sh --episode-selection first
```

The release uses a batched evaluation adapter around RoboMME's environment builder, with the benchmark test metadata and success criteria. Base inference uses bfloat16, FlashAttention 2, and 10 denoising steps; the residual actor uses its deterministic tanh-mean action. Keep GPU worker assignment and group size fixed when reproducing the reference because they affect the base policy's sampling RNG assignment. Invalid test resets raise errors rather than being skipped.

- [Per-task reference results](https://github.com/zhudongwork/GR-SAC/blob/main/release/residual_sac/reference_results.json)
- [All 800 episode outcomes and environment seeds](https://github.com/zhudongwork/GR-SAC/blob/main/release/residual_sac/reference_episodes.jsonl)
- [Checkpoint hashes, evaluation configuration, and metadata provenance](https://github.com/zhudongwork/GR-SAC/blob/main/release/residual_sac/provenance.json)

### Released Checkpoints

- Frozen base model: [yinchenghust/simplememvla_robomme](https://huggingface.co/yinchenghust/simplememvla_robomme).
- Residual actor: [`actor_00035000.pt`](https://github.com/zhudongwork/GR-SAC/blob/main/release/residual_sac/actor_00035000.pt), SHA-256 `bf6cf83631580dfe951e2adbf31222eefe0149ba5b0a602ff389ea0392428201`.
- Training and evaluation code: [zhudongwork/GR-SAC](https://github.com/zhudongwork/GR-SAC).

### Citation

```bibtex
@techreport{zhu2026grsac,
  title       = {Gated Residual Reinforcement Learning for Vision-Language-Action Models},
  author      = {Zhu, Dong and Sun, Yue and Zhu, Jinglun and Wang, Ya},
  institution = {Zoellent Technology},
  year        = {2026},
  url         = {https://github.com/zhudongwork/GR-SAC}
}
```

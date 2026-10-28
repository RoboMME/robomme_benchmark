# Can Astra Guide a VLA to Solve RoboMME?

**A Self-Evolving Harness for Memory-Dependent Manipulation**

### [Website](https://legend717.github.io/astra-guide-vla-website/) | [Paper](https://legend717.github.io/astra-guide-vla-website/report/) | [Code](https://github.com/Legend717/astra-guide-vla)

## Introduction

We present a self-evolving harness combining GPT-6 Astra with the official RoboMME
GroundSG vision-language-action model. Astra interprets task history and selects
grounded subgoals, while the VLA handles their execution. Astra chooses finite
control-step budgets for delegation, can select robot-state monitors for earlier
review, and intervenes directly when correction is needed. Starting from subtask
duration priors, execution experience is accumulated and revised across episodes
to guide planning, delegation, review and recovery. Model weights remain fixed.

## Results

We evaluate all 16 tasks in the official RoboMME test split, with 50 episodes per
task and 800 in total. The same Astra model and GroundSG checkpoint are used across
tasks. Official environment feedback determines success. The technical report
records **777/800 successes (97.125%, displayed as 97.1%)**. We report one run;
the table is not an average over multiple independent seeds. The benchmark seeds
are fixed internally.

The VLA executes absolute joint actions, with bounded end-effector interventions
from Astra. Inputs include task instructions, front/wrist RGB images and robot
state, with access to recorded visual history and candidate forward kinematics.
Planning and control receive no hidden object states or oracle subgoals and do not
use privileged multi-choice options. Official success and termination feedback
is used for outcome assessment and post-episode review.

### Table

| Suite | Task | Successes / 50 | Run 1 (%) | Avg (%) |
|---|---|---:|---:|---:|
| Counting | BinFill | 48 | 96.0 | 96.0 |
| Counting | PickXtimes | 50 | 100.0 | 100.0 |
| Counting | SwingXtimes | 48 | 96.0 | 96.0 |
| Counting | StopCube | 48 | 96.0 | 96.0 |
| Permanence | VideoUnmask | 50 | 100.0 | 100.0 |
| Permanence | VideoUnmaskSwap | 49 | 98.0 | 98.0 |
| Permanence | ButtonUnmask | 50 | 100.0 | 100.0 |
| Permanence | ButtonUnmaskSwap | 46 | 92.0 | 92.0 |
| Reference | PickHighlight | 50 | 100.0 | 100.0 |
| Reference | VideoRepick | 50 | 100.0 | 100.0 |
| Reference | VideoPlaceButton | 49 | 98.0 | 98.0 |
| Reference | VideoPlaceOrder | 45 | 90.0 | 90.0 |
| Imitation | MoveCube | 50 | 100.0 | 100.0 |
| Imitation | InsertPeg | 45 | 90.0 | 90.0 |
| Imitation | PatternLock | 50 | 100.0 | 100.0 |
| Imitation | RouteStick | 49 | 98.0 | 98.0 |
| **Overall** | **16 tasks** | **777 / 800** | **97.125** | **97.125** |

### Training and evaluation details

- **Models:** GPT-6 Astra with high reasoning effort; official GroundSG pi0.5
  checkpoint at training step 79,999, inference seed 42 and independent random
  state per policy connection. The VLA weights are fixed; no additional weight
  training is introduced by the harness.
- **Initialization:** duration recommendations are derived from subtask
  completion-step statistics in the training demonstrations.
- **Experience:** the first five episodes of each task allow updates after either
  outcome; the remaining 45 allow updates only after unsuccessful outcomes. All
  50 episodes contribute to the score. Global advice is shared, task experience
  is separate, and the experience input stays fixed within each episode.
- **Execution:** 20 Hz control frequency; up to the first 16 predicted actions
  per VLA chunk, with a maximum of 1,300 control steps per episode. Delegation
  budgets range from 30 to 200 actual control steps in increments of 10.
  Robot-state monitors can trigger earlier review.
- **Action space:** hybrid `joint_angle` / `ee_pose` control. The VLA emits
  absolute joint actions; bounded Astra corrections use the official
  end-effector-pose wrapper. The evaluation does not use `multi_choice`.
- **History:** sparse queries return up to eight uniformly sampled frames,
  including endpoints; current front and wrist views are supplied to Astra.
- **API settings:** 80 calls per episode, 8,192 output tokens per call and a
  300,000-token context limit.

Across the reported evaluation, 54.9% of control steps execute without an
additional Astra review. This measures execution allocation, not a percentage
reduction in total cost.

### Released checkpoints and code

The VLA checkpoint is available in the official
[MME-VLA collection](https://huggingface.co/Yinpei/mme_vla_suite). Use
`symbolic-grounded-subgoal/79999` with its `history_config.txt` and normalization
assets. Astra is accessed through the API; no Astra checkpoint is distributed.

The public [code repository](https://github.com/Legend717/astra-guide-vla)
provides the harness, prompts, configuration, duration-prior builder, and
installation/evaluation instructions. The [project website](https://legend717.github.io/astra-guide-vla-website/)
hosts the [online technical report](https://legend717.github.io/astra-guide-vla-website/report/).
Results and descriptions in this document follow that report; no new benchmark
run was performed during release preparation.

### Citations

An arXiv version is not yet available. Please cite the online technical report:

```bibtex
@misc{chen2026astraguide,
  title        = {Can Astra Guide a VLA to Solve RoboMME? A Self-Evolving Harness for Memory-Dependent Manipulation},
  author       = {Xingping Chen and Zihao Mo and Yimin Chen and Yu Lu and Zhanghui Kuang and Guanbin Li},
  year         = {2026},
  howpublished = {Online technical report},
  url          = {https://legend717.github.io/astra-guide-vla-website/report/}
}
```

# RoboMME OOD test split

## Motivation

The regular RoboMME test split measures performance within the released task
configurations. `ood_test` probes whether that performance extends to larger
counts, longer sequences and changed geometry beyond the hard settings, while
preserving the original task mechanisms, instruction templates and success checks.

## Split design

### Compared with the regular test split

The regular test split has **50 episodes per task** across easy, medium and hard
settings. `ood_test` has **20 episodes per task across all 16 tasks (320 total)**,
using `config_ood` in the [original task implementations](../src/robomme/robomme_env/).
It is for testing only, with no new training split; the original train, validation
and test splits are unchanged.

### Compared with RoboMME-Plus

[RoboMME-Plus](../src/robomme_plus/README.md) is a separate benchmark that also
changes task mechanisms, such as moving targets and additional manipulation
methods, and provides its own training and test splits. `ood_test` instead
studies configuration extrapolation within regular RoboMME's existing tasks.

### Episode verification

Seeds exclude the original splits, and every selected episode passed both
native-expert execution and end-effector-action replay. Replay screening used
1,300 steps, so the selection is conditioned on expert replay solvability
within that budget.

## Usage

Choose `ood_test` through the regular RoboMME interface:

```python
from robomme.env_record_wrapper import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder(
    "BinFill", dataset="ood_test", action_space="joint_angle"
)
env = builder.make_env_for_episode(0)  # Episode indices: 0–19.
try:
    obs, info = env.reset()
    # Run your policy through env.step(action).
finally:
    env.close()
```

The default limit is 1,500 execution steps, excluding the demonstration. In-episode
demonstration videos and cues remain part of the policy's observations.

## Baseline results

Released RoboMME checkpoints are evaluated without OOD fine-tuning. Entries are
successful / total episodes under the native success check; the overall row also
shows the success rate. Each task has 20 episodes.

| Task | FrameSamp+Modul | SimpleMemVLA | PonderPounce |
|---|---:|---:|---:|
| BinFill | 0/20 | 0/20 | 0/20 |
| ButtonUnmask | 0/20 | 19/20 | 18/20 |
| ButtonUnmaskSwap | 1/20 | 5/20 | 2/20 |
| InsertPeg | 0/20 | 12/20 | 0/20 |
| MoveCube | 13/20 | 17/20 | 10/20 |
| PatternLock | 0/20 | 0/20 | 0/20 |
| PickHighlight | 0/20 | 3/20 | 1/20 |
| PickXtimes | 0/20 | 0/20 | 16/20 |
| RouteStick | 0/20 | 10/20 | 1/20 |
| StopCube | 0/20 | 0/20 | 0/20 |
| SwingXtimes | 0/20 | 4/20 | 5/20 |
| VideoPlaceButton | 5/20 | 9/20 | 9/20 |
| VideoPlaceOrder | 4/20 | 6/20 | 4/20 |
| VideoRepick | 0/20 | 0/20 | 7/20 |
| VideoUnmask | 0/20 | 19/20 | 17/20 |
| VideoUnmaskSwap | 3/20 | 2/20 | 6/20 |
| **Overall** | 26/320 (8.1%) | 106/320 (33.1%) | 96/320 (30.0%) |

Completed episodes retained a 1,300-step limit for SimpleMemVLA (217 episodes)
and PonderPounce (156 episodes). All remaining episodes, including all
FrameSamp+Modul episodes, use 1,500 steps. These are exploratory results with
mixed step budgets.

# RoboMME OOD test split

The regular RoboMME test split measures performance within the released task
configurations. `ood_test` probes whether that performance extends to larger
counts, longer sequences and changed geometry beyond the hard settings, while
preserving the original task mechanisms, instruction templates and success checks.

The regular test split has **50 episodes per task** across easy, medium and hard
settings. `ood_test` has **20 episodes per task across all 16 tasks (320 total)**,
using `config_ood` in the [original task implementations](../src/robomme/robomme_env/).
It is for testing only, with no new training split; the original train, validation
and test splits are unchanged. Seeds exclude the original splits, and every
selected episode passed both native-expert execution and end-effector-action
replay. Replay screening used 1,300 steps, so the selection is conditioned on
expert replay solvability within that budget.

[RoboMME-Plus](../src/robomme_plus/README.md) is a separate benchmark that also
changes task mechanisms, such as moving targets and additional manipulation
methods, and provides its own training and test splits. `ood_test` instead
studies configuration extrapolation within regular RoboMME's existing tasks.

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

Evaluation is ongoing. Per-task and overall results for FrameSamp+Modul,
SimpleMemVLA and PonderPounce will be added after all three finish.

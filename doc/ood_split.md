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

Use `ood_test` to test whether a model trained on regular RoboMME generalizes to
extrapolated parameters without further training. It changes counts, sequence
lengths and geometry while retaining the original task mechanisms and success
criteria.

Use [RoboMME-Plus](../src/robomme_plus/README.md) to evaluate robot memory across
a broader range of demanding conditions, including moving targets, stronger
occlusion and additional manipulation methods. Its separate training and test
splits support both zero-shot evaluation and few-shot adaptation.

### Episode verification

Seeds exclude the original splits, and every selected episode passed both
native-expert execution and end-effector-action replay. Replay screening used
1,300 steps, so the selection is conditioned on expert replay solvability
within that budget.

## Parameter changes

Each `config_ood` extends the hard configuration or the task's original fixed
settings. Unlisted parameters and success checks stay unchanged. Ranges below
describe the configurations; the selected seeds may not cover every value.

| Task | Regular hard/default setting | `ood_test` setting |
|---|---|---|
| BinFill | Spawn 10–12 cubes; collect 3–5 | Spawn 13–18 cubes; collect 6–8 |
| ButtonUnmask / VideoUnmask | 0.4 × 0.4 m sampling area | 0.5 × 0.5 m sampling area |
| ButtonUnmaskSwap / VideoUnmaskSwap | 2–3 swaps | 4–8 swaps |
| InsertPeg | 3 pegs | 4 pegs |
| MoveCube | Original cube dimensions; 10 cm peg | Cube dimensions scaled to 80%; 8 cm peg |
| PatternLock | 4–8 nodes on a 5 × 5 grid | 9–16 nodes on a 6 × 6 grid |
| PickHighlight | 6 cubes, 3 highlights; 0.4 × 0.4 m area | 8–12 cubes, 4–5 highlights; 0.5 × 0.5 m area |
| PickXtimes | 4–5 pick-and-place repetitions | 6–10 repetitions |
| RouteStick | Route length 4–7 | Route length 8–12 |
| StopCube | Stop count 2–5; 5 available passes | Stop count 6–10; 10 available passes |
| SwingXtimes | 3 back-and-forth repetitions | 4–8 repetitions |
| VideoPlaceButton / VideoPlaceOrder | 1 swap | 2–4 swaps |
| VideoRepick | 1–3 pickup/drop cycles | 4–5 cycles |

ButtonUnmask and VideoUnmask retain two pickups and a bin cap of 15; actual bin
counts depend on placement. Their swap variants retain four bins and two pickups.
VideoPlaceButton and VideoPlaceOrder retain four targets.

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

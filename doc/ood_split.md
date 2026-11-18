# RoboMME OOD-test split

## Motivation

The regular RoboMME test split measures performance within the released task
configurations. `ood_test` probes whether that performance extends to larger
counts, longer sequences and changed geometry beyond the hard settings, while
preserving the original task mechanisms, instruction templates and success checks.

## Split design

### Compared with the regular RoboMME test split

The regular test split has **50 episodes per task** across easy, medium and hard
settings. `ood_test` has **20 episodes per task across all 16 tasks (320 total)**,
using `config_ood` in the [original task implementations](../src/robomme/robomme_env/).
It is for testing only, with no new training split; the original train, validation
and test splits are unchanged.

### Difference from RoboMME-Plus

Use `ood_test` to test whether a model trained on regular RoboMME generalizes to
extrapolated parameters without further training. It changes counts, sequence
lengths and geometry while retaining the original task mechanisms and success
criteria.

[RoboMME-Plus](../src/robomme_plus/README.md) evaluates memory generalization and
adaptation with changed task mechanisms and perceptual variations. Its 50 training
demos and 800 test episodes support zero-shot transfer (ZS), adaptation from demos
(AfD), training-seed interactions (AfI), or both (AfD+I). Interaction budgets cap
attempts per task at R1, R3, R5 or R10. It is separate from `ood_test`.

## Parameter changes

The changes below are relative to each task's regular hard configuration, or
its original fixed settings. Each task inherits all other parameters. Ranges
describe the configuration bounds; the 20 selected episodes may not cover
every value. Task names link to their implementations.

| Task | Parameter changes (regular → OOD) | Retained settings |
|---|---|---|
| [BinFill](../src/robomme/robomme_env/BinFill.py) | Spawned cubes (`spawn_cubes`): **10–12 → 13–18**; cubes to collect (`put_in_numbers`): **3–5 → 6–8**. | Three available colors; requests span 2–3 colors. |
| [ButtonUnmask](../src/robomme/robomme_env/ButtonUnmask.py) | Sampling half-width (`region_half_size`): **0.20 → 0.25 m**. | Two pickups; bin cap remains 15. The button/cue and masking rules are unchanged. |
| [ButtonUnmaskSwap](../src/robomme/robomme_env/ButtonUnmaskSwap.py) | Swap count (`swap_min`, `swap_max`): **2–3 → 4–8**. | Four bins, two pickups and the original 50-step swap animation. |
| [InsertPeg](../src/robomme/robomme_env/InsertPeg.py) | Number of pegs (`pegs`): **3 → 4**. | Original peg geometry and insertion requirements. |
| [MoveCube](../src/robomme/robomme_env/MoveCube.py) | Cube scale (`cube_scale`): **1.0 → 0.8**; tool length parameter (`peg_length`): **0.10 → 0.08 m**. | The demonstrated manipulation method and existing goal tolerances. |
| [PatternLock](../src/robomme/robomme_env/PatternLock.py) | Grid size (`grid`): **5 × 5 → 6 × 6**; pattern length (`length`): **4–8 → 9–16 nodes**. | The original pattern-tracing mechanism and success rules. |
| [PickHighlight](../src/robomme/robomme_env/PickHighlight.py) | Cube count (`spawn`): **6 → 8–12**; highlighted targets (`pickup`): **3 → 4–5**; sampling half-width: **0.20 → 0.25 m**. | The original highlight cues and pickup mechanism. |
| [PickXtimes](../src/robomme/robomme_env/PickXtimes.py) | Pick-and-place repetitions (`number_min`, `number_max`): **4–5 → 6–10**. | Three colors and the original counting rule. |
| [RouteStick](../src/robomme/robomme_env/RouteStick.py) | Route length parameter (`length`): **4–7 → 8–12**. | Backtracking remains enabled; drawing and success rules are unchanged. |
| [StopCube](../src/robomme/robomme_env/StopCube.py) | Requested stop count (`stop_count`): **2–5 → 6–10**; available passes (`passes`): **5 → 10**. | The same motion path and button-timing rule. |
| [SwingXtimes](../src/robomme/robomme_env/SwingXtimes.py) | Back-and-forth repetitions (`number_min`, `number_max`): **3 → 4–8**. | Three colors and the original swing/counting mechanism. |
| [VideoPlaceButton](../src/robomme/robomme_env/VideoPlaceButton.py) | Target swaps: **1 → 2–4**, using `swap_min` and `swap_max`. | Four targets, the same workspace and the original placement goal. |
| [VideoPlaceOrder](../src/robomme/robomme_env/VideoPlaceOrder.py) | Target swaps: **1 → 2–4**, using `swap_min` and `swap_max`. | Four targets and the original demonstrated-order query and placement goal. |
| [VideoRepick](../src/robomme/robomme_env/VideoRepick.py) | Pickup/drop cycles (`number_min`, `number_max`): **1–3 → 4–5**. | The hard setting's 15-cube clutter and the same demonstrated target. |
| [VideoUnmask](../src/robomme/robomme_env/VideoUnmask.py) | Sampling half-width (`region_half_size`): **0.20 → 0.25 m**. | Two pickups; bin cap remains 15. The video cue and masking rules are unchanged. |
| [VideoUnmaskSwap](../src/robomme/robomme_env/VideoUnmaskSwap.py) | Swap count (`swap_min`, `swap_max`): **2–3 → 4–8**. | Four bins, two pickups and the original 50-step swap animation. |

Increasing the sampling half-width from 0.20 to 0.25 m expands the square
sampling area from **0.4 × 0.4 m to 0.5 × 0.5 m**. Unmask bin counts depend on
successful placement and need not reach the cap of 15. VideoPlace swap phases
extend to accommodate the extra swaps while retaining their original motion.

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

Released fine-tuned policy checkpoints are evaluated without OOD fine-tuning. Entries are
successful / total episodes under the native success check; the overall row also
shows the success rate. Each task has 20 episodes.

Source code:

- FrameSamp+Modul: [RoboMME/robomme_policy_learning](https://github.com/RoboMME/robomme_policy_learning/tree/ac846a9a4aec841e483b1dc5d620dfe426bde853).
- SimpleMemVLA: [OpenBMB/SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA).
- PonderPounce: [worv-ai/PonderPounce](https://github.com/worv-ai/PonderPounce/tree/723df35762bb641e1d520e4fa9359b98644adc21).

| Task | FrameSamp+Modul | SimpleMemVLA | PonderPounce |
|---|---:|---:|---:|
| BinFill | 0/20 | 2/20 | 0/20 |
| ButtonUnmask | 0/20 | 20/20 | 18/20 |
| ButtonUnmaskSwap | 1/20 | 5/20 | 2/20 |
| InsertPeg | 0/20 | 12/20 | 0/20 |
| MoveCube | 13/20 | 19/20 | 11/20 |
| PatternLock | 0/20 | 0/20 | 0/20 |
| PickHighlight | 0/20 | 4/20 | 2/20 |
| PickXtimes | 0/20 | 0/20 | 18/20 |
| RouteStick | 0/20 | 10/20 | 1/20 |
| StopCube | 0/20 | 0/20 | 0/20 |
| SwingXtimes | 0/20 | 4/20 | 5/20 |
| VideoPlaceButton | 5/20 | 9/20 | 9/20 |
| VideoPlaceOrder | 4/20 | 6/20 | 4/20 |
| VideoRepick | 0/20 | 0/20 | 7/20 |
| VideoUnmask | 0/20 | 19/20 | 17/20 |
| VideoUnmaskSwap | 3/20 | 2/20 | 6/20 |
| **Overall** | 26/320 (8.1%) | 112/320 (35.0%) | 100/320 (31.2%) |

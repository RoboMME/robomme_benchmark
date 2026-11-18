# Mechanism variation

Thirteen tasks have **8 hard test episodes each** marked with
`mechanism_variation` in [`env_metadata/test`](../../env_metadata/test/).
Seed values and task goals are unchanged. These 104 episodes do not overlap
perceptual variations. Unmarked episodes retain their original behavior.

| Tasks | Variation |
|---|---|
| BinFill | Nearby conveyor distractors at the original belt speed. |
| PickXtimes | Variable speed and late forward/reverse dodges, with `N // 2` dodges. |
| StopCube | Reversals and speed changes; the target lies behind the button with a visible expert counting view. |
| SwingXtimes | Larger moving target circles and a cube start in front of a target. |
| ButtonUnmask / ButtonUnmaskSwap | Fast cube-cluster swaps after pressing; native press deadlines remain. |
| VideoUnmask / VideoUnmaskSwap | Visible cube-cluster swaps before masking; VideoUnmaskSwap also changes container motion. |
| VideoRepick | Transferred continuous motion with recurring swaps. |
| VideoPlaceButton / VideoPlaceOrder | Alternating continuous motion and concurrent pair swaps; execution motion stops permanently when the requested cube is grasped. |
| PatternLock | Seeded neighbor activation with wraparound, longer holds and/or smaller nodes, demonstrated before execution. |
| RouteStick | Longer contact holds, also shown during demonstration. |

Every Unmask target must show at least 12 pixels in a front or wrist cue frame
before masking in these test mechanisms. Button-task cube swaps run at twice the
reviewed initial speed.

Two training episodes use separate atomic profiles, without appearance changes:

- **ButtonUnmaskSwap:** `container_speed_2x` plays its original finite container
  routes at twice the speed. It adds no pre-mask cube shuffle and preserves the
  layout, route waypoints and native button deadlines.
- **RouteStick:** `longer_activation` uses the longer contact hold above in both
  the demonstration and execution.

These use `{"version": 1, "atomic": "<preset>"}` in `mechanism_variation` and may
appear at any training difficulty. Their new expert timings are stored alongside
the original timings, so the opt-out switch restores the original budget.

```python
builder = BenchmarkEnvBuilder("StopCube", mechanism_variations=False)
env = builder.make_env_for_episode(27)
```

The switch defaults to `True` and is also available per episode. Disabling it
restores the original oracle timing. Marked episodes otherwise use their new
expert execution count: `max_steps = ceil(1.5 * oracle_policy_step) + 200`.
Demonstration steps are excluded.

Mechanism environments reuse native task source in isolated module namespaces.
Overrides cannot modify the shared classes used by normal or perceptual episodes.
Only runtime mechanics and required demonstration helpers live here; recording,
evaluation experiments and expert execution policies remain outside this package.

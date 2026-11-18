# Variations

Fourteen tasks mark 5 medium and 8 hard test episodes each with
`perceptual_variation` in `env_metadata/test`. Existing seeds, oracle timings,
task geometry, motion and scoring are unchanged. Unmarked episodes retain
their original observations. Separately, 13 tasks mark 8 hard episodes each for
[mechanism variation](mechanism/). The two selections do not overlap.

- `non_lighting` (medium): seeded natural table materials, applicable object,
  button and target colors, and low depth/calibration noise.
  VideoUnmask also enables seeded lighting for these marked medium episodes.
- `full` (hard): the same factors plus a seeded lighting mode. Motion uses
  simulation time; bright local spots move continuously without flashing.
- PickHighlight uses normal or dim lighting; spotlights and projected patterns
  are excluded so they cannot imitate or obscure its highlight cue.
- StopCube adds a shallow rectangular wooden coaster, a stationary elongated
  block, or one of each, selected by seed on marked medium and hard episodes.
  Props stay clear of the full cube route, button and target, affect RGB/depth,
  and have no colliders.
- Toggled target rings keep their native idle colors and use a seeded active
  color. BinFill retains its black inner bottom.
- Containers use one plain, matte medium-to-dark gray shade per marked episode
  (seeded base RGB 0.12–0.45), avoiding near-white shades that resemble native covers.
- When other-colored distractor cubes are present, colored cube materials blend
  50% native color with 50% seeded shade variation; lighting applies normally.
  RGB-only episodes retain the broader shade variation.
- Low noise: depth σ = 1 mm with 0.25% dropout; reported camera calibration has
  0.25° rotation, 1 mm translation, ±0.25% focal and ±0.25 px principal-point bias.
  True camera poses stay unchanged; noise modifies only reported sensor data.

## Atomic training variations

One existing training episode per non-MoveCube task is marked with a versioned
`{"version": 1, "atomic": "<preset>"}` profile. Seeds and difficulty levels are
unchanged. Thirteen episodes change only the appearance factors below; depth and
calibration remain exact, and other object colors retain their native settings.

| Factor | Training task |
|---|---|
| Dim light | BinFill |
| Marble table | PickXtimes |
| Ring colors | StopCube, VideoPlaceButton |
| Dark navy base / ivory cap | ButtonUnmask |
| Disco spotlights | SwingXtimes |
| Moving white ring-pattern light | VideoUnmask |
| Board color and light gray containers (RGB 0.75) | VideoUnmaskSwap |
| Light oak table | PickHighlight |
| Moving colored patches | VideoRepick |
| Steel table | PatternLock |
| Static colored light | VideoPlaceOrder |
| Moving local overexposure | InsertPeg |

The other two selected episodes use [atomic mechanisms](mechanism/):
ButtonUnmaskSwap doubles its original container-swap speed, without a pre-mask
shuffle; RouteStick requires longer contact holds. The remaining 35 episodes
have no added variation flags. The metadata selects the episode and preset; the existing seed
determines any random parameters. Test profiles and selections are unchanged.

## Controls

Variations are enabled by default for marked episodes. To disable them:

```python
builder = BenchmarkEnvBuilder("StopCube", perceptual_variations=False,
                              mechanism_variations=False)
env = builder.make_env_for_episode(12)
```

The switches are independent and also available on `make_env_for_episode`.
Each task also exposes
`PERCEPTUAL_VARIATIONS = True` in its registered class. Change it to `False`
before constructing environments to disable variations for that task.

# Variation implementation

See the [benchmark overview](../README.md#variations) for the split and controls.

`perceptual_variation` and `mechanism_variation` in episode metadata select the
profiles. Parameters are seeded; unmarked episodes keep their original behavior.
Perceptual profiles preserve task mechanics. Atomic training profiles expose
selected factors, with exact depth and calibration.

Appearance configuration and application live in this directory;
[mechanism overrides](mechanism) are kept separately. The builder enables both
by default and provides independent opt-out switches for ablations.

# RoboMME-Plus variations and training exposure

Training exposes selected atomic variations; test episodes also combine factors
and include held-out mechanisms. The 50 training seeds and all test selections
are unchanged. This document describes the current local code and recordings;
the updated training files have not yet been uploaded to Hugging Face.

## Training exposure

One existing episode per non-MoveCube task is selected, without adding seeds.
Appearance profiles enable only the listed factors, with exact depth/calibration.
ButtonUnmaskSwap and RouteStick use mechanism profiles without appearance changes.
The other 35 episodes have no added variation flags.

Episode indices below are zero-based. Selection uses RNG seed `20261114`;
VideoUnmaskSwap is selected from medium/hard because its easy scene has no board.

| Task | Episode / difficulty | Seed | Added training variation |
|---|---|---:|---|
| BinFill | 0 / easy | 1380945633 | Dim light |
| PickXtimes | 0 / easy | 2083218884 | Marble table |
| StopCube | 2 / hard | 1637987667 | Seeded ring colors |
| ButtonUnmask | 2 / hard | 1878721550 | Dark navy button base / ivory cap |
| SwingXtimes | 1 / medium | 1822653288 | Disco spotlights |
| VideoUnmask | 2 / hard | 1497627596 | Moving white ring-pattern light |
| VideoUnmaskSwap | 1 / medium | 1900429819 | Board color + light gray containers |
| ButtonUnmaskSwap | 1 / medium | 438646560 | 2× original container-swap speed; no pre-mask shuffle |
| PickHighlight | 0 / easy | 1115689320 | Light oak table |
| VideoRepick | 1 / medium | 1284088509 | Moving colored patches |
| VideoPlaceButton | 0 / easy | 1155046303 | Seeded ring colors |
| PatternLock | 1 / medium | 1583389125 | Steel table |
| VideoPlaceOrder | 2 / hard | 2124331336 | Static colored light |
| InsertPeg | 0 / easy | 1365734056 | Moving local overexposure |
| RouteStick | 0 / easy | 1882199712 | Longer contact holds |

All 15 replacement recordings passed expert execution and physical replay through
the public builder. Training flags use `{"version": 1, "atomic": "<preset>"}`.
Presets do not change the existing test-profile sampling pools or RNG domains.

## Current split

| Split / group | Episodes |
|---|---:|
| Training, normal | 35 |
| Training, atomic appearance | 13 |
| Training, atomic mechanism | 2 |
| Test, normal | 514 |
| Test, perceptual | 182 |
| Test, mechanism | 104 |

Training contains easy/medium/hard for 15 tasks and one example per MoveCube
method. Perceptual test variations select 5 medium + 8 hard seeds in each of
14 tasks; mechanism variations select 8 different hard seeds in 13 tasks.
The test lists do not overlap. Both switches default to enabled; only marked
metadata activates variations. Random parameters are determined by episode seed.

The 14 perceptual test tasks are BinFill, PickXtimes, SwingXtimes, StopCube,
VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap, PickHighlight,
VideoRepick, VideoPlaceButton, VideoPlaceOrder, PatternLock and RouteStick.
The mechanism test list is the same except PickHighlight. InsertPeg has only
its atomic training light variation; MoveCube has neither added family.

## Perceptual variation inventory

These changes alter rendering or reported sensor data while preserving the
task's physical trajectories, identities and success rules. Categories apply to
the relevant objects actually present in each selected episode.

| Category | Implemented behavior | Training exposure / held-out part |
|---|---|---|
| Table | One seeded material: original brown wood, light oak, brushed steel, marble or concrete. Steel/marble/concrete have generated textures; material reflectance also changes. No ring-pattern tablecloth. | Train: light oak, steel and marble; native wood is already present. Concrete remains test-only. |
| Lighting | One seeded mode: dim, static colored lights, moving colored lights, disco spotlights, moving projected ring patterns, moving local overexposure, or moving colored patches. | Train: dim, static color, disco, moving rings, local overexposure and colored patches. The separate moving-color light mode remains test-only. |
| Moving colored patches | A projected field of 85–120 soft colored regions. Colors derive from task objects and target rings, creating visual similarity. | Train: VideoRepick. |
| Moving colored lights / disco | Colors are matched to the scene's object/target palette. Light positions and motion parameters are seeded. | Train: disco in SwingXtimes. Moving-color light mode remains test-only. |
| Ring-pattern lighting | A moving/rotating projection of colored concentric rings, using scene colors. These are light patterns, not additional physical targets. | Train: white rings in VideoUnmask; colored ring projections remain test-only. |
| Local overexposure | Two bright, moving warm-white spots with constant light intensity. This is not flashing or strobing. | Train: InsertPeg. |
| Cubes | Seeded color/shade changes. When other-colored distractor cubes are present, materials blend 50% original color with 50% varied shade to preserve color-name distinctions. RGB-only scenes retain the broader shade variation. Lighting still affects the rendered colors normally. | Material shade variation remains test-only. Training lighting still changes rendered cube appearance. |
| StopCube moving cube | A separate seeded color change, rather than the semantic RGB-cube shading rule. | Recoloring remains test-only; its training atom changes only the ring. |
| Containers | One plain matte gray shade per episode; base RGB is 0.12–0.45. | Train: fixed light gray (RGB 0.75) in VideoUnmaskSwap. The darker seeded test range remains held out. |
| Boards / bins | Seeded surface colors. BinFill's inner bottom stays black. RouteStick's board-like obstacles also receive these colors. | Train: VideoUnmaskSwap board. Bin and RouteStick obstacle recoloring remain test-only. |
| Conveyor | BinFill belt/treads receive a color matched to the selected table material; the contrasting stripe/textured appearance is removed. Motion remains unchanged. | Test only. |
| Buttons | Button base/body and cap colors change wherever a button is present. | Train: dark navy body / ivory cap in ButtonUnmask, outside the test color pool. |
| Ordinary target rings | Seeded ring palettes: teal/ivory, amber/charcoal, blue/gray or rose/cream, with shade variation. This includes PickXtimes, StopCube and VideoPlace targets. | Train: seeded palettes in StopCube and VideoPlaceButton; exact unseen seeds and combinations remain for test. |
| Activated target feedback | SwingXtimes, PatternLock and RouteStick retain their idle appearance and change the active feedback color. Feedback no longer has to be red. The simulator's activation state is unchanged. | Test only; native active feedback colors remain in training. |
| Tools | Seeded recoloring of applicable tool surfaces, including the stick; distinct parts preserve their original pairwise RGB distances. Geometry is unchanged. | Test only. |
| StopCube visual distractors | A shallow rectangular wooden coaster, an elongated stationary block, or one of each. They stay clear of the route/button/target and have no colliders. | Test only. |
| Front and wrist depth | Gaussian noise with 1 mm standard deviation and 0.25% dropout, when depth is requested. | Test only; training sensor data remain exact. |
| Reported camera extrinsics | A fixed seeded bias per camera: 0.25° rotation and 1 mm translation. Physical cameras do not move. | Test only; training calibration remains exact. |
| Reported camera intrinsics | Focal values receive ±0.25% bias; principal point receives ±0.25 pixel bias. | Test only; training calibration remains exact. |

Medium marked test episodes normally use the non-lighting profile. **VideoUnmask
medium also samples lighting.** Hard marked episodes use the full profile.
**PickHighlight is restricted to normal or dim lighting** so projected patterns
cannot imitate its highlight cue. A full profile samples one lighting mode;
it does not apply every mode simultaneously.

## Mechanism variation inventory

The following table describes the marked hard test seeds. Only the RouteStick
row is also used unchanged in atomic training. ButtonUnmaskSwap training uses
the separate 2× container-speed profile described above. Task goals retain their
normal RoboMME-Plus wording.

| Task | Implemented mechanism | Training exposure / held-out part |
|---|---|---|
| BinFill | Adds nearby non-target conveyor cubes beside targets, with planned surface gaps of 1.2–1.6 cm. Target supply and spawn times remain fixed; belt speed stays original. The proposed conveyor dodge/reversal was not retained. | Test only; training changes only lighting. |
| PickXtimes | Target stays on its original ellipse but varies speed and pauses. Near placement, it makes a rapid forward or reverse displacement along that ellipse over 6 control steps (0.3 s). Trigger requires the cube bottom within 1 cm of the table and within 6 cm horizontally of the target. Dodges alternate with non-dodge eligible attempts, starting with a dodge, up to `N // 2`. Each dodge has a seeded 50% reverse choice; reversing changes subsequent travel direction. | Test-only late-placement trigger and persistent reversal; native SwingXtimes already demonstrates dodges. |
| StopCube | Cube pauses, reverses and changes speed along its seeded route. The route/target is repositioned behind the button, with the button more directly in front. The stored layout keeps an expert preparation view countable. Speed/direction changes are gated away from the target approach zone so the final approach remains predictable. | Test-only added reversals/pauses and geometry; baseline training already includes ordinary mixed-speed routes. |
| SwingXtimes | Targets orbit in larger, faster circles while staying on their respective sides. Requested radii are 4–5 cm, reduced if needed for clearance; periods are 48–72 control steps (2.4–3.6 s). Reactive dodges remain active, and circular motion continues afterward. The pickup cube starts in front of one seeded target, encouraging temporary gripper occlusion. | Test-only composition of larger circles, reactive dodges and pickup occlusion. |
| ButtonUnmask | The first button press triggers 1–2 visible multi-cluster cube swaps, then quick masking by stationary containers. Each round takes 20 steps (1 s), twice the earlier shuffle speed. The native 90-step press deadline remains. | Test-only pre-mask shuffle; training recolors the button. |
| ButtonUnmaskSwap | The first accepted button press triggers 2–3 visible multi-cluster cube swaps, then masking. Rounds take 32 steps (1.6 s), twice the earlier shuffle speed. After masking, containers retain their original RoboMME-Plus motion behavior. Both required presses still have the 180-step deadline. | Test-only pre-mask shuffle. Training only doubles native container-swap speed, with no cube shuffle. |
| VideoUnmask | The demonstration shows 2–3 visible cube-cluster swaps before stationary masking. Each round takes 40 steps (2 s). The cube-to-container mapping follows the cubes' real final positions. | Test-only pre-mask shuffle; training changes only lighting. |
| VideoUnmaskSwap | Adds 2–3 pre-mask cube-cluster swaps, 64 steps (3.2 s) each. Container motion uses a transferred VideoRepick or ButtonUnmaskSwap style, with continued travel and recurring swaps during execution. The eight selected seeds split evenly between those two source styles. | Test-only transferred container motion and pre-mask shuffle; training recolors board/containers. |
| VideoRepick | Uses transferred VideoUnmaskSwap or ButtonUnmaskSwap motion, plus recurring physical swaps. Movement continues through execution, with local proximity/grasp pauses to permit picking. The eight selected seeds split evenly between the two source styles. | Test-only transferred motion and recurring swaps; training changes only lighting. |
| VideoPlaceButton | Alternates gathering/spreading continuous motion for 50–80 steps with 2–3 global pair-swap rounds. Each swap takes 40 steps, with up to two concurrent one-to-one pairs. Ring edges maintain at least 3.5 cm clearance. Execution movement stops permanently on the first grasp of the requested cube. No dodge. | Test only; training changes ring colors. |
| VideoPlaceOrder | The same added ring-motion mechanism as VideoPlaceButton, while retaining the native order-memory task. Execution movement stops permanently on grasp; no dodge. | Test only; training changes lighting. |
| PatternLock | A seeded nonempty combination of: neighbor activation, longer contact holds, and smaller nodes. Neighbor direction is fixed per episode: left, right or forward, with wraparound at boundaries. Longer holds require 12–18 steps (0.6–0.9 s), versus 6 steps without that factor. Smaller targets scale visible and contact radii to 45–60% of original. The demonstration uses the same rule, hold and size as execution. | Neighbor activation and smaller nodes remain test-only. Longer holds are exposed through RouteStick. |
| RouteStick | Requires 12–18 consecutive contact intervals (0.6–0.9 s) before activation, also shown in the demonstration. Target size remains unchanged: visible radius 2 cm and contact threshold 3 cm. No new tilt requirement. | Train: the same longer-hold mechanism in its selected easy episode. |

Shared constraints:

- Cube/cover identity is preserved during motion; swap routes are checked for
  clearance rather than changing identities in place.
- Each target in the four mechanism Unmask tasks must have at least **12 visible
  pixels in one front or wrist cue frame before masking**. This is not a 40-pixel
  guarantee, nor a requirement for visibility in both cameras at once.
- Mechanism timing uses its validated expert execution reference:
  `max_steps = ceil(1.5 * oracle_policy_step) + 200`; demo steps are excluded.
- PatternLock and RouteStick retain their normal task constraints except for
  the specific mechanism factors above. No added cube-size variation is currently
  integrated; PatternLock target size is the size variation in this mechanism set.



PatternLock and RouteStick show the demo's final activated target for two control
steps (0.1 s). Execution keeps its original success checks and each variation's
activation rules, without an added final wait.

## Interpreting held-out variation

- Distinguish unseen parameters, transfer between tasks, and unseen rules.
  Ordinary training already includes moving objects, swaps, occlusion, short
  sticks and SwingXtimes dodges; these families are not wholly unseen.
- Test-only factors include cube/tool shading, active-cue recoloring, concrete,
  moving-color lighting, sensor noise, belt camouflage and visual clutter.
  Other appearance families are seen individually but combined in test.
- New pre-mask shuffles, transferred recurring motion, late placement dodges,
  neighbor activation and smaller PatternLock nodes remain test-only mechanisms.
  Longer activation holds are introduced through RouteStick training.
- Keep earlier model results labeled by their training-data version. Do not
  interpret all perceptual or all mechanism episodes as unseen after this update.

## Sources

- [Training metadata](../src/robomme_plus/env_metadata/train/)
- [Test metadata](../src/robomme_plus/env_metadata/test/)
- [Atomic appearance presets](../src/robomme_plus/variation/atomic.py)
- [Appearance configuration and controls](../src/robomme_plus/variation/)
- [Mechanism implementation and controls](../src/robomme_plus/variation/mechanism/)
- [Native difficulty differences](../src/robomme_plus/README.md)

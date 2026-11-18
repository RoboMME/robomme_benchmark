# RoboMME-Plus: Benchmarking Memory Generalization and Adaptation for Robot Policies

<p align="center">
  <img src="../../assets/RoboMME-Plus-logo-v2.png" alt="RoboMME-Plus" width="640">
</p>

RoboMME-Plus tests robot memory under more demanding conditions. It retains
RoboMME's 16 tasks across counting, permanence, reference and imitation, with
changes to object counts, motion, occlusion, demonstration histories and tools.

## Splits and training data

- **Train:** 50 successful episodes: easy/medium/hard for 15 tasks, plus one example
  for each of MoveCube's five methods. One episode per non-MoveCube task has a
  selected atomic variation: 13 appearance changes and 2 mechanism changes;
  the other 35 retain their original settings. All recordings use 20 Hz control.
- **Test:** 800 episodes, 50 per task. Tiered tasks have 10 easy, 15 medium and
  25 hard episodes; MoveCube has 10 per method. Train/test seeds are disjoint.
- **Downloads:** [H5](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5) and
  [LeRobot](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot).
  The current `main` contains v2 with atomic training variations and successful
  joint and end-effector replay. The original data is preserved on each dataset's `v1` branch.

Training seeds and episode settings are in
[`env_metadata/train/record_dataset_<Task>_metadata.json`](env_metadata/train).
Test settings are in [`env_metadata/test/`](env_metadata/test).
Use `dataset="train"` or `dataset="test"` with `BenchmarkEnvBuilder`.
The [data guide](../../doc/plus_training_data_format.md) describes fields and mappings.
MoveCube's packaged [demonstration controls](assets/demonstrations/MoveCube)
are replayed through the simulator at reset to produce its conditioning video.

## Evaluation protocol

Choose the setting before development and report each setting and budget separately.

| Setting | Released training demonstrations | Simulator starts per task |
|---|---|---|
| ZS: direct transfer | None | 0 |
| AfD: adaptation from demonstrations | Up to all 50 | 0 (R0) |
| AfI: adaptation from interaction | None | At most 1, 3, 5 or 10 |
| AfD+I: adaptation from both | Up to all 50 | At most 1, 3, 5 or 10 |

- **Training experience:** Use only the released demonstrations and
  [training seeds](env_metadata/train). Demonstrations may be reused without limit;
  any RoboMME-Plus fine-tuning uses only these 50 recordings. Every simulator start
  counts toward R, including previews, debugging, verification, retries and sandbox
  rehearsals. R is a rollout budget, not a control-step budget.
- **Permitted inputs:** Learn from task instructions, public observations and
  simulator feedback. Do not expose task source, oracle annotations or privileged
  object state to the agent. A test episode's conditioning video is a normal
  observation in every setting, distinct from an expert training demonstration.
- **Shared method:** Use the same model or framework across tasks. Task-specific
  prompts or strategies must be learned from the permitted training experience.
- **Frozen testing:** Freeze weights, programs, prompt templates, tools and
  libraries before testing. Reset adaptive memory between episodes. Test data,
  seeds and rollout outcomes cannot be used for development or policy selection.

Evaluate all **50 test episodes per task (800 total)** with the default variation
settings and execution limits below. Report changed test conditions separately.
See [adaptation details and reporting requirements](../../doc/plus_agentic_evaluation.md)
and [results by setting](../../doc/robomme_plus_baseline_results.md).

## Task changes

Triples below mean **easy / medium / hard**. Ranges describe the task code;
the fixed seeds may not realize every endpoint. Goals count objects to retrieve.

Selected episodes add either perceptual variations (appearance and sensor changes)
or mechanism variations (motion and interaction changes). They are enabled by
default; training exposes individual factors while testing includes held-out
factors and combinations.

| Task | RoboMME setup | RoboMME-Plus setup |
| --- | --- | --- |
| **BinFill** | Collect **1–5 cubes** | Collect **6–8 / 3–6 / 3–6 cubes** |
| **PickXtimes** | **1–5** pick-and-place cycles | **6–8 / 5–6 / 7–8** pick-and-place cycles |
| **SwingXtimes** | **1–3** swing cycles; **0 target dodges** | **4–7 / 3–4 / 4–6** cycles; **0 / 0–2 / 0–4** target dodges per visit |
| **StopCube** | Stop on pass **2–5**; **1 trajectory type** | Stop on pass **3–8 / 5–10 / 5–10**; **4 / 4 / 5** trajectory types |
| **VideoUnmask** | **3, 5 or 15 containers**; **1–2 goals** | **5–10 / 11–15 / 16–28 containers**; **3–5 goals** at every level |
| **ButtonUnmask** | **3, 5 or 15 containers**; **1–2 goals** | **6–8 / 8–12 / 16–18 containers**; **3–5 goals** at every level |
| **VideoUnmaskSwap** | **3–4 containers**; **1–2 goals** | **6–8 / 8–10 / 9–12 containers**; **3–5 goals** at every level |
| **ButtonUnmaskSwap** | **3–4 containers**; **1–2 goals** | **6–8 / 9–10 / 11–13 containers**; **3 / 3–4 / 3–4 goals** |
| **PickHighlight** | **3–6 objects**; **1–3 goals**; **90-frame** highlights | **9–10 / 11–13 / 14–17 objects**; **3–5 / 4–6 / 5–9 goals**; **45 / 30–45 / 16–45-step** highlights |
| **VideoRepick** | **3 or 15 cubes**; **1 goal** | **4–6 / 7–10 / 6–10 objects**; **1 goal** at every level |
| **VideoPlaceButton** | **3–4 target rings**; **1 demonstrated cube** | **4 / 6 / 7–8 rings**; **1 / 1 / 2 demonstrated cubes** |
| **VideoPlaceOrder** | **4 rings**; **2–4 placements**; request **1st–4th** | **4 / 6 / 7–8 rings**; **4 / 5 / 5 placements per cube**; request **1st–4th / 1st–5th / 1st–5th** |
| **MoveCube** | **3 manipulation methods** | **5 methods**, **10 test episodes each**; **3 methods available per phase** |
| **InsertPeg** | **34 mm hole**; **20 mm peg**; **0 mm hole shift** | **23–25 / 22–23 / 20.4 mm hole** (easy / medium / hard); **20 mm peg**; **5–10 mm shift along each hole-face axis** |
| **PatternLock** | **2–8 nodes**; **3×3–5×5 grids**; **10 cm** stick | **4–5 / 6–8 / 9–12 nodes**; **5×5 grid**; **10 / 5–7 / 3.5–6 cm** execution stick |
| **RouteStick** | **2–7 segments**; **10 cm** drawing stick | **8–13 / 4–7 / 6–8 curves**; **10 / 2.5–5 / 2.5–5 cm** execution stick |

Counting and reference tasks add motion, distractors and rearrangement. PickXtimes
and VideoPlace require release and a 5 cm upward retreat. SwingXtimes credits
visits after dodges settle; PickHighlight rejects wrong-object pickups through
the final button press.

Permanence tasks increase occlusion and container motion. ButtonUnmask requires
a press before step 90; ButtonUnmaskSwap requires both buttons before step 180
and before pickup. Missing a deadline fails the task.

MoveCube requires the demonstrated tool and method; InsertPeg requires the
demonstrated peg end and box side. See [MoveCube's method criteria](tasks/imitation/move_cube_support/SUCCESS_CRITERIA.md).

PatternLock uses tilted mounts and requires the stick within 8° of downward
vertical; RouteStick adds moving obstacles. Both allow 450 control steps without
valid drawing, subject to the overall execution budget.

## Evaluate

Recoverable setup errors retry at most three times with the same episode settings.
Budgeted agent development uses one attempt per reserved rollout. Policy execution
is not retried automatically; successful reset info reports `reset_attempts`.

Follow the [installation](../../readme.md#-installation) and
[baseline running instructions](../../scripts/plus_eval/README.md), or use your own policy:

```python
from robomme_plus import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder("VideoRepick", dataset="test", action_space="joint_angle")
env = builder.make_env_for_episode(0)
try:
    obs, info = env.reset()
    goal = info["task_goal"][0]
    while True:
        action = policy.predict(obs, goal, error_message=info.get("error_message"))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = info.get("status") == "success"
            break
finally:
    env.close()
```

| Interface | Convention |
| --- | --- |
| Action spaces | `joint_angle`, `ee_pose`, `waypoint` |
| Observations | Lists of front/wrist RGB, joints, TCP and gripper state; latest sample is `[-1]` |
| Joint actions | 7 arm joints + gripper; stick tasks use 7 joints |
| Pose/waypoint actions | World-frame `[x, y, z, roll, pitch, yaw, gripper]`, metres/radians |
| Waypoint gripper | `-1` close, `0` keep, `1` open |
| Execution budget | `ceil(1.5 * oracle_policy_step) + 200` control steps; reset/demo excluded |
| Recoverable action error | `info["error_message"]`; 100-step penalty; continue until termination |
| Task failure | `info["failure_reason"]` explains the violated rule |

Preserve reset demonstration frames. Construct a new environment for each episode.
MoveCube ball/cue methods allow two trials; other methods allow one. A retry resets
that episode's scene and execution budget internally.

Only `status == "success"` counts as success. Include failures, timeouts and errors
in the denominator. Keep oracle subgoals and test metadata out of policy inputs.
Report each evaluation setting and interaction budget separately using the
[submission template](../../doc/submission/model_example_plus.md).

Published dataset URLs and recording format identifiers retain their original
names. Use `robomme_plus` and `RoboMME-Plus/<Task>-v0` in code.

# RoboMME-Plus: Benchmarking Memory Generalization and Adaptation for Robot Policies

<p align="center">
  <img src="../../assets/RoboMME-Plus-logo-v2.png" alt="RoboMME-Plus" width="640">
</p>

RoboMME-Plus tests whether robot memory skills transfer when task conditions
change, and how much new experience helps policies adapt. It retains RoboMME's
16 tasks across Counting, Permanence, Reference and Imitation, with more objects,
repetitions, continuous motion, occlusion and limited correction. The 800 held-out
tests add perceptual and mechanism variations that are mostly unseen in training.

## Getting started

1. **Choose your setting** using the [evaluation protocol](#evaluation-protocol) below.
2. **Install and prepare inputs:** follow [installation](../../readme.md#-installation)
   and the [training-data guide](../../doc/robomme_plus/plus_training_data_format.md).
   Pin the benchmark and dataset revisions; use only inputs allowed by your setting.
3. **Set up the method:** follow the [baseline instructions](../../scripts/plus_eval/README.md)
   or integrate your own agent with the [development interface](../../doc/robomme_plus/agentic_development.md).
   Before development, run the [harness audit](../../doc/robomme_plus/auditing.md#1-review-the-harness).
4. **Develop and retain every solution revision and its logs**, including failed
   and interrupted work. Freeze the final policy, prompts, tools and harness.
5. **Audit the saved development record** with the
   [instructions and example prompt](../../doc/robomme_plus/auditing.md#2-review-the-completed-development-record).
6. **[Evaluate all 800 test episodes](#evaluate) and [submit results](../../doc/robomme_plus/submission/model_example_plus.md)**
   with code, complete development logs and both audit reports for agentic development.

The two audits apply to agentic development. The
[complete guide](../../doc/robomme_plus/README.md) links the audit skills,
ASPIRE/RAX setup and submission examples.

## Splits and training data

- **Train:** 50 successful episodes: easy/medium/hard for 15 tasks, plus one example
  for each of MoveCube's five methods. All recordings use 20 Hz control.
- **Test:** 800 episodes, 50 per task. Tiered tasks have 10 easy, 15 medium and
  25 hard episodes; MoveCube has 10 per method. Train/test seeds are disjoint within each task.
- **Downloads:** [H5](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5) and
  [LeRobot](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot).
  The current `main` contains v2 with atomic training variations and successful
  joint and end-effector replay. The original data is preserved on each dataset's `v1` branch.

Training seeds and episode settings are in
[`env_metadata/train/record_dataset_<Task>_metadata.json`](env_metadata/train).
Test settings are in [`env_metadata/test/`](env_metadata/test).
Use `dataset="train"` or `dataset="test"` with `BenchmarkEnvBuilder`.
The [data guide](../../doc/robomme_plus/plus_training_data_format.md) describes fields and mappings.

## Evaluation protocol

The same rules apply to **Non-Agentic** policies trained or fine-tuned for control
and **Agentic** systems that prompt frontier models to plan, act or write code. Choose a setting before development.

| Setting | What it may use before testing | Maximum Plus rollouts per task |
| --- | --- | --- |
| `zero-shot` | No Plus data: test directly, or develop only on original RoboMME | 0 |
| `demo-only` | The 50 released demonstrations, used offline | 0 |
| `interaction-only` | Simulator rollouts on provided training seeds; no released demonstrations | 10 |
| `demo&interaction` | Released demonstrations and training-seed rollouts | 10 |

`zero-shot` uses the original policy without Plus adaptation. Released training
demonstrations are recorded data for offline analysis and learning. Replaying
their actions through the simulator counts as development interaction and is
not permitted in the `demo-only` setting. Interaction budgets are
**at most 10 development attempts per task, across all its training seeds
combined**. Reporting results for 1, 3 and 5 attempts is optional; the leaderboard uses 10 attempts for interaction settings. Each result evaluates all 800 test episodes.

1. **Keep the full development history.** Record the whole development process
   so it can be audited. Every simulator start on a Plus training seed counts,
   including previews, failed setup, retries, demonstration replays, debugging,
   verification and model selection. All share the task's budget and follow its
   episode step limits. Agentic methods follow the
   [shared development interface](../../doc/robomme_plus/agentic_development.md)
   and submit complete logs with the
   [harness audit](../../doc/skills/audit_robomme_plus_harness.md) and
   [development-log audit](../../doc/skills/audit_robomme_plus_development.md).

2. **Use only permitted resources.** Adapt to Plus using released demonstrations
   and/or training seeds, according to the setting. Do not access privileged
   information, reverse-engineer Plus task code, use test seeds during development,
   or handcraft task solutions from implementation details. The original RoboMME
   [dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5) and
   [simulator](../robomme) may be used freely in every setting, outside the Plus
   rollout budget. A task's conditioning video is a normal observation in every
   setting, distinct from an expert training recording.

3. **Freeze everything before evaluation.** Finalize weights, programs, prompts,
   tools, libraries and the harness before testing. Reset memory between test
   episodes. Do not learn from test results or use them for policy selection.
   Report test-time optimization separately.

4. **Report resource usage.** Include the setting, demonstrations used, rollout
   budget, actual rollouts and simulator steps per task, and development/evaluation
   API costs separately in USD. Compare methods under the same setting and budget.

Evaluate all **50 test episodes per task (800 total)** with default variations
and execution limits. Keep failures, timeouts and errors in the denominator;
report changed test conditions separately. Submit per-task results, code,
development logs and audits through a pull request using the
[submission template](../../doc/robomme_plus/submission/model_example_plus.md).
See the [baseline results](../../doc/robomme_plus/robomme_plus_baseline_results.md)
and [running instructions](../../scripts/plus_eval/README.md).

### Guidance for coding agents

During RoboMME-Plus development:

- **Respect the setting.** Use released Plus demonstrations only in demo settings.
  Plus development rollouts use the provided training seeds and episode settings
  within the per-task budget. `demo-only` permits no Plus rollouts. Original
  RoboMME data and simulator use remains allowed in every setting.
- **Keep testing separate.** Never use Plus test seeds, data or outcomes for development
  or policy selection. Freeze the policy and agentic framework before evaluation.
- **Preserve audit evidence.** Save complete prompts, model responses, tool calls
  and outputs, including child and resumed sessions. Retain code revisions,
  rollout ledgers with task/seed/outcome records (including failures), frozen
  artifact hashes, and separate development/evaluation API usage and costs.

Use the [harness audit](../../doc/skills/audit_robomme_plus_harness.md) and
[development-log audit](../../doc/skills/audit_robomme_plus_development.md)
to check development compliance against this protocol.
Follow the [audit procedure and example prompts](../../doc/robomme_plus/auditing.md)
to review the harness before development and the saved logs before submission.
Include both audit reports and their supporting evidence in the
[agentic development audit record](../../doc/robomme_plus/submission/model_example_plus.md#agentic-development-audit-record).

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
[submission template](../../doc/robomme_plus/submission/model_example_plus.md).

Published dataset URLs and recording format identifiers retain their original
names. Use `robomme_plus` and `RoboMME-Plus/<Task>-v0` in code.



## Task changes

Triples below mean **easy / medium / hard**. Ranges describe the task code;
the fixed seeds may not realize every endpoint. Goals count objects to retrieve.
Medium and hard also change motion, occlusion or tool geometry, so difficulty
does not depend on counts alone.

| Task | RoboMME setup | RoboMME-Plus setup |
| --- | --- | --- |
| **BinFill** | Collect **1–5 cubes** | Collect **6–8 / 3–6 / 3–6 cubes**. Medium/hard use a conveyor at **5 / 6.5 cm/s**; hard requires all three colors. |
| **PickXtimes** | **1–5** pick-and-place cycles | **6–8 / 5–6 / 7–8** cycles. Medium/hard use a moving target; hard has a larger, faster path. |
| **SwingXtimes** | **1–3** swing cycles; **0 target dodges** | **4–7 / 3–4 / 4–6** cycles; **0 / 0–2 / 0–4** target dodges per visit |
| **StopCube** | Stop on pass **2–5**; **1 trajectory type** | Stop on pass **3–8 / 5–10 / 5–10**; **4 / 4 / 5** trajectory types. Medium moves faster; hard changes speed within/across passes and shrinks the target. |
| **VideoUnmask** | **3, 5 or 15 containers**; **1–2 goals** | **5–10 / 11–15 / 16–28 containers**; **3–5 goals**. Medium/hard add an occluding board and empty covers. |
| **ButtonUnmask** | **3, 5 or 15 containers**; **1–2 goals** | **6–8 / 8–12 / 16–18 containers**; **3–5 goals**. Medium/hard add occlusion and empty covers; hard adds central distractors. |
| **VideoUnmaskSwap** | **3–4 containers**; **1–2 goals** | **6–8 / 8–10 / 9–12 containers**; **3–5 goals**. Containers move independently; medium/hard add an occluding board. |
| **ButtonUnmaskSwap** | **3–4 containers**; **1–2 goals** | **6–8 / 9–10 / 11–13 containers**; **3 / 3–4 / 3–4 goals**. Medium/hard add occlusion; hard has faster container traffic. |
| **PickHighlight** | **3–6 objects**; **1–3 goals**; **90-frame** highlights | **9–10 / 11–13 / 14–17 objects**; **3–5 / 4–6 / 5–9 goals**; **45 / 30–45 / 16–45-step** highlights. Higher levels rearrange larger groups more often. |
| **VideoRepick** | **3 or 15 cubes**; **1 goal** | **4–6 / 7–10 / 6–10 objects**; **1 goal**. Medium/hard exchange moving objects' destinations; hard adds an occluding board. |
| **VideoPlaceButton** | **3–4 target rings**; **1 demonstrated cube** | **4 / 6 / 7–8 rings**; **1 / 1 / 2 demonstrated cubes**. Medium/hard swap rings more frequently. |
| **VideoPlaceOrder** | **4 rings**; **2–4 placements**; request **1st–4th** | **4 / 6 / 7–8 rings**; **4 / 5 / 5 placements per cube**; request **1st–4th / 1st–5th / 1st–5th**. Medium/hard swap rings more frequently; hard tracks two cubes. |
| **MoveCube** | **3 manipulation methods** | **5 methods**, **10 test episodes each**; **3 methods available per phase** |
| **InsertPeg** | **34 mm hole**; **20 mm peg**; **0 mm hole shift** | **23–25 / 22–23 / 20.4 mm hole** (easy / medium / hard); **20 mm peg**; **5–10 mm shift along each hole-face axis** |
| **PatternLock** | **2–8 nodes**; **3×3–5×5 grids**; **10 cm** stick | **4–5 / 6–8 / 9–12 nodes**; **5×5 grid**; **10 / 5–7 / 3.5–6 cm** execution stick. Medium/hard shift, rotate and rescale the grid, with **10–20° / 20–30°** tool-mount tilt. |
| **RouteStick** | **2–7 segments**; **10 cm** drawing stick | **8–13 / 4–7 / 6–8 curves**; **10 / 2.5–5 / 2.5–5 cm** execution stick. Medium/hard change the route layout; hard can move obstacle boards. |

PickXtimes and VideoPlace require release and a 5 cm upward retreat. SwingXtimes credits
visits after dodges settle; PickHighlight rejects wrong-object pickups through
the final button press.

ButtonUnmask requires a press before step 90; ButtonUnmaskSwap requires both buttons before step 180
and before pickup. Missing a deadline fails the task.

MoveCube requires the demonstrated tool and method; InsertPeg requires the
demonstrated peg end and box side. See [MoveCube's method criteria](tasks/imitation/move_cube_support/SUCCESS_CRITERIA.md).

PatternLock requires the stick within 8° of downward vertical. Both drawing tasks
allow 450 control steps without valid drawing, subject to the overall execution budget.

**Mechanism variations** add the following changes to selected episodes, beyond
their difficulty level. They are enabled by default.

| Task(s) | Additional mechanism change |
| --- | --- |
| BinFill | Closely spaced conveyor neighbors at the original speed. |
| PickXtimes | Target pauses, speed changes and forward/reverse dodges during placement. |
| SwingXtimes | Larger, faster target orbits that continue after dodges. |
| StopCube | Relocated route and target, pauses, reversals and changing speed. |
| VideoUnmask, ButtonUnmask | Visible cube shuffles before stationary covers hide them. |
| VideoUnmaskSwap | Visible cube shuffles, followed by mixed container motion and recurring swaps. |
| ButtonUnmaskSwap | Visible cube shuffles before covering; training includes doubled container speed. |
| VideoRepick | Motion patterns transferred from other tasks, with recurring swaps. |
| VideoPlaceButton, VideoPlaceOrder | Alternating ring gathering/spreading and global swaps. |
| PatternLock | Neighbor-node activation, longer contact holds, smaller targets, or combinations. |
| RouteStick | Longer contact holds to activate targets. |

**Perceptual variations** change textures, object appearance, lighting and, in
selected test episodes, depth/calibration noise or visual clutter. They preserve
task mechanics. Training exposes individual factors; testing includes held-out
factors and combinations. Episode metadata selects both variation types.

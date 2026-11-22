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
5. **Prepare runnable frozen code** with installation and [evaluation instructions](#evaluate).
   The RoboMME team evaluates all 800 test episodes for verification.
6. **Audit and submit:** run the [development-log audit](../../doc/robomme_plus/auditing.md#2-review-the-completed-development-record)
   if you use an agentic method to iteratively improve your policies. Follow
   [What to submit](../../doc/robomme_plus/submission/model_example_plus.md#what-to-submit).

The audits in steps 3 and 6 apply only to [agentic development](#evaluation-protocol).
The [complete guide](../../doc/robomme_plus/README.md) links the audit skills,
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

Choose a setting before development. These rules cover **Non-Agentic** control
policies and **Agentic** systems that prompt frontier models to plan, act or code.
**Agentic development** uses agents to improve policies, prompts, skills or memory
from Plus demonstrations or rollouts. It requires development logs and two audits.

| Setting | What it may use before testing | Maximum Plus rollouts per task |
| --- | --- | --- |
| `zero-shot` | No Plus data: test directly, or develop only on original RoboMME | 0 |
| `demo-only` | The 50 released demonstrations, used offline | 0 |
| `interaction-only` | Simulator rollouts on provided training seeds; no released demonstrations | 10 |
| `demo&interaction` | Released demonstrations and training-seed rollouts | 10 |

Demonstrations are offline recordings. Replaying their actions in the simulator
counts as interaction and is forbidden in `demo-only`. The interaction budget is
**at most 10 attempts per task across all its training seeds combined**.
The leaderboard uses this budget; 1, 3 and 5 are optional. Each result covers all 800 tests.

1. **Keep the full development history.** Record the whole development process
   so it can be audited. Every simulator start on a Plus training seed counts,
   including previews, failed setup, retries, demonstration replays, debugging,
   verification and model selection. All share the task's budget and follow its
   episode step limits. Agentic development uses the
   [shared development interface](../../doc/robomme_plus/agentic_development.md),
   publishes its text logs and code history, and includes the
   [harness](../../doc/skills/audit_robomme_plus_harness.md) and
   [development-log](../../doc/skills/audit_robomme_plus_development.md) audits.

2. **Use only permitted resources.** Adapt to Plus using released demonstrations
   and/or training seeds, according to the setting. Do not access privileged
   information, reverse-engineer Plus task code, use test seeds during development,
   or handcraft task solutions from implementation details. Agent-readable docs
   are this README, the [training-data format](../../doc/robomme_plus/plus_training_data_format.md),
   the [development interface](../../doc/robomme_plus/agentic_development.md), and
   the Markdown files in the [public prompt bundle](../../scripts/plus_eval/benchmark_io/prompts/current/README.md).
   This permission does not extend to linked files. Baseline results, submissions,
   audit evidence and other methods' solutions are for operators and reviewers only.
   Agents must not be able to read or import the `robomme_plus` package beyond this
   README. That covers task code, variations, scoring, success criteria and
   simulator internals, including installed copies. They must also be unable to
   reach test metadata (`env_metadata/test/`), test results or other methods'
   Plus solutions. **Any agent access to these is a violation.** Prompts, skills
   and notes given to agents must not restate task-code details, such as exact
   parameters, thresholds or success criteria, beyond this public documentation.
   List the documents given to agents in the audit record.
   The original RoboMME
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
report changed test conditions separately. Submit through a pull request using the
[submission template](../../doc/robomme_plus/submission/model_example_plus.md#what-to-submit).
Provide runnable frozen code; agentic development also requires complete
development text logs and code history. Development trajectories, images and
videos are optional uploads. The RoboMME team runs verification evaluation and
updates the leaderboard after reviewing the submission.
See the [baseline results](../../doc/robomme_plus/robomme_plus_baseline_results.md)
and [running instructions](../../scripts/plus_eval/README.md).

### Guidance for coding agents

During RoboMME-Plus development:

- **Respect the setting.** Use released Plus demonstrations only in demo settings,
  and Plus rollouts only on the provided training seeds within the per-task budget.
  Original RoboMME data and simulator remain allowed in every setting.
- **Keep testing separate.** Never use Plus test seeds, data, results or task source
  for development or policy selection. Freeze the policy and framework before evaluation.
- **Preserve audit evidence** as listed in the
  [development interface](../../doc/robomme_plus/agentic_development.md#preserve-evidence).

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
in the denominator. Use a shared model checkpoint or agentic framework across all
16 tasks. For repeated runs, report each inference seed and the mean for the same
setting and budget. Keep oracle subgoals and test metadata out of policy inputs.
Report each evaluation setting and interaction budget separately using the
[submission template](../../doc/robomme_plus/submission/model_example_plus.md).

Published dataset URLs and recording format identifiers retain their original
names. Use `robomme_plus` and `RoboMME-Plus/<Task>-v0` in code.



## Task changes

Easy, medium and hard differ in counts, motion, occlusion and tool geometry.
This summary is qualitative; do not recover exact parameters
from Plus task code ([rule 2](#evaluation-protocol)).

| Task | RoboMME setup | RoboMME-Plus changes |
| --- | --- | --- |
| **BinFill** | Collect **1–5 cubes** | More cubes. Medium/hard add a moving conveyor; hard requires all three colors. |
| **PickXtimes** | **1–5** pick-and-place cycles | More cycles. Medium/hard move the target; hard uses a larger, faster path. |
| **SwingXtimes** | **1–3** swing cycles; no target dodges | More cycles. At medium/hard the target can dodge during visits. |
| **StopCube** | Stop on pass **2–5**; one trajectory type | Later stopping passes and several trajectory types. Medium moves faster; hard varies speed within and across passes and shrinks the target. |
| **VideoUnmask** | **3, 5 or 15 containers**; **1–2 goals** | More containers and goals. Medium/hard add an occluding board and empty covers. |
| **ButtonUnmask** | **3, 5 or 15 containers**; **1–2 goals** | More containers and goals. Medium/hard add occlusion and empty covers; hard adds central distractors. |
| **VideoUnmaskSwap** | **3–4 containers**; **1–2 goals** | More containers and goals, moving independently. Medium/hard add an occluding board. |
| **ButtonUnmaskSwap** | **3–4 containers**; **1–2 goals** | More containers and goals. Medium/hard add occlusion; hard has faster container traffic. |
| **PickHighlight** | **3–6 objects**; **1–3 goals**; **90-frame** highlights | More objects and goals with shorter highlights. Higher levels rearrange larger groups more often. |
| **VideoRepick** | **3 or 15 cubes**; **1 goal** | Varied object counts; still one goal. Medium/hard exchange moving objects' destinations; hard adds an occluding board. |
| **VideoPlaceButton** | **3–4 target rings**; **1 demonstrated cube** | More rings, swapped more often at medium/hard; hard demonstrates two cubes. |
| **VideoPlaceOrder** | **4 rings**; **2–4 placements**; request **1st–4th** | More rings and placements, with later placements requested. Medium/hard swap rings more often; hard tracks two cubes. |
| **MoveCube** | **3 manipulation methods** | **5 methods**, with a subset available in each phase. |
| **InsertPeg** | **34 mm hole**; **20 mm peg**; no hole shift | Tighter clearance at higher levels; the hole is offset on its face. |
| **PatternLock** | **2–8 nodes**; **3×3–5×5 grids**; **10 cm** stick | More nodes at higher levels and a shorter execution stick at medium/hard. Medium/hard also shift, rotate and rescale the grid and tilt the tool mount. |
| **RouteStick** | **2–7 segments**; **10 cm** drawing stick | Curved routes and a shorter execution stick at medium/hard. Medium/hard change the route layout; hard can move obstacle boards. |

**Success rules** beyond the language goal:

PickXtimes and VideoPlace require releasing the object and retreating upward.
SwingXtimes credits visits after dodges settle; PickHighlight rejects wrong-object
pickups through the final button press.

ButtonUnmask and ButtonUnmaskSwap require their button presses within an early
deadline; ButtonUnmaskSwap needs both buttons, before pickup. Missing a deadline
fails the task. Exact deadlines are in the [public timing rules](../../scripts/plus_eval/benchmark_io/prompts/current/PUBLIC_RULES.md).

MoveCube requires the demonstrated tool and method; InsertPeg requires the
demonstrated peg end and box side.

PatternLock requires the stick to stay close to vertical. Both drawing tasks fail
after a long period without valid drawing, within the overall execution budget.

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
| ButtonUnmaskSwap | Visible cube shuffles before covering; training includes faster container motion. |
| VideoRepick | Motion patterns transferred from other tasks, with recurring swaps. |
| VideoPlaceButton, VideoPlaceOrder | Alternating ring gathering/spreading and global swaps. |
| PatternLock | Neighbor-node activation, longer contact holds, smaller targets, or combinations. |
| RouteStick | Longer contact holds to activate targets. |

**Perceptual variations** change textures, object appearance, lighting and, in
selected test episodes, depth/calibration noise or visual clutter. They preserve
task mechanics. Training exposes individual factors; testing includes held-out
factors and combinations. Episode metadata selects both variation types.

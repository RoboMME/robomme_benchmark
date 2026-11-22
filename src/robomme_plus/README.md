# RoboMME-Plus: Benchmarking Memory Generalization and Adaptation for Robot Policies

<p align="center">
  <img src="../../assets/RoboMME-Plus-logo-v2.png" alt="RoboMME-Plus" width="640">
</p>

RoboMME-Plus tests memory generalization and adaptation across RoboMME's 16 tasks
in Counting, Permanence, Reference and Imitation. Tasks add objects, repetitions,
motion, occlusion and limited correction. The 800 held-out tests include perceptual
and mechanism variations that are mostly unseen in training.

## Getting started

### Fine-tuned policies or direct model control

For fine-tuning a robot policy or calling a frontier model with a general prompt
to control the robot (e.g., Embodied Astra). Development audits are not required.

1. **Choose a setting** and follow the [evaluation protocol](#evaluation-protocol).
2. **Install and prepare data:** follow [installation](../../readme.md#-installation)
   and the [training-data guide](../../doc/robomme_plus/plus_training_data_format.md).
   Use only data allowed by your setting; see the [baseline instructions](../../scripts/plus_eval/README.md).
3. **Freeze and submit:** fix the policy, prompt and inference code. Provide
   runnable code and [evaluation instructions](#evaluate) using the
   [submission template](../../doc/robomme_plus/submission/model_example_plus.md#what-to-submit).

### Agentic development

For agents developing task solutions from demonstrations or rollouts, learning
task-specific prompts, or improving themselves (e.g., ASPIRE and RAX). A harness
audit and a development-log audit are required, even if the final solution
controls the robot directly.

1. **Choose a setting and prepare data** using the protocol, installation and
   training-data links above.
2. **Build and audit the harness:** use the
   [development interface](../../doc/robomme_plus/agentic_development.md) and run the
   [harness audit](../../doc/robomme_plus/auditing.md#1-review-the-harness) before development.
3. **Develop and freeze:** retain every code/prompt revision, command, output and
   rollout record, including failed or interrupted work. Freeze the complete method
   before testing.
4. **Audit and submit:** run the
   [development-log audit](../../doc/robomme_plus/auditing.md#2-review-the-completed-development-record).
   Include frozen code, development logs and both audit reports in the
   [submission template](../../doc/robomme_plus/submission/model_example_plus.md#what-to-submit).

The RoboMME team verifies submissions on all 800 tests and updates the leaderboard.
The [complete guide](../../doc/robomme_plus/README.md) links further instructions and examples.

## Splits and training data

- **Train:** 50 successful episodes: easy/medium/hard for 15 tasks, plus one example
  for each of MoveCube's five methods. All recordings use 20 Hz control.
- **Test:** 800 episodes, 50 per task. Tiered tasks have 10 easy, 15 medium and
  25 hard episodes; MoveCube has 10 per method. Train/test seeds are disjoint within
  each task. Test seeds cannot be used during training.
- **Downloads:** [H5](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5) and
  [LeRobot](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot).

Training seeds and episode settings are in
[`env_metadata/train/record_dataset_<Task>_metadata.json`](env_metadata/train).
Test settings are in [`env_metadata/test/`](env_metadata/test).
Use `dataset="train"` or `dataset="test"` with `BenchmarkEnvBuilder`.
The [data guide](../../doc/robomme_plus/plus_training_data_format.md) describes fields and mappings.

## Evaluation protocol

Choose a setting before development. These rules apply to both tracks;
audit requirements follow the [development approach](#getting-started).

| Setting | What it may use before testing | Maximum Plus rollouts per task |
| --- | --- | --- |
| `Zero-shot` | No Plus data: test directly, or develop only on original RoboMME | 0 |
| `Demo-only` | The 50 released demonstrations, used offline | 0 |
| `Interaction-only` | Simulator rollouts on provided training seeds; no released demonstrations | 10 |
| `Hybrid` | Released demonstrations and training-seed rollouts | 10 |

Demonstrations are offline recordings; simulator replay counts as interaction
and is forbidden in `Demo-only`. The interaction leaderboard allows **at most
10 attempts per task across all its training seeds combined**. Smaller budgets
are optional under rule 5.

1. **Count rollouts and keep the full history.** Every Plus simulator start counts,
   including previews, failed setup, retries, demonstration replays, debugging,
   verification and model selection. Respect the per-task budget and episode step
   limits. Agentic development uses the
   [shared development interface](../../doc/robomme_plus/agentic_development.md),
   publishes complete text logs and code history, and includes the
   [harness](../../doc/skills/audit_robomme_plus_harness.md) and
   [development-log](../../doc/skills/audit_robomme_plus_development.md) audits.

2. **Use only permitted resources.** Use the setting's allowed data and training
   seeds. Do not use privileged information, reverse-engineer Plus task code or
   handcraft solutions from implementation details. Allowed Plus documentation is
   this README, the [training-data format](../../doc/robomme_plus/plus_training_data_format.md),
   the [development interface](../../doc/robomme_plus/agentic_development.md), and
   the Markdown files in the [public prompt bundle](../../scripts/plus_eval/benchmark_io/prompts/current/README.md).
   Linked files are not automatically allowed. Keep the rest of `robomme_plus`
   (including installed source, variations, scoring and simulator internals),
   test seeds/metadata/results, other methods' Plus solutions, baseline results,
   submissions and audit evidence inaccessible to developing agents.
   Prompts, skills and notes must not reveal task parameters, thresholds or success
   criteria beyond the allowed docs. List agent-visible documents in the audit.
   Original RoboMME's
   [dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5) and
   [simulator](../robomme) may be used freely in every setting, outside the Plus
   rollout budget. Conditioning videos are normal episode observations in every
   setting, separate from training demonstrations.

3. **Freeze everything before evaluation.** Finalize weights, programs, prompts,
   tools, libraries and the harness before testing. Reset memory between test
   episodes. Do not learn from test results or use them for policy selection.
   Report test-time optimization separately.

4. **Report resource usage.** Include the setting, demonstrations used, rollout
   budget, actual rollouts and simulator steps per task, and development/evaluation
   API costs separately in USD. Compare methods under the same setting and budget.

5. **Optional smaller budgets.** For budget R, use an independent run capped at R
   attempts per task or extract checkpoints from a 10-attempt run; state which.
   Save each checkpoint after at most R attempts and **before that task's (R+1)th
   start**. Offline revisions and saving immediately after an attempt are allowed.
   Tasks may freeze at different times. Keep exact code and dependency versions,
   including shared components; none may use
   experience beyond the reported per-task budget. Another task's later start
   alone does not invalidate a saved checkpoint.

Evaluate all **50 test episodes per task (800 total)** with default variations
and execution limits. Only `status == "success"` counts as success; failures,
timeouts and errors stay in the denominator. Report changed test conditions separately.
Submit by pull request with runnable frozen code and the required development
evidence; development trajectories, images and videos are optional uploads.
See the [baseline results](../../doc/robomme_plus/robomme_plus_baseline_results.md).

### Guidance for coding agents

Follow rules 1–5 above and the [evidence checklist](../../doc/robomme_plus/agentic_development.md#preserve-evidence).
Use the [audit walkthrough](../../doc/robomme_plus/auditing.md) for review prompts and report formats.

## Evaluate

Recoverable setup errors retry at most three times with the same episode settings.
Budgeted agent development uses one attempt per reserved rollout. Policy execution
is not retried automatically; successful reset info reports `reset_attempts`.

Use a [baseline runner](../../scripts/plus_eval/README.md) or your own policy:

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

Use a shared model checkpoint or agentic framework across all 16 tasks. For repeated
runs, report each inference seed and the mean for the same setting and budget.
Keep oracle subgoals and test metadata out of policy inputs.
Package and environment IDs are `robomme_plus` and `RoboMME-Plus/<Task>-v0`.

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

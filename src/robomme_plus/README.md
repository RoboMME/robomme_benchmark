# RoboMME-Plus: Benchmarking Memory Generalization and Adaptation for Robot Policies

<p align="center">
  <img src="../../assets/RoboMME-Plus-logo-v2.png" alt="RoboMME-Plus" width="640">
</p>

RoboMME-Plus tests memory generalization and adaptation across RoboMME's 16 tasks
in. Tasks add more objects, repetitions,
motion, occlusion and limited correction. The 800 held-out tests include perceptual and mechanism variations that are mostly unseen in training.

## Getting started

### Fine-tuned policies or direct model control

For fine-tuning a robot policy or calling a frontier model with a general prompt
to control the robot (e.g., Embodied Astra). Development audits are not required.

1. **Choose a setting** and follow the [evaluation protocol](#evaluation-protocol).
2. **Install and prepare data:** follow [installation](../../readme.md#-installation)
   and the [training-data guide](../../doc/robomme_plus/plus_training_data_format.md).
   Use only data allowed by your setting; refer to the [baselines](../../scripts/plus_eval/README.md).
3. **Submit:** release the final policy, prompt and inference code. Provide
   runnable code and [evaluation instructions](#evaluate) using the
   [submission template](../../doc/robomme_plus/submission/model_example_plus.md#what-to-submit).

### Agentic development

For agents developing task solutions from demonstrations or rollouts, learning
task-specific prompts, or improving themselves (e.g., ASPIRE and RAX). One
development audit is required, even if the final solution controls the robot directly.
It checks two things: the agent had no access to forbidden information, and it used
at most the allowed simulator rollouts.

Follow the **[agentic development guide](../../doc/robomme_plus/agentic_development.md)**:
create the run's ledger, build the harness, give the agent its instructions, develop,
evaluate, audit the development record and submit your final code.

### Submission requirements

For PR acceptance and leaderboard inclusion:

- **Code and checkpoints:** publicly release runnable code, all method-specific
  weights and final programs/prompts at fixed revisions, with setup and evaluation instructions.
- **Agentic development:** also release final evaluation solutions and complete
  text logs, code/prompt history and rollout accounting, including failed and
  interrupted work. Media and raw trajectories are optional.
- **Method description:** link at least one paper, blog post or project website.
- **Evaluation access:** for closed APIs or gated models needed at test time,
  email [robomme2026@gmail.com](mailto:robomme2026@gmail.com) with the PR link and access
  confirmation. Arrange private API credentials with quota for **800 episodes per
  submitted setting**; never post credentials in the PR.

Use the [submission template](../../doc/robomme_plus/submission/model_example_plus.md).
The RoboMME team verifies all 800 tests before leaderboard inclusion.

**Reference:** [baseline results](../../doc/robomme_plus/robomme_plus_baseline_results.md) · [ASPIRE/RAX setup](../../scripts/plus_eval/benchmark_io/README.md) · [release manifest](../../doc/robomme_plus/plus_release_manifest.json)

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
10 attempts per task across all its training seeds combined**.

1. **Count rollouts.** Every Plus simulator start counts, including previews, failed
   setup, retries, demonstration replays, debugging, verification and model
   selection. Respect the per-task budget and episode step limits. Agentic
   development uses the [agentic development guide](../../doc/robomme_plus/agentic_development.md),
   keeps logs of every agent session and simulator start, and includes the
   [development audit](../../doc/robomme_plus/agentic_development.md#step-8-audit).

2. **Use only permitted resources.** Use the setting's allowed data and training
   seeds. Do not use privileged information, reverse-engineer Plus task code or
   handcraft solutions from implementation details. Allowed Plus documentation is
   this README, the [training-data format](../../doc/robomme_plus/plus_training_data_format.md),
   the [agentic development guide](../../doc/robomme_plus/agentic_development.md), and
   the Markdown files in the [public prompt bundle](../../scripts/plus_eval/benchmark_io/prompts/current/README.md).
   Linked files are not automatically allowed. Keep the rest of `robomme_plus`
   (including installed source, variations, scoring and simulator internals),
   test seeds/metadata/results, other methods' Plus solutions, baseline results,
   submissions and audit evidence inaccessible to developing agents.
   Prompts, skills and notes must not reveal task parameters, thresholds or success
   criteria beyond the allowed docs. Do not use test results during development or
   to choose among policies.
   Original RoboMME's
   [dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5) and
   [simulator](../robomme) may be used freely in every setting, outside the Plus
   rollout budget.

3. **Submit the final code.** The RoboMME team re-runs it on all 800 tests, and the
   leaderboard uses that run. Reset memory between test episodes. If evaluation
   calls a closed API or gated model, email the RoboMME team
   ([robomme2026@gmail.com](mailto:robomme2026@gmail.com)) with the PR link and prepare an API
   key with enough quota for 800 test episodes per submitted setting. Never post
   keys in the PR.

4. **Report resource usage.** Include the setting, demonstrations used, rollout
   budget, actual rollouts per task, and estimated development and evaluation API
   costs in USD. A reasonable estimate is enough. Compare methods under the same
   setting and budget.

Evaluate all **50 test episodes per task (800 total)** with default variations
and execution limits. Only `status == "success"` counts as success; failures,
timeouts and errors stay in the denominator. Report changed test conditions separately.
Follow the [submission requirements](#submission-requirements) and include the
required development evidence; development trajectories, images and videos are
optional uploads.
See the [baseline results](../../doc/robomme_plus/robomme_plus_baseline_results.md).

### Guidance for coding agents

Follow rules 1–4 above and the [agentic development guide](../../doc/robomme_plus/agentic_development.md).

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

Medium and hard levels increase memory and control demands through more complex
sequences, motion, occlusion and geometry. This qualitative overview may be used
in a general prompt; do not recover exact parameters from Plus task code
([rule 2](#evaluation-protocol)).

| Task | RoboMME setup | RoboMME-Plus changes | Success rules beyond the language goal |
| --- | --- | --- | --- |
| **BinFill** | Collect cubes | More objects and moving scenes at higher difficulty. | — |
| **PickXtimes** | Repeat pick-and-place cycles | More repetitions and moving targets. | Release the object and retreat upward after placement. |
| **SwingXtimes** | Repeat target visits | More repetitions and variable target motion. | Visits count after target dodges settle. |
| **StopCube** | Stop an object on a requested pass | Longer sequences and more varied motion. | — |
| **VideoUnmask** | Track objects under covers | More objects, occlusion and distractors. | — |
| **ButtonUnmask** | Reveal and track hidden objects | More objects, occlusion and distractors. | Press the button within the early deadline. |
| **VideoUnmaskSwap** | Track objects under moving covers | More complex motion and occlusion. | — |
| **ButtonUnmaskSwap** | Reveal and track objects under moving covers | More complex motion and occlusion. | Press both buttons before pickup and within the early deadline. |
| **PickHighlight** | Remember highlighted objects | More objects, shorter cues and more rearrangements. | Avoid wrong-object pickups through the final button press. |
| **VideoRepick** | Retrieve a previously demonstrated object | More varied object motion and visibility. | — |
| **VideoPlaceButton** | Place using a demonstrated target | More targets and scene changes to track. | Release the object and retreat upward after placement. |
| **VideoPlaceOrder** | Recall a placement from a sequence | Longer sequences and more target motion. | Release the object and retreat upward after placement. |
| **MoveCube** | Reproduce a manipulation method | A broader range of methods and execution conditions. | Use the demonstrated tool and method. |
| **InsertPeg** | Reproduce an insertion | More varied geometry and tighter tolerances. | Use the demonstrated peg end and box side. |
| **PatternLock** | Reproduce a drawn pattern | More complex patterns and varied tool/target geometry. | Keep the stick near vertical; prolonged gaps in valid drawing fail. |
| **RouteStick** | Reproduce a drawn route | More complex paths and varied tool/obstacle layouts. | Prolonged gaps in valid drawing fail. |

**Mechanism variations** change scene behavior and responses to actions.
**Perceptual variations** change appearance and sensing conditions. Testing covers
a broader range of conditions and combinations than the small training set.

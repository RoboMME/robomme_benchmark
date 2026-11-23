# RoboMME-Plus: Benchmarking Memory Generalization and Adaptation for Robot Policies

<p align="center">
  <img src="../../assets/RoboMME-Plus-logo-v2.png" alt="RoboMME-Plus" width="640">
</p>

RoboMME-Plus tests how well robot policies generalize and adapt their memory on
RoboMME's 16 tasks. The tasks add more objects, more repetitions, moving scenes,
occlusion and fewer chances to correct mistakes. The 800 held-out test episodes
include perceptual and mechanism variations that are mostly unseen in training.

## Getting started

### Fine-tuned policies or direct model control

For fine-tuned robot policies, or a frontier model that controls the robot with a
general prompt (e.g., Embodied Astra). No development audit is needed.

1. **Choose a setting** and follow the [evaluation protocol](#evaluation-protocol).
2. **Install and prepare data:** follow [installation](../../readme.md#-installation)
   and the [training-data guide](../../doc/robomme_plus/plus_training_data_format.md).
   Use only the data your setting allows; the [baselines](../../scripts/plus_eval/README.md)
   show examples.
3. **Submit:** release the final policy, prompts and runnable inference code with
   [evaluation instructions](#evaluate), using the
   [submission template](../../doc/robomme_plus/submission/model_example_plus.md#what-to-submit).

### Agentic development

For methods where an agent writes task solutions from demonstrations or rollouts,
learns task-specific prompts, or improves itself over iterations (e.g., ASPIRE and
RAX). One **development audit** is required, even if the final solution controls
the robot directly. It checks three things: the agent had no access to forbidden
information, it stayed within the rollout budget, and the policy did not change
during testing.

Follow the **[agentic development guide](../../doc/robomme_plus/agentic_development.md)**:
set up rollout counting, build the harness, give the agent its instructions, develop,
evaluate, audit the development record and submit your final code.

### Submission requirements

For PR acceptance and leaderboard inclusion:

- **Method description:** link at least one paper, blog post or project website.
- **Code and checkpoints:** publicly release runnable code, all method-specific
  weights and the final programs and prompts at fixed revisions, with setup and
  evaluation instructions.
- **Agentic development:** also release the final solutions used for testing and
  the full development record: text logs, every code and prompt version, and every
  simulator start, including failed and interrupted work. Videos, images and raw
  trajectories are optional.
- **Evaluation access:** if testing needs a closed API or gated model, email
  [robomme2026@gmail.com](mailto:robomme2026@gmail.com) with the PR link and
  provide an API key with enough quota for **800 episodes per submitted setting**
  ([rule 3](#evaluation-protocol)).

Use the [submission template](../../doc/robomme_plus/submission/model_example_plus.md).
The RoboMME team verifies all 800 tests before leaderboard inclusion.

## Splits and training data

- **Train:** 50 successful demonstrations in total: one easy, one medium and one
  hard episode for each of 15 tasks, plus one for each of MoveCube's five methods.
  All recordings use 20 Hz control.
- **Test:** 800 episodes, 50 per task: 10 easy, 15 medium and 25 hard, or 10 per
  method for MoveCube. Train and test seeds never overlap, and test seeds must not
  be used during development.
- **Downloads:** [H5](https://huggingface.co/datasets/Yinpei/robomme_plus_data_h5) and
  [LeRobot](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot).

Training seeds and episode settings are in
[`env_metadata/train/record_dataset_<Task>_metadata.json`](env_metadata/train).
Test settings are in [`env_metadata/test/`](env_metadata/test).
Use `dataset="train"` or `dataset="test"` with `BenchmarkEnvBuilder`.
The [data guide](../../doc/robomme_plus/plus_training_data_format.md) describes fields and mappings.

## Evaluation protocol

Choose a setting before development. These rules apply to every method; only
agentic development needs an audit ([getting started](#getting-started)).

| Setting | What it may use before testing | Maximum Plus rollouts per task |
| --- | --- | --- |
| `Zero-shot` | No Plus data: test directly, or develop only on original RoboMME | 0 |
| `Demo-only` | The 50 released demonstrations, used offline | 0 |
| `Interaction-only` | Simulator rollouts on provided training seeds; no released demonstrations | 10 |
| `Hybrid` | Released demonstrations and training-seed rollouts | 10 |

Demonstrations are offline recordings. Replaying them in the simulator counts as
interaction, so it is not allowed in `Demo-only`. Interaction settings allow **at
most 10 simulator starts per task, shared across all of its training seeds**.

1. **Count rollouts.** Every Plus simulator start counts, including previews, failed
   setup, retries, demonstration replays, debugging, verification and model
   selection. Stay within the per-task budget and the episode step limits. Agentic
   development follows the [agentic development guide](../../doc/robomme_plus/agentic_development.md):
   log every agent session and simulator start, and include the
   [development audit](../../doc/robomme_plus/agentic_development.md#step-7-audit).

2. **Use only permitted resources.** Use only the data and training seeds your
   setting allows. Do not use privileged information, reverse-engineer the Plus task
   code or hand-write solutions from its implementation details.
   - **Allowed Plus documents:** this README, the
     [training-data format](../../doc/robomme_plus/plus_training_data_format.md),
     the [agentic development guide](../../doc/robomme_plus/agentic_development.md)
     and the Markdown files in the
     [public prompt bundle](../../scripts/plus_eval/benchmark_io/prompts/current/README.md).
     Files these documents link to are not allowed unless listed here.
   - **Keep away from developing agents:** the rest of `robomme_plus` (including
     installed copies, variation, scoring and simulator code), test seeds, test
     metadata and test results, other methods' Plus solutions, baseline results,
     submissions and audit evidence.
   - **No hidden task details:** prompts, skills and notes must not state task
     parameters, thresholds or success criteria beyond the allowed documents.
   - **No test feedback:** do not use test results during development or to choose
     among policies.
   - The original RoboMME [dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5)
     and [simulator](../robomme) may be used freely in every setting and do not
     count toward the Plus budget.

3. **Submit the final code.** The RoboMME team re-runs it on all 800 tests, and the
   leaderboard uses that run. Keep the policy unchanged during testing and clear its
   memory between test episodes. If evaluation
   calls a closed API or gated model, email the RoboMME team
   ([robomme2026@gmail.com](mailto:robomme2026@gmail.com)) with the PR link and prepare an API
   key with enough quota for 800 test episodes per submitted setting. Never post
   keys in the PR.

4. **Report resource usage.** Report the setting, the demonstrations used, the
   rollout budget, the rollouts actually used per task, and estimated development
   and evaluation API costs in USD. A reasonable estimate is enough. Compare methods
   only under the same setting and budget.

Evaluate all **50 test episodes per task (800 total)** with the default variations
and step limits. Only `status == "success"` counts as a success; failures, timeouts
and errors stay in the denominator. Report results under changed test conditions
separately. See the [baseline results](../../doc/robomme_plus/robomme_plus_baseline_results.md).

### Guidance for coding agents

Follow rules 1–4 above and the [agentic development guide](../../doc/robomme_plus/agentic_development.md).

## Evaluate

If an episode fails to set up, `reset()` rebuilds the same episode and tries
again, up to three times; `info["reset_attempts"]` reports how many tries it took.
Budgeted development with `benchmark_io` makes one try per counted start. Policy
actions are never retried.

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
| Step limit | `ceil(1.5 * oracle_policy_step) + 200` control steps, not counting reset or demonstration frames |
| Invalid action | Reported in `info["error_message"]`; costs 100 steps of the limit; the episode continues |
| Task failure | `info["failure_reason"]` explains the violated rule |

Keep the demonstration frames returned by `reset()`; video-conditioned tasks
depend on them. Build a new environment for each episode. MoveCube's ball and cue
methods allow two trials and the other methods one; a second trial resets the scene
and step limit automatically.

Use one model checkpoint or agentic framework for all 16 tasks. For repeated runs,
report each inference seed and the mean. Do not give the policy oracle subgoals or
test metadata. The package is `robomme_plus`; environment IDs are
`RoboMME-Plus/<Task>-v0`.

## Task changes

Medium and hard levels demand more memory and finer control, through longer
sequences, more motion, occlusion and harder geometry. You may use this overview in
a general prompt, but do not recover exact parameters from the Plus task code
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

**Mechanism variations** change how the scene behaves and responds to actions.
**Perceptual variations** change how the scene looks and is sensed. The test set
covers a wider range of conditions and combinations than the small training set.

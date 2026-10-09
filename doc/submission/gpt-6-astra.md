# GPT-6-Astra (end-to-end, waypoint)

### [Code](https://github.com/jackbjed/astra-robomme)

## Introduction
A general-purpose vision-language model (OpenAI `gpt-6-astra`, accessed through the API) used directly as the
policy: at every step it receives the task goal, the camera frames and the robot's own state, and outputs the next
end-effector waypoint (position, orientation) and gripper command. RoboMME's motion planner executes each waypoint.
Memory is the model's own context: all sampled observations and previous model responses within the episode stay
in one conversation. No training or fine-tuning on RoboMME.

## Results

> Multi-task setting with a single model for all 16 tasks, on the **test** split, **`waypoint`** action space,
> `max_steps=1300`, 50 episodes per task, evaluated with the loop of RoboMME's `scripts/evaluation.py`.
> We report a single run (the leaderboard page recommends multiple runs; the submission template asks for three). The benchmark seed is fixed internally. Model version as reported by the API: `gpt-6-astra`;
> evaluation dates 2026-10-08 to 2026-10-09 (UTC).

### Table

<table>
<tr>
  <th rowspan="2">Suite</th>
  <th rowspan="2">Task</th>
</tr>
<tr>
  <th>Run 1</th><th><b>Avg</b></th>
</tr>
<tr>
  <td rowspan="4">Counting</td>
  <td>BinFill</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>PickXtimes</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>SwingXtimes</td><td>0.0</td><td><b>0.0</b></td></tr>
<tr><td>StopCube</td><td>56.0</td><td><b>56.0</b></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr><td>ButtonUnmask</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>ButtonUnmaskSwap</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td>98.0</td><td><b>98.0</b></td>
</tr>
<tr><td>VideoRepick</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceButton</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceOrder</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td>54.0</td><td><b>54.0</b></td>
</tr>
<tr><td>InsertPeg</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr><td>PatternLock</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>RouteStick</td><td>98.0</td><td><b>98.0</b></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td><b>87.4</b></td><td><b>87.4</b></td>
</tr>
</table>

### Evaluation Details
- **Inputs per step** (identical for all tasks): task goal; front-camera frames of the demonstration/history video and of
  every executed motion, one in every 3 control steps plus the last frame; the current wrist-camera frame
  (256 x 256 RGB); TCP pose, finger positions and elapsed control steps; front/wrist camera intrinsics and extrinsics
  (RoboMME's documented optional observations `include_*_camera_*`). No depth, no simulator state, no available
  multi-choice options, no oracle subgoals.
- **Prompt**: one system prompt for all tasks (released with the code), describing the waypoint action format, the
  world frame (as in RoboMME's `env_format.md`) and general facts about the robot interface (timing of motions, gripper
  behaviour). The only task-dependent sentence states the hardware of PatternLock and RouteStick (the robot holds a
  stick, so the gripper command is ignored).
- **Episode endings besides the environment's** (all counted as failures): the model declares the task complete before
  the environment judges success; 3 consecutive actions that execute no steps; an unusable answer;
  150 model calls (in practice the 1300-step limit binds first). Environment errors (e.g. unreachable poses)
  count as failures, as in `evaluation.py`.
- No RoboMME demonstrations were used for model training or fine-tuning.

### Released Checkpoints
The model is a closed API model; no checkpoint can be released. Evaluation code, prompts, configuration and all
per-episode logs are released at [github.com/jackbjed/astra-robomme](https://github.com/jackbjed/astra-robomme).

### Citations
```
```

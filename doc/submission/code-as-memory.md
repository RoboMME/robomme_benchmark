# Code-as-Memory (CaM, 25 demos, `ee_pose`)

### [Code](https://github.com/YinpeiDai/Code-as-Memory) | [Policies](https://github.com/YinpeiDai/Code-as-Memory/tree/main/solutions/robomme/demo_only_25) | [Demos](https://huggingface.co/datasets/Yinpei/robomme-25-demos) | [Development Logs](https://huggingface.co/datasets/Yinpei/CaM_develop_logs)

## Introduction
Code-as-Memory (CaM) stores robot memory in an executable Python program. A coding agent (Claude Code with
Claude Opus 5.5) reads offline RoboMME training demonstrations and writes one Python policy for each task. The
policy keeps its memory in typed variables such as counts, colour lists, positions and keyframe references. At
each step it perceives the scene, updates that memory, decides what to do, and sends `ee_pose` commands. After
development, every policy is frozen. Evaluation uses only Python execution, with no learned weights, model calls
or network access.

## Results

**Track (select one):**

- [ ] Standard Track
- [x] Agentic Track

**API Cost — Agentic Track:** **US$0 total; US$0 per episode** for the reported 800-episode evaluation. The
frozen policies make no model or API calls. One-time development cost was **≈US$29.44 across all 16 tasks**
(≈US$1.84/task). This figure comes from Claude Code CLI session cost estimates for Claude Opus 5.5 on AWS
Bedrock, not audited billing, and excludes local compute and data preparation. There are no model parameters at
inference.

> Evaluated on the **test** split with the **`ee_pose`** action space and the original wrapper at `max_steps=1300`;
> 50 episodes per task, 800 in total. We report a single run; the benchmark seed is fixed internally. Only
> simulator success counts. All 800 episodes stay in the
> denominator, including failures and episodes stopped by the policy's decision cap. Evaluated November 5–6, 2026.

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
<tr><td>SwingXtimes</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>StopCube</td><td>94.0</td><td><b>94.0</b></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>ButtonUnmask</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>ButtonUnmaskSwap</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>VideoRepick</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceButton</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>VideoPlaceOrder</td><td>96.0</td><td><b>96.0</b></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td>100.0</td><td><b>100.0</b></td>
</tr>
<tr><td>InsertPeg</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>PatternLock</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr><td>RouteStick</td><td>100.0</td><td><b>100.0</b></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td><b>99.4</b></td><td><b>99.4</b></td>
</tr>
</table>

**Overall:** 795/800 successes (**99.38%**). StopCube: 47/50, with two task failures and one episode stopped by
the policy's 40-decision cap. VideoPlaceOrder: 48/50, with two task failures. The other 14 tasks: 50/50.

### Training Details
- **No weight training and no simulator interaction during development.** One agent configuration is shared
  across all 16 tasks: the same coding agent, model, prompts and public RoboMME interface. Claude Code with
  `claude-opus-5-5` at reasoning effort `high` writes a separate Python policy for each task; this is analogous to
  MEMI's per-task prompts.
- **Demonstrations:** 25 per task (9 easy / 8 medium / 8 hard) from RoboMME's released training H5, 400 in total
  out of the 1,600 released. They include front/wrist RGB, aligned depth, camera calibration, TCP, joints, fingers
  and recorded actions. The coding agent can analyse them only with offline scripts. No test-split episodes,
  benchmark source code, object poses, masks or oracle subgoals are exposed to it.
- **Policy inputs during evaluation:** the task goal; the native task-provided demonstration video, when the task
  has one; and every frame since the previous decision. Each frame gives front/wrist RGB, depth, intrinsics and
  extrinsics (RoboMME's optional `include_*` observations), TCP pose, joints and finger widths. No privileged
  simulator state or `multi_choice` options are used.
- **Policy outputs:** absolute world TCP targets (`ee_pose`: x, y, z, roll, pitch, yaw, gripper), one per native
  control step. Each episode allows at most 40 policy decisions; reaching the cap counts as failure. Scoring is
  RoboMME's own.
- **Computation:** averaged over 80 recorded test episodes, the frozen policies use **1.413 ms per control step**
  for policy computation on one CPU core (AMD EPYC 9334). This includes perception, memory updates and action
  helpers, and excludes simulation.

### Released Checkpoints
CaM has no learned checkpoint; the memory policies are plain Python files. The 16 frozen policies
(`solutions/robomme/demo_only_25/<Task>/policy.py`) have SHA-256 checksums and provenance in
`configs/solutions/robomme/demo_only_25.json` in the [Code](https://github.com/YinpeiDai/Code-as-Memory)
repository. The repository also has evaluation commands. The exact 25-demo subset is released as
[Yinpei/robomme-25-demos](https://huggingface.co/datasets/Yinpei/robomme-25-demos). Full coding-agent
development sessions are released as [Yinpei/CaM_develop_logs](https://huggingface.co/datasets/Yinpei/CaM_develop_logs).

Evaluate the released policies:
```bash
python scripts/evaluate_solutions.py solutions/robomme/demo_only_25 \
  --output runs/robomme_full --gpus 0 1 --workers-per-gpu 5
```

### Citations
- [Code-as-Memory: Self-improved Robot Memory via Coding Agents](https://github.com/YinpeiDai/Code-as-Memory).
- [RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](https://arxiv.org/abs/2603.04639).

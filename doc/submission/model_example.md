# Your Cool Model Name

### [Website]() | [Paper]() | [Code]()

## Introduction
My cool model leverages a novel representation of historical keyframes and maintains a memory cache integrated with a diffusion policy.

## Results 

> We evaluate in a multi-task setting, using a single model checkpoint for all tasks, and require at least three runs with different random seeds to reduce performance variance. 
> The benchmark seed is fixed internally.

> For the action space, only `joint_angle`, `ee_pose`, and `waypoint` are allowed. `multi_choice` is disallowed because it leverages ground-truth information and is mainly designed for Video-QA.

### Table

<table>
<tr>
  <th rowspan="2">Suite</th>
  <th rowspan="2">Task</th>
</tr>
<tr>
  <th>Seed 7</th><th>Seed 42</th><th>Seed 0</th><th><b>Avg</b></th>
</tr>
<tr>
  <td rowspan="4">Counting</td>
  <td>BinFill</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>PickXtimes</td><td></td><td></td><td></td><td></td></tr>
<tr><td>SwingXtimes</td><td></td><td></td><td></td><td></td></tr>
<tr><td>StopCube</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td rowspan="4">Permanence</td>
  <td>VideoUnmask</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>VideoUnmaskSwap</td><td></td><td></td><td></td><td></td></tr>
<tr><td>ButtonUnmask</td><td></td><td></td><td></td><td></td></tr>
<tr><td>ButtonUnmaskSwap</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td rowspan="4">Reference</td>
  <td>PickHighlight</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>VideoRepick</td><td></td><td></td><td></td><td></td></tr>
<tr><td>VideoPlaceButton</td><td></td><td></td><td></td><td></td></tr>
<tr><td>VideoPlaceOrder</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td rowspan="4">Imitation</td>
  <td>MoveCube</td><td></td><td></td><td></td><td></td>
</tr>
<tr><td>InsertPeg</td><td></td><td></td><td></td><td></td></tr>
<tr><td>PatternLock</td><td></td><td></td><td></td><td></td></tr>
<tr><td>RouteStick</td><td></td><td></td><td></td><td></td></tr>
<tr>
  <td colspan="2"><b>Overall</b></td><td></td><td></td><td></td><td></td>
</tr>
</table>


### Training Details

Share any hyperparameters you would like to include.

### Released Checkpoints

List any fine-tuned checkpoints you would like to release.

> We highly encourage authors to fully release their training/eval code and checkpoints to help the community accelerate memory-augmented manipulation.

### Citations
```
```

# 验证覆盖度表格

本表汇总每个 task 的 init / segmentation / task_goal 三类信息当前是否已经
进入 `scripts/dev3/inspect-stat.py` 的可视化产物。三类的定义如下：

1. **没有验证、需要读 init 信息**（或通过其它规则筛选）—— 这些字段当前
   既不在 segmentation XY 图里，也不在 task_goal CSV 里，要确认正确性
   还得回到 setup 元数据 / 规则筛选。
2. **已经通过 segmentation 验证** —— 在
   `runs/replay_videos/inspect-stat/xy/*_xy.png` 中能看到分布
   （`VideoRepick` 拆 `_xy_easy_medium.png` / `_xy_hard.png` 两张）。
3. **已经通过 task_goal 验证** —— 在
   `runs/replay_videos/inspect-stat/task-goal/*_distribution.png`
   与 `episode_task_metadata.csv` 中能看到。


## 表格


| Task | Init-only（缺验证） | Segmentation 已验证 | Task-Goal 已验证 |
| --- | --- | --- | --- |
| **Counting** | | | |
| BinFill | 选择的 pickup cube 位置、顺序、颜色 | cube / bin / button 颜色与位置 | 放入颜色 × 数量 |
| PickXtimes | 所 pick cube 的选择（颜色 / 位置） | cube / button / target 初始化颜色与位置 | pickup 次数、颜色 |
| SwingXtimes | 所 swing cube 的选择（颜色 / 位置） | cube / button / target 颜色与位置 | swing 次数、所选 cube 颜色 |
| StopCube | — | cube swing 朝向、target / button 位置 | stop 次数 |
| **Permanence** | | | |
| VideoUnmask | bin 选择顺序、内部 cube 颜色 | bin 位置、button 位置 | 第一 / 二次 pickup 颜色 |
| VideoUnmaskSwap | bin 选择顺序、内部 cube 颜色、swap 对象 | bin 位置、button 位置 | 第一 / 二次 pickup 颜色 |
| ButtonUnmask | bin 选择顺序、内部 cube 颜色 | bin 位置、button 位置 | 第一 / 二次 pickup 颜色 |
| ButtonUnmaskSwap | bin 选择顺序、内部 cube 颜色、swap 对象 | bin 位置、button 位置 | 第一 / 二次 pickup 颜色 |
| **Reference** | | | |
| PickHighlight | 高亮 cube、pick 顺序对应的 cube 颜色 / 位置 | cube颜色与位置 | pickup 颜色 |
| VideoRepick | easy/medium 的 swap 顺序<br>easy/medium/hard pick 对应的颜色 / 位置 | cube 颜色与位置 | pick 次数、所选 cube 颜色 |
| VideoPlaceButton | target 选择顺序与位置、swap 位置 | target / cube / button 颜色与位置 | pick 颜色、before/after 第几个 target |
| VideoPlaceOrder | target 选择顺序与位置、swap 位置 | target / cube / button 颜色与位置 | pick 颜色、第几个 target |
| **Imitation** | | | |
| MoveCube | 三种机制的选择 | target / stick / cube 初始化位置 | 固定 goal |
| InsertPeg | 选择的方向、插入方向、所选 peg 位置 | peg / box 初始化位置 | 固定 goal |
| PatternLock | 游走顺序 | — | 固定 goal |
| RouteStick | 游走顺序  | 初始化旋转 | 固定 goal |

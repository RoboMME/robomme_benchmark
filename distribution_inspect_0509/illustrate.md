
---

## MoveCube

<img src="MoveCube_xy.png" alt="MoveCube xy" width="70%">

**第 1 行（散点图）**

1. **第 1 行第 1 列** — 把 cube / peg / goal_site 全部对象画到同一张图，观察 reset 阶段整体的物体在桌面 xy 上的覆盖范围。
2. **第 1 行第 2 列** — 仅画 peg 的 head 与 tail 两个端点，看 peg 的位置 + 摆放朝向分布。
3. **第 1 行第 3 列** — 仅画 goal site也就是cube的目标位置（绿色三角），看目标点在 xy 平面上的分布。
4. **第 1 行第 4 列** — 仅画 cube 起始位置，看 cube 的初始 xy 分布。

**第 2 行（跨 4 列柱状图）**

5. **第 2 行（跨 4 列）** — 300 episode 中 reset 阶段被选中的执行方式 `peg_push` / `gripper_push` / `grasp_putdown` 的次数与占比，验证三种方式是否均衡覆盖。

---

## InsertPeg

<img src="InsertPeg_xy.png" alt="InsertPeg xy" width="70%">

**第 1 行（散点图）**

1. **第 1 行第 1 列** — peg + box_with_hole 全部点位的 xy 总览。
2. **第 1 行第 2 列** — peg 两端 head / tail 的 xy 分布，看 peg 摆放姿态范围。
3. **第 1 行第 3 列** — 带孔盒子（红色十字）xy 分布。

**第 2 行（任务选择分布 + 选中端散点）**

4. **第 2 行第 1 列** — 插入方向 `left` vs `right` 的次数与占比。
5. **第 2 行第 2 列** — 用 peg 的 `head` 还是 `tail` 端去插孔的次数与占比。
6. **第 2 行第 3 列** — 实际被选中作为插入端的 xy 散点。颜色区分 head（蓝）/ tail（橙），marker 区分 direction `left`（圆）/ `right`（方）；灰底为所有 peg head + tail 备选点；红 X 为 box 中心。综合看选端与选向在空间上是否有偏。

---

## PatternLock

<img src="PatternLock_xy.png" alt="PatternLock xy" width="70%">


**第 1 行（跨整行）**

1. **第 1 行（跨整行）** — 全部 episode 的 target（按钮）点位汇总，验证按钮排成规则的 6×6 网格。

**第 2 行**

2. **第 2 行第 1 列**-**起终点可视化（画完整path太乱了）** — 每个 episode 的 walk-path 起点（绿圆 marker `o`）和终点（红三角 marker `^`）落在哪个按钮上；灰底为所有候选按钮。看起点 / 终点是否均匀分布。
3. **第 2 行第 2 列** —**相对位置（subgoal）分布**— 每条 walk-path 中相邻两步的相对方向计数（`forward` / `backward` / `left` / `right` + 4 个对角 `forward-left` / `forward-right` / `backward-left` / `backward-right`），看 8 方向是否均衡。

---

## RouteStick

<img src="RouteStick_xy.png" alt="RouteStick xy" width="70%">

**第 1 行（visible-objects 散点 collage）**

1. **第 1 行第 1 列** —  obstacle + target 全部点位的 xy 总览。
2. **第 1 行第 2 列** — obstacle的 xy 分布。
3. **第 1 行第 3 列** — target 弧上的 xy 分布。

**第 2 行**

4. **第 2 行第 1 列** —**起终点可视化（画完整path太乱了）**：每个 episode walk-path 的起点（绿圆）与终点（红三角）落在哪些 target 上；灰底为所有候选按钮。看起点 / 终点的覆盖范围。
5. **第 2 行第 2 列** —**相对位置（subgoal）分布**— 4 种组合 `left+clockwise` / `left+counterclockwise` / `right+clockwise` / `right+counterclockwise` （stick 在哪一侧 × 摆动方向）的步数计数与占比，看 4 组合是否均衡覆盖。

---

## PickHighlight




<img src="PickHighlight_distribution.png" alt="PickHighlight distribution" width="70%">

**第 1 行**

1. **第 1 行第 1 列（1st Target Color）** — 第 1 个 highlight target 的颜色（red / blue / green）的 episode 数与占比；所有 300 个 episode 都至少有 1 个 target，因此没有 `none` 列。
2. **第 1 行第 2 列（2nd Target Color）** — 第 2 个 target 的颜色，包含 `none`（K=1 的 episode 在这里落到 `none`）。

**第 2 行**

3. **第 2 行第 1 列（3rd Target Color）** — 第 3 个 target 的颜色，包含 `none`（K=1 / K=2 的 episode 都落到 `none`，所以 `none` 占比明显更高）。


### xy

<img src="PickHighlight_xy.png" alt="PickHighlight xy" width="70%">


**第 1 行（visible-objects 散点 collage）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button）xy 总览。
2. **第 1 行第 2 列** — Cube 按颜色（red / blue / green）拆开的 xy 分布。
3. **第 1 行第 3 列** — Button xy 分布。

**第 2 行（任务选择分布 + 选中端散点）**

4. **第 2 行第 1 列（Panel A）** — `highlighted_cube_color` 计数：把 300 个 episode 中**所有** highlight target 的颜色都展开累计。
5. **第 2 行第 2 列（Panel B）** — `task_count K` 计数：每个 episode 的 highlight target 数 K=1 / 2 / 3 的 episode 数与占比，看 K 是否均衡。
6. **第 2 行第 3 列（Panel C）** — `task_target XY`：所有被选作 target 的 cube 的 xy 散点（橙色）。

---

## VideoPlaceButton


<img src="VideoPlaceButton_distribution.png" alt="VideoPlaceButton distribution" width="70%">


**第 1 行（Easy 难度，n=100）**

1. **第 1 行第 1 列（Target Color）** — Easy 档内选中 target cube 颜色（red / blue / green）的 episode 数与占比，验证 Easy 档三色均衡且无 `unknown`。
2. **第 1 行第 2 列（Before / After）** — Easy 档内 task goal 实际是第几个。

**第 2 行（Medium 难度，n=100）**

3. **第 2 行第 1 列** — Medium 档 Target Color，三色均衡情况。
4. **第 2 行第 2 列** — Medium 档内 task goal 实际是第几个。

**第 3 行（Hard 难度，n=100）**

5. **第 3 行第 1 列** — Hard 档 Target Color，三色均衡情况。
6. **第 3 行第 2 列** — Hard 档内 task goal 实际是第几个。



<img src="VideoPlaceButton_xy.png" alt="VideoPlaceButton xy" width="70%">


**第 1 行（visible-objects 散点 collage，4 列）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button + target）xy 总览。
2. **第 1 行第 2 列** — Cube 按颜色拆开的 xy 分布。
3. **第 1 行第 3 列** — Button xy 分布。
4. **第 1 行第 4 列** — Target xy 分布。

**第 2 行（任务选择柱状 + 选中端散点）**

5. **第 2 行第 1 列（Panel A）** — before after 整体计数。
6. **第 2 行第 2 列（Panel B）** — 选中的target实际是哪一个对象。
7. **第 2 行第 3 列（Panel C）** — 选中 target 的 xy 散点（橙色），看 target 在桌面 xy 上是否均匀覆盖。

**第 3 行（swap-pair 连线）**

8. **第 3 行（跨整行，Panel D）** — `swap_pair XY` 连线图：每个 swap 记录两端的 cube 位置用线段连起来，颜色表示该 episode 的难度或 swap 次数，整体看 swap 几何关系覆盖是否充分。

---

## VideoPlaceOrder


<img src="VideoPlaceOrder_distribution.png" alt="VideoPlaceOrder distribution" width="70%">

**第 1 行（Easy 难度，n=100）**

1. **第 1 行第 1 列（Target Color）** — Easy 档内选中 target cube 颜色（red / blue / green）的 episode 数与占比，验证 Easy 档三色均衡且无 `unknown`。
2. **第 1 行第 2 列（Target Order）** — Easy 档内目标 cube 在 reference 视频中是第几个被放上 button 的（`1st` / `2nd` / `3rd` / `4th` / `5th` / `6th`）的 episode 数与占比。

**第 2 行（Medium 难度，n=100）**

3. **第 2 行第 1 列** — Medium 档 Target Color，三色均衡情况。
4. **第 2 行第 2 列** — Medium 档 Target Order 槽位分布。

**第 3 行（Hard 难度，n=100）**

5. **第 3 行第 1 列** — Hard 档 Target Color，三色均衡情况。
6. **第 3 行第 2 列** — Hard 档 Target Order 槽位分布；Easy/Medium 通常 1st-4th 都有，5th/6th 视任务序列长度而定。



<img src="VideoPlaceOrder_xy.png" alt="VideoPlaceOrder xy" width="70%">

**第 1 行（visible-objects 散点 collage，4 列）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button + target）xy 总览。
2. **第 1 行第 2 列** — Cube 按颜色拆开的 xy 分布。
3. **第 1 行第 3 列** — Button xy 分布。
4. **第 1 行第 4 列** — Target xy 分布。

**第 2 行（任务选择柱状 + 选中端散点）**

5. **第 2 行第 1 列（Panel A）** — `order_position`（target 在 reference 序列中的次序 1st-Nth）的 episode 计数。
6. **第 2 行第 2 列（Panel B）** — `target_index` 计数（target_1 / target_2 ...）。
7. **第 2 行第 3 列（Panel C）** — 选中 target 的 xy 散点（橙色），看 target 是否在桌面 xy 上均匀覆盖。

**第 3 行（swap-pair 连线）**

8. **第 3 行（跨整行，Panel D）** — `swap_pair XY` 连线图：与 VideoPlaceButton 同一表征（每个 swap 两端 cube 位置连线），整体看 swap 几何关系覆盖。

---

## VideoRepick


<img src="VideoRepick_distribution.png" alt="VideoRepick distribution" width="70%">

**第 1 行（Easy 难度，n=100）**

1. **第 1 行第 1 列（Target Color）** — Easy 档内选中 target cube 颜色（red / blue / green）的 episode 数与占比，验证 Easy 档三色均衡。
2. **第 1 行第 2 列（Repeat Count）** — Easy 档内该 cube 在 reference 视频里被反复拾取的次数（`1` / `2` / `3` / `4` / `5` / `6`）的 episode 数与占比。

**第 2 行（Medium 难度，n=100）**

3. **第 2 行第 1 列** — Medium 档 Target Color，三色均衡情况。
4. **第 2 行第 2 列** — Medium 档 Repeat Count 分布。

**第 3 行（Hard 难度，n=100）**

5. **第 3 行第 1 列** — Hard 档 Target Color，三色均衡情况。
6. **第 3 行第 2 列** — Hard 档 Repeat Count 分布；当前数据集 1-3 次为主要分布、4-6 次为 0，体现任务设定的上限。


<img src="VideoRepick_xy.png" alt="VideoRepick xy" width="70%">

1. **第 1 行第 1 列** — All Visible Objects xy 总览（button + 其他 actor）。
2. **第 1 行第 2 列** — Button xy 分布。
3. **第 2 行（跨整行）** — `Pickup target XY`：每个 episode 的目标 cube xy 位置，按颜色（red / blue / green）拆开标记，看三色 pickup 目标在桌面 xy 上是否均匀覆盖。
4. **第 3 行（跨整行）** — `VideoRepick swaps XY`：所有 swap 记录两端 cube 位置的连线图（每个 episode 的 swap_pairs ≈ 1-2 条），整体看 swap 几何关系覆盖是否充分。

---

## BinFill


<img src="BinFill_distribution.png" alt="BinFill distribution" width="70%">

**第 1 行（Easy 难度，n=100）**

1. **第 1 行第 1 列（Put-In Color Count）** — Easy 档内本 episode 需要装入 bin 的颜色种数（`1` / `2` / `3` / `unknown`），Easy 全部为单色。
2. **第 1 行第 2 列（Put-In Cube Count）** — Easy 档内本 episode 需要装入的 cube 总数（`1` / `2` / `3` / `4` / `5` / `6` / `unknown`），主要落在 1-3 个。

**第 2 行（Medium 难度，n=100）**

3. **第 2 行第 1 列** — Medium 档颜色种数分布，落在 1-2 色。
4. **第 2 行第 2 列** — Medium 档 cube 总数分布，主要落在 2-4 个。

**第 3 行（Hard 难度，n=100）**

5. **第 3 行第 1 列** — Hard 档颜色种数分布，落在 2-3 色（无单色）。
6. **第 3 行第 2 列** — Hard 档 cube 总数分布，主要落在 3-5 个。


<img src="BinFill_xy.png" alt="BinFill xy" width="70%">

**第 1 行（visible-objects 散点 collage，4 列）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button + board_with_hole）xy 总览。
2. **第 1 行第 2 列** — Cube 按颜色（red / blue / green）拆开的 xy 分布。
3. **第 1 行第 3 列** — Button xy 分布。
4. **第 1 行第 4 列** — `board_with_hole`（BinFill 的目标板）xy 分布，红色三角 marker；BinFill 此 panel 是目标板而非 cube target。

---

## PickXtimes


<img src="PickXtimes_distribution.png" alt="PickXtimes distribution" width="70%">

**第 1 行（Easy 难度，n=100）**

1. **第 1 行第 1 列（Target Color）** — Easy 档内目标 cube 颜色（red / blue / green）分布，三色基本均衡。
2. **第 1 行第 2 列（Repeat Count）** — Easy 档内目标 cube 被 pick 的总次数（`1` / `2` / `3` / `4` / `5` / `6` / `unknown`），落在 1-3 次。

**第 2 行（Medium 难度，n=100）**

3. **第 2 行第 1 列** — Medium 档 Target Color，三色均衡。
4. **第 2 行第 2 列** — Medium 档 Repeat Count，落在 1-3 次。

**第 3 行（Hard 难度，n=100）**

5. **第 3 行第 1 列** — Hard 档 Target Color，三色均衡。
6. **第 3 行第 2 列** — Hard 档 Repeat Count，全部落在 4-5 次（各约 50%）。


<img src="PickXtimes_xy.png" alt="PickXtimes xy" width="70%">

**第 1 行（visible-objects 散点 collage，4 列）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button + target）xy 总览。
2. **第 1 行第 2 列** — Cube 按颜色（red / blue / green）拆开的 xy 分布。
3. **第 1 行第 3 列** — Button xy 分布。
4. **第 1 行第 4 列** — Target xy 分布（红色三角 marker）。

---

## SwingXtimes


<img src="SwingXtimes_distribution.png" alt="SwingXtimes distribution" width="70%">

**第 1 行（Easy 难度，n=100）**

1. **第 1 行第 1 列（Target Color）** — Easy 档内目标 cube 颜色（red / blue / green）分布，三色基本均衡。
2. **第 1 行第 2 列（Repeat Count）** — Easy 档内目标 cube 被 swing 的次数（`1` / `2` / `3` / `4` / `5` / `6` / `unknown`），落在 1-3 次。

**第 2 行（Medium 难度，n=100）**

3. **第 2 行第 1 列** — Medium 档 Target Color，三色均衡。
4. **第 2 行第 2 列** — Medium 档 Repeat Count，落在 1-2 次（无 3 次）。

**第 3 行（Hard 难度，n=100）**

5. **第 3 行第 1 列** — Hard 档 Target Color，三色均衡。
6. **第 3 行第 2 列** — Hard 档 Repeat Count，全部落在 3 次（100%）。


<img src="SwingXtimes_xy.png" alt="SwingXtimes xy" width="70%">

**第 1 行（visible-objects 散点 collage，4 列）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button + target）xy 总览。
2. **第 1 行第 2 列** — Cube 按颜色（red / blue / green）拆开的 xy 分布。
3. **第 1 行第 3 列** — Button xy 分布。
4. **第 1 行第 4 列** — Target xy 分布（红色三角 marker）。

---

## StopCube


<img src="StopCube_distribution.png" alt="StopCube distribution" width="70%">

**单图（不按难度分行，n=300）**

1. **Stop Visit** — policy 应当在目标 cube 第几次被 visit 时停下（`2` / `3` / `4` / `5`），4 个槽位大致均衡（约 26 / 27 / 24 / 23%）。整张图只有 1 行 1 列，**不**按难度档分行。


<img src="StopCube_xy.png" alt="StopCube xy" width="70%">

**第 1 行（visible-objects 散点 collage，4 列）**

1. **第 1 行第 1 列** — All Visible Objects（cube + button + target）xy 总览；cube 与 target 在桌面上各成一团，与 button 区域分离。
2. **第 1 行第 2 列** — Cube xy 分布，**全部为 `cube_unknown`（灰色）**：StopCube 的 cube 不带颜色 label，不拆 red / blue / green。
3. **第 1 行第 3 列** — Button xy 分布。
4. **第 1 行第 4 列** — Target xy 分布（红色三角 marker）。

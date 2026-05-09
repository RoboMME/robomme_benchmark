
---

## MoveCube

![MoveCube xy](MoveCube_xy.png)

**第 1 行（散点图）**

1. **第 1 行第 1 列** — 把 cube / peg / goal_site 全部对象画到同一张图，观察 reset 阶段整体的物体在桌面 xy 上的覆盖范围。
2. **第 1 行第 2 列** — 仅画 peg 的 head 与 tail 两个端点，看 peg 的位置 + 摆放朝向分布。
3. **第 1 行第 3 列** — 仅画 goal site也就是cube的目标位置（绿色三角），看目标点在 xy 平面上的分布。
4. **第 1 行第 4 列** — 仅画 cube 起始位置，看 cube 的初始 xy 分布。

**第 2 行（跨 4 列柱状图）**

5. **第 2 行（跨 4 列）** — 300 episode 中 reset 阶段被选中的执行方式 `peg_push` / `gripper_push` / `grasp_putdown` 的次数与占比，验证三种方式是否均衡覆盖。

---

## InsertPeg

![InsertPeg xy](InsertPeg_xy.png)

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

![PatternLock xy](PatternLock_xy.png)


**第 1 行（跨整行）**

1. **第 1 行（跨整行）** — 全部 episode 的 target（按钮）点位汇总，验证按钮排成规则的 6×6 网格。

**第 2 行**

2. **第 2 行第 1 列**-**起终点可视化（画完整path太乱了）** — 每个 episode 的 walk-path 起点（绿圆 marker `o`）和终点（红三角 marker `^`）落在哪个按钮上；灰底为所有候选按钮。看起点 / 终点是否均匀分布。
3. **第 2 行第 2 列** —**相对位置（subgoal）分布**— 每条 walk-path 中相邻两步的相对方向计数（`forward` / `backward` / `left` / `right` + 4 个对角 `forward-left` / `forward-right` / `backward-left` / `backward-right`），看 8 方向是否均衡。

---

## RouteStick

![RouteStick xy](RouteStick_xy.png)

布局：2 行 × 3 列。

**第 1 行（visible-objects 散点 collage）**

1. **第 1 行第 1 列** —  obstacle + target 全部点位的 xy 总览。
2. **第 1 行第 2 列** — obstacle的 xy 分布。
3. **第 1 行第 3 列** — target 弧上的 xy 分布。

**第 2 行**

4. **第 2 行第 1 列** —**起终点可视化（画完整path太乱了）**：每个 episode walk-path 的起点（绿圆）与终点（红三角）落在哪些 target 上；灰底为所有候选按钮。看起点 / 终点的覆盖范围。
5. **第 2 行第 2 列** —**相对位置（subgoal）分布**— 4 种组合 `left+clockwise` / `left+counterclockwise` / `right+clockwise` / `right+counterclockwise` （stick 在哪一侧 × 摆动方向）的步数计数与占比，看 4 组合是否均衡覆盖。

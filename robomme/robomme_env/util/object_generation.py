import numpy as np
import torch
import sapien
from typing import Optional, Tuple, Sequence, Union
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
import mani_skill.envs.utils.randomization as randomization  # 仅用于其他地方需要的话
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from transforms3d.euler import euler2quat
from mani_skill.utils import sapien_utils
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from typing import Optional, Union

def _color_to_rgba(color: Union[str, Sequence[float]]) -> Tuple[float, float, float, float]:
    """Convert a hex string or RGB/RGBA tuple to an RGBA tuple accepted by SAPIEN."""
    if isinstance(color, str):
        return sapien_utils.hex2rgba(color)
    if len(color) == 3:
        return (float(color[0]), float(color[1]), float(color[2]), 1.0)
    if len(color) == 4:
        return tuple(float(c) for c in color)
    raise ValueError("color must be a hex string or a sequence of 3/4 floats")


def build_peg(
    env_or_scene,
    length: float,
    radius: float,
    *,
    initial_pose: Optional["sapien.Pose"] = None,
    head_color: str = "#EC7357",
    tail_color: str = "#F5F5F5",
    density: float = 1200.0,
    name: str = "peg",
) -> Tuple["sapien.Articulation", "sapien.Link", "sapien.Link"]:
    """Construct a peg articulation with head and tail links tied by a fixed joint.

    Args:
        env_or_scene: Environment or scene providing `create_articulation_builder`.
        length: Total length of the peg (meters).
        radius: Half-width of the rectangular cross section (meters).
        initial_pose: Optional pose for the articulation root; defaults to placing
            the head centered at positive x.
        head_color: Hex color for the head visual.
        tail_color: Hex color for the tail visual.
        density: Collision density (kg/m^3) shared by both links.
        name: Name assigned to the articulation.

    Returns:
        The articulation along with the head and tail links.
    """

    scene = getattr(env_or_scene, "scene", env_or_scene)
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[length / 2, 0.0, radius], q=[1, 0, 0, 0])

    builder = scene.create_articulation_builder()
    builder.initial_pose = initial_pose

    head_builder = builder.create_link_builder()
    head_builder.set_name("peg_head")
    head_builder.add_box_collision(
        half_size=[length / 2 * 0.9, radius, radius], density=density
    )
    head_material = sapien.render.RenderMaterial(
        base_color=_color_to_rgba(head_color),
        roughness=0.5,
        specular=0.5,
    )
    head_builder.add_box_visual(
        half_size=[length / 2, radius, radius],
        material=head_material,
    )

    tail_builder = builder.create_link_builder(head_builder)
    tail_builder.set_name("peg_tail")
    tail_builder.set_joint_name("peg_fixed_joint")
    tail_builder.set_joint_properties(
        type="fixed",
        limits=[[0.0, 0.0]],
        pose_in_parent=sapien.Pose(p=[-length, 0.0, 0.0], q=[1, 0, 0, 0]),
        pose_in_child=sapien.Pose(p=[0.0, 0.0, 0.0], q=[1, 0, 0, 0]),
        friction=0.0,
        damping=0.0,
    )
    tail_builder.add_box_collision(
        half_size=[length / 2 * 0.9, radius, radius], density=density
    )
    tail_material = sapien.render.RenderMaterial(
        base_color=_color_to_rgba(tail_color),
        roughness=0.5,
        specular=0.5,
    )
    tail_builder.add_box_visual(
        half_size=[length / 2, radius, radius],
        material=tail_material,
    )

    peg = builder.build(name=name, fix_root_link=False)
    link_map = {link.get_name(): link for link in peg.get_links()}
    peg_head = link_map["peg_head"]
    peg_tail = link_map["peg_tail"]
    return peg, peg_head, peg_tail


def build_box_with_hole(self, inner_radius, outer_radius, depth, center=(0, 0)):
    builder = self.scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    box=builder.build_kinematic(f"box_with_hole")
    return box
def _safe_unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def _trimesh_box_to_obb2d(obb_box, extra_pad=0.0):
    """
    将 trimesh.primitives.Box（世界系）转为 2D OBB 表示：中心 c(2,), 轴 A(2x2), 半长 h(2,)
    extra_pad: 在 XY 平面上各向外膨胀的边距（米）
    """
    # 兼容 obb 可能被包装在 .primitive 里
    b = getattr(obb_box, "primitive", obb_box)
    T = np.asarray(b.transform, dtype=np.float64)  # 4x4
    ex = np.asarray(b.extents, dtype=np.float64)   # 3

    R = T[:3, :3]
    t = T[:3, 3]

    c = t[:2].copy()

    # 取 X、Y 轴在平面上的投影作为 2D OBB 的两个轴
    u = _safe_unit(R[:2, 0])  # x 轴投影
    v = _safe_unit(R[:2, 1])  # y 轴投影
    A = np.stack([u, v], axis=1)  # 2x2，每列是一个轴

    h = 0.5 * ex[:2].astype(np.float64)
    if extra_pad > 0:
        h = h + float(extra_pad)
    return c, A, h

def _obb2d_intersect(c1, A1, h1, c2, A2, h2):
    """
    2D OBB SAT 检测。c*: (2,), A*: (2x2) 列为轴，h*: (2,)
    返回 True 表示相交（含接触），False 表示分离
    """
    d = c2 - c1
    axes = [A1[:, 0], A1[:, 1], A2[:, 0], A2[:, 1]]

    for a in axes:
        a = _safe_unit(a)
        # 投影半径
        r1 = abs(np.dot(A1[:, 0], a)) * h1[0] + abs(np.dot(A1[:, 1], a)) * h1[1]
        r2 = abs(np.dot(A2[:, 0], a)) * h2[0] + abs(np.dot(A2[:, 1], a)) * h2[1]
        dist = abs(np.dot(d, a))
        if dist > (r1 + r2):
            return False  # 存在分离轴 -> 不相交
    return True  # 所有轴都重叠 -> 相交/接触

def _yaw_to_quat_tensor(yaw: float, device):
    """
    用 z 轴欧拉角得到与 ManiSkill/你的转换工具一致的四元数（形状 [1,4]，float32，设备对齐）
    """
    # euler_angles_to_matrix 接受 [roll, pitch, yaw]（弧度），返回 Nx3x3
    angles = torch.tensor([[0.0, 0.0, float(yaw)]], dtype=torch.float32, device=device)
    R = euler_angles_to_matrix(angles,convention="XYZ")            # (1, 3, 3)
    q = matrix_to_quaternion(R)                   # (1, 4) 约定同 ManiSkill
    return q

def _build_new_cube_obb2d(x, y, half_size_xy, yaw, pad_xy=0.0):
    """
    构造“准备落位的方块”的 2D OBB：中心/轴/半长
    half_size_xy: float，正方体在 XY 上的半边长
    yaw: 绕 z 轴旋转（弧度）
    pad_xy: 额外在 XY 上膨胀的半长（用于最小间隙）
    """
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]], dtype=np.float64)  # 列为轴
    h = np.array([half_size_xy + pad_xy, half_size_xy + pad_xy], dtype=np.float64)
    return c, A, h

def spawn_random_cube(
        self,
        region_center=[0, 0],
        region_half_size=0.1,
        half_size=0.01,
        color=(1, 0, 0, 1),
        name_prefix="cube_extra",
        min_gap=0.005,
        max_trials=256,
        avoid=None,
        random_yaw=True,
        include_existing=True,
        include_goal=True,
        generator=None
    ):
    """
    在矩形区域内用拒绝采样投放一个 cube(落在桌面上),并返回该 cube actor。
    - 使用 OBB 精确碰撞(2D 投影 + SAT),满足 min_gap 才落位。
    - avoid: 物体输入一个 list。可为 [actor, ...] 或 [(actor, pad), ...](pad 单位米)。
    - generator: 必须传入 torch.Generator 进行随机化。
    """
    # 缓存
    if not hasattr(self, "_spawned_cubes"):
        self._spawned_cubes = []
        self._spawned_count = 0

    center = np.array(region_center if region_center is not None else self.cube_spawn_center, dtype=np.float64)

    # 支持两种输入：标量或二维数组
    if region_half_size is None:
        region_half_size = self.cube_spawn_half_size

    # 兼容两种输入格式
    if isinstance(region_half_size, (list, tuple, np.ndarray)):
        # 二维数组输入：独立控制xy
        area_half = np.array(region_half_size, dtype=np.float64)
        if area_half.shape == ():  # 处理0维数组
            area_half = np.array([float(area_half), float(area_half)], dtype=np.float64)
        elif len(area_half) == 1:
            area_half = np.array([float(area_half[0]), float(area_half[0])], dtype=np.float64)
        elif len(area_half) != 2:
            raise ValueError("region_half_size数组必须包含1或2个元素 [x_half, y_half]")
    else:
        # 标量输入：xy保持一致
        area_half = np.array([float(region_half_size), float(region_half_size)], dtype=np.float64)

    hs_new = float(half_size if half_size is not None else self.cube_half_size)

    # 让方块完整落在区域内（独立控制xy）
    x_low = center[0] - area_half[0] + hs_new
    x_high = center[0] + area_half[0] - hs_new
    y_low = center[1] - area_half[1] + hs_new
    y_high = center[1] + area_half[1] - hs_new
    if x_low > x_high or y_low > y_high:
        raise ValueError("spawn_random_cube: 采样区域过小,无法放下该尺寸的 cube。")

    # === 组装障碍物 OBB(2D)列表 ===
    obb2d_list = []  # [(c, A, h), ...]

    def _push_actor_as_obb2d(actor, pad=0.0):
        try:
            # 特殊处理board_with_hole
            if hasattr(actor, '_board_side') and hasattr(actor, '_hole_side'):
                # 这是我们创建的带洞板子,手动添加其OBB
                board_side = actor._board_side
                hole_side = actor._hole_side

                # 获取板子的世界位置
                actor_pos = actor.pose.p
                if isinstance(actor_pos, torch.Tensor):
                    actor_pos = actor_pos[0].detach().cpu().numpy()

                board_center = np.array(actor_pos[:2], dtype=np.float64)
                board_half = board_side / 2
                hole_half = hole_side / 2

                # 添加四个矩形条的OBB
                # 上条
                if board_half > hole_half:  # 确保有足够空间
                    top_height = board_half - hole_half
                    top_center = board_center + np.array([0, hole_half + top_height / 2])
                    A_top = np.eye(2)  # 无旋转
                    h_top = np.array([board_half + pad, top_height / 2 + pad])
                    obb2d_list.append((top_center, A_top, h_top))

                    # 下条
                    bottom_center = board_center + np.array([0, -(hole_half + top_height / 2)])
                    obb2d_list.append((bottom_center, A_top, h_top))

                    # 左条
                    left_width = board_half - hole_half
                    left_center = board_center + np.array([-(hole_half + left_width / 2), 0])
                    h_left = np.array([left_width / 2 + pad, hole_half + pad])
                    obb2d_list.append((left_center, A_top, h_left))

                    # 右条
                    right_center = board_center + np.array([hole_half + left_width / 2, 0])
                    obb2d_list.append((right_center, A_top, h_left))
                return

            obb = get_actor_obb(actor, to_world_frame=True, vis=False)
            obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
            obb2d_list.append(obb2d)
        except Exception:
            # 某些对象(如 site/marker)没有物理 mesh,则忽略或在下方用圆近似
            pass

    if include_existing:
        # 主 cube
        if hasattr(self, "cube") and self.cube is not None:
            _push_actor_as_obb2d(self.cube, pad=0.0)

        # 历史生成的 cubes
        for ac in self._spawned_cubes:
            _push_actor_as_obb2d(ac, pad=0.0)

    # 用户额外指定的避让
    if avoid:
        for it in avoid:
            if isinstance(it, tuple):
                # Check if it's a pre-made OBB tuple (c, A, h) or (actor, pad)
                if len(it) == 3 and isinstance(it[0], np.ndarray) and isinstance(it[1], np.ndarray):
                    # Pre-made OBB: (center, axes, half_sizes)
                    obb2d_list.append(it)
                else:
                    # Actor with padding
                    act_i, pad_i = it
                    _push_actor_as_obb2d(act_i, pad=float(pad_i))
            else:
                _push_actor_as_obb2d(it, pad=0.0)

    # 目标点(如果没有 mesh),用"圆 + 外接圆"的保守近似补充一下(可选)
    circle_list = []  # [(xy(2,), R)], 用于没有 mesh 的对象
    def _actor_xy(actor):
        p = actor.pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        return np.array(p[:2], dtype=np.float64)

    if include_goal and hasattr(self, "goal_site") and self.goal_site is not None:
        try:
            # 如果 goal_site 有 mesh,会在 _push_actor_as_obb2d 里覆盖,这里仅作兜底
            _push_actor_as_obb2d(self.goal_site, pad=0.0)
        except Exception:
            # 退化为圆近似:goal 半径 + 新 cube 外接圆半径
            R_goal = float(getattr(self, "goal_thresh", 0.03))
            R_new_ext = np.sqrt(2.0) * hs_new
            circle_list.append((_actor_xy(self.goal_site), R_goal + R_new_ext + min_gap))

    # === 采样迭代 ===
    if generator is None:
        raise ValueError("spawn_random_cube: 必须显式传入generator参数进行随机化")

    device = self.device

    for trial in range(int(max_trials)):
        # 使用简单均匀采样确保良好的空间覆盖
        # 复杂的采样策略往往降低覆盖率

        u1 = torch.rand(1, generator=generator).item()
        u2 = torch.rand(1, generator=generator).item()

        # 直接映射到采样区域 - 均匀分布提供最佳空间覆盖
        x = float(x_low + u1 * (x_high - x_low))
        y = float(y_low + u2 * (y_high - y_low))

        if random_yaw:
            # 使用更随机的yaw生成，从完整的[0, 2π]范围采样
            yaw_sample = torch.rand(1, generator=generator).item()
            yaw = float(yaw_sample * 2 * np.pi)
        else:
            yaw = 0.0

        # 新 cube 的 2D OBB(把 min_gap 体现在 "新物体半长的膨胀" 上,避免对双方都加导致双倍)
        c_new, A_new, h_new = _build_new_cube_obb2d(x, y, hs_new, yaw, pad_xy=float(min_gap))

        # 与 OBB 障碍逐一检测
        hit = False
        for (c_obs, A_obs, h_obs) in obb2d_list:
            if _obb2d_intersect(c_obs, A_obs, h_obs, c_new, A_new, h_new):
                hit = True
                break
        if hit:
            continue

        # 与圆形兜底障碍检测(如果存在)
        for (xy_c, R_c) in circle_list:
            if np.linalg.norm(np.asarray([x, y], dtype=np.float64) - xy_c) < R_c:
                hit = True
                break
        if hit:
            continue

        # 通过检测,创建 cube(姿态与碰撞检测使用同一 yaw,保证一致)
        q = _yaw_to_quat_tensor(yaw, device=device)

        cube = actors.build_cube(
            self.scene,
            half_size=hs_new,
            color=color,
            name=name_prefix,  # 直接使用name_prefix,不添加计数器
            initial_pose=Pose.create_from_pq(
                torch.tensor([[x, y, hs_new]], device=device, dtype=torch.float32),
                q,
            ),
        )
        cube._cube_half_size = hs_new
        self._spawned_cubes.append(cube)
        self._spawned_count += 1
        return cube

    raise RuntimeError("spawn_random_cube: 区域拥挤或约束过紧,未找到可行位置。可尝试:放大区域/减小方块/减小 min_gap。")

def _build_new_target_obb2d(x, y, half_size_xy, yaw, pad_xy=0.0):
    """
    构造"准备落位的target"的 2D OBB：中心/轴/半长
    half_size_xy: float，target在 XY 上的半边长
    yaw: 绕 z 轴旋转（弧度）
    pad_xy: 额外在 XY 上膨胀的半长（用于最小间隙）
    """
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]], dtype=np.float64)  # 列为轴
    h = np.array([half_size_xy + pad_xy, half_size_xy + pad_xy], dtype=np.float64)
    return c, A, h

def spawn_random_target(
        self,
        region_center=[0, 0],
        region_half_size=0.1,
        radius=0.01,
        thickness=0.005,
        name_prefix="target_extra",
        min_gap=0.005,
        max_trials=256,
        avoid=None,          # 支持 [actor, ...] 或 [(actor, pad), ...]
        include_existing=True,   # 是否自动避让已有主 target 与已生成的额外 targets
        include_goal=True,       # 是否把 goal_site 也作为障碍（用圆近似，保守）
        generator=None,
        randomize=True,      # 控制是否随机生成位置
        target_style="purple",  # 选择创建哪种配色的 target
    ):
    """
    在矩形区域内用拒绝采样投放一个 target（落在桌面上），并返回该 target actor。
    - 使用 OBB 精确碰撞（2D 投影 + SAT），满足 min_gap 才落位。
    - avoid: 物体输入一个 list。可为 [actor, ...] 或 [(actor, pad), ...]（pad 单位米）。
    - generator: 必须传入 torch.Generator 进行随机化（当randomize=True时）。
    - randomize: 控制是否随机生成位置。若为False，则直接在region_center位置生成。
    """
    # 缓存
    random_yaw=False
    if not hasattr(self, "_spawned_targets"):
        self._spawned_targets = []
        self._spawned_target_count = 0

    center = np.array(region_center if region_center is not None else getattr(self, 'target_spawn_center', [0, 0]), dtype=np.float64)
    area_half = float(region_half_size if region_half_size is not None else getattr(self, 'target_spawn_half_size', 0.1))
    target_radius = float(radius if radius is not None else getattr(self, 'target_radius', 0.01))
    target_thickness = float(thickness if thickness is not None else getattr(self, 'target_thickness', 0.005))

    # 让target完整落在区域内
    x_low = center[0] - area_half + target_radius
    x_high = center[0] + area_half - target_radius
    y_low = center[1] - area_half + target_radius
    y_high = center[1] + area_half - target_radius
    if x_low > x_high or y_low > y_high:
        raise ValueError("spawn_random_target: 采样区域过小，无法放下该尺寸的 target。")

    # === 组装障碍物 OBB（2D）列表 ===
    obb2d_list = []  # [(c, A, h), ...]

    def _push_actor_as_obb2d(actor, pad=0.0):
        try:
            # 特殊处理board_with_hole
            if hasattr(actor, '_board_side') and hasattr(actor, '_hole_side'):
                # 这是我们创建的带洞板子，手动添加其OBB
                board_side = actor._board_side
                hole_side = actor._hole_side

                # 获取板子的世界位置
                actor_pos = actor.pose.p
                if isinstance(actor_pos, torch.Tensor):
                    actor_pos = actor_pos[0].detach().cpu().numpy()

                board_center = np.array(actor_pos[:2], dtype=np.float64)
                board_half = board_side / 2
                hole_half = hole_side / 2

                # 添加四个矩形条的OBB
                # 上条
                if board_half > hole_half:  # 确保有足够空间
                    top_height = board_half - hole_half
                    top_center = board_center + np.array([0, hole_half + top_height / 2])
                    A_top = np.eye(2)  # 无旋转
                    h_top = np.array([board_half + pad, top_height / 2 + pad])
                    obb2d_list.append((top_center, A_top, h_top))

                    # 下条
                    bottom_center = board_center + np.array([0, -(hole_half + top_height / 2)])
                    obb2d_list.append((bottom_center, A_top, h_top))

                    # 左条
                    left_width = board_half - hole_half
                    left_center = board_center + np.array([-(hole_half + left_width / 2), 0])
                    h_left = np.array([left_width / 2 + pad, hole_half + pad])
                    obb2d_list.append((left_center, A_top, h_left))

                    # 右条
                    right_center = board_center + np.array([hole_half + left_width / 2, 0])
                    obb2d_list.append((right_center, A_top, h_left))
                return

            obb = get_actor_obb(actor, to_world_frame=True, vis=False)
            obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
            obb2d_list.append(obb2d)
        except Exception:
            # 某些对象（如 site/marker）没有物理 mesh，则忽略或在下方用圆近似
            pass

    if include_existing:
        # 主 cube
        if hasattr(self, "cube") and self.cube is not None:
            _push_actor_as_obb2d(self.cube, pad=0.0)

        # 主 target
        if hasattr(self, "target") and self.target is not None:
            _push_actor_as_obb2d(self.target, pad=0.0)

        # 历史生成的 cubes
        if hasattr(self, "_spawned_cubes"):
            for ac in self._spawned_cubes:
                _push_actor_as_obb2d(ac, pad=0.0)

    # 目标点（如果没有 mesh），用"圆 + 外接圆"的保守近似补充一下（可选）
    circle_list = []  # [(xy(2,), R)], 用于没有 mesh 的对象
    def _actor_xy(actor):
        p = actor.pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        return np.array(p[:2], dtype=np.float64)

    # 历史生成的 targets - 作为圆形障碍处理
    if include_existing:
        for ac in self._spawned_targets:
            target_r = getattr(ac, "_target_radius", target_radius)
            circle_list.append((_actor_xy(ac), target_r))

    # 用户额外指定的避让
    if avoid:
        for it in avoid:
            if isinstance(it, tuple):
                # Check if it's a pre-made OBB tuple (c, A, h) or (actor, pad)
                if len(it) == 3 and isinstance(it[0], np.ndarray) and isinstance(it[1], np.ndarray):
                    # Pre-made OBB: (center, axes, half_sizes)
                    obb2d_list.append(it)
                else:
                    # Actor with padding
                    act_i, pad_i = it
                    # 检查是否是target（圆形）
                    if hasattr(act_i, "_target_radius"):
                        target_r = getattr(act_i, "_target_radius", target_radius)
                        circle_list.append((_actor_xy(act_i), target_r + float(pad_i)))
                    else:
                        _push_actor_as_obb2d(act_i, pad=float(pad_i))
            else:
                # 检查是否是target（圆形）
                if hasattr(it, "_target_radius"):
                    target_r = getattr(it, "_target_radius", target_radius)
                    circle_list.append((_actor_xy(it), target_r))
                else:
                    _push_actor_as_obb2d(it, pad=0.0)

    if include_goal and hasattr(self, "goal_site") and self.goal_site is not None:
        try:
            # 如果 goal_site 有 mesh，会在 _push_actor_as_obb2d 里覆盖，这里仅作兜底
            _push_actor_as_obb2d(self.goal_site, pad=0.0)
        except Exception:
            # 退化为圆近似：goal 半径 + 新 target 外接圆半径
            R_goal = float(getattr(self, "goal_thresh", 0.03))
            R_new_ext = target_radius
            circle_list.append((_actor_xy(self.goal_site), R_goal + R_new_ext + min_gap))

    # === 采样迭代 ===
    if generator is None:
        raise ValueError("spawn_random_target: 必须显式传入generator参数进行随机化")

    device = self.device

    target_builders = {
        "purple": build_purple_white_target,
        "gray": build_gray_white_target,
        "green": build_green_white_target,
        "red": build_red_white_target,
    }
    if isinstance(target_style, str):
        builder_key = target_style.lower()
        if builder_key not in target_builders:
            raise ValueError(f"spawn_random_target: 未知 target_style '{target_style}'. 支持: {list(target_builders.keys())}")
        target_builder = target_builders[builder_key]
    elif callable(target_style):
        target_builder = target_style
    else:
        raise ValueError("spawn_random_target: target_style 需为字符串或可调用的 builder 函数")

    for _ in range(int(max_trials)):
        x = float(torch.rand(1, generator=generator).item() * (x_high - x_low) + x_low)
        y = float(torch.rand(1, generator=generator).item() * (y_high - y_low) + y_low)

        if random_yaw:
            yaw = float(torch.rand(1, generator=generator).item() * 2 * np.pi - np.pi)
        else:
            yaw = 0.0

        # 新 target 的圆形碰撞检测（target是圆形的，用圆形检测更准确）
        target_pos = np.array([x, y], dtype=np.float64)
        target_collision_radius = target_radius + min_gap

        # 与 OBB 障碍检测（将圆形target与方形障碍物检测）
        hit = False
        for (c_obs, A_obs, h_obs) in obb2d_list:
            # 计算圆心到OBB的最近距离
            # 将圆心转换到OBB的局部坐标系
            local_pos = A_obs.T @ (target_pos - c_obs)
            # 计算圆心到OBB的最近点
            closest_point = np.clip(local_pos, -h_obs, h_obs)
            # 转换回世界坐标系
            closest_world = c_obs + A_obs @ closest_point
            # 计算距离
            dist = np.linalg.norm(target_pos - closest_world)
            if dist < target_collision_radius:
                hit = True
                break
        if hit:
            continue

        # 与圆形障碍检测（圆与圆的检测）
        for (xy_c, R_c) in circle_list:
            if np.linalg.norm(target_pos - xy_c) < (target_collision_radius + R_c):
                hit = True
                break
        if hit:
            continue

        # 通过检测，创建 target（姿态与碰撞检测使用同一 yaw，保证一致）
        rotate = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])  # z轴旋转的四元数
        angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))  # (3,)
        rotate = matrix_to_quaternion(
            euler_angles_to_matrix(angles, convention="XYZ")
        )
        target = target_builder(
            scene=self.scene,
            radius=target_radius,
            thickness=target_thickness,
            name=name_prefix,  # 直接使用name_prefix，不添加计数器
            body_type="kinematic",  # 仅可视化
            add_collision=False,  # 关闭碰撞
            initial_pose=sapien.Pose(p=[x, y, target_thickness], q=rotate),
        )
        target._target_radius = target_radius
        self._spawned_targets.append(target)
        self._spawned_target_count += 1
        return target

    raise RuntimeError("spawn_random_target: 区域拥挤或约束过紧，未找到可行位置。可尝试：放大区域/减小target/减小 min_gap。")


def create_button_obb(center_xy=(-0.3, 0), half_size=0.05):
    """
    Create a manual OBB for button collision avoidance.

    Args:
        center_xy: Button center position (x, y)
        half_size: Safe zone half-size around button (default 0.05m)

    Returns:
        Tuple (center, axes, half_sizes) for use in avoid lists
    """
    return (
        np.array(center_xy, dtype=np.float64),  # center
        np.eye(2, dtype=np.float64),  # axes (identity for axis-aligned)
        np.array([half_size, half_size], dtype=np.float64)  # half-sizes
    )

def build_button(
            self,
            center_xy=(0.15, 0.10),  # 按钮在桌面上的 (x,y)
            base_half=[0.025, 0.025, 0.005],  # 基座半尺寸 [x,y,z]
            cap_radius=0.015,  # 按钮帽半径
            cap_half_len=0.006,  # 按钮帽半长
            travel=None,  # 可按行程
            stiffness=800.0,
            damping=40.0,
            scale: float = None,  # ⭐ 新增：放缩倍数
            generator=None,
            name: str = "button",  # ⭐ 新增：按钮名称
            randomize: bool = True,  # ⭐ 新增：是否随机化位置
            randomize_range=(0.1, 0.4),  # ⭐ 新增：随机化范围，(range_x, range_y)
    ):
        # ------- 放缩倍数与行程 -------
        if scale is None:
            # 若未传，使用环境上的默认缩放
            scale = getattr(self, "button_scale", 1.0)
        scale = float(scale)

        # 行程优先级：入参 > 环境基准
        if travel is None:
            # 用基准行程按比例放缩
            base_travel = getattr(self, "_button_travel_base", 0.1)
            travel = base_travel * scale
        else:
            # 如果显式传了 travel，也跟随 scale 放缩（想保持绝对值，可把下面这行改成 pass）
            travel = float(travel) * scale

        # 尺寸放缩
        base_half = [bh * scale for bh in base_half]
        cap_radius = float(cap_radius) * scale
        cap_half_len = float(cap_half_len) * scale

        # 记录当前按钮行程供其它函数使用
        self.button_travel = float(travel)

        # ------- 位置随机化 -------
        cx, cy = float(center_xy[0]), float(center_xy[1])

        if randomize:
            if not isinstance(randomize_range, (tuple, list, np.ndarray)):
                raise TypeError("randomize_range must be a sequence of length 2.")
            if len(randomize_range) != 2:
                raise ValueError("randomize_range must contain exactly two elements.")
            range_x, range_y = float(randomize_range[0]), float(randomize_range[1])
            offset = torch.rand(2, generator=generator) - 0.5
            cx += float(offset[0]) * range_x
            cy += float(offset[1]) * range_y
        center_xy = (cx, cy)

        scene = self.scene
        builder = scene.create_articulation_builder()

        # 初始位姿：基座中心抬到 z=base_half[2]
        builder.initial_pose = sapien.Pose(p=[cx, cy, base_half[2]])

        # Root：基座
        base = builder.create_link_builder()
        base.set_name("button_base")
        base.add_box_collision(half_size=base_half, density=200000)
        base.add_box_visual(half_size=base_half)

        # Child：按钮帽（竖直滑动）
        cap = builder.create_link_builder(base)
        cap.set_name("button_cap")
        cap.set_joint_name("button_joint")

        R_up = euler2quat(0, -np.pi / 2, 0)  # 将关节 x 轴对齐世界 z

        cap.set_joint_properties(
            type="prismatic",
            limits=[[-travel, 0.0]],  # 负向为按下
            pose_in_parent=sapien.Pose(p=[0, 0, base_half[2]], q=R_up),
            pose_in_child=sapien.Pose(p=[0, 0, 0.0], q=R_up),
            friction=0.0,
            damping=0.0,
        )

        cap.add_cylinder_collision(
            half_length=cap_half_len, radius=cap_radius,
            pose=sapien.Pose(p=[0, 0, cap_half_len], q=R_up), density=1500
        )
        material = sapien.render.RenderMaterial()
        material.set_base_color([0.5, 0.5, 0.5, 1.0])
        cap.add_cylinder_visual(
            half_length=cap_half_len, radius=cap_radius,
            pose=sapien.Pose(p=[0, 0, cap_half_len], q=R_up), material=material
        )



        button = builder.build(name=name, fix_root_link=True)

        j = {j.name: j for j in button.get_joints()}["button_joint"]
        j.set_drive_properties(stiffness=stiffness, damping=damping)
        j.set_drive_target(0.0)

        self.button = button
        self.button_joint = j

        cap_link = next(
            link for link in button.get_links()
            if link.get_name() == "button_cap"
        )
        cap_link = next(link for link in button.get_links()
                if link.get_name() == "button_cap")
        if not hasattr(self, "cap_links"):
            self.cap_links = {}
        self.cap_links[name] = [cap_link]   # name 就是 “button_left”“button_right” 等
        self.cap_link = self.cap_links[name]  # 兼容旧逻辑

        # Provide an OBB for downstream placement logic using the scaled button footprint
        button_obb = create_button_obb(
            center_xy=center_xy,
            half_size=max(base_half[0], base_half[1]) * 1.5,
        )
        return button_obb
def build_bin(
        self,
        *,
        inner_side: float = 0.04,  # 内部开口边长（整长，米），原来 2*inner_side_half_len = 0.04
        wall_thickness: float = 0.005,  # 围墙厚度（整厚，米）
        wall_height: float = 0.05,  # 围墙高度（整高，米）
        floor_thickness: float = 0.004,  # 底板厚度（整厚，米）
        callsign=None,
        position=None,  # 添加位置参数
        z_rotation_deg=0.0  # 添加z轴旋转角度参数（度）
):
    """
    由 1 底板 + 4 墙条拼出一个"敞口方盒"。
    所有尺寸用"整尺寸（米）"，内部自动换算为半尺寸（half-size）。
    参考cube的生成方式，让bin底部贴在桌面(z=0)上。
    """
    inner_side = self.cube_half_size * 2.5
    wall_height = self.cube_half_size * 2.5

    # ---- 将整尺寸转为半尺寸（与 add_box_* 接口一致）----
    inner_half = inner_side * 0.5
    t = wall_thickness * 0.5  # 墙厚半
    h = wall_height * 0.5  # 墙高半
    tf = floor_thickness * 0.5  # 底板厚半

    # ---- 组件半尺寸（按世界坐标 [x, y, z]）----
    # 底板：覆盖内部开口 + 两侧墙厚
    bottom_half = [inner_half + t, inner_half + t, tf]
    # 左/右墙：厚度沿 x，高度沿 z，长度沿 y
    lr_wall_half = [t, inner_half + t, h]
    # 前/后墙：厚度沿 y，高度沿 z，长度沿 x
    fb_wall_half = [inner_half + t, t, h]

    # ---- 确定bin的位置（参考cube的方式）----
    if position is None:
        base_pos = [0.0, 0.0, 0.0]
    else:
        base_pos = list(position)

    # 在局部坐标系里仍按“开口向上”搭建几何，最后整体翻转到“开口向下”
    # 底板在桌面上，墙体从底板顶面开始向上延伸（翻转后即向下）
    base_z = tf  # 底板中心高度（底板厚度的一半）

    # ---- 组件摆放位置（相对于bin的builder原点）----
    # 墙中心的水平偏移量 = 内部半边 + 墙厚半
    offset = inner_half + t
    # 墙中心的竖直位置 = 底板厚度 + 墙高半
    z_wall = tf + h

    poses = [
        sapien.Pose([0.0, 0.0, 0]),
        # 底板：在桌面上，厚度的一半高度
        sapien.Pose([0.0, 0.0, base_z]),
        # 左、右墙（±x 方向）：从底板顶面开始向上
        sapien.Pose([-offset, 0.0, z_wall]),
        sapien.Pose([+offset, 0.0, z_wall]),
        # 前、后墙（±y 方向）：从底板顶面开始向上
        sapien.Pose([0.0, -offset, z_wall]),
        sapien.Pose([0.0, +offset, z_wall]),
    ]
    half_sizes = [
        [self.cube_half_size,self.cube_half_size,self.cube_half_size],
        bottom_half,
        lr_wall_half,  # 左
        lr_wall_half,  # 右
        fb_wall_half,  # 前
        fb_wall_half,  # 后
    ]

    builder = self.scene.create_actor_builder()

    # 让bin"扣"在桌面上：整体绕 x 轴翻转 180°，让开口朝下，然后绕z轴旋转
    angles = torch.deg2rad(torch.tensor([180.0, 0.0, z_rotation_deg], dtype=torch.float32))  # (3,)
    rotate = matrix_to_quaternion(
        euler_angles_to_matrix(angles, convention="XYZ")
    )
    # 旋转后最低点位于 -(tf + 2h)，将其平移到 z=0 贴在桌面
    builder.set_initial_pose(
        sapien.Pose(
            p=[base_pos[0], base_pos[1], tf + 2 * h],
            q=rotate,
        )
    )

    for pose, half_size in zip(poses, half_sizes):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size)

    bin_actor = builder.build_dynamic(name=callsign)

    return bin_actor

def spawn_random_bin(
        self,
        avoid=None,
        region_center=[-0.1, 0],
        region_half_size=0.3,
        min_gap=0.05,
        name_prefix="bin",
        max_trials=256,
        generator=None
):
    """
    在矩形区域内用拒绝采样投放一个 bin，并返回该 bin actor。
    使用 OBB 精确碰撞检测，满足 min_gap 才落位。
    """
    if avoid is None:
        avoid = []

    center = np.array(region_center, dtype=np.float64)
    area_half = float(region_half_size)

    # 计算bin的大小（用于碰撞检测）
    inner_side = self.cube_half_size * 2.5
    wall_thickness = 0.005
    bin_half_size = (inner_side + wall_thickness) * 0.5  # bin总尺寸的一半

    # 让bin完整落在区域内
    x_low = center[0] - area_half + bin_half_size
    x_high = center[0] + area_half - bin_half_size
    y_low = center[1] - area_half + bin_half_size
    y_high = center[1] + area_half - bin_half_size

    if x_low > x_high or y_low > y_high:
        raise ValueError("_spawn_random_bin: 采样区域过小，无法放下该尺寸的 bin。")

    # === 组装障碍物 OBB（2D）列表 ===
    obb2d_list = []  # [(c, A, h), ...]

    def _push_actor_as_obb2d(actor, pad=0.0):
        try:
            obb = get_actor_obb(actor, to_world_frame=True, vis=False)
            obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
            obb2d_list.append(obb2d)
        except Exception:
            # 某些对象（如 site/marker）没有物理 mesh，则忽略
            pass

    # 收集避让物体的OBB
    for item in avoid:
        if isinstance(item, tuple):
            # Check if it's a pre-made OBB tuple (c, A, h) or (actor, pad)
            if len(item) == 3 and isinstance(item[0], np.ndarray) and isinstance(item[1], np.ndarray):
                # Pre-made OBB: (center, axes, half_sizes)
                obb2d_list.append(item)
            else:
                # Actor with padding
                actor, pad = item
                _push_actor_as_obb2d(actor, pad)
        else:
            _push_actor_as_obb2d(item, min_gap)

    for trial in range(int(max_trials)):
        x = float(torch.rand(1, generator=generator).item() * (x_high - x_low) + x_low)
        y = float(torch.rand(1, generator=generator).item() * (y_high - y_low) + y_low)

        # 新 bin 的方形碰撞检测
        bin_pos = np.array([x, y], dtype=np.float64)
        bin_collision_half_size = bin_half_size + min_gap

        # 与其他OBB障碍检测
        hit = False
        for (c_obs, A_obs, h_obs) in obb2d_list:
            # 简化：将bin视为方形，检测与OBB的碰撞
            # 计算bin中心到OBB的最近距离
            local_pos = A_obs.T @ (bin_pos - c_obs)
            closest_point = np.clip(local_pos, -h_obs, h_obs)
            closest_world = c_obs + A_obs @ closest_point
            dist = np.linalg.norm(bin_pos - closest_world)
            if dist < bin_collision_half_size:
                hit = True
                break

        if hit:
            continue

        # 通过检测，创建 bin（在指定位置），带随机z轴旋转
        z_rotation = float(torch.rand(1, generator=generator).item() * 90.0)  # 0-360度
        bin_actor = build_bin(self, callsign=name_prefix, position=[x, y, 0.002], z_rotation_deg=z_rotation)

        return bin_actor

    raise RuntimeError("_spawn_random_bin: 区域拥挤或约束过紧，未找到可行位置。可尝试：放大区域/减小bin/减小 min_gap。")

def spawn_fixed_cube(
        self,
        position,  # [x, y, z] 固定位置
        half_size=None,
        color=(1, 0, 0, 1),
        name_prefix="fixed_cube",
        yaw=0.0,  # 绕z轴旋转角度（弧度）
        dynamic=False,
    ):
    """
    在固定位置生成一个cube，不进行碰撞检测。
    使用builder模式创建动态物体，参考build_bin的实现。
    """
    hs = float(half_size if half_size is not None else self.cube_half_size)

    # 确保位置是数组格式
    pos = np.array(position, dtype=np.float64)
    if len(pos) == 2:
        # 如果只提供x,y，z设置为cube的半高度（让cube底部贴在桌面上）
        pos = np.append(pos, hs)

    # 创建actor builder
    builder = self.scene.create_actor_builder()

    # 生成旋转四元数（绕z轴旋转yaw角度）
    if yaw != 0.0:
        angles = torch.tensor([0.0, 0.0, float(yaw)], dtype=torch.float32)
        R = euler_angles_to_matrix(angles.unsqueeze(0), convention="XYZ")[0]
        q = matrix_to_quaternion(R.unsqueeze(0))[0]
        rotate = q
    else:
        rotate = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 无旋转四元数

    # 设置初始位置和旋转
    builder.set_initial_pose(
        sapien.Pose(
            p=[pos[0], pos[1], pos[2]],
            q=rotate.numpy() if isinstance(rotate, torch.Tensor) else rotate
        )
    )

    # 添加box几何体（碰撞和视觉）
    half_size_list = [hs, hs, hs]
    if  dynamic==True:
        # Collision geometry stays at builder origin; initial pose already positions the actor
        builder.add_box_collision(sapien.Pose([0, 0, 0]), half_size_list)

    # 创建材质
    material = sapien.render.RenderMaterial()
    material.set_base_color(color)
    builder.add_box_visual(sapien.Pose([0, 0, 0]), half_size_list, material=material)

    # 根据dynamic参数选择构建方式
    if dynamic==True:
        cube = builder.build_dynamic(name=name_prefix)
    else:
        cube = builder.build_kinematic(name=name_prefix)

    # 设置cube属性
    cube._cube_half_size = hs

    return cube

def build_board_with_hole(
        self,
        *,
        board_side=0.01,  # 正方形板子的边长
        hole_side=0.06,  # 正方形洞的边长
        thickness=0.02,  # 板子厚度
        position=None,  # 板子位置 [x, y] 或 [x, y, z]
        rotation_quat=None,  # 旋转四元数 [w, x, y, z]
        name="board_with_hole"
):
    """
    创建一个带正方形洞的正方形板子
    使用四个矩形条组合：上、下、左、右

    Args:
        height: 如果提供，则覆盖position中的z坐标
    """
    if position is None:
        position = [0.3, 0, 0]  # 默认位置，底部贴在桌面


    # 板子和洞的半边长
    board_half = board_side / 2
    hole_half = hole_side / 2
    thickness_half = thickness / 2

    # 将输入位置作为板子底面，计算板子中心位置
    # 输入position是底面位置，需要加上thickness_half得到中心位置
    center_position = [position[0], position[1], position[2] + thickness_half]

    # 创建actor builder
    builder = self.scene.create_actor_builder()

    # 设置板子的初始位置（使用中心位置）
    if rotation_quat is None:
        rotation_quat = [1.0, 0.0, 0.0, 0.0]  # 无旋转
    builder.set_initial_pose(
        sapien.Pose(
            p=center_position,
            q=rotation_quat
        )
    )

    # 创建材质 - 棕色板子
    material = sapien.render.RenderMaterial()
    material.set_base_color([0.8, 0.6, 0.4, 1.0])  # 明亮棕色

    # 四个矩形条的尺寸和位置
    # 上条 (top strip)
    top_width = board_side  # 整个板子宽度
    top_height = board_half - hole_half  # 从洞上边到板子上边
    top_center_y = hole_half + top_height / 2
    builder.add_box_collision(
        sapien.Pose([0, top_center_y, 0]),
        [top_width / 2, top_height / 2, thickness_half]
    )
    builder.add_box_visual(
        sapien.Pose([0, top_center_y, 0]),
        [top_width / 2, top_height / 2, thickness_half],
        material=material
    )

    # 下条 (bottom strip)
    bottom_width = board_side  # 整个板子宽度
    bottom_height = board_half - hole_half  # 从板子下边到洞下边
    bottom_center_y = -(hole_half + bottom_height / 2)
    builder.add_box_collision(
        sapien.Pose([0, bottom_center_y, 0]),
        [bottom_width / 2, bottom_height / 2, thickness_half]
    )
    builder.add_box_visual(
        sapien.Pose([0, bottom_center_y, 0]),
        [bottom_width / 2, bottom_height / 2, thickness_half],
        material=material
    )

    # 左条 (left strip) - 只在洞的高度范围内
    left_width = board_half - hole_half  # 从板子左边到洞左边
    left_height = hole_side  # 洞的高度
    left_center_x = -(hole_half + left_width / 2)
    builder.add_box_collision(
        sapien.Pose([left_center_x, 0, 0]),
        [left_width / 2, left_height / 2, thickness_half]
    )
    builder.add_box_visual(
        sapien.Pose([left_center_x, 0, 0]),
        [left_width / 2, left_height / 2, thickness_half],
        material=material
    )

    # 右条 (right strip) - 只在洞的高度范围内
    right_width = board_half - hole_half  # 从洞右边到板子右边
    right_height = hole_side  # 洞的高度
    right_center_x = hole_half + right_width / 2
    builder.add_box_collision(
        sapien.Pose([right_center_x, 0, 0]),
        [right_width / 2, right_height / 2, thickness_half]
    )
    builder.add_box_visual(
        sapien.Pose([right_center_x, 0, 0]),
        [right_width / 2, right_height / 2, thickness_half],
        material=material
    )

    # 在洞的中心添加一个与洞同样大小但高度减半的黑色cube（仅视觉，无碰撞）
    hole_cube_half_size_xy = hole_half  # cube的长宽半尺寸与洞相同
    hole_cube_half_height = thickness_half / 2  # cube高度为板子厚度的一半

    # 创建黑色材质
    black_material = sapien.render.RenderMaterial()
    black_material.set_base_color([0.0, 0.0, 0.0, 1.0])  # 黑色

    # 添加黑色cube（只有视觉，无碰撞）
    # 位置设置：cube底部贴在板子底面，所以cube中心在 -thickness_half + hole_cube_half_height
    cube_center_z = -thickness_half + hole_cube_half_height
    builder.add_box_visual(
        sapien.Pose([0, 0, cube_center_z]),  # 黑色cube底部贴在板子底面
        [hole_cube_half_size_xy, hole_cube_half_size_xy, hole_cube_half_height],
        material=black_material
    )

    # 构建actor
    board_actor = builder.build_kinematic(name=name)

    # 存储板子属性
    board_actor._board_side = board_side
    board_actor._hole_side = hole_side
    board_actor._thickness = thickness

    return board_actor


def build_purple_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_PURPLE = (np.array([160, 32, 240, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_PURPLE),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_PURPLE),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_PURPLE),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_gray_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_GRAY = (np.array([128, 128, 128, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_GRAY),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GRAY),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GRAY),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_green_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_GREEN = (np.array([34, 139, 34, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_GREEN),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GREEN),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_GREEN),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_red_white_target(
    scene: ManiSkillScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_RED = (np.array([200, 33, 33, 255]) / 255).tolist()
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def _build_by_type(
    builder: ActorBuilder,
    name,
    body_type,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor

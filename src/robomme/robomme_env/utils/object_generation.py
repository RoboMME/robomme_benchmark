import numpy as np
import torch
import sapien
from typing import Optional, Tuple, Sequence, Union
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
import mani_skill.envs.utils.randomization as randomization  # Only used if needed elsewhere
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
    Convert trimesh.primitives.Box (world frame) to 2D OBB representation: center c(2,), axes A(2x2), half-extents h(2,)
    extra_pad: Margins to expand outward on XY plane (meters)
    """
    # Compatible with obb potentially wrapped in .primitive
    b = getattr(obb_box, "primitive", obb_box)
    T = np.asarray(b.transform, dtype=np.float64)  # 4x4
    ex = np.asarray(b.extents, dtype=np.float64)   # 3

    R = T[:3, :3]
    t = T[:3, 3]

    c = t[:2].copy()

    # Take projection of X, Y axes on plane as two axes of 2D OBB
    u = _safe_unit(R[:2, 0])  # x-axis projection
    v = _safe_unit(R[:2, 1])  # y-axis projection
    A = np.stack([u, v], axis=1)  # 2x2, each column is an axis

    h = 0.5 * ex[:2].astype(np.float64)
    if extra_pad > 0:
        h = h + float(extra_pad)
    return c, A, h

def _obb2d_intersect(c1, A1, h1, c2, A2, h2):
    """
    2D OBB SAT detection. c*: (2,), A*: (2x2) columns are axes, h*: (2,)
    Returns True indicating intersection (including contact), False indicating separation
    """
    d = c2 - c1
    axes = [A1[:, 0], A1[:, 1], A2[:, 0], A2[:, 1]]

    for a in axes:
        a = _safe_unit(a)
        # Projected radius
        r1 = abs(np.dot(A1[:, 0], a)) * h1[0] + abs(np.dot(A1[:, 1], a)) * h1[1]
        r2 = abs(np.dot(A2[:, 0], a)) * h2[0] + abs(np.dot(A2[:, 1], a)) * h2[1]
        dist = abs(np.dot(d, a))
        if dist > (r1 + r2):
            return False  # Separating axis exists -> No intersection
    return True  # All axes overlap -> Intersection/Contact

def _yaw_to_quat_tensor(yaw: float, device):
    """
    Get quaternion consistent with ManiSkill/your conversion tools using z-axis Euler angle (shape [1,4], float32, device aligned)
    """
    # euler_angles_to_matrix accepts [roll, pitch, yaw] (radians), returns Nx3x3
    angles = torch.tensor([[0.0, 0.0, float(yaw)]], dtype=torch.float32, device=device)
    R = euler_angles_to_matrix(angles,convention="XYZ")            # (1, 3, 3)
    q = matrix_to_quaternion(R)                   # (1, 4) Convention same as ManiSkill
    return q

def _build_new_cube_obb2d(x, y, half_size_xy, yaw, pad_xy=0.0):
    """
    Construct 2D OBB for "cube ready to be placed": center/axes/half-extents
    half_size_xy: float, half length of cube on XY
    yaw: rotation around z-axis (radians)
    pad_xy: extra padding on half length on XY (for minimum gap)
    """
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]], dtype=np.float64)  # Columns are axes
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
    Drop a cube (onto table) in rectangular region using rejection sampling, and return the cube actor.
    - Uses OBB precise collision (2D projection + SAT), places only if min_gap is satisfied.
    - avoid: Input a list of objects. Can be [actor, ...] or [(actor, pad), ...] (pad in meters).
    - generator: Must pass torch.Generator for randomization.
    """
    # Cache
    if not hasattr(self, "_spawned_cubes"):
        self._spawned_cubes = []
        self._spawned_count = 0

    center = np.array(region_center if region_center is not None else self.cube_spawn_center, dtype=np.float64)

    # Support two types of input: scalar or 2D array
    if region_half_size is None:
        region_half_size = self.cube_spawn_half_size

    # Compatible with two input formats
    if isinstance(region_half_size, (list, tuple, np.ndarray)):
        # 2D array input: independent control for xy
        area_half = np.array(region_half_size, dtype=np.float64)
        if area_half.shape == ():  # Handle 0-dim array
            area_half = np.array([float(area_half), float(area_half)], dtype=np.float64)
        elif len(area_half) == 1:
            area_half = np.array([float(area_half[0]), float(area_half[0])], dtype=np.float64)
        elif len(area_half) != 2:
            raise ValueError("region_half_size array must contain 1 or 2 elements [x_half, y_half]")
    else:
        # Scalar input: xy remain consistent
        area_half = np.array([float(region_half_size), float(region_half_size)], dtype=np.float64)

    hs_new = float(half_size if half_size is not None else self.cube_half_size)

    # Let cube fall completely inside region (independent control for xy)
    x_low = center[0] - area_half[0] + hs_new
    x_high = center[0] + area_half[0] - hs_new
    y_low = center[1] - area_half[1] + hs_new
    y_high = center[1] + area_half[1] - hs_new
    if x_low > x_high or y_low > y_high:
        raise ValueError("spawn_random_cube: Sampling region too small, cannot fit cube of this size.")

    # === Assemble Obstacle OBB (2D) List ===
    obb2d_list = []  # [(c, A, h), ...]

    def _push_actor_as_obb2d(actor, pad=0.0):
        try:
            # Special handling for board_with_hole
            if hasattr(actor, '_board_side') and hasattr(actor, '_hole_side'):
                # This is our board with hole, manually add its OBB
                board_side = actor._board_side
                hole_side = actor._hole_side

                # Get board world position
                actor_pos = actor.pose.p
                if isinstance(actor_pos, torch.Tensor):
                    actor_pos = actor_pos[0].detach().cpu().numpy()

                board_center = np.array(actor_pos[:2], dtype=np.float64)
                board_half = board_side / 2
                hole_half = hole_side / 2

                # Add OBBs for four rectangular strips
                # Top strip
                if board_half > hole_half:  # Ensure enough space
                    top_height = board_half - hole_half
                    top_center = board_center + np.array([0, hole_half + top_height / 2])
                    A_top = np.eye(2)  # No rotation
                    h_top = np.array([board_half + pad, top_height / 2 + pad])
                    obb2d_list.append((top_center, A_top, h_top))

                    # Bottom strip
                    bottom_center = board_center + np.array([0, -(hole_half + top_height / 2)])
                    obb2d_list.append((bottom_center, A_top, h_top))

                    # Left strip
                    left_width = board_half - hole_half
                    left_center = board_center + np.array([-(hole_half + left_width / 2), 0])
                    h_left = np.array([left_width / 2 + pad, hole_half + pad])
                    obb2d_list.append((left_center, A_top, h_left))

                    # Right strip
                    right_center = board_center + np.array([hole_half + left_width / 2, 0])
                    obb2d_list.append((right_center, A_top, h_left))
                return

            obb = get_actor_obb(actor, to_world_frame=True, vis=False)
            obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
            obb2d_list.append(obb2d)
        except Exception:
            # Some objects (like site/marker) do not have physical mesh, ignore or use circle approximation below
            pass

    if include_existing:
        # Main cube
        if hasattr(self, "cube") and self.cube is not None:
            _push_actor_as_obb2d(self.cube, pad=0.0)

        # Historically spawned cubes
        for ac in self._spawned_cubes:
            _push_actor_as_obb2d(ac, pad=0.0)

    # User specified extra avoidance
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

    # Target point (if no mesh), supplement with "circle + circumscribed circle" conservative approximation (optional)
    circle_list = []  # [(xy(2,), R)], for objects without mesh
    def _actor_xy(actor):
        p = actor.pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        return np.array(p[:2], dtype=np.float64)

    if include_goal and hasattr(self, "goal_site") and self.goal_site is not None:
        try:
            # If goal_site has mesh, it will be covered in _push_actor_as_obb2d, here only as a fallback
            _push_actor_as_obb2d(self.goal_site, pad=0.0)
        except Exception:
            # Degrade to circle approximation: goal radius + new cube circumscribed circle radius
            R_goal = float(getattr(self, "goal_thresh", 0.03))
            R_new_ext = np.sqrt(2.0) * hs_new
            circle_list.append((_actor_xy(self.goal_site), R_goal + R_new_ext + min_gap))

    # === Sampling Iteration ===
    if generator is None:
        raise ValueError("spawn_random_cube: generator argument must be explicitly passed for randomization")

    device = self.device

    for trial in range(int(max_trials)):
        # Use simple uniform sampling to ensure good spatial coverage
        # Complex sampling strategies often reduce coverage

        u1 = torch.rand(1, generator=generator).item()
        u2 = torch.rand(1, generator=generator).item()

        # Map directly to sampling region - Uniform distribution provides best spatial coverage
        x = float(x_low + u1 * (x_high - x_low))
        y = float(y_low + u2 * (y_high - y_low))

        if random_yaw:
            # Use more random yaw generation, sampling from full [0, 2π] range
            yaw_sample = torch.rand(1, generator=generator).item()
            yaw = float(yaw_sample * 2 * np.pi)
        else:
            yaw = 0.0

        # New cube's 2D OBB (reflect min_gap in "new object half length expansion", avoid adding to both sides causing double)
        c_new, A_new, h_new = _build_new_cube_obb2d(x, y, hs_new, yaw, pad_xy=float(min_gap))

        # Check collision one by one with OBB obstacles
        hit = False
        for (c_obs, A_obs, h_obs) in obb2d_list:
            if _obb2d_intersect(c_obs, A_obs, h_obs, c_new, A_new, h_new):
                hit = True
                break
        if hit:
            continue

        # Check circular conservative obstacles (if present)
        for (xy_c, R_c) in circle_list:
            if np.linalg.norm(np.asarray([x, y], dtype=np.float64) - xy_c) < R_c:
                hit = True
                break
        if hit:
            continue

        # Passing detection, create cube (pose and collision detection use same yaw to ensure consistency)
        q = _yaw_to_quat_tensor(yaw, device=device)

        cube = actors.build_cube(
            self.scene,
            half_size=hs_new,
            color=color,
            name=name_prefix,  # Use name_prefix directly, do not add counter
            initial_pose=Pose.create_from_pq(
                torch.tensor([[x, y, hs_new]], device=device, dtype=torch.float32),
                q,
            ),
        )
        cube._cube_half_size = hs_new
        self._spawned_cubes.append(cube)
        self._spawned_count += 1
        return cube

    raise RuntimeError("spawn_random_cube: Region crowded or constraints too tight, no feasible position found. Try: increase region/decrease cube/decrease min_gap.")

def _build_new_target_obb2d(x, y, half_size_xy, yaw, pad_xy=0.0):
    """
    Construct 2D OBB for "target ready to be placed": center/axes/half-extents
    half_size_xy: float, half length of target on XY
    yaw: rotation around z-axis (radians)
    pad_xy: extra padding on half length on XY (for minimum gap)
    """
    c = np.array([x, y], dtype=np.float64)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    A = np.array([[cos_y, -sin_y],
                  [sin_y,  cos_y]], dtype=np.float64)  # Each column is an axis
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
        avoid=None,          # Supports [actor, ...] or [(actor, pad), ...]
        include_existing=True,   # Whether to automatically avoid existing main target and generated extra targets
        include_goal=True,       # Whether to treat goal_site as obstacle (approximate with circle, conservative)
        generator=None,
        randomize=True,      # Control whether to randomize position
        target_style="purple",  # Choose which color scheme target to create
    ):
    """
    Drop a target (onto table) in rectangular region using rejection sampling, and return the target actor.
    - Uses OBB precise collision (2D projection + SAT), places only if min_gap is satisfied.
    - avoid: Input a list of objects. Can be [actor, ...] or [(actor, pad), ...] (pad in meters).
    - generator: Must pass torch.Generator for randomization (when randomize=True).
    - randomize: Control whether to randomize position. If False, generate directly at region_center.
    """
    # Cache
    random_yaw=False
    if not hasattr(self, "_spawned_targets"):
        self._spawned_targets = []
        self._spawned_target_count = 0

    center = np.array(region_center if region_center is not None else getattr(self, 'target_spawn_center', [0, 0]), dtype=np.float64)
    area_half = float(region_half_size if region_half_size is not None else getattr(self, 'target_spawn_half_size', 0.1))
    target_radius = float(radius if radius is not None else getattr(self, 'target_radius', 0.01))
    target_thickness = float(thickness if thickness is not None else getattr(self, 'target_thickness', 0.005))

    # Let target fall completely inside region
    x_low = center[0] - area_half + target_radius
    x_high = center[0] + area_half - target_radius
    y_low = center[1] - area_half + target_radius
    y_high = center[1] + area_half - target_radius
    if x_low > x_high or y_low > y_high:
        raise ValueError("spawn_random_target: Sampling region too small, cannot fit target of this size.")

    # === Assemble Obstacle OBB (2D) List ===
    obb2d_list = []  # [(c, A, h), ...]

    def _push_actor_as_obb2d(actor, pad=0.0):
        try:
            # Special handling for board_with_hole
            if hasattr(actor, '_board_side') and hasattr(actor, '_hole_side'):
                # This is our board with hole, manually add its OBB
                board_side = actor._board_side
                hole_side = actor._hole_side

                # Get board world position
                actor_pos = actor.pose.p
                if isinstance(actor_pos, torch.Tensor):
                    actor_pos = actor_pos[0].detach().cpu().numpy()

                board_center = np.array(actor_pos[:2], dtype=np.float64)
                board_half = board_side / 2
                hole_half = hole_side / 2

                # Add OBBs for four rectangular strips
                # Top strip
                if board_half > hole_half:  # Ensure enough space
                    top_height = board_half - hole_half
                    top_center = board_center + np.array([0, hole_half + top_height / 2])
                    A_top = np.eye(2)  # No rotation
                    h_top = np.array([board_half + pad, top_height / 2 + pad])
                    obb2d_list.append((top_center, A_top, h_top))

                    # Bottom strip
                    bottom_center = board_center + np.array([0, -(hole_half + top_height / 2)])
                    obb2d_list.append((bottom_center, A_top, h_top))

                    # Left strip
                    left_width = board_half - hole_half
                    left_center = board_center + np.array([-(hole_half + left_width / 2), 0])
                    h_left = np.array([left_width / 2 + pad, hole_half + pad])
                    obb2d_list.append((left_center, A_top, h_left))

                    # Right strip
                    right_center = board_center + np.array([hole_half + left_width / 2, 0])
                    obb2d_list.append((right_center, A_top, h_left))
                return

            obb = get_actor_obb(actor, to_world_frame=True, vis=False)
            obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
            obb2d_list.append(obb2d)
        except Exception:
            # Some objects (like site/marker) do not have physical mesh, ignore or use circle approximation below
            pass

    if include_existing:
        # Main cube
        if hasattr(self, "cube") and self.cube is not None:
            _push_actor_as_obb2d(self.cube, pad=0.0)

        # Main target
        if hasattr(self, "target") and self.target is not None:
            _push_actor_as_obb2d(self.target, pad=0.0)

        # Historically spawned cubes
        if hasattr(self, "_spawned_cubes"):
            for ac in self._spawned_cubes:
                _push_actor_as_obb2d(ac, pad=0.0)

    # Target point (if no mesh), supplement with "circle + circumscribed circle" conservative approximation (optional)
    circle_list = []  # [(xy(2,), R)], for objects without mesh
    def _actor_xy(actor):
        p = actor.pose.p
        if isinstance(p, torch.Tensor):
            p = p[0].detach().cpu().numpy()
        return np.array(p[:2], dtype=np.float64)

    # Historically spawned targets - Treat as circular obstacles
    if include_existing:
        for ac in self._spawned_targets:
            target_r = getattr(ac, "_target_radius", target_radius)
            circle_list.append((_actor_xy(ac), target_r))

    # User specified extra avoidance
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
                    # Check if it is a target (circular)
                    if hasattr(act_i, "_target_radius"):
                        target_r = getattr(act_i, "_target_radius", target_radius)
                        circle_list.append((_actor_xy(act_i), target_r + float(pad_i)))
                    else:
                        _push_actor_as_obb2d(act_i, pad=float(pad_i))
            else:
                # Check if it is a target (circular)
                if hasattr(it, "_target_radius"):
                    target_r = getattr(it, "_target_radius", target_radius)
                    circle_list.append((_actor_xy(it), target_r))
                else:
                    _push_actor_as_obb2d(it, pad=0.0)

    if include_goal and hasattr(self, "goal_site") and self.goal_site is not None:
        try:
            # If goal_site has mesh, it will be covered in _push_actor_as_obb2d, here only as a fallback
            _push_actor_as_obb2d(self.goal_site, pad=0.0)
        except Exception:
            # Degrade to circle approximation: goal radius + new target circumscribed circle radius
            R_goal = float(getattr(self, "goal_thresh", 0.03))
            R_new_ext = target_radius
            circle_list.append((_actor_xy(self.goal_site), R_goal + R_new_ext + min_gap))

    # === Sampling Iteration ===
    if generator is None:
        raise ValueError("spawn_random_target: generator argument must be explicitly passed for randomization")

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
            raise ValueError(f"spawn_random_target: Unknown target_style '{target_style}'. Supported: {list(target_builders.keys())}")
        target_builder = target_builders[builder_key]
    elif callable(target_style):
        target_builder = target_style
    else:
        raise ValueError("spawn_random_target: target_style must be a string or callable builder function")

    for _ in range(int(max_trials)):
        x = float(torch.rand(1, generator=generator).item() * (x_high - x_low) + x_low)
        y = float(torch.rand(1, generator=generator).item() * (y_high - y_low) + y_low)

        if random_yaw:
            yaw = float(torch.rand(1, generator=generator).item() * 2 * np.pi - np.pi)
        else:
            yaw = 0.0

        # New target's circular collision detection (target is circular, circular detection is more accurate)
        target_pos = np.array([x, y], dtype=np.float64)
        target_collision_radius = target_radius + min_gap

        # Check collision with OBB obstacles (check circular target against square obstacles)
        hit = False
        for (c_obs, A_obs, h_obs) in obb2d_list:
            # Calculate minimum distance from circle center to OBB
            # Convert circle center to OBB local coordinate system
            local_pos = A_obs.T @ (target_pos - c_obs)
            # Calculate closest point from circle center to OBB
            closest_point = np.clip(local_pos, -h_obs, h_obs)
            # Convert back to world coordinate system
            closest_world = c_obs + A_obs @ closest_point
            # Calculate distance
            dist = np.linalg.norm(target_pos - closest_world)
            if dist < target_collision_radius:
                hit = True
                break
        if hit:
            continue

        # Check collision with circular obstacles (circle vs circle)
        for (xy_c, R_c) in circle_list:
            if np.linalg.norm(target_pos - xy_c) < (target_collision_radius + R_c):
                hit = True
                break
        if hit:
            continue

        # Passed detection, create target (pose and collision detection use same yaw to ensure consistency)
        rotate = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])  # Quaternion for z-axis rotation
        angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))  # (3,)
        rotate = matrix_to_quaternion(
            euler_angles_to_matrix(angles, convention="XYZ")
        )
        target = target_builder(
            scene=self.scene,
            radius=target_radius,
            thickness=target_thickness,
            name=name_prefix,  # Use name_prefix directly, do not add counter
            body_type="kinematic",  # Visualization only
            add_collision=False,  # Disable collision
            initial_pose=sapien.Pose(p=[x, y, target_thickness], q=rotate),
        )
        target._target_radius = target_radius
        self._spawned_targets.append(target)
        self._spawned_target_count += 1
        return target

    raise RuntimeError("spawn_random_target: Region crowded or constraints too tight, no feasible position found. Try: increase region/decrease target/decrease min_gap.")


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
            center_xy=(0.15, 0.10),  # Button (x,y) on table
            base_half=[0.025, 0.025, 0.005],  # Base half-size [x,y,z]
            cap_radius=0.015,  # Button cap radius
            cap_half_len=0.006,  # Button cap half-length
            travel=None,  # Press travel
            stiffness=800.0,
            damping=40.0,
            scale: float = None,  # ⭐ New: scaling factor
            generator=None,
            name: str = "button",  # ⭐ New: button name
            randomize: bool = True,  # ⭐ New: whether to randomize position
            randomize_range=(0.1, 0.4),  # ⭐ New: randomization range, (range_x, range_y)
    ):
        # ------- Scaling and Travel -------
        if scale is None:
            # If not passed, use default scaling from environment
            scale = getattr(self, "button_scale", 1.0)
        scale = float(scale)

        # Travel priority: argument > environment base
        if travel is None:
            # Scale proportionally using base travel
            base_travel = getattr(self, "_button_travel_base", 0.1)
            travel = base_travel * scale
        else:
            # If travel explicitly passed, also follow scale (to keep absolute value, change next line to pass)
            travel = float(travel) * scale

        # Size scaling
        base_half = [bh * scale for bh in base_half]
        cap_radius = float(cap_radius) * scale
        cap_half_len = float(cap_half_len) * scale

        # Record current button travel for other functions
        self.button_travel = float(travel)

        # ------- Position Randomization -------
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

        # Initial pose: lift base center to z=base_half[2]
        builder.initial_pose = sapien.Pose(p=[cx, cy, base_half[2]])

        # Root: Base
        base = builder.create_link_builder()
        base.set_name("button_base")
        base.add_box_collision(half_size=base_half, density=200000)
        base.add_box_visual(half_size=base_half)

        # Child: Button cap (vertical sliding)
        cap = builder.create_link_builder(base)
        cap.set_name("button_cap")
        cap.set_joint_name("button_joint")

        R_up = euler2quat(0, -np.pi / 2, 0)  # Align joint x-axis with world z

        cap.set_joint_properties(
            type="prismatic",
            limits=[[-travel, 0.0]],  # Negative direction is pressed
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
        self.cap_links[name] = [cap_link]   # name is "button_left", "button_right", etc.
        self.cap_link = self.cap_links[name]  # Compatible with old logic

        # Provide an OBB for downstream placement logic using the scaled button footprint
        button_obb = create_button_obb(
            center_xy=center_xy,
            half_size=max(base_half[0], base_half[1]) * 1.5,
        )
        return button_obb
def build_bin(
        self,
        *,
        inner_side: float = 0.04,  # Inner opening side length (full length, meters), originally 2*inner_side_half_len = 0.04
        wall_thickness: float = 0.005,  # Wall thickness (full thickness, meters)
        wall_height: float = 0.05,  # Wall height (full height, meters)
        floor_thickness: float = 0.004,  # Floor thickness (full thickness, meters)
        callsign=None,
        position=None,  # Add position argument
        z_rotation_deg=0.0  # Add z-axis rotation angle argument (degrees)
):
    """
    Assemble an "open box" using 1 floor + 4 wall strips.
    All dimensions use "full size (meters)", automatically converted to half-size internally.
    Refer to cube generation method, let bin bottom sit on table (z=0).
    """
    inner_side = self.cube_half_size * 2.5
    wall_height = self.cube_half_size * 2.5

    # ---- Convert full size to half size (consistent with add_box_* interface) ----
    inner_half = inner_side * 0.5
    t = wall_thickness * 0.5  # Half wall thickness
    h = wall_height * 0.5  # Half wall height
    tf = floor_thickness * 0.5  # Half floor thickness

    # ---- Component half sizes (in world coordinates [x, y, z]) ----
    # Floor: covers inner opening + two side wall thicknesses
    bottom_half = [inner_half + t, inner_half + t, tf]
    # Left/Right Wall: thickness along x, height along z, length along y
    lr_wall_half = [t, inner_half + t, h]
    # Front/Back Wall: thickness along y, height along z, length along x
    fb_wall_half = [inner_half + t, t, h]

    # ---- Determine bin position (refer to cube way) ----
    if position is None:
        base_pos = [0.0, 0.0, 0.0]
    else:
        base_pos = list(position)

    # Build geometry as "opening up" in local coordinate system, then flip to "opening down" globally
    # Floor on table, walls extend up from floor top (flipped becomes down)
    base_z = tf  # Floor center height (half of floor thickness)

    # ---- Component placement positions (relative to bin builder origin) ----
    # Wall center horizontal offset = inner half + half wall thickness
    offset = inner_half + t
    # Wall center vertical position = floor thickness + half wall height
    z_wall = tf + h

    poses = [
        sapien.Pose([0.0, 0.0, 0]),
        # Floor: on table, half thickness height
        sapien.Pose([0.0, 0.0, base_z]),
        # Left/Right Wall (+/- x direction): extend up from floor top
        sapien.Pose([-offset, 0.0, z_wall]),
        sapien.Pose([+offset, 0.0, z_wall]),
        # Front/Back Wall (+/- y direction): extend up from floor top
        sapien.Pose([0.0, -offset, z_wall]),
        sapien.Pose([0.0, +offset, z_wall]),
    ]
    half_sizes = [
        [self.cube_half_size,self.cube_half_size,self.cube_half_size],
        bottom_half,
        lr_wall_half,  # Left
        lr_wall_half,  # Right
        fb_wall_half,  # Front
        fb_wall_half,  # Back
    ]

    builder = self.scene.create_actor_builder()

    # Let bin "clasp" on table: flip 180 degrees around x-axis, opening down, then rotate around z-axis
    angles = torch.deg2rad(torch.tensor([180.0, 0.0, z_rotation_deg], dtype=torch.float32))  # (3,)
    rotate = matrix_to_quaternion(
        euler_angles_to_matrix(angles, convention="XYZ")
    )
    # Lowest point after rotation is -(tf + 2h), translate to z=0 to sit on table
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
    Drop a bin in rectangular region using rejection sampling, and return the bin actor.
    Use OBB precise collision detection, place only if min_gap is satisfied.
    """
    if avoid is None:
        avoid = []

    center = np.array(region_center, dtype=np.float64)
    area_half = float(region_half_size)

    # Calculate bin size (for collision detection)
    inner_side = self.cube_half_size * 2.5
    wall_thickness = 0.005
    bin_half_size = (inner_side + wall_thickness) * 0.5  # Half of bin total size

    # Let bin fall completely inside region
    x_low = center[0] - area_half + bin_half_size
    x_high = center[0] + area_half - bin_half_size
    y_low = center[1] - area_half + bin_half_size
    y_high = center[1] + area_half - bin_half_size

    if x_low > x_high or y_low > y_high:
        raise ValueError("_spawn_random_bin: Sampling region too small, cannot fit bin of this size.")

    # === Assemble Obstacle OBB (2D) List ===
    obb2d_list = []  # [(c, A, h), ...]

    def _push_actor_as_obb2d(actor, pad=0.0):
        try:
            obb = get_actor_obb(actor, to_world_frame=True, vis=False)
            obb2d = _trimesh_box_to_obb2d(obb, extra_pad=float(pad))
            obb2d_list.append(obb2d)
        except Exception:
            # Some objects (like site/marker) do not have physical mesh, ignore
            pass

    # Collect avoidance object OBBs
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

        # New bin square collision detection
        bin_pos = np.array([x, y], dtype=np.float64)
        bin_collision_half_size = bin_half_size + min_gap

        # Detect collision with other OBB obstacles
        hit = False
        for (c_obs, A_obs, h_obs) in obb2d_list:
            # Simplify: treat bin as square, detect collision with OBB
            # Calculate bin center to OBB closest distance
            local_pos = A_obs.T @ (bin_pos - c_obs)
            closest_point = np.clip(local_pos, -h_obs, h_obs)
            closest_world = c_obs + A_obs @ closest_point
            dist = np.linalg.norm(bin_pos - closest_world)
            if dist < bin_collision_half_size:
                hit = True
                break

        if hit:
            continue

        # Passing detection, create bin (at specified position), with random z-axis rotation
        z_rotation = float(torch.rand(1, generator=generator).item() * 90.0)  # 0-360 degrees
        bin_actor = build_bin(self, callsign=name_prefix, position=[x, y, 0.002], z_rotation_deg=z_rotation)

        return bin_actor

    raise RuntimeError("_spawn_random_bin: Region crowded or constraints too tight, no feasible position found. Try: increase region/decrease bin/decrease min_gap.")

def spawn_fixed_cube(
        self,
        position,  # [x, y, z] fixed position
        half_size=None,
        color=(1, 0, 0, 1),
        name_prefix="fixed_cube",
        yaw=0.0,  # rotation around z-axis (radians)
        dynamic=False,
    ):
    """
    Generate a cube at fixed position, no collision detection.
    Use builder pattern to create dynamic object, refer to build_bin implementation.
    """
    hs = float(half_size if half_size is not None else self.cube_half_size)

    # Ensure position is array format
    pos = np.array(position, dtype=np.float64)
    if len(pos) == 2:
        # If only x,y provided, set z to cube half height (let cube bottom sit on table)
        pos = np.append(pos, hs)

    # Create actor builder
    builder = self.scene.create_actor_builder()

    # Generate rotation quaternion (rotate yaw angle around z-axis)
    if yaw != 0.0:
        angles = torch.tensor([0.0, 0.0, float(yaw)], dtype=torch.float32)
        R = euler_angles_to_matrix(angles.unsqueeze(0), convention="XYZ")[0]
        q = matrix_to_quaternion(R.unsqueeze(0))[0]
        rotate = q
    else:
        rotate = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    # Set initial position and rotation
    builder.set_initial_pose(
        sapien.Pose(
            p=[pos[0], pos[1], pos[2]],
            q=rotate.numpy() if isinstance(rotate, torch.Tensor) else rotate
        )
    )

    # Add box geometry (collision and visual)
    half_size_list = [hs, hs, hs]
    if  dynamic==True:
        # Collision geometry stays at builder origin; initial pose already positions the actor
        builder.add_box_collision(sapien.Pose([0, 0, 0]), half_size_list)

    # Create material
    material = sapien.render.RenderMaterial()
    material.set_base_color(color)
    builder.add_box_visual(sapien.Pose([0, 0, 0]), half_size_list, material=material)

    # Choose build method based on dynamic argument
    if dynamic==True:
        cube = builder.build_dynamic(name=name_prefix)
    else:
        cube = builder.build_kinematic(name=name_prefix)

    # Set cube attribute
    cube._cube_half_size = hs

    return cube

def build_board_with_hole(
        self,
        *,
        board_side=0.01,  # Square board side length
        hole_side=0.06,  # Square hole side length
        thickness=0.02,  # Board thickness
        position=None,  # Board position [x, y] or [x, y, z]
        rotation_quat=None,  # Rotation quaternion [w, x, y, z]
        name="board_with_hole"
):
    """
    Create a square board with a square hole
    Combine four rectangular strips: top, bottom, left, right

    Args:
        height: If provided, overwrite z coordinate in position
    """
    if position is None:
        position = [0.3, 0, 0]  # Default position, bottom on table


    # Board and hole half lengths
    board_half = board_side / 2
    hole_half = hole_side / 2
    thickness_half = thickness / 2

    # Use input position as board bottom, calculate board center position
    # Input position is bottom position, need to add thickness_half to get center position
    center_position = [position[0], position[1], position[2] + thickness_half]

    # Create actor builder
    builder = self.scene.create_actor_builder()

    # Set board initial position (use center position)
    if rotation_quat is None:
        rotation_quat = [1.0, 0.0, 0.0, 0.0]  # No rotation
    builder.set_initial_pose(
        sapien.Pose(
            p=center_position,
            q=rotation_quat
        )
    )

    # Create material - brown board
    material = sapien.render.RenderMaterial()
    material.set_base_color([0.8, 0.6, 0.4, 1.0])  # Light brown

    # Four rectangular strips dimensions and positions
    # Top strip
    top_width = board_side  # Full board width
    top_height = board_half - hole_half  # From hole top to board top
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

    # Bottom strip
    bottom_width = board_side  # Full board width
    bottom_height = board_half - hole_half  # From board bottom to hole bottom
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

    # Left strip - only within hole height range
    left_width = board_half - hole_half  # From board left to hole left
    left_height = hole_side  # Hole height
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

    # Right strip - only within hole height range
    right_width = board_half - hole_half  # From hole right to board right
    right_height = hole_side  # Hole height
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

    # Add a black cube at hole center with same size as hole but half height (visual only, no collision)
    hole_cube_half_size_xy = hole_half  # cube half size same as hole
    hole_cube_half_height = thickness_half / 2  # cube height is half of board thickness

    # Create black material
    black_material = sapien.render.RenderMaterial()
    black_material.set_base_color([0.0, 0.0, 0.0, 1.0])  # Black

    # Add black cube (visual only, no collision)
    # Position: cube bottom at board bottom, so cube center at -thickness_half + hole_cube_half_height
    cube_center_z = -thickness_half + hole_cube_half_height
    builder.add_box_visual(
        sapien.Pose([0, 0, cube_center_z]),  # Black cube bottom at board bottom
        [hole_cube_half_size_xy, hole_cube_half_size_xy, hole_cube_half_height],
        material=black_material
    )

    # Build actor
    board_actor = builder.build_kinematic(name=name)

    # Store board attributes
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

"""Instance-local geometry, scoring and oracle methods for imitation tasks."""

import math

import numpy as np
from mani_skill.envs.sapien_env import BaseEnv

from robomme_plus.common.sim import (
    position,
    scalar,
    set_position,
    step_number,
)
from robomme_plus.scoring.subgoals import evaluate_tasks, task
from robomme_plus.tasks.imitation.utils.geometry import (
    INSERT_GRASP_OFFSET_M,
    INSERT_LIFT_M,
    INSERT_PHASES,
    INSERT_STANDOFF_M,
    INSERT_TIP_OFFSETS_M,
    ROUTE_YAW_OFFSETS_DEG,
    STAND_HALF_DEPTH_M,
    STAND_HALF_WIDTH_M,
    STAND_HEIGHT_M,
    frame_from_axis,
    half_ellipse_route,
    insertion_choice,
    insertion_hole,
    insertion_metrics,
    insertion_peg_roots,
    insertion_stand_boxes,
    insertion_stand_center,
    insertion_stand_frame,
    insertion_valid,
    lateral_occluder_y,
    ray_intersects_box,
    resolve_config,
)


def rotation(actor):
    from transforms3d.quaternions import quat2mat

    return quat2mat(actor.pose.q.detach().cpu().numpy().reshape(-1, 4)[0])


def move_pose(planner, point, orientation):
    import sapien
    from transforms3d.quaternions import mat2quat

    result = planner.move_to_pose_with_screw(
        sapien.Pose(point, mat2quat(orientation)), dry_run=True
    )
    if isinstance(result, int) and result == -1:
        result = planner.move_to_pose_with_RRTStar(
            sapien.Pose(point, mat2quat(orientation)), dry_run=True
        )
        if isinstance(result, int) and result == -1:
            raise RuntimeError(
                f"Imitation waypoint planning failed at {np.asarray(point).tolist()}"
            )
    # The planner can return a successful empty path for an already reached
    # pose; its native follow_path assumes at least one step.
    return planner.follow_path(result) if len(result["position"]) else None


def relative_tcp(env, actor):
    r = rotation(actor)
    return (
        r.T @ (position(env.agent.tcp, False) - position(actor, False)),
        r.T @ rotation(env.agent.tcp),
    )


def box_actor(env, name, center, half, color=(0.52, 0.58, 0.66, 1)):
    import sapien

    builder = env.scene.create_actor_builder()
    builder.add_box_collision(half_size=half)
    builder.add_box_visual(
        half_size=half,
        material=sapien.render.RenderMaterial(base_color=color, roughness=0.8),
    )
    builder.initial_pose = sapien.Pose(center)
    return builder.build_kinematic(name)


# Contact generation distance for the peg and the stand (PhysX default 0.02 m).
INSERTION_CONTACT_OFFSET_M = 0.00005
# The inactive stand waits 3 m beside the table, on the floor, out of both views.
STAND_PARK_Y_M = -3.0


def setup_insert(cls):
    """Two stands at one table position: A in the demonstration, B in execution.

    Both stands share a seeded yaw and have different straight through-holes (see
    ``geometry._resolve_insert``). The "NO RECORD" reset that ends the
    demonstration parks stand A and puts stand B in place; scoring, the oracle,
    and the success cue use the active stand (``env.box``).
    """
    import sapien
    from transforms3d.axangles import axangle2mat, mat2axangle
    from transforms3d.quaternions import mat2quat

    from robomme_plus.control.planning import solve_strong_reset
    from robomme_plus.scoring.predicates import (
        is_A_pickup_notB,
        is_any_obj_pickup,
        reset_check,
    )

    original_load, original_init = cls._load_scene, cls._initialize_episode

    def park_pose(env):
        floor = -float(getattr(env.table_scene, "table_height", 0.9196429))
        return sapien.Pose([0, STAND_PARK_Y_M, floor])

    def build_stand(env, phase):
        """A kinematic stand whose four walls surround that phase's hole."""
        config = env._imitation_config
        stand = config["stands"][phase]
        builder = env.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(
            base_color=[1, 210 / 255, 137 / 255, 1], roughness=0.5, specular=0.5
        )
        for center, half in insertion_stand_boxes(
            stand["hole_height_m"],
            stand["hole_lateral_m"],
            config["aperture_half_size_m"],
        ):
            builder.add_box_collision(sapien.Pose(center), half)
            builder.add_box_visual(sapien.Pose(center), half, material=material)
        builder.initial_pose = park_pose(env)
        actor = builder.build_kinematic(f"imitation_insertion_stand_{phase}")
        # PhysX's default 2 cm contact offset creates contacts across a
        # sub-millimetre gap; at the hole's edge they wedge a centered peg.
        for entity in actor._objs:
            body = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
            for shape in body.get_collision_shapes():
                shape.set_contact_offset(INSERTION_CONTACT_OFFSET_M)
        return actor

    def fixture(env, inner_radius, outer_radius, depth, center=(0, 0)):
        """Native hook: build both stands and return stand A as the native ``box``."""
        config = env._imitation_config
        env._imitation_native_aperture = float(inner_radius)
        if (
            abs(outer_radius - STAND_HALF_WIDTH_M) > 1e-9
            or abs(depth - STAND_HALF_DEPTH_M) > 1e-9
        ):
            raise ValueError("InsertPeg stands keep the native 10 x 8 cm footprint")
        if not float(env.radius) < config["aperture_half_size_m"]:
            raise ValueError("Insertion aperture must clear the peg")
        env._imitation_aperture = config["aperture_half_size_m"]
        env._imitation_depth = STAND_HALF_DEPTH_M  # hole center to the entry face
        # The native success cue is a red circle on top of the fixture. Keep
        # it visual-only and flush to that surface so it cannot plug the hole.
        marker_builder = env.scene.create_actor_builder()
        marker_builder.add_cylinder_visual(
            pose=sapien.Pose(q=[math.sqrt(0.5), 0, math.sqrt(0.5), 0]),
            radius=0.015,
            half_length=0.001,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
        )
        marker_builder.initial_pose = sapien.Pose([0, 0, -10])
        env._imitation_success_marker = marker_builder.build_kinematic(
            "insertion_success_circle"
        )
        env._imitation_stands = {
            phase: build_stand(env, phase) for phase in INSERT_PHASES
        }
        env._imitation_stand_dimensions = [
            2 * STAND_HALF_DEPTH_M,
            2 * STAND_HALF_WIDTH_M,
            STAND_HEIGHT_M,
        ]
        return env._imitation_stands["demonstration"]

    def activate_stand(env, phase):
        """Put ``phase``'s stand at its table pose and park the other one."""
        env._imitation_active_phase = phase
        env.box = env._imitation_stands[phase]
        for other, stand in env._imitation_stands.items():
            stand.set_pose(
                env._imitation_layouts[other]["pose"]
                if other == phase
                else park_pose(env)
            )

    def load(env, *args, **kwargs):
        env._imitation_config = resolve_config(env.imitation_profile, env.seed, env.level)
        if not env._oracle_wrist_visible_grasp:
            # Archived easier metadata can replay its original grasp/layout.
            env._imitation_config.pop("oracle_grasp_offset_m", None)
        original_load(env, *args, **kwargs)
        # Match the high-friction finger pads on the peg's contact surfaces.
        # This is an ordinary frictional grasp, with no attachment or drive.
        material = sapien.physx.PhysxMaterial(2.0, 2.0, 0.0)
        for peg in env.pegs:
            for link in peg.get_links():
                for body in link._objs:
                    for shape in body.get_collision_shapes():
                        shape.set_physical_material(material)
                        shape.set_contact_offset(INSERTION_CONTACT_OFFSET_M)

    def initialize(env, *args, **kwargs):
        # The native initialization poses ``box``: always start from stand A.
        env.box = env._imitation_stands["demonstration"]
        original_init(env, *args, **kwargs)
        config = env._imitation_config
        env._limit_failed = False
        env._imitation_events = []
        env._imitation_seen = set()
        env._imitation_grasp_reference = None
        env._imitation_grasp_trace = []
        env._imitation_highlight_start = None
        env._imitation_success_marker.set_pose(sapien.Pose([0, 0, -10]))
        env._limit_diagnostics = {"config": config, "events": env._imitation_events}
        # Both stands share one table position (seeded, within the native
        # +-10 cm square, and within the oracle's reach) and the native (left
        # or right) entry face and orientation; each has its own hole position.
        stand_xy = insertion_stand_center(
            env.seed, [config["stands"][p] for p in INSERT_PHASES], env.direction,
            config.get("oracle_grasp_offset_m", INSERT_GRASP_OFFSET_M),
        )
        config["stand_xy_m"] = stand_xy.tolist()
        env._imitation_layouts = {}
        for phase in INSERT_PHASES:
            frame = insertion_stand_frame(
                config["stands"][phase]["yaw_offset_deg"], env.direction
            )
            hole, outward = insertion_hole(
                stand_xy, config["stands"][phase], env.direction
            )
            env._imitation_layouts[phase] = {
                "pose": sapien.Pose([*stand_xy, 0.0], mat2quat(frame)),
                "hole_center": hole,
                "outward": outward,
                "yaw": math.atan2(frame[1, 0], frame[0, 0]),
            }
        activate_stand(env, "demonstration")
        peg_yaws = [
            math.atan2(rotation(peg)[1, 0], rotation(peg)[0, 0]) for peg in env.pegs
        ]
        roots = insertion_peg_roots(
            env.seed,
            stand_xy,
            [env._imitation_layouts[p]["yaw"] for p in INSERT_PHASES],
            peg_yaws,
            config.get("oracle_grasp_offset_m", INSERT_GRASP_OFFSET_M),
        )
        for peg, root in zip(env.pegs, roots):
            set_position(peg, [*root, float(env.radius)])
        config["peg_start_xy_m"] = roots
        env._imitation_initial_poses = [
            sapien.Pose(
                position(p, False), p.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
            )
            for p in env.pegs
        ]
        selected, grasp_end = insertion_choice(env.seed, len(env.pegs))
        config.update(
            selected_peg_index=selected,
            grasp_end=grasp_end,
            available_pegs=len(env.pegs),
        )
        obj_flag = -1 if grasp_end == "head" else 1
        env.obj_flag = obj_flag
        env.peg, env.peg_head, env.peg_tail = (
            env.pegs[selected],
            env.peg_heads[selected],
            env.peg_tails[selected],
        )
        env.grasp_target = env.peg_head if obj_flag == -1 else env.peg_tail
        env.insert_target = env.peg_tail if obj_flag == -1 else env.peg_head
        layouts = env._imitation_layouts
        env._limit_diagnostics.update(
            stands={
                phase: {
                    "center_xyz_m": [*map(float, stand_xy), 0.0],
                    "yaw_deg": math.degrees(layouts[phase]["yaw"]),
                    "yaw_offset_deg": config["stands"][phase]["yaw_offset_deg"],
                    "hole_center_xyz_m": layouts[phase]["hole_center"].tolist(),
                    "hole_lateral_m": config["stands"][phase]["hole_lateral_m"],
                    "hole_height_m": config["stands"][phase]["hole_height_m"],
                    "outward_axis": layouts[phase]["outward"].tolist(),
                }
                for phase in INSERT_PHASES
            },
            stand_swap={
                "parked_stand_pose": "y = -3 m beside the table, on the floor",
                "swap_point": "end of the NO RECORD reset after the demonstration (arm at home)",
                "yaw_difference_deg": abs(
                    config["stands"]["demonstration"]["yaw_offset_deg"]
                    - config["stands"]["execution"]["yaw_offset_deg"]
                ),
                "hole_shift_m": float(
                    np.hypot(
                        config["stands"]["demonstration"]["hole_lateral_m"]
                        - config["stands"]["execution"]["hole_lateral_m"],
                        config["stands"]["demonstration"]["hole_height_m"]
                        - config["stands"]["execution"]["hole_height_m"],
                    )
                ),
            },
            entry_side="left" if env.direction == -1 else "right",
            insertion_motion="horizontal",
            aperture_half_size_m=env._imitation_aperture,
            peg_order=[selected],
            entry_distance_m=env._imitation_depth,
            hole_type="straight square through-hole",
            stand_dimensions_m=env._imitation_stand_dimensions,
            stand_shape="upright rectangular cuboid",
            stand_appearance={
                "base_color_hex": "#FFD289",
                "roughness": 0.5,
                "specular": 0.5,
                "reference": "native InsertPeg build_box_with_hole",
            },
            grasp_motion={
                "mode": "fixed peg-to-gripper transform; wrist rotation",
                "peg_friction": {"static": 2.0, "dynamic": 2.0},
                "artificial_attachment": False,
                "trace": env._imitation_grasp_trace,
            },
            native_aperture_half_size_m=env._imitation_native_aperture,
            success_marker={
                "shape": "red circle",
                "radius_m": 0.015,
                "duration_s": config["success_marker_seconds"],
                "physical_collision": False,
                "placement": "center of the active stand's top face",
                "local_center_xyz_m": [0.0, 0.0, STAND_HEIGHT_M + 0.001],
                "trigger": "each valid insertion in demonstration and execution",
            },
            grasp_end="head" if obj_flag == -1 else "tail",
            scoring="one seeded peg and end, axis alignment, aperture clearance, insertion depth, correct side",
        )

        def ends(i):
            return (
                (env.peg_tails[i], env.peg_heads[i])
                if obj_flag == -1
                else (env.peg_heads[i], env.peg_tails[i])
            )

        def event(kind, i, demo, **data):
            key = (kind, i, demo)
            if key not in env._imitation_seen:
                env._imitation_seen.add(key)
                if kind == "insert":
                    env._imitation_highlight_start = step_number(env)
                env._imitation_events.append(
                    dict(
                        kind=kind,
                        peg_index=i,
                        demonstration=demo,
                        step=step_number(env),
                        **data,
                    )
                )

        def inserted(i, phase):
            tip, grasp = ends(i)
            hole = env._imitation_layouts[phase]
            metrics = insertion_metrics(
                position(tip, False),
                position(grasp, False),
                hole["hole_center"],
                hole["outward"],
            )
            valid = (
                env._imitation_active_phase == phase
                and scalar(env.agent.is_grasping(grasp))
                and insertion_valid(
                    metrics, env._imitation_aperture, env.radius, env._imitation_depth
                )
            )
            env._limit_diagnostics["last_insertion_check"] = dict(
                metrics,
                phase=phase,
                held=scalar(env.agent.is_grasping(grasp)),
                peg_index=i,
                tip_xyz_m=position(tip, False).tolist(),
                grasp_xyz_m=position(grasp, False).tolist(),
            )
            if valid:
                event("insert", i, phase == "demonstration", **metrics)
            return valid

        def wrong_pick():
            return [
                is_A_pickup_notB(env, env.insert_target, env.grasp_target),
                is_any_obj_pickup(
                    env,
                    [
                        end
                        for i in range(len(env.pegs))
                        if i != selected
                        for end in (env.peg_heads[i], env.peg_tails[i])
                    ],
                ),
            ]

        def ik_near(planner, point, orientation, q, threshold=1e-4):
            """IK solution closest to ``q`` (one seed at ``q``, then nine random seeds), or None."""
            mp = planner.planner
            goal = mp.transform_goal_to_wrt_base(
                np.concatenate([point, mat2quat(orientation)])
            )
            for seeds in (1, 10):
                status, solutions = mp.IK(
                    goal, q, n_init_qpos=seeds, threshold=threshold
                )
                if status == "Success":
                    return min(solutions, key=lambda s: np.abs(s[:7] - q[:7]).max())
            return None

        def joint_margin(planner, poses, q):
            """Smallest distance to an arm joint limit over IK solutions for ``poses`` (-1 if one fails)."""
            limits = np.asarray(planner.planner.joint_limits).reshape(-1, 2)[:7]
            margin = np.inf
            for point, orientation in poses:
                solution = ik_near(planner, point, orientation, q, threshold=1e-3)
                if solution is None:
                    return -1.0
                margin = min(
                    margin,
                    float(
                        np.min(
                            np.minimum(
                                solution[:7] - limits[:, 0], limits[:, 1] - solution[:7]
                            )
                        )
                    ),
                )
            return margin

        def pick(e, planner, i, phase):
            tip, grasp = ends(i)
            from transforms3d.euler import euler2mat

            # Easier tiers grip farther toward the selected end and keep the
            # opposite end toward +TCP X, inside the fixed wrist-camera view.
            # Hard retains its legacy grasp point and joint-margin selection.
            # Wrist rotation happens only after the peg is clear of the table.
            visible_grasp = env._oracle_wrist_visible_grasp
            reach = env._imitation_config.get("oracle_grasp_offset_m", INSERT_GRASP_OFFSET_M)
            axis = position(grasp, False) - position(tip, False)
            point = position(grasp, False) + axis / np.linalg.norm(axis) * reach
            # Evaluate both top grasps at the pick and insertion poses. Hard
            # selects by joint margin; easier tiers require the orientation
            # that keeps the opposite end inside the wrist-camera view.
            hole = env._imitation_layouts[phase]
            peg_r = frame_from_axis(
                hole["outward"] if obj_flag == -1 else -hole["outward"]
            )
            tip_p, tip_r = position(tip, False), rotation(tip)
            q_now = env.agent.robot.get_qpos().cpu().numpy()[0].copy()
            options = []
            for flip in (0, 1):
                r = rotation(grasp) @ euler2mat(np.pi, 0, np.pi * flip)
                rel_p, rel_r = tip_r.T @ (point - tip_p), tip_r.T @ r
                poses = [(point, r)] + [
                    (
                        hole["hole_center"] + hole["outward"] * offset + peg_r @ rel_p,
                        peg_r @ rel_r,
                    )
                    for offset in INSERT_TIP_OFFSETS_M
                ]
                options.append((joint_margin(planner, poses, q_now), flip, r))
            if visible_grasp:
                # Head and tail link frames share their X axis. This selects
                # the top grasp with the other half in front of the wrist
                # camera, rather than below the image after a 180-degree flip.
                margin, flip, r = options[int(obj_flag == -1)]
                if margin < 0:
                    raise RuntimeError("No reachable wrist-visible InsertPeg end grasp")
            else:
                margin, flip, r = max(options, key=lambda option: option[0])
            env._limit_diagnostics.setdefault("grasp_choice", []).append(
                {
                    "phase": phase,
                    "flipped": bool(flip),
                    "joint_margins_rad": [o[0] for o in options],
                }
            )
            if visible_grasp:
                env._limit_diagnostics["grasp_choice"][-1].update(
                    wrist_visible=True, grasp_offset_m=reach,
                )
            above = point.copy()
            above[2] = 0.24
            planner.open_gripper()
            move_pose(planner, above, r)
            axis = position(grasp, False) - position(tip, False)
            point = position(grasp, False) + axis / np.linalg.norm(axis) * reach
            move_pose(planner, point, r)
            planner.close_gripper(t=24)
            if not scalar(env.agent.is_grasping(grasp)):
                # Correct an initial miss on the table before lifting. Once
                # lifted, the fingers remain closed for the entire insertion.
                planner.open_gripper()
                point = position(grasp, False).copy()
                if visible_grasp:
                    axis = point - position(tip, False)
                    point += axis / np.linalg.norm(axis) * reach
                r = rotation(grasp) @ euler2mat(np.pi, 0, np.pi * flip)
                move_pose(planner, point, r)
                planner.close_gripper(t=24)
                if not scalar(env.agent.is_grasping(grasp)) and env.level in ("easy", "medium"):
                    # Make one fresh lift/approach attempt after a miss,
                    # preserving the wrist-visible orientation and end offset.
                    # This recovery is never entered by legacy hard episodes.
                    planner.open_gripper()
                    recovery_flip = flip if visible_grasp else 1 - flip
                    r = rotation(grasp) @ euler2mat(np.pi, 0, np.pi * recovery_flip)
                    above = position(grasp, False).copy()
                    above[2] = 0.24
                    move_pose(planner, above, r)
                    axis = position(grasp, False) - position(tip, False)
                    point = position(grasp, False) + axis / np.linalg.norm(axis) * reach
                    r = rotation(grasp) @ euler2mat(np.pi, 0, np.pi * recovery_flip)
                    move_pose(planner, point, r)
                    planner.close_gripper(t=24)
                    env._limit_diagnostics["grasp_choice"][-1]["recovery_flipped"] = bool(recovery_flip)
                if not scalar(env.agent.is_grasping(grasp)):
                    raise RuntimeError(
                        "Could not establish a firm grasp of the selected peg end"
                    )
            # Capture once after closure, before lift or wrist rotation. The
            # peg remains an ordinary dynamic body held by finger friction.
            env._imitation_grasp_reference = relative_tcp(env, tip)
            env._imitation_grasp_reference_tip = tip
            move_pose(planner, above, r)
            planner.close_gripper(t=12)
            if not scalar(env.agent.is_grasping(grasp)):
                raise RuntimeError("Peg lost during lift")

        def insert(e, planner, i, phase):
            tip, grasp = ends(i)
            hole = env._imitation_layouts[phase]
            center, outward = hole["hole_center"], hole["outward"]
            desired_r = frame_from_axis(outward if obj_flag == -1 else -outward)
            rel_p, rel_r = env._imitation_grasp_reference
            tcp_r = desired_r @ rel_r
            # Turn the wrist above the table, keeping the grasp closed. All
            # subsequent waypoints retain exactly the same wrist orientation.
            tcp = position(env.agent.tcp, False).copy()
            velocity = planner.planner.joint_vel_limits.copy()
            acceleration = planner.planner.joint_acc_limits.copy()
            planner.planner.joint_vel_limits = velocity * 0.5
            planner.planner.joint_acc_limits = acceleration * 0.5
            try:
                move_pose(planner, tcp, tcp_r)
            finally:
                planner.planner.joint_vel_limits = velocity
                planner.planner.joint_acc_limits = acceleration
            # Come down beside the stand, on its approach corridor (kept free
            # of distractor pegs), then align and insert.
            for offset, lift in [
                (INSERT_STANDOFF_M, INSERT_LIFT_M),
                (INSERT_STANDOFF_M, 0),
            ]:
                desired_tip = center + outward * offset + [0, 0, lift]
                move_pose(planner, desired_tip + desired_r @ rel_p, tcp_r)
            guided_insert(planner, tip, desired_r, center, outward)
            planner.close_gripper(t=24)
            env._imitation_grasp_reference = None

        def straight_move(planner, point, orientation, step=0.003, settle=3):
            """Move the gripper along a straight line through IK waypoints.

            The screw planner's Jacobian integration drifts by millimetres on
            small turns and long lines; solving IK at every control step
            (``step`` metres or 0.3 degrees apart) keeps the closed grasp on the
            commanded line. Falls back to the screw planner if IK fails.
            """
            q = env.agent.robot.get_qpos().cpu().numpy()[0].copy()
            start_p, start_r = position(env.agent.tcp, False), rotation(env.agent.tcp)
            axis, angle = mat2axangle(orientation @ start_r.T)
            count = int(
                max(
                    1,
                    np.ceil(
                        max(
                            np.linalg.norm(point - start_p) / step,
                            angle / math.radians(0.3),
                        )
                    ),
                )
            )
            path = []
            for k in range(1, count + 1):
                r = axangle2mat(axis, angle * k / count) @ start_r
                solution = ik_near(
                    planner, start_p + (point - start_p) * k / count, r, q
                )
                if solution is None or np.abs(solution[:7] - q[:7]).max() > 0.1:
                    return move_pose(planner, point, orientation)
                q = solution
                path.append(q[:7].copy())
            planner.follow_path(
                {"position": np.array(path), "velocity": np.zeros((len(path), 7))},
                refine_steps=settle,
            )

        def align_tip(
            planner, tip, desired_tip, peg_r, center, outward, command=None, attempts=5
        ):
            """Put the held peg's tip on the hole axis with the hole's orientation.

            Measures the peg pose in the closed fingers and moves the whole
            gripper by the rigid correction (a turn about the tip plus a
            translation); the grasp itself never changes. ``command`` is the
            last commanded gripper pose: the arm's steady tracking offset from
            it is fed back, so repeated corrections converge.
            """
            for _ in range(attempts):
                tip_p = position(tip, False)
                axis, angle = mat2axangle(peg_r @ rotation(tip).T)
                if np.linalg.norm(
                    desired_tip - tip_p
                ) < 0.00015 and angle < math.radians(0.15):
                    break
                # Bound each turn: a peg pivoting in the fingers must not
                # drive the wrist into ever larger corrections.
                turn = axangle2mat(axis, min(angle, math.radians(2)))
                tcp_p, tcp_r = position(env.agent.tcp, False), rotation(env.agent.tcp)
                goal = (desired_tip + turn @ (tcp_p - tip_p), turn @ tcp_r)
                if command is not None:
                    goal = (
                        command[0] + goal[0] - tcp_p,
                        goal[1] @ tcp_r.T @ command[1],
                    )
                straight_move(planner, *goal)
                command = goal
            env._limit_diagnostics.setdefault("oracle_alignment", []).append(
                {
                    "axial_m": float((position(tip, False) - center) @ outward),
                    "tip_error_m": float(
                        np.linalg.norm(desired_tip - position(tip, False))
                    ),
                    "angle_error_deg": float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    (np.trace(peg_r @ rotation(tip).T) - 1) / 2, -1, 1
                                )
                            )
                        )
                    ),
                }
            )
            return command

        def servo_along_axis(
            planner, tracked, line_start, outward, distance, step, command, gain=0.6
        ):
            """Move ``tracked`` (a link) ``distance`` metres along ``outward``, ``step`` per control step.

            While the arm moves, its tracking lag pushes the hand 0.1-0.5 mm
            off the line, as much as the hole's clearance. Each control step
            therefore feeds the measured off-line error of ``tracked`` back
            into the commanded gripper position (integral correction) and
            solves IK for that single step; the motion along the line is
            feed-forward.
            """
            across = frame_from_axis(outward)[:, 1:]
            tcp_p, tcp_r = command
            q = env.agent.robot.get_qpos().cpu().numpy()[0].copy()
            count = max(1, int(np.ceil(abs(distance) / step)))
            errors = []
            for k in range(1, count + 1):
                error = across.T @ (
                    line_start
                    + outward * distance * k / count
                    - position(tracked, False)
                )
                errors.append(float(np.abs(error).max()))
                tcp_p = tcp_p + outward * distance / count + gain * (across @ error)
                solution = ik_near(planner, tcp_p, tcp_r, q)
                if solution is None or np.abs(solution[:7] - q[:7]).max() > 0.1:
                    raise RuntimeError("Insertion servo IK failed")
                q = solution
                planner.follow_path(
                    {"position": q[None, :7].copy(), "velocity": np.zeros((1, 7))}
                )
            env._limit_diagnostics.setdefault("oracle_servo", []).append(
                {
                    "tracked": tracked.name,
                    "distance_m": float(distance),
                    "steps": count,
                    "max_off_axis_error_m": max(errors),
                }
            )
            return tcp_p, tcp_r

        def guided_insert(planner, tip, peg_r, center, outward):
            """Slow, aligned insertion along the hole axis.

            The peg tip is aligned on the axis at a standoff 1 cm outside the
            hole's mouth and again 2 mm before it. The closed gripper then
            servos the tip along the axis until the peg end is 1 cm inside,
            and servos the hand along its own line on to the insertion depth.
            """
            end = env.length / 2  # tip link center to the peg's (visual) end face
            outside = env._imitation_depth + end + 0.010
            entry = env._imitation_depth + end + 0.002
            through = env._imitation_depth + end - 0.010
            insertion_offset = INSERT_TIP_OFFSETS_M[-1]
            command = None
            for offset in (outside, entry):
                command = align_tip(
                    planner,
                    tip,
                    center + outward * offset,
                    peg_r,
                    center,
                    outward,
                    command,
                )
            # 0.4 mm per control step through the mouth, then 1 mm per step.
            command = servo_along_axis(
                planner,
                tip,
                center + outward * entry,
                outward,
                through - entry,
                0.0004,
                command,
            )
            command = servo_along_axis(
                planner,
                env.agent.tcp,
                position(env.agent.tcp, False),
                outward,
                insertion_offset - through,
                0.001,
                command,
            )

        # Upstream RoboMME subgoal wording: near/far end from the robot base, side from the hole.
        base_xy = position(env.agent.robot, False)[:2]
        tip, grasp = ends(selected)
        near = np.linalg.norm(position(grasp, False)[:2] - base_xy) <= np.linalg.norm(
            position(tip, False)[:2] - base_xy
        )
        end, side = (
            "near" if near else "far",
            "left" if env.direction == -1 else "right",
        )

        tasks = []
        for demo in (True, False):
            i = selected
            phase = "demonstration" if demo else "execution"
            tip, grasp = ends(i)
            tasks.append(
                task(
                    f"Pick up the peg by grasping the {end} end",
                    lambda a=grasp, b=tip: is_A_pickup_notB(env, a, b),
                    lambda e, p, i=i, ph=phase: pick(e, p, i, ph),
                    demo=demo,
                    segment=grasp,
                    failure=wrong_pick,
                    failure_reason=["wrong_peg_end", "wrong_peg"],
                    grounded=f"Pick up the peg by grasping the {end} end at <>",
                )
            )
            insert_name = f"Insert the peg from the {side} side" + (
                " of the box" if demo else ""
            )
            tasks.append(
                task(
                    insert_name,
                    lambda i=i, ph=phase: inserted(i, ph),
                    lambda e, p, i=i, ph=phase: insert(e, p, i, ph),
                    demo=demo,
                    segment=env._imitation_stands[phase],
                    failure=wrong_pick,
                    failure_reason=["wrong_peg_end", "wrong_peg"],
                    grounded=insert_name + " at <>",
                )
            )
            if demo:

                def reset(e, planner):
                    env._imitation_resetting = True
                    env._imitation_grasp_reference = None
                    env._imitation_highlight_start = None
                    try:
                        return solve_strong_reset(e, planner, timestep=60)
                    finally:
                        # Swap once the arm is back home, before the first
                        # execution frame: stand A leaves, stand B appears.
                        activate_stand(env, "execution")
                        env._imitation_resetting = False

                tasks.append(
                    task("NO RECORD", lambda: reset_check(env), reset, demo=True)
                )
        env.task_list = tasks
        env._imitation_resetting = False

    def on_execution_start(env):
        # The reset task performs the swap; this guards other reset paths.
        if getattr(env, "_imitation_active_phase", "execution") != "execution":
            activate_stand(env, "execution")

    def step(env, action):
        if getattr(env, "_imitation_resetting", False):
            for peg, pose in zip(env.pegs, env._imitation_initial_poses):
                peg.set_pose(pose)
                peg.set_root_linear_velocity([0, 0, 0])
                peg.set_root_angular_velocity([0, 0, 0])
        return BaseEnv.step(env, action)

    def evaluate(env, solve_complete_eval=False):
        if getattr(env, "_imitation_grasp_reference", None) is not None:
            p, r = relative_tcp(env, env._imitation_grasp_reference_tip)
            p0, r0 = env._imitation_grasp_reference
            angle = np.degrees(np.arccos(np.clip((np.trace(r0.T @ r) - 1) / 2, -1, 1)))
            env._imitation_grasp_trace.append(
                {
                    "step": step_number(env),
                    "demonstration": bool(env.demonstration_record_traj),
                    "translation_slip_m": float(np.linalg.norm(p - p0)),
                    "rotation_slip_deg": float(angle),
                    "held": scalar(env.agent.is_grasping(env.grasp_target)),
                    "stand_contacts": [
                        link.name
                        for link in env.agent.robot.get_links()
                        if float(
                            env.scene.get_pairwise_contact_impulses(link, env.box)
                            .abs()
                            .sum()
                        )
                        > 1e-7
                    ],
                }
            )
        result = evaluate_tasks(env, solve_complete_eval)
        # BaseEnv evaluates before rendering observations, so the success cue
        # appears on the same frame as the insertion, including the last one.
        start = env._imitation_highlight_start
        duration = round(
            env._imitation_config["success_marker_seconds"] / env.control_timestep
        )
        active = start is not None and 0 <= step_number(env) - start < duration
        # Anchor to the active stand's top center, independently of its hole.
        local_center = env._limit_diagnostics["success_marker"]["local_center_xyz_m"]
        marker_pose = (
            sapien.Pose(
                position(env.box, False) + rotation(env.box) @ local_center,
                env.box.pose.sp.q,
            )
            if active
            else sapien.Pose([0, 0, -10])
        )
        env._imitation_success_marker.set_pose(marker_pose)
        return result

    cls._build_insertion_fixture = fixture
    cls._load_scene = load
    cls._initialize_episode = initialize
    cls.evaluate = evaluate
    cls.step = step
    cls.on_execution_start = on_execution_start


def hand_angle(env):
    """World yaw of the hand's local x (short) axis."""
    links = {link.name: link for link in env.agent.robot.get_links()}
    x = rotation(links["panda_hand"])[:2, 0]
    return math.atan2(x[1], x[0])


def trace_rotation(env):
    """RouteStick planning frame: current yaw with the hand upright."""
    return upright_rotation(rotation(env.agent.tcp))


def upright_rotation(frame):
    """Keep world yaw while pointing the straight drawing stick down."""
    return turned(np.diag([1., -1., -1.]), math.atan2(frame[1, 0], frame[0, 0]))


def turned(rotation_matrix, delta):
    """Rotate an orientation about the world vertical by ``delta``."""
    c, s = math.cos(delta), math.sin(delta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rotation_matrix


def row_angle(env):
    """World yaw of the RouteStick target row."""
    targets = [position(t, False)[:2] for t in env.targets_grid]
    d = targets[-1] - targets[0]
    return math.atan2(d[1], d[0])


def nearest_equivalent(angle, reference):
    """``angle`` + k*pi closest to ``reference`` (the hand footprint is symmetric)."""
    return angle + math.pi * round((reference - angle) / math.pi)


WRIST_LIMIT_RAD = 2.8973  # Panda joint 7
UNWIND_LIFT_M = (
    0.035  # tip lift that brings the hand's bottom 1.7 cm above the board tops
)


def route_unwind_lift(env):
    """Preserve hand clearance above taller execution boards."""
    layout = getattr(env, "execution_route_layout", None)
    return UNWIND_LIFT_M + (.1 * (layout["board_height_scale"] - 1) if layout else 0)


def _ik_path(planner, poses, max_joint_step=0.3, wrist_margin=0.15, start=None):
    """Chain IK through ``poses`` (planner goals) for as long as the arm can follow them.

    Starts from ``start`` (joint positions) or the robot's current ones. Stops
    before a pose that has no solution, needs a joint jump, or turns the wrist
    further into the last ``wrist_margin`` before its limit. Returns the joint
    path so far and why it stopped (None when it reached the last pose).
    """
    full = planner.planner.pad_qpos(
        planner.robot.get_qpos().cpu().numpy().reshape(-1).copy()
        if start is None
        else np.asarray(start, dtype=float).copy()
    )
    path = []
    for pose in poses:
        goal = planner._transform_pose_for_planning(pose)
        goal = planner.planner.transform_goal_to_wrt_base(
            np.concatenate([goal.p, goal.q])
        )
        status, solutions = planner.planner.IK(goal, full.copy())
        where = {
            "tip_xyz_m": np.round(np.asarray(pose.p, float), 3).tolist(),
            "wrist_rad": round(float(full[6]), 2),
        }
        if status != "Success" or not len(solutions):
            return path, dict(where, reason="no IK solution")
        collision_free = getattr(planner, "_route_collision_free", None)
        if collision_free is not None:
            # Prefer the continuous arm branch and reject geometry collisions
            # before executing any part of a candidate route.
            candidates = sorted(solutions, key=lambda s: np.linalg.norm(s - full[:len(s)]))
            q = None
            for solution in candidates:
                candidate = full.copy()
                candidate[:len(solution)] = solution
                if np.max(np.abs(candidate[:7] - full[:7])) > max_joint_step:
                    continue
                count = max(1, int(np.ceil(np.max(np.abs(candidate[:7] - full[:7])) / .04)))
                if all(collision_free(full + f * (candidate - full))
                       for f in np.linspace(0, 1, count + 1)[1:]):
                    q = candidate
                    break
            if q is None:
                return path, dict(where, reason="obstacle collision or joint jump")
        else:
            q = full.copy()
            q[: solutions[0].shape[0]] = solutions[0]
        jump = np.abs(q[:7] - full[:7])
        if jump.max() > max_joint_step:
            return path, dict(where, reason=f"joint {int(jump.argmax()) + 1} jump")
        if abs(q[6]) > WRIST_LIMIT_RAD - wrist_margin and abs(q[6]) > abs(full[6]):
            return path, dict(where, reason="wrist limit")
        path.append(q)
        full = q
    return path, None


def _follow(planner, path):
    result = {"status": "Success", "position": np.stack(path)}
    if planner.control_mode == "pd_joint_pos_vel":
        result["velocity"] = np.zeros_like(result["position"])
    return planner.follow_path(result)


def _poses(moves):
    """Planner goals from ``(tip point, orientation matrix)`` pairs."""
    import sapien
    from transforms3d.quaternions import mat2quat

    return [sapien.Pose([float(v) for v in p], mat2quat(r)) for p, r in moves]


def _wrist_room(path):
    return min(WRIST_LIMIT_RAD - abs(q[6]) for q in path) if path else -1.0


def _turn_in_place(tip, r, turn, lift=0.0, steps_deg=5):
    """Lift by ``lift``, turn about the vertical by ``turn``, come back down."""
    up = np.asarray(tip, float) + [0, 0, lift]
    moves = (
        [(tip + f * (up - tip), r) for f in np.linspace(0, 1, 9)[1:]] if lift else []
    )
    count = max(1, int(round(abs(math.degrees(turn)) / steps_deg)))
    moves += [(up, turned(r, turn * f)) for f in np.linspace(0, 1, count + 1)[1:]]
    if lift:
        moves += [
            (up + f * (tip - up), turned(r, turn)) for f in np.linspace(0, 1, 9)[1:]
        ]
    return moves


def _route_moves(env, tip, target, direction, r, start_yaw_offset=0.0, end_yaw_offset=0.0):
    """Half-ellipse tip moves from ``tip`` around the board to ``target``, the hand
    turning with the path from orientation ``r`` (plus a short hold at the end);
    yaw offsets as in ``half_ellipse_route``."""
    xy, yaw = half_ellipse_route(
        tip,
        position(target, False),
        env._imitation_config["route_depth_m"],
        str(direction).lower() == "counterclockwise",
        start_yaw_offset=start_yaw_offset,
        end_yaw_offset=end_yaw_offset,
    )
    moves = [([x, y, tip[2]], turned(r, a)) for (x, y), a in zip(xy, yaw)]
    return moves + moves[-1:] * 5


def _best_plan(planner, options, enough_room=0.6):
    """``options``: per option, lists of moves run one after another. Returns the
    index of the feasible option with the most wrist room, counting at most
    ``enough_room`` (beyond that, earlier options win), and its joint path per
    part, or ``(None, None)``."""
    best = None
    for index, parts in enumerate(options):
        paths, q = [], None
        for moves in parts:
            part, stop = _ik_path(planner, _poses(moves), start=q)
            if stop:
                paths = None
                break
            paths.append(part)
            q = part[-1] if part else q
        if paths is not None:
            key = (
                round(
                    min(_wrist_room([q for part in paths for q in part]), enough_room),
                    1,
                ),
                -index,
            )
            if best is None or key > best[0]:
                best = (key, index, paths)
    return (None, None) if best is None else best[1:]


def _count_unwind(env, kind):
    counts = env._limit_diagnostics.setdefault(
        "route_unwinds", {"at_target": 0}
    )
    counts[kind] += 1


def route_turning(env, planner, target=None, radius=0.1, direction="counterclockwise"):
    """Circle the board between two targets on a half-ellipse, turning the hand with the path.

    The hand arrives at a target with its long side across the target row. On
    the way to the next target, its long side stays along the direction of
    travel (``half_ellipse_route``): away from the row, along the row beside the
    board, and back across the row, so the hand yaw turns gradually by half a
    revolution in the requested direction. At the targets the hand may be
    rotated horizontally from straight across the row by up to 20° (``ROUTE_YAW_OFFSETS_DEG``).
    Before leaving a target, the oracle checks (inverse kinematics, no motion)
    which arrival yaw offset lets the wrist make the turn, preferring small yaw offsets once
    the wrist keeps enough room. If none does, it unwinds at the target: lifts
    the tip 3.5 cm plus any extra board height (the drawing pauses), turns
    the hand by half a revolution plus a yaw adjustment in place, and comes back down.
    Goals are for the stick's real tip; ``planner._transform_pose_for_planning``
    turns them into TCP goals.
    """
    from robomme_plus.tasks.imitation.utils.split_stick import tip_position

    tip, r = tip_position(env).copy(), trace_rotation(env)
    # Trace at the demonstration's drawing height, not wherever the tip settled.
    tip[2] = env._trace_start_pose[0][2]
    if np.linalg.norm(position(target, False)[:2] - tip[:2]) < 1e-6:
        return None
    yaw_offset = math.remainder(hand_angle(env) - row_angle(env), math.pi)  # current yaw offset
    yaw_offsets = np.radians(ROUTE_YAW_OFFSETS_DEG)
    _, paths = _best_plan(
        planner,
        [[_route_moves(env, tip, target, direction, r, yaw_offset, end)] for end in yaw_offsets],
    )
    if paths is None:
        turns = [
            (sign * math.pi + new - yaw_offset, new) for sign in (1, -1) for new in yaw_offsets
        ]
        _, paths = _best_plan(
            planner,
            [
                [
                    _turn_in_place(tip, r, turn, route_unwind_lift(env)),
                    _route_moves(
                        env, tip, target, direction, turned(r, turn), new, end
                    ),
                ]
                for turn, new in turns
                for end in yaw_offsets
            ],
        )
        if paths is not None:
            _count_unwind(env, "at_target")
    if paths is not None:
        return _follow(planner, [q for part in paths for q in part])
    raise RuntimeError("No upright RouteStick route after reorienting at the target")


def start_drawing(env, planner):
    """Move from home to the first drawing point.

    PatternLock goes directly to the short stick's drawing pose.
    Other configurations first approach 10 cm above the start target.
    RouteStick with the short stick then turns the hand in place (still above the
    boards) so its long side lies across the target row, with whichever such yaw
    (offset by at most 20°) leaves the wrist room for the first segment's turn.
    Then the hand moves down until the tip draws (the start target toggles red)."""
    from robomme_plus.tasks.imitation.utils.split_stick import tip_position

    start_p, start_r = env._trace_start_pose
    if env.task_name == "PatternLock":
        # The oracle's planner transform already compensates for the short tip.
        move_pose(planner, start_p, start_r)
        return
    start_r = upright_rotation(start_r)
    above = start_p.copy()
    above[2] += 0.10
    move_pose(planner, above, start_r)
    if not (
        env._imitation_config.get("board_collision", True)
        and getattr(env, "_stick_short", False)
    ):
        move_pose(planner, start_p, start_r)
        return
    tip, r = tip_position(env).copy(), trace_rotation(env)
    current = hand_angle(env)
    wanted = nearest_equivalent(row_angle(env), current)  # short side along the row
    # Move down to the drawing height, then hold a few steps so the arm settles.
    descend = lambda o: (
        [(tip + f * (start_p - tip), o) for f in np.linspace(0, 1, 16)[1:]]
        + [(start_p, o)] * 6
    )
    target, direction = env.selected_buttons[1], env.swing_directions[0]
    options = []
    for yaw_offset in np.radians(ROUTE_YAW_OFFSETS_DEG):
        for half in (0, math.pi, -math.pi):
            turn = wanted - current + half + yaw_offset
            o = turned(r, turn)
            options.append(
                [
                    _turn_in_place(tip, r, turn),
                    descend(o),
                    _route_moves(env, start_p, target, direction, o, yaw_offset),
                ]
            )
    _, paths = _best_plan(planner, options)
    if paths is None:
        if getattr(planner, "_route_collision_free", None) is not None:
            raise RuntimeError("No collision-free approach and first RouteStick segment")
        o = turned(r, wanted - current)
        move_pose(planner, tip, o)
        move_pose(planner, start_p, o)
        return
    # Turn and descend; the first segment is planned again from there.
    _follow(planner, paths[0] + paths[1])


def setup_occlusion(cls):
    original_load, original_init = cls._load_scene, cls._initialize_episode
    original_after = cls._after_simulation_step

    if cls.task_name == "PatternLock":
        from robomme_plus.control.planning import solve_swingonto as original_swing
        from .trace_reference import linear_trace

        def swing(e, planner, target=None, record_swing_qpos=False):
            try:
                value = original_swing(e, planner, target=target, record_swing_qpos=record_swing_qpos)
            except RuntimeError as exc:
                if str(exc) != "Fail to parameterize path":
                    raise
                value = -1
            if isinstance(value, int) and value == -1:
                value = linear_trace(e, planner, target)
            # A pass through the target is not a selection. Keep the final arm
            # pose until the same stationary-dwell rule used by agents accepts it.
            env = e.unwrapped
            node = env.buttons_grid.index(target)
            qpos = env.agent.robot.qpos.cpu().numpy()[0].copy()
            hold = {"position": qpos[None], "velocity": np.zeros_like(qpos)[None]}
            for _ in range(env._pattern_dwell.dwell_steps + 20):
                if node in env._pattern_dwell.accepted or scalar(env.failureflag):
                    break
                value = planner.follow_path(hold)
            if node not in env._pattern_dwell.accepted and not scalar(env.failureflag):
                raise RuntimeError("PatternLock oracle did not settle on its target")
            if record_swing_qpos:
                env.swing_qpos = env.agent.robot.qpos.clone()
            return value

        cls._solve_swingonto = staticmethod(swing)

    def load(env, *args, **kwargs):
        config = resolve_config(env.imitation_profile, env.seed, env.level,
                                stick_change_version=getattr(env, 'stick_change_version', 1),
                                execution_stick_angle_deg=getattr(env, 'execution_stick_angle_deg', None))
        layout = getattr(env, "execution_route_layout", None)
        if cls.task_name == "RouteStick" and layout is not None:
            config["board_collision"] = layout.get("board_collision", True)
            config["obstacle_contact_fails"] = (layout.get("board_contact_fails", True)
                                                 and config["board_collision"])
        env._imitation_config = config
        decision = env._sampling["decision"]
        if cls.task_name == "PatternLock":
            if env.execution_grid_scale != 1:
                config['execution_grid_scale'] = env.execution_grid_scale
            count = config["path_node_count"]
            decision["grid"][env.difficulty] = config["grid_size"]
            decision["path_length_range"][env.difficulty] = [count, count]
            expected_moves = count  # includes a fresh confirmation of the start node
        else:
            count = config["segment_count"]
            decision[env.difficulty]["segment_count_range"] = [count, count]
            expected_moves = count + 1  # Confirm the start before drawing the curves.
            if config.get("target_spacing_m"):
                # Keep the same route spacing at every level; medium/hard
                # need this clearance for the lowered hand to turn.
                env._sampling["positions"]["grid_spacing_y"] = config[
                    "target_spacing_m"
                ]
        original_load(env, *args, **kwargs)
        # The demonstration starts at home, and its first move (to the start
        # target, which toggles red) is recorded; upstream drops it as "NO RECORD".
        first = env.task_list[0]
        if first.get("demonstration") and first.get("name") == "NO RECORD":
            first["name"] = first["subgoal_segment"] = "move to the starting target"
        # Keep the demo's last activated target in its recording. H5 images
        # precede their actions; execution retains its original termination.
        from .tracing import finish_reference_feedback
        last = [entry for entry in env.task_list
                if entry['demonstration'] and entry['name'] != 'NO RECORD'][-1]
        solve = last['solve']
        def finish(wrapped, planner):
            result = solve(wrapped, planner)
            if not (isinstance(result, int) and result == -1):
                finish_reference_feedback(wrapped, planner)
            return result
        last['solve'] = finish
        from robomme_plus.tasks.imitation.utils import (
            split_stick as split_stick,
        )

        env._stick_lower, env._stick_short_tip = split_stick.links(env)
        actual_moves = sum(not entry["demonstration"] for entry in env.task_list)
        if actual_moves != expected_moves:
            raise RuntimeError(
                f"Seeded tracing length mismatch: expected {expected_moves}, got {actual_moves}"
            )
        env._imitation_config = config
        if not config["board"]:
            # Easy: the demonstration is fully visible.
            env._limit_diagnostics = {"config": config, "occluder": None}
            return
        targets = np.array([position(t, False) for t in env.targets_grid])
        eye = np.asarray(env._default_sensor_configs[0].pose.p).reshape(3)
        # Keep the physical board beyond the route's manipulation region, then
        # place it on a camera ray to an interior target.
        target = targets[np.argmin(np.linalg.norm(targets[:, :2], axis=1))]
        x = max(0.20, float(targets[:, 0].max()) + 0.11)
        if x >= eye[0] - 0.02:
            raise ValueError("No room for an occlusion board between camera and route")
        fraction = (x - target[0]) / (eye[0] - target[0])
        ray = target + fraction * (eye - target)
        height = float(ray[2] + 0.035)
        center, size = (
            [x, float(ray[1]), height / 2],
            [0.02, config["board_width_m"], height],
        )
        board = box_actor(env, "imitation_front_board", center, np.asarray(size) / 2)
        env._limit_occluder = board
        env._imitation_board_eye = eye
        env._imitation_board_center = np.array(center)
        env._imitation_config = config
        env._limit_diagnostics = {
            "config": config,
            "occluder": {
                "center_xyz_m": center,
                "size_xyz_m": size,
                "physical_collision": True,
                "present_during": "demonstration only",
                "camera_attached": False,
                "motion": config["board_motion"],
                "tracking": [],
                "target_centers_occluded_by_geometry": [
                    i
                    for i, p in enumerate(targets)
                    if ray_intersects_box(eye, p, center, np.asarray(size) / 2)
                ],
                "minimum_target_x_clearance_m": x - 0.01 - float(targets[:, 0].max()),
            },
        }

    def start_execution(env):
        # The demonstration just ended back at its first drawing pose (``swing_qpos``,
        # the stick on the start target). Execution starts from the home pose, as
        # the demonstration did: the robot moves to the start target itself, and
        # with the stick cut short it must also go lower for the tip to draw (and
        # the target to toggle).
        import torch

        import robomme_plus.control.reset as reset_panda
        from robomme_plus.simulation.choreography import highlight_obj
        from robomme_plus.tasks.imitation.utils import (
            split_stick as split_stick,
        )

        env._trace_start_pose = (
            position(env.agent.tcp, False).copy(),
            rotation(env.agent.tcp),
        )
        prepare_layout = getattr(env, "_prepare_execution_layout", None)
        if callable(prepare_layout):
            prepare_layout()
        fraction = env._imitation_config["execution_stick_fraction"]
        short = fraction < 1
        if short:
            split_stick.set_short(env, True)
        home = reset_panda.get_reset_panda_param("qpos", gripper="stick")
        start = torch.as_tensor(np.asarray(home, dtype=np.float32)).reshape(1, -1)
        env.agent.robot.set_qpos(start)
        env.agent.robot.set_qvel(torch.zeros_like(start))
        env.swing_qpos = start  # the wrapper's initial execution step holds this pose
        # A target the demonstration touched stays red for 40 steps. Clear the
        # demonstration's highlights so execution starts with no target lit.
        env.highlight_starts = {index: -(10**6) for index in env.highlight_starts}
        for button in env.buttons_grid:
            highlight_obj(
                env,
                button,
                start_step=-(10**6),
                end_step=-(10**6) + 40,
                cur_step=step_number(env),
                highlight_color=[1.0, 0.0, 0.0, 1.0],
                use_target_style=True,
            )
        env._limit_diagnostics["stick"] = {
            "kept_fraction": fraction,
            "demonstration_length_m": split_stick.STICK_LENGTH_M,
            "execution_length_m": split_stick.STICK_LENGTH_M * fraction,
            "removed_length_m": split_stick.STICK_LENGTH_M * (1 - fraction),
            "execution_start": "home pose",
            "start_tip_xyz_m": split_stick.tip_position(env).tolist(),
            "drawing_start_tcp_xyz_m": env._trace_start_pose[0].tolist(),
            "mount_tilt_deg": env._imitation_config.get('execution_stick_tilt_deg'),
            "drawing_max_angle_deg": env._imitation_config.get('drawing_max_angle_deg'),
        }
        if short:
            env._limit_diagnostics["stick"]["shortened_at_step"] = step_number(env)
        if cls.task_name == "RouteStick":
            env._route_trace.begin_execution([position(node, False)[:2] for node in env.selected_buttons])
        env._drawing.begin("execution", step_number(env))
        env._trace_feedback_step = None

    def restart_first_route(env):
        # RouteStick judges each segment's side (clockwise or not) from the gripper
        # path recorded since the previous judgement. Execution now starts at home,
        # so until the first segment is judged, the path restarts whenever the tip
        # is on the start target at drawing height: the approach does not count.
        if (
            cls.task_name != "RouteStick"
            or env.demonstration_record_traj
            or getattr(env, "_drawing", None) is None
            or "execution" not in env._drawing.phases
        ):
            return
        first = next(
            (i for i, entry in enumerate(env.task_list)
             if not entry.get("demonstration") and "expected_dir" in entry),
            None,
        )
        if first is None or int(env.timestep) > first:
            return
        from robomme_plus.tasks.imitation.utils.split_stick import (
            tip_position,
        )

        tip, start = tip_position(env), position(env.selected_buttons[0], False)
        if (
            np.linalg.norm(tip[:2] - start[:2]) <= 0.03
            and tip[2] <= env._drawing.height_m
        ):
            env._gripper_xy_trace = []

    def update_stick(env):
        # Drawing rules run in the patched native trail (split_stick.patch_scoring).
        # RouteStick execution: any robot contact with a standing board fails.
        if (
            not env._imitation_config.get("obstacle_contact_fails")
            or not getattr(env, "_stick_short", False)
            or env.demonstration_record_traj
            or getattr(env, "_drawing", None) is None
            or "execution" not in env._drawing.phases
            or env._drawing.failure is not None
        ):
            return
        for board in env.cubes_on_targets:
            for link in env.agent.robot.get_links():
                if (
                    float(
                        env.scene.get_pairwise_contact_impulses(link, board).abs().sum()
                    )
                    > 1e-7
                ):
                    env._drawing.failure = {
                        "reason": "obstacle_contact",
                        "step": step_number(env),
                        "link": link.name,
                        "board": board.name,
                    }
                    return

    def follow_gripper(env):
        restart_first_route(env)
        update_stick(env)
        if not hasattr(env, "_imitation_board_center"):
            return
        tcp = position(env.agent.tcp, False)
        center = env._imitation_board_center.copy()
        center[1] = lateral_occluder_y(env._imitation_board_eye, tcp, center[0])
        visible = bool(env.demonstration_record_traj)
        set_position(env._limit_occluder, center if visible else [0, 0, -10])
        if getattr(env, "_imitation_board_trace_step", -1) != step_number(env):
            env._imitation_board_trace_step = step_number(env)
            env._limit_diagnostics["occluder"]["tracking"].append(
                {
                    "step": step_number(env),
                    "phase": "demonstration"
                    if env.demonstration_record_traj
                    else "execution",
                    "board_y_m": float(position(env._limit_occluder, False)[1]),
                    "visible": visible,
                    "gripper_xyz_m": tcp.tolist(),
                }
            )

    def initialize(env, *args, **kwargs):
        original_init(env, *args, **kwargs)
        from robomme_plus.tasks.imitation.utils import (
            split_stick as split_stick,
        )

        split_stick.set_short(env, False)
        from robomme_plus.tasks.imitation.utils.drawing import Drawing

        env._drawing = Drawing(
            env._imitation_config.get("drawing_height_m"),
            env._imitation_config.get("drawing_continuity_m"),
            env._imitation_config.get("drawing_idle_limit_steps"),
            max_angle_deg=env._imitation_config.get('drawing_max_angle_deg'),
        )
        if cls.task_name == "RouteStick":
            from .route_trace import RouteTrace
            env._route_trace = RouteTrace(
                [position(node, False)[:2] for node in env.selected_buttons],
                env._imitation_config["trace_lateral_ratio"], env._imitation_config["trace_endpoint_margin_m"])
            env._route_curve_tasks = {}
            for phase, demonstration in (("demonstration", True), ("execution", False)):
                entries = [i for i, entry in enumerate(env.task_list)
                           if entry.get("demonstration") == demonstration and "expected_dir" in entry]
                env._route_curve_tasks[phase] = dict(zip(entries, range(len(entries))))
        env._trace_feedback_step = None
        env._trace_completion_started = {}
        env._limit_diagnostics.pop("stick", None)
        if env._limit_diagnostics["occluder"] is None:
            return
        env._limit_diagnostics["occluder"]["tracking"] = []
        env._imitation_board_trace_step = -1
        follow_gripper(env)

    def after(env):
        original_after(env)
        follow_gripper(env)

    original_evaluate = cls.evaluate

    def evaluate(env, solve_complete_eval=False):
        from robomme_plus.tasks.imitation.utils.tracing import final_feedback_ready

        drawing = getattr(env, "_drawing", None)
        phase = drawing.phase if drawing is not None else None
        curve = getattr(env, "_route_curve_tasks", {}).get(phase, {}).get(int(getattr(env, "timestep", -1)))
        result = original_evaluate(env, solve_complete_eval)
        # BaseEnv renders after evaluate(). Updating here includes the final red
        # target and drawing point in the successful observation and recording.
        env._update_trace_feedback()
        trace = getattr(env, "_route_trace", None)
        if trace is not None and drawing is not None and drawing.failure is None:
            state = drawing.phases.get(phase, {})
            dots = state.get("points", [])
            draws = bool(state.get("drawing") and dots and dots[-1][0] == step_number(env))
            from .split_stick import tip_position
            xy = tip_position(env)[:2]
            to_rest = getattr(env, "_route_to_rest", None)
            if callable(to_rest):
                xy = to_rest(xy)
            trace.update(phase, curve, step_number(env), xy, draws)
            if trace.failure is not None:
                drawing.failure = trace.failure
        if scalar(result["success"]) and not final_feedback_ready(env):
            env.successflag = env.successflag.new_tensor([False])
            result = dict(result, success=env.successflag)
        if scalar(result["success"]) and trace is not None:
            trace.finish(step_number(env))
            if drawing.failure is None:
                drawing.failure = trace.failure
        if (
            getattr(env, "_drawing", None) is not None
            and env._drawing.failure is not None
        ):
            import torch
            from robomme_plus.scoring.failures import record_failure

            record_failure(env, env._drawing.failure["reason"])

            env.successflag, env.failureflag = (
                torch.tensor([False]),
                torch.tensor([True]),
            )
            result = dict(result, success=env.successflag, fail=env.failureflag)
        return result

    if cls.task_name == "RouteStick":
        # Execution with the short stick: the lowered hand turns around the boards.
        import importlib

        module = importlib.import_module(
            "robomme_plus.tasks.imitation.route_stick"
        )
        native_route = module.solve_swingonto_withDirection
        if not getattr(native_route, "_turning", False):

            def route(
                env, planner, target=None, radius=0.1, direction="counterclockwise"
            ):
                native = env.unwrapped
                motion = getattr(native, "_route_motion_parameters", None)
                if (motion is not None and motion["mode"] in ("arrival_boards", "departure_boards")
                        and native.demonstration_record_traj
                        and "demonstration" in motion["phases"]):
                    from .route_departure_controller import demonstrate_departure_route
                    return demonstrate_departure_route(native, planner, native_route, target, radius, direction)
                if (motion is not None and motion["mode"] == "crossing_boards"
                        and native.demonstration_record_traj
                        and "demonstration" in motion["phases"]):
                    from .route_crossing_controller import demonstrate_crossing_route
                    return demonstrate_crossing_route(native, planner, native_route, target, radius, direction)
                if (motion is not None and motion["mode"] == "rigid_layout"
                        and native.demonstration_record_traj
                        and "demonstration" in motion["phases"]):
                    from .route_motion import demonstrate_moving_route
                    return demonstrate_moving_route(native, planner, target, radius, direction)
                if getattr(
                    native, "_stick_short", False
                ) and native._imitation_config.get("board_collision", True):
                    return route_turning(
                        native,
                        planner,
                        target=target,
                        radius=radius,
                        direction=direction,
                    )
                return native_route(
                    env, planner, target=target, radius=radius, direction=direction
                )

            route._turning = True
            module.solve_swingonto_withDirection = route

    def get_flag(env):
        return env.__dict__.get("_demonstration_flag", False)

    def set_flag(env, value):
        previous = env.__dict__.get("_demonstration_flag", False)
        env.__dict__["_demonstration_flag"] = value
        if not previous and value and getattr(env, "_drawing", None) is not None:
            env._drawing.begin("demonstration", step_number(env))
            env._demonstration_start_qpos = env.agent.robot.qpos.clone()
        if previous and not value:
            start_execution(env)

    cls._load_scene = load
    cls._initialize_episode = initialize
    cls._after_simulation_step = after
    cls.evaluate = evaluate
    cls.demonstration_record_traj = property(get_flag, set_flag)

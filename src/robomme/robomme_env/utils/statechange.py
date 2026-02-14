from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

#Robomme
import matplotlib.pyplot as plt

import random
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

TARGET_GRAY = (np.array([128, 128, 128, 255]) / 255).tolist()



def _vec3_of(self, actor):
    """Get actor's (x,y,z), compatible with torch / numpy / sapien"""
    p = actor.pose.p if hasattr(actor, "pose") else actor.get_pose().p
    try:
        import torch, numpy as np
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        p = np.asarray(p).reshape(-1)
    except Exception:
        p = [float(p[0]), float(p[1]), float(p[2])]
    return float(p[0]), float(p[1]), float(p[2])


def swap_flat_two_lane(
        self,
        cube_a, cube_b,
        start_step: int, end_step: int,
        cur_step: int,  # ✅ Explicitly pass current timestep
        ax=None, ay=None, bx=None, by=None,
        z=None,
        lane_offset=0.05,
        keep_upright=True,
        smooth=True,
        lock_cube_offset=True,  # Reserved parameter
        other_cube=None,  # New: Extra cube(s), can be single cube or list of cubes, set pose at each timestep to prevent collision
):
    """
    Perform "lane change" swap on same plane (A<->B):
    - Use [start_step, end_step] to control time window;
    - Automatically capture endpoint poses when entering window for first time, fixed for entire animation;
    - Auto no-op outside window; clear cache when reaching end_step.
    - If other_bins provided, keep them in original position at each timestep to prevent collision
    - other_bins can be single bin or list of bins
    """

    import math
    import numpy as np
    import sapien

    # Return directly outside window (allow calling every step)
    if cur_step < int(start_step) or cur_step > int(end_step):
        return

    # Lazy initialize cache container
    if not hasattr(self, "_two_lane_swaps"):
        self._two_lane_swaps = {}
    #print("swap!")
    key = (id(cube_a), id(cube_b), int(start_step), int(end_step))

    # --- Endpoint capture (first frame entering window or cache missing) ---
    if cur_step == int(start_step) or key not in self._two_lane_swaps:
        ax0, ay0, az0 = _vec3_of(self,cube_a)
        bx0, by0, bz0 = _vec3_of(self,cube_b)

        # Capture initial quaternion (keep object's original rotation state)
        qa0 = cube_a.pose.q if hasattr(cube_a, "pose") else cube_a.get_pose().q
        qb0 = cube_b.pose.q if hasattr(cube_b, "pose") else cube_b.get_pose().q

        # Process torch tensor format and ensure quaternion is 1D array
        try:
            import torch
            if isinstance(qa0, torch.Tensor):
                qa0 = qa0.detach().cpu().numpy()
            if isinstance(qb0, torch.Tensor):
                qb0 = qb0.detach().cpu().numpy()

            # Ensure quaternion is 1D array [w, x, y, z]
            qa0 = np.asarray(qa0, dtype=np.float32).flatten()
            qb0 = np.asarray(qb0, dtype=np.float32).flatten()
        except Exception:
            pass

        if ax is None or ay is None or bx is None or by is None:
            ax_c, ay_c, bx_c, by_c = ax0, ay0, bx0, by0
        else:
            ax_c, ay_c, bx_c, by_c = float(ax), float(ay), float(bx), float(by)

        if z is None:
            z_a0, z_b0 = az0, bz0
        else:
            z_a0 = z_b0 = float(z)

        # If other_bins provided, capture initial poses (support single bin or list)
        other_bins_poses = []
        if other_cube is not None:
            # Ensure other_bins is list
            bins_to_lock = other_cube if isinstance(other_cube, list) else [other_cube]

            for bin_obj in bins_to_lock:
                bx0, by0, bz0 = _vec3_of(self, bin_obj)
                qb_lock = bin_obj.pose.q if hasattr(bin_obj, "pose") else bin_obj.get_pose().q

                try:
                    import torch
                    if isinstance(qb_lock, torch.Tensor):
                        qb_lock = qb_lock.detach().cpu().numpy()
                    qb_lock = np.asarray(qb_lock, dtype=np.float32).flatten()
                except Exception:
                    pass

                other_bins_poses.append({
                    "bin": bin_obj,
                    "x": bx0,
                    "y": by0,
                    "z": bz0,
                    "q": qb_lock
                })

        self._two_lane_swaps[key] = {
            "ax": ax_c, "ay": ay_c, "az": z_a0, "qa": qa0,
            "bx": bx_c, "by": by_c, "bz": z_b0, "qb": qb0,
            "other_bins_poses": other_bins_poses
        }

    # Get cached endpoints
    ax = self._two_lane_swaps[key]["ax"]
    ay = self._two_lane_swaps[key]["ay"]
    z_a0 = self._two_lane_swaps[key]["az"]
    qa0 = self._two_lane_swaps[key]["qa"]
    bx = self._two_lane_swaps[key]["bx"]
    by = self._two_lane_swaps[key]["by"]
    z_b0 = self._two_lane_swaps[key]["bz"]
    qb0 = self._two_lane_swaps[key]["qb"]
    other_bins_poses = self._two_lane_swaps[key]["other_bins_poses"]

    # Normalized progress alpha in [0,1]
    denom = max(1, int(end_step) - int(start_step))
    alpha = (int(cur_step) - int(start_step)) / denom
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if smooth:
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

    # Main direction and normal
    dx, dy = (bx - ax), (by - ay)
    nx, ny = -dy, dx
    n_norm = (nx * nx + ny * ny) ** 0.5
    if n_norm > 1e-9:
        nx, ny = nx / n_norm, ny / n_norm
    else:
        nx, ny = 0.0, 0.0  # Start and end points coincide: no normal offset

    # Bell-shaped offset (max at middle)
    offset = float(lane_offset) * math.sin(math.pi * alpha)

    # At each timestep, if other_bins list provided, keep them in original position to prevent collision
    if other_bins_poses:
        for pose_data in other_bins_poses:
            try:
                pose_data["bin"].set_pose(sapien.Pose(
                    p=[pose_data["x"], pose_data["y"], pose_data["z"]],
                    q=pose_data["q"]
                ))
            except Exception as e:
                print(f"Failed to set pose for locked bin: {e}")

    # Check if end time reached
    if int(cur_step) >= int(end_step):
        cube_a.set_pose(sapien.Pose(p=[bx, by, z_a0], q=qb0))
        cube_b.set_pose(sapien.Pose(p=[ax, ay, z_b0], q=qa0))

        # Clear cache
        if key in self._two_lane_swaps:
            del self._two_lane_swaps[key]
        return

    # Interpolation path: A forward + offset; B backward - offset
    xa = ax + dx * alpha + nx * offset
    ya = ay + dy * alpha + ny * offset
    xb = bx - dx * alpha - nx * offset
    yb = by - dy * alpha - ny * offset

    # During swap, quaternions should also swap: A tends to B's rotation, B tends to A's rotation
    # Use slerp for quaternion interpolation (simplified as linear blending)
    try:
        # Calculate current quaternion to use (transition from original rotation to target rotation)
        qa_current = qa0 * (1 - alpha) + qb0 * alpha  # A transitions from qa0 to qb0
        qb_current = qb0 * (1 - alpha) + qa0 * alpha  # B transitions from qb0 to qa0

        # Normalize quaternion
        qa_norm = np.linalg.norm(qa_current)
        qb_norm = np.linalg.norm(qb_current)
        if qa_norm > 1e-6:
            qa_current = qa_current / qa_norm
        if qb_norm > 1e-6:
            qb_current = qb_current / qb_norm

        cube_a.set_pose(sapien.Pose(p=[xa, ya, z_a0], q=qa_current))
        cube_b.set_pose(sapien.Pose(p=[xb, yb, z_b0], q=qb_current))

    except Exception as e:
        # If quaternion interpolation fails, fallback to original scheme
        cube_a.set_pose(sapien.Pose(p=[xa, ya, z_a0], q=qa0))
        cube_b.set_pose(sapien.Pose(p=[xb, yb, z_b0], q=qb0))
    
def highlight_obj(self, obj, start_step: int, end_step: int, cur_step: int,
                     highlight_color=None, disk_radius=0.05,disk_half_length=0.001,
                     use_target_style: bool = False):
        """
        Highlight an object during specified timesteps by adding/removing a visible disk below it.

        Args:
            obj: The object to highlight
            start_step: Start timestep for highlighting
            end_step: End timestep for highlighting
            cur_step: Current timestep
            highlight_color: RGBA color for the disk (default: white)
            disk_radius: Radius of the highlight disk
            use_target_style: When True, draw concentric target-style rings instead of a solid disk
        """
        if highlight_color is None:
            highlight_color = TARGET_GRAY if use_target_style else [1.0, 1.0, 1.0, 1.0]

        # Initialize highlight cache if needed
        if not hasattr(self, "_highlight_cache"):
            self._highlight_cache = {}

        obj_id = id(obj)

        # Initialize cache entry for this object
        if obj_id not in self._highlight_cache:
            self._highlight_cache[obj_id] = {
                'highlight_disk': None,
                'is_highlighted': False,
                'highlight_color_key': None,
            }

        def _to_int(value):
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return 0
                value = value.reshape(-1)[0].item()
            elif isinstance(value, np.ndarray):
                value = float(np.asarray(value).reshape(-1)[0])
            return int(value)

        def _set_visibility(disk_actor, visible: bool):
            try:
                for visual_body in disk_actor.visual_bodies:
                    visual_body.set_visibility(1.0 if visible else 0.0)
            except Exception:
                pass

        def _ensure_disk_position(disk_actor):
            try:
                obj_pos = obj.pose.p
                if hasattr(obj_pos, "cpu"):
                    obj_pos = obj_pos.cpu().numpy()
                elif hasattr(obj_pos, "numpy"):
                    obj_pos = obj_pos.numpy()
                obj_pos = np.asarray(obj_pos, dtype=np.float32).flatten()
                if len(obj_pos) < 3:
                    print(f"Warning: Object position has insufficient coordinates: {obj_pos}")
                    return None
                disk_pos = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) - 0.01]
                disk_actor.set_pose(sapien.Pose(p=disk_pos))
                return disk_pos
            except Exception as e:
                print(f"Failed to update highlight disk position: {e}")
                return None

        def _rgba_from_spec(color_spec):
            if isinstance(color_spec, sapien.render.RenderMaterial):
                color_value = getattr(color_spec, "base_color", None)
                if color_value is None:
                    getter = getattr(color_spec, "get_base_color", None)
                    if callable(getter):
                        color_value = getter()
                if color_value is None:
                    raise ValueError("RenderMaterial missing base_color information")
            else:
                color_value = color_spec

            if isinstance(color_value, torch.Tensor):
                color_value = color_value.detach().cpu().reshape(-1).tolist()
            elif isinstance(color_value, np.ndarray):
                color_value = np.asarray(color_value, dtype=np.float32).reshape(-1).tolist()
            elif isinstance(color_value, (tuple, list)):
                color_value = list(color_value)
            elif isinstance(color_value, str) and color_value.startswith("#"):
                color_value = list(sapien_utils.hex2rgba(color_value))

            if not isinstance(color_value, (list, tuple)) or len(color_value) == 0:
                raise ValueError(f"Unsupported highlight color specification: {color_spec}")

            color_list = [float(x) for x in color_value]
            if len(color_list) == 3:
                color_list.append(1.0)
            elif len(color_list) > 4:
                color_list = color_list[:4]
            return color_list

        def _resolve_material_and_key(color_spec):
            if isinstance(color_spec, sapien.render.RenderMaterial):
                key = None
                base_color_attr = getattr(color_spec, "base_color", None)
                if base_color_attr is not None:
                    try:
                        color_tuple = tuple(float(x) for x in base_color_attr)
                        if len(color_tuple) == 3:
                            color_tuple = (*color_tuple, 1.0)
                        key = ("rgba", color_tuple)
                    except Exception:
                        key = None
                if key is None:
                    maybe_getter = getattr(color_spec, "get_base_color", None)
                    if callable(maybe_getter):
                        try:
                            color_tuple = tuple(float(x) for x in maybe_getter())
                            if len(color_tuple) == 3:
                                color_tuple = (*color_tuple, 1.0)
                            key = ("rgba", color_tuple)
                        except Exception:
                            key = None
                if key is None:
                    key = ("material", id(color_spec))
                return color_spec, key

            color_value = color_spec
            if isinstance(color_value, torch.Tensor):
                color_value = color_value.detach().cpu().reshape(-1).tolist()
            elif isinstance(color_value, np.ndarray):
                color_value = np.asarray(color_value, dtype=np.float32).reshape(-1).tolist()
            elif isinstance(color_value, (tuple, list)):
                color_value = list(color_value)
            elif isinstance(color_value, str) and color_value.startswith("#"):
                color_value = list(sapien_utils.hex2rgba(color_value))

            if not isinstance(color_value, (list, tuple)) or len(color_value) == 0:
                raise ValueError(f"Unsupported highlight color specification: {color_spec}")

            color_list = [float(x) for x in color_value]
            if len(color_list) == 3:
                color_list.append(1.0)
            elif len(color_list) > 4:
                color_list = color_list[:4]

            material = sapien.render.RenderMaterial()
            material.set_base_color(color_list)
            return material, ("rgba", tuple(color_list))

        def _add_target_visuals(builder, orientation, radius, half_length, ring_color):
            target_material = sapien.render.RenderMaterial()
            target_material.set_base_color(ring_color)
            white_material = sapien.render.RenderMaterial()
            white_material.set_base_color([1.0, 1.0, 1.0, 1.0])
            offsets = [0.0, 1e-5, 2e-5, 3e-5, 4e-5]
            radii = [radius, radius * 4 / 5, radius * 3 / 5, radius * 2 / 5, radius * 1 / 5]
            materials = [target_material, white_material, target_material, white_material, target_material]
            for r, offset, mat in zip(radii, offsets, materials):
                builder.add_cylinder_visual(
                    radius=r,
                    half_length=half_length + offset,
                    material=mat,
                    pose=sapien.Pose(q=orientation),
                )

        start_step = _to_int(start_step)
        end_step = _to_int(end_step)
        cur_step = _to_int(cur_step)

        # Check if we should highlight
        should_highlight = start_step <= cur_step <= end_step
        cache_entry = self._highlight_cache[obj_id]
        disk = cache_entry['highlight_disk']
        is_currently_highlighted = cache_entry['is_highlighted']

        # Apply highlighting if needed
        if should_highlight:
            style_key = "target" if use_target_style else "solid"
            ring_color = None
            material = None
            if use_target_style:
                ring_color = _rgba_from_spec(highlight_color)
                color_key = (style_key, tuple(ring_color))
            else:
                material, material_key = _resolve_material_and_key(highlight_color)
                color_key = (style_key, material_key)
            needs_new_actor = False

            if cache_entry.get('highlight_color_key') != color_key and disk is not None:
                if style_key == "solid":
                    updated = False
                    try:
                        for visual_body in getattr(disk, "visual_bodies", []):
                            setter = getattr(visual_body, "set_render_material", None)
                            if callable(setter):
                                setter(material)
                                updated = True
                            else:
                                setter = getattr(visual_body, "set_material", None)
                                if callable(setter):
                                    setter(material)
                                    updated = True
                        if updated:
                            cache_entry['highlight_color_key'] = color_key
                        else:
                            needs_new_actor = True
                    except Exception:
                        needs_new_actor = True
                else:
                    needs_new_actor = True

            if disk is None or needs_new_actor:
                if disk is not None:
                    try:
                        remove_actor = getattr(self.scene, "remove_actor", None)
                        if callable(remove_actor):
                            remove_actor(disk)
                    except Exception:
                        pass
                try:
                    angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
                    disk_orientation = matrix_to_quaternion(
                        euler_angles_to_matrix(angles, convention="XYZ")
                    )
                    builder = self.scene.create_actor_builder()
                    disk_orientation_np = disk_orientation.cpu().numpy()
                    if use_target_style:
                        _add_target_visuals(
                            builder,
                            disk_orientation_np,
                            disk_radius,
                            disk_half_length,
                            ring_color,
                        )
                    else:
                        builder.add_cylinder_visual(
                            radius=disk_radius,
                            half_length=disk_half_length,
                            material=material,
                            pose=sapien.Pose(q=disk_orientation_np),
                        )
                    suffix = cache_entry.get('disk_instance_counter', 0) + 1
                    cache_entry['disk_instance_counter'] = suffix
                    disk_name = f"highlight_disk_{obj_id}_{suffix}"
                    disk = builder.build_kinematic(name=disk_name)
                    cache_entry['highlight_disk'] = disk
                    cache_entry['highlight_color_key'] = color_key
                except Exception as e:
                    print(f"Failed to create highlight disk: {e}")
                    cache_entry['highlight_disk'] = None
                    return
            else:
                cache_entry.setdefault('highlight_color_key', color_key)

            _set_visibility(disk, True)
            _ensure_disk_position(disk)
            cache_entry['is_highlighted'] = True
            cache_entry['highlight_disk'] = disk

        elif is_currently_highlighted and disk is not None:
            # Keep disk but hide it to avoid name conflicts on recreation
            _set_visibility(disk, False)
            try:
                disk.set_pose(sapien.Pose(p=[0.0, 0.0, -10.0]))
            except Exception:
                pass
            cache_entry['is_highlighted'] = False

# def highlight_obj(self, obj, start_step: int, end_step: int, cur_step: int,
#                      highlight_color=None, disk_radius=0.05,disk_half_length=0.001):
#         """
#         Highlight an object during specified timesteps by adding/removing a visible disk below it.

#         Args:
#             obj: The object to highlight
#             start_step: Start timestep for highlighting
#             end_step: End timestep for highlighting
#             cur_step: Current timestep
#             highlight_color: RGBA color for the disk (default: white)
#             disk_radius: Radius of the highlight disk
#         """
#         if highlight_color is None:
#             highlight_color = [1.0, 1.0, 1.0, 1.0]  # White color

#         # Initialize highlight cache if needed
#         if not hasattr(self, "_highlight_cache"):
#             self._highlight_cache = {}

#         obj_id = id(obj)

#         # Initialize cache entry for this object
#         if obj_id not in self._highlight_cache:
#             self._highlight_cache[obj_id] = {
#                 'highlight_disk': None,
#                 'is_highlighted': False,
#                 'highlight_color_key': None,
#             }

#         def _to_int(value):
#             if isinstance(value, torch.Tensor):
#                 if value.numel() == 0:
#                     return 0
#                 value = value.reshape(-1)[0].item()
#             elif isinstance(value, np.ndarray):
#                 value = float(np.asarray(value).reshape(-1)[0])
#             return int(value)

#         def _set_visibility(disk_actor, visible: bool):
#             try:
#                 for visual_body in disk_actor.visual_bodies:
#                     visual_body.set_visibility(1.0 if visible else 0.0)
#             except Exception:
#                 pass

#         def _ensure_disk_position(disk_actor):
#             try:
#                 obj_pos = obj.pose.p
#                 if hasattr(obj_pos, "cpu"):
#                     obj_pos = obj_pos.cpu().numpy()
#                 elif hasattr(obj_pos, "numpy"):
#                     obj_pos = obj_pos.numpy()
#                 obj_pos = np.asarray(obj_pos, dtype=np.float32).flatten()
#                 if len(obj_pos) < 3:
#                     print(f"Warning: Object position has insufficient coordinates: {obj_pos}")
#                     return None
#                 disk_pos = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) - 0.01]
#                 disk_actor.set_pose(sapien.Pose(p=disk_pos))
#                 return disk_pos
#             except Exception as e:
#                 print(f"Failed to update highlight disk position: {e}")
#                 return None

#         def _resolve_material_and_key(color_spec):
#             if isinstance(color_spec, sapien.render.RenderMaterial):
#                 key = None
#                 base_color_attr = getattr(color_spec, "base_color", None)
#                 if base_color_attr is not None:
#                     try:
#                         color_tuple = tuple(float(x) for x in base_color_attr)
#                         if len(color_tuple) == 3:
#                             color_tuple = (*color_tuple, 1.0)
#                         key = ("rgba", color_tuple)
#                     except Exception:
#                         key = None
#                 if key is None:
#                     maybe_getter = getattr(color_spec, "get_base_color", None)
#                     if callable(maybe_getter):
#                         try:
#                             color_tuple = tuple(float(x) for x in maybe_getter())
#                             if len(color_tuple) == 3:
#                                 color_tuple = (*color_tuple, 1.0)
#                             key = ("rgba", color_tuple)
#                         except Exception:
#                             key = None
#                 if key is None:
#                     key = ("material", id(color_spec))
#                 return color_spec, key

#             color_value = color_spec
#             if isinstance(color_value, torch.Tensor):
#                 color_value = color_value.detach().cpu().reshape(-1).tolist()
#             elif isinstance(color_value, np.ndarray):
#                 color_value = np.asarray(color_value, dtype=np.float32).reshape(-1).tolist()
#             elif isinstance(color_value, (tuple, list)):
#                 color_value = list(color_value)
#             elif isinstance(color_value, str) and color_value.startswith("#"):
#                 color_value = list(sapien_utils.hex2rgba(color_value))

#             if not isinstance(color_value, (list, tuple)) or len(color_value) == 0:
#                 raise ValueError(f"Unsupported highlight color specification: {color_spec}")

#             color_list = [float(x) for x in color_value]
#             if len(color_list) == 3:
#                 color_list.append(1.0)
#             elif len(color_list) > 4:
#                 color_list = color_list[:4]

#             material = sapien.render.RenderMaterial()
#             material.set_base_color(color_list)
#             return material, ("rgba", tuple(color_list))

#         start_step = _to_int(start_step)
#         end_step = _to_int(end_step)
#         cur_step = _to_int(cur_step)

#         # Check if we should highlight
#         should_highlight = start_step <= cur_step <= end_step
#         cache_entry = self._highlight_cache[obj_id]
#         disk = cache_entry['highlight_disk']
#         is_currently_highlighted = cache_entry['is_highlighted']

#         # Apply highlighting if needed
#         if should_highlight:
#             material, color_key = _resolve_material_and_key(highlight_color)
#             needs_new_actor = False

#             if cache_entry.get('highlight_color_key') != color_key and disk is not None:
#                 updated = False
#                 try:
#                     for visual_body in getattr(disk, "visual_bodies", []):
#                         setter = getattr(visual_body, "set_render_material", None)
#                         if callable(setter):
#                             setter(material)
#                             updated = True
#                         else:
#                             setter = getattr(visual_body, "set_material", None)
#                             if callable(setter):
#                                 setter(material)
#                                 updated = True
#                     if updated:
#                         cache_entry['highlight_color_key'] = color_key
#                     else:
#                         needs_new_actor = True
#                 except Exception:
#                     needs_new_actor = True

#             if disk is None or needs_new_actor:
#                 if disk is not None:
#                     try:
#                         remove_actor = getattr(self.scene, "remove_actor", None)
#                         if callable(remove_actor):
#                             remove_actor(disk)
#                     except Exception:
#                         pass
#                 try:
#                     angles = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0], dtype=torch.float32))
#                     disk_orientation = matrix_to_quaternion(
#                         euler_angles_to_matrix(angles, convention="XYZ")
#                     )
#                     builder = self.scene.create_actor_builder()
#                     builder.add_cylinder_visual(
#                         radius=disk_radius,
#                         half_length=disk_half_length,
#                         material=material,
#                         pose=sapien.Pose(q=disk_orientation.cpu().numpy()),
#                     )
#                     suffix = cache_entry.get('disk_instance_counter', 0) + 1
#                     cache_entry['disk_instance_counter'] = suffix
#                     disk_name = f"highlight_disk_{obj_id}_{suffix}"
#                     disk = builder.build_kinematic(name=disk_name)
#                     cache_entry['highlight_disk'] = disk
#                     cache_entry['highlight_color_key'] = color_key
#                 except Exception as e:
#                     print(f"Failed to create highlight disk: {e}")
#                     cache_entry['highlight_disk'] = None
#                     return
#             else:
#                 cache_entry.setdefault('highlight_color_key', color_key)

#             _set_visibility(disk, True)
#             _ensure_disk_position(disk)
#             cache_entry['is_highlighted'] = True
#             cache_entry['highlight_disk'] = disk

#         elif is_currently_highlighted and disk is not None:
#             # Keep disk but hide it to avoid name conflicts on recreation
#             _set_visibility(disk, False)
#             try:
#                 disk.set_pose(sapien.Pose(p=[0.0, 0.0, -10.0]))
#             except Exception:
#                 pass
#             cache_entry['is_highlighted'] = False


def highlight_position(
        self,
        position,
        start_step: int,
        end_step: int,
        cur_step: int,
        highlight_color=None,
        disk_radius: float = 0.02,
        disk_half_length: float = 0.01,
):
    """
    Highlight an arbitrary 3D position for a finite timestep window by spawning small spheres.
    Each call can introduce a new highlight that remains visible for the requested duration.
    Note: `disk_radius` now controls the radius of the highlight sphere. `disk_half_length`
    is kept for backward compatibility but is unused.
    """
    if highlight_color is None:
        highlight_color =[1.0, 1.0, 1.0, 1.0]

    # Only draw the highlight when the gripper is near
    if self.agent.tcp.pose.p[0][2] <0.1:
          gripper_close_flag = True
    else:
        gripper_close_flag = False


    if not hasattr(self, "_position_highlight_cache"):
        self._position_highlight_cache = {}
        self._position_highlight_counter = 0

    def _to_int(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0
            value = value.reshape(-1)[0].item()
        elif isinstance(value, np.ndarray):
            value = float(np.asarray(value).reshape(-1)[0])
        return int(value)

    def _to_vec3(value):
        if value is None:
            return np.zeros(3, dtype=np.float32)
        if isinstance(value, sapien.Pose):
            value = value.p
        if hasattr(value, "p"):
            value = value.p
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "tolist"):
            arr = np.asarray(value.tolist(), dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
        arr = arr.reshape(-1)
        if arr.size < 3:
            pad = np.zeros(3, dtype=np.float32)
            pad[: arr.size] = arr
            arr = pad
        return np.asarray(arr[:3], dtype=np.float32)

    def _set_visibility(actor, visible: bool):
        try:
            for visual_body in actor.visual_bodies:
                visual_body.set_visibility(0.1 if visible else 0.0)
        except Exception:
            pass

    def _resolve_material(color_spec):
        if isinstance(color_spec, sapien.render.RenderMaterial):
            return color_spec

        color_value = color_spec
        if isinstance(color_value, torch.Tensor):
            color_value = color_value.detach().cpu().reshape(-1).tolist()
        elif isinstance(color_value, np.ndarray):
            color_value = np.asarray(color_value, dtype=np.float32).reshape(-1).tolist()
        elif isinstance(color_value, (tuple, list)):
            color_value = list(color_value)
        elif isinstance(color_value, str) and color_value.startswith("#"):
            color_value = list(sapien_utils.hex2rgba(color_value))

        if not isinstance(color_value, (list, tuple)) or len(color_value) == 0:
            raise ValueError(f"Unsupported highlight color specification: {color_spec}")

        color_list = [float(x) for x in color_value]
        if len(color_list) == 3:
            color_list.append(1.0)
        elif len(color_list) > 4:
            color_list = color_list[:4]

        material = sapien.render.RenderMaterial()
        material.set_base_color(color_list)
        return material

    def _set_sphere_pose(actor, pos_vec):
        sphere_pos = [
            float(pos_vec[0]),
            float(pos_vec[1]),
            float(0.01),
        ]
        try:
            actor.set_pose(sapien.Pose(p=sphere_pos))
        except Exception:
            pass

    def _teleport_away(actor):
        try:
            actor.set_pose(sapien.Pose(p=[0.0, 0.0, -10.0]))
        except Exception:
            pass

    start_step_i = _to_int(start_step)
    end_step_i = _to_int(end_step)
    cur_step_i = _to_int(cur_step)

    cache = self._position_highlight_cache

    # Clean up expired highlights and update visibility for active ones.
    for key, entry in list(cache.items()):
        actor = entry.get("actor")
        if actor is None:
            cache.pop(key, None)
            continue

        if cur_step_i > entry["end_step"]:
            _set_visibility(actor, False)
            # teleport out of the scene before cleanup to avoid lingering visuals
            _teleport_away(actor)
            try:
                if hasattr(self.scene, "remove_actor"):
                    self.scene.remove_actor(actor)
            except Exception:
                pass
            cache.pop(key, None)
            continue

        if entry["start_step"] <= cur_step_i <= entry["end_step"]:
            _set_visibility(actor, True)
            _set_sphere_pose(actor, entry["position"])
        else:
            _set_visibility(actor, False)

    if not gripper_close_flag:
        # Existing highlight actors are already updated above; just block new spawns.
        return

    if cur_step_i < start_step_i or cur_step_i > end_step_i:
        return

    pos_vec = _to_vec3(position)
    material = _resolve_material(highlight_color)

    builder = self.scene.create_actor_builder()
    builder.add_sphere_visual(
        radius=float(disk_radius),
        material=material,
    )

    self._position_highlight_counter += 1
    actor_name = f"position_highlight_{self._position_highlight_counter}"
    initial_pose = sapien.Pose(
        p=[float(pos_vec[0]), float(pos_vec[1]), float(0.01)]
    )
    builder.set_initial_pose(initial_pose)
    disk_actor = builder.build_kinematic(name=actor_name)

    _set_visibility(disk_actor, True)
    _set_sphere_pose(disk_actor, pos_vec)

    cache[self._position_highlight_counter] = {
        "actor": disk_actor,
        "start_step": start_step_i,
        "end_step": end_step_i,
        "position": pos_vec,
    }

def lift_and_drop_objects_back_to_original(
        self,
        obj,
        start_step: int,
        end_step: int,
        cur_step: int,
):
    """Temporarily move object away within window, and release above original position at midpoint."""

    import numpy as np
    import sapien

    start_step = int(start_step)
    end_step = int(end_step)
    cur_step = int(cur_step)

    if not hasattr(self, "_lift_drop_cache"):
        self._lift_drop_cache = {}

    key = (id(obj), start_step, end_step)

    if cur_step > end_step:
        self._lift_drop_cache.pop(key, None)
        return

    if cur_step < start_step:
        return

    cache = self._lift_drop_cache.get(key)
    if cache is None:
        x0, y0, z0 = _vec3_of(self, obj)
        quat = obj.pose.q if hasattr(obj, "pose") else obj.get_pose().q

        try:
            import torch
            if isinstance(quat, torch.Tensor):
                quat = quat.detach().cpu().numpy()
        except Exception:
            pass

        quat = np.asarray(quat, dtype=np.float32).flatten()

        cache = {
            "origin": np.array([x0, y0, z0], dtype=np.float32),
            "quat": quat,
            "height": 0.0,
            "drop_done": False,
        }

        drop_height = cache["height"]

        # Temporary position away from table: fixed teleport to (10, 10, 10)
        away = np.array([10.0, 10.0, 10.0], dtype=np.float32)

        # Release position: return to directly above origin
        drop_target = cache["origin"].copy()
        drop_target[2] += drop_height

        duration = max(1, end_step - start_step)
        half_window = max(1, duration // 2)
        drop_step = min(end_step, start_step + half_window)

        cache["away_pos"] = away
        cache["drop_pos"] = drop_target
        cache["drop_step"] = drop_step

        self._lift_drop_cache[key] = cache

    def _teleport(target_pos):
        pose = sapien.Pose(p=target_pos.tolist(), q=cache["quat"])
        try:
            obj.set_pose(pose)
            try:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        except Exception as exc:
            print(f"Failed to teleport object: {exc}")

    drop_step = cache["drop_step"]

    if cur_step < drop_step:
        _teleport(cache["away_pos"])
        return

    if cur_step == drop_step:
        _teleport(cache["drop_pos"])
        return

    if cur_step > drop_step:
        self._lift_drop_cache.pop(key, None)


def lift_and_drop_objectA_onto_objectB(
        self,
        obj_a,
        obj_b,
        start_step: int,
        end_step: int,
        cur_step: int,
):
    """
    Teleport obj_a away during [start_step, end_step), then place it on top of obj_b at end_step.

    Args:
        obj_a: Object to be moved
        obj_b: Target object (obj_a will be placed on top of this)
        start_step: Start timestep (obj_a teleported away)
        end_step: End timestep (obj_a placed on obj_b)
        cur_step: Current timestep
        z_offset: Additional height offset above obj_b (default: 0.0)
    """

    import numpy as np
    import sapien

    start_step = int(start_step)
    end_step = int(end_step)
    cur_step = int(cur_step)

    # Initialize cache
    if not hasattr(self, "_lift_drop_onto_cache"):
        self._lift_drop_onto_cache = {}

    key = (id(obj_a), id(obj_b), start_step, end_step)

    # Before window: do nothing
    if cur_step < start_step:
        return

    # After window: clean up cache
    if cur_step > end_step:
        self._lift_drop_onto_cache.pop(key, None)
        return

    # Initialize cache on first entry into window
    cache = self._lift_drop_onto_cache.get(key)
    if cache is None:
        # Capture obj_a's quaternion
        quat_a = obj_a.pose.q if hasattr(obj_a, "pose") else obj_a.get_pose().q

        try:
            import torch
            if isinstance(quat_a, torch.Tensor):
                quat_a = quat_a.detach().cpu().numpy()
        except Exception:
            pass

        quat_a = np.asarray(quat_a, dtype=np.float32).flatten()

        cache = {
            "quat_a": quat_a,
            "away_pos": np.array([10.0, 10.0, 10.0], dtype=np.float32),  # Far away position
            # Preserve original height so we can put it back exactly where it was
            "origin_z": _vec3_of(self, obj_a)[2],
        }
        self._lift_drop_onto_cache[key] = cache

    def _teleport(target_pos, quat):
        """Helper function to teleport object and zero velocities"""
        pose = sapien.Pose(p=target_pos.tolist(), q=quat)
        try:
            obj_a.set_pose(pose)
            try:
                obj_a.set_linear_velocity(np.zeros(3))
                obj_a.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        except Exception as exc:
            print(f"Failed to teleport object: {exc}")

    # During window (before end_step): teleport away
    if cur_step < end_step:
        _teleport(cache["away_pos"], cache["quat_a"])
        return

    # At end_step: place obj_a on top of obj_b
    if cur_step == end_step:
        # Get obj_b's position
        bx, by, bz = _vec3_of(self, obj_b)

        # Calculate obj_a's half size (height)
        obj_a_half_size = getattr(obj_a, "_cube_half_size", 0.02)

        # Get obj_b's dimensions to calculate top surface
        # For bins, the height is typically stored or can be calculated
        if hasattr(obj_b, "_bin_height"):
            obj_b_height = obj_b._bin_height
        else:
            # Default: assume bin height based on cube_half_size * 2.5 (from build_bin)
            obj_b_height = getattr(self, "cube_half_size", 0.02) * 2.5

        # Place obj_a back using its original height (keeps it from sinking into the table)
        target_z = cache["origin_z"]

        target_pos = np.array([bx, by, target_z], dtype=np.float32)
        _teleport(target_pos, cache["quat_a"])

        # Clean up cache
        self._lift_drop_onto_cache.pop(key, None)
        return


def rotate_points_random(points, angle_range, generator=None):
    """
    Generate randomly rotated points

    Args:
        points: torch.Tensor or list, shape (N, 2) - N 2D points
        angle_range: tuple - (min_angle, max_angle) angle range (radians)
        generator: torch.Generator, optional - random number generator

    Returns:
        tuple: (angle, rotated_points) - rotation angle and rotated points
    """
    # Convert to tensor if input is a list
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    min_angle, max_angle = angle_range

    # Generate random rotation angle within specified range
    angle = torch.rand(1, generator=generator) * (max_angle - min_angle) + min_angle

    # Build 2D rotation matrix
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ], dtype=points.dtype).squeeze()

    # Rotate all points
    rotated_points = torch.matmul(points, rotation_matrix.T)

    return angle.item(), rotated_points.tolist()


def move_straight_line(self, cube, start_step, end_step, cur_step, start_pos, end_pos,stop=False):
    """
    Move a cube in a straight line from start_pos to end_pos during [start_step, end_step].

    Args:
        cube: The cube object to move
        start_step: Start timestep for movement
        end_step: End timestep for movement
        cur_step: Current timestep
        start_pos: Starting position [x, y, z] or tuple (x, y, z)
        end_pos: Ending position [x, y, z] or tuple (x, y, z)
    """
    import numpy as np
    import sapien
    if not stop:
        start_step = int(start_step)
        end_step = int(end_step)
        cur_step = int(cur_step)

        # Outside the time window, do nothing
        if cur_step < start_step:
            return

        # Initialize cache for this movement
        if not hasattr(self, "_move_straight_cache"):
            self._move_straight_cache = {}

        key = (id(cube), start_step, end_step)

        if cur_step > end_step:
            self._move_straight_cache.pop(key, None)
            return

        # Initialize cache on first entry
        if cur_step == start_step or key not in self._move_straight_cache:
            # Capture initial quaternion to maintain rotation
            quat = cube.pose.q if hasattr(cube, "pose") else cube.get_pose().q

            try:
                import torch
                if isinstance(quat, torch.Tensor):
                    quat = quat.detach().cpu().numpy()
            except Exception:
                pass

            quat = np.asarray(quat, dtype=np.float32).flatten()

            # Convert positions to numpy arrays
            start = np.array(start_pos, dtype=np.float32)
            end = np.array(end_pos, dtype=np.float32)

            self._move_straight_cache[key] = {
                "start_pos": start,
                "end_pos": end,
                "quat": quat,
            }

        # Get cached values
        cache = self._move_straight_cache[key]
        start = cache["start_pos"]
        end = cache["end_pos"]
        quat = cache["quat"]

        # Calculate interpolation factor alpha in [0, 1]
        duration = max(1, end_step - start_step)
        alpha = (cur_step - start_step) / duration
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Smooth interpolation (smoothstep)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        # Linear interpolation between start and end positions
        current_pos = start + (end - start) * alpha

        # Set the cube's pose
        try:
            cube.set_pose(sapien.Pose(p=current_pos.tolist(), q=quat))
        except Exception as e:
            print(f"Failed to set cube pose in move_straight_line: {e}")

        # Clean up cache if we've reached the end and should stop
        if cur_step >= end_step:
            self._move_straight_cache.pop(key, None)

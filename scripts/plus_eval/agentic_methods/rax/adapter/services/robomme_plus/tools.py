# Derived from RAX LIBERO adapter d49c8ad; benchmark IO adaptation only.
"""RoboMME-Plus MCP tool schemas, host-side request validation and depth-pixel geometry.

Pure Python + numpy: safe to import in the host harness interpreter. The action and state layouts
are the public LIBERO evaluation interface: 7-D OSC_POSE delta rows, and the 8-D proprioceptive
state the public runners give a policy.
"""

from __future__ import annotations

import math

import numpy as np

from services.benchmarks import ROBOMME_PLUS_TOOLS
from services.robomme_plus import tasks as TK
from services.simulator.io import RejectedBeforeAction, reject_unless

STATUS, OBSERVE = "robomme_status", "robomme_observe"
STEP = "robomme_step"
PIXEL_TO_POSITION = "robomme_pixel_to_position"

ROBOT_TOOLS = frozenset({STATUS, OBSERVE, STEP, PIXEL_TO_POSITION})
assert ROBOT_TOOLS == ROBOMME_PLUS_TOOLS, "services/benchmarks.py and the RoboMME-Plus tool set disagree"
MUTATING = frozenset({STEP})
MAX_ACTION_ROWS = 100
FRAME_STRIDE = TK.FRAME_STRIDE
CAMERAS = TK.CAMERAS
# Geometry-only fields removed by the RGB-only profile.
GEOMETRY_KEYS = frozenset({"camera_poses", "camera_intrinsics", *(f"{camera}_depth_m" for camera in CAMERAS)})
CONTROL_TEXT = f"{TK.CONTROL_HZ} Hz, {1000 // TK.CONTROL_HZ} ms per step"


def action_layout():
    """The official action row (RAX normalized deltas translated to RoboMME public ee_pose)."""
    return {"dim": TK.ACTION_DIM, "fields": [
        {"slice": [0, 3], "name": "end_effector_position_delta", "frame": "world",
         "scale": f"1.0 = {TK.POSITION_SCALE_M} m"},
        {"slice": [3, 6], "name": "end_effector_rotation_delta", "frame": "world", "note": "axis-angle",
         "scale": f"1.0 = {TK.ROTATION_SCALE_RAD} rad"},
        {"slice": [6, 7], "name": "gripper", "values": "-1 opens, +1 closes"}],
        "range": [-1.0, 1.0],
        "semantics": "each row is a relative command: the public ee_pose wrapper targets the measured "
                     "end-effector pose plus the scaled delta for one control step"}


def state_layout():
    """The 8-D proprioceptive state the public runners give a policy."""
    return {"dim": TK.STATE_DIM, "fields": [
        {"slice": [0, 3], "name": "end_effector_position_m", "frame": "world"},
        {"slice": [3, 6], "name": "end_effector_orientation_axis_angle_rad", "frame": "world"},
        {"slice": [6, 8], "name": "gripper_finger_positions_m"}]}


def tool_definitions():
    return [
        {"name": STATUS, "description": (
            "Episode status, step budget, the official action and state layouts, cameras and instruction. No physics."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": OBSERVE, "description": (
            f"Current observation: the {TK.IMAGE_SIZE}x{TK.IMAGE_SIZE} RGB images of the third-person and wrist "
            "cameras (upright, as each camera sees) with depth attachments and calibration, the 8-D proprioceptive "
            "state (end-effector position and axis-angle orientation, gripper finger positions) and the "
            "instruction. No physics; the observation is cached after the last motion."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": STEP, "description": (
            f"Execute 1..{MAX_ACTION_ROWS} rows of the official 7-D action, one control step each ({CONTROL_TEXT}): "
            f"[dx, dy, dz, drx, dry, drz, gripper], each in [-1, 1]. Position deltas are scaled by "
            f"{TK.POSITION_SCALE_M} m and rotation deltas (axis-angle, world frame) by {TK.ROTATION_SCALE_RAD} rad; "
            "the controller targets the measured pose plus the scaled delta at every step, so a zero row keeps the "
            "pose approximately. gripper -1 opens, +1 closes. Stops early when the episode ends. Returns the final "
            f"frame and saves every {FRAME_STRIDE}th frame."),
         "inputSchema": {"type": "object", "required": ["actions"], "additionalProperties": False, "properties": {
             "actions": {"type": "array", "minItems": 1, "maxItems": MAX_ACTION_ROWS,
                         "items": {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1},
                                   "minItems": TK.ACTION_DIM, "maxItems": TK.ACTION_DIM}}}}},
        {"name": PIXEL_TO_POSITION, "description": (
            "Back-project 1..32 pixels of the current observation's depth images into world-frame points "
            "(metres), using each camera's calibration. Pass the observation's step_id."),
         "inputSchema": {"type": "object", "required": ["observation_step_id", "selections"],
                         "additionalProperties": False, "properties": {
                             "observation_step_id": {"type": "integer"},
                             "selections": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
                                 "type": "object", "required": ["camera", "u", "v"], "additionalProperties": False,
                                 "properties": {"camera": {"type": "string", "enum": list(CAMERAS)},
                                                "u": {"type": "number"}, "v": {"type": "number"}}}},
                             "depth_window_radius": {"type": "integer", "minimum": 0, "maximum": 5}}}},
    ]


def _only(arguments, allowed):
    unknown = set(arguments) - set(allowed)
    reject_unless(not unknown, f"Unknown arguments: {sorted(unknown)}")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_actions(actions):
    reject_unless(isinstance(actions, list) and 1 <= len(actions) <= MAX_ACTION_ROWS,
                  f"actions must be a list of 1..{MAX_ACTION_ROWS} rows")
    for row in actions:
        reject_unless(isinstance(row, list) and len(row) == TK.ACTION_DIM,
                      f"each action row must have {TK.ACTION_DIM} numbers")
        reject_unless(all(_finite(v) for v in row), "action rows must contain finite numbers")
        # robosuite would clip silently; an out-of-range row is a mistake worth reporting.
        reject_unless(all(-1.0 <= v <= 1.0 for v in row), "action values must be in [-1, 1]")


def validate_request(name, arguments):
    """Complete schema/shape validation before dispatch; raises RejectedBeforeAction."""
    reject_unless(isinstance(arguments, dict), "arguments must be an object")
    if name in (STATUS, OBSERVE):
        _only(arguments, ())
    elif name == STEP:
        _only(arguments, ("actions",))
        validate_actions(arguments.get("actions"))
    elif name == PIXEL_TO_POSITION:
        _only(arguments, ("observation_step_id", "selections", "depth_window_radius"))
        reject_unless(type(arguments.get("observation_step_id")) is int, "observation_step_id must be an integer")
        selections = arguments.get("selections")
        reject_unless(isinstance(selections, list) and 1 <= len(selections) <= 32, "selections must have 1..32 items")
        for item in selections:
            reject_unless(isinstance(item, dict) and set(item) == {"camera", "u", "v"}, "selection needs camera, u, v")
            reject_unless(item["camera"] in CAMERAS, f"unknown camera {item['camera']!r}")
            reject_unless(_finite(item["u"]) and _finite(item["v"]), "pixel coordinates must be finite numbers")
        radius = arguments.get("depth_window_radius", 1)
        reject_unless(type(radius) is int and 0 <= radius <= 5, "depth_window_radius must be 0..5")
    else:
        raise RejectedBeforeAction(f"Unknown RoboMME-Plus tool: {name}")


def quaternion_matrix(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pixel_to_world(depth_m, intrinsics, pose, u, v, radius=1):
    """Back-project one pixel of a linear-depth image into the world frame.

    MuJoCo cameras look down their -Z axis with +Y up (OpenGL); the images are upright (row 0 at the
    top). Depth is distance along the optical axis (robosuite's get_real_depth_map). Uses the median
    valid depth in a (2r+1)^2 window.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape[:2]
    reject_unless(0 <= u < width and 0 <= v < height, f"pixel ({u}, {v}) outside {width}x{height}")
    iu, iv = int(round(u)), int(round(v))
    window = depth[max(0, iv - radius):iv + radius + 1, max(0, iu - radius):iu + radius + 1]
    valid = window[np.isfinite(window) & (window > 0)]
    if valid.size == 0:
        return {"valid": False, "reason": "no valid depth near the pixel"}
    z = float(np.median(valid))
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    point_camera = np.array([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z])
    point = quaternion_matrix(pose["quaternion_wxyz"]) @ point_camera + np.asarray(pose["position_m"], dtype=np.float64)
    return {"valid": True, "position_m": point.tolist(), "depth_m": z, "frame": "world",
            "depth_spread_m": float(valid.max() - valid.min())}


def world_to_pixel(point, intrinsics, pose):
    """Project a world-frame point into a camera (inverse of pixel_to_world)."""
    local = quaternion_matrix(pose["quaternion_wxyz"]).T @ (np.asarray(point, dtype=np.float64)
                                                            - np.asarray(pose["position_m"], dtype=np.float64))
    if local[2] >= 0:
        return None
    z = -local[2]
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    return float(K[0, 0] * local[0] / z + K[0, 2]), float(-K[1, 1] * local[1] / z + K[1, 2]), float(z)

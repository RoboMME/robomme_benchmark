"""Public failure feedback: fixed messages, never private task state or answers."""


MESSAGES = {
    "wrong_cube": "Picked up the wrong cube.",
    "wrong_object": "Picked up the wrong object.",
    "wrong_container": "Picked up the wrong container for the current step.",
    "container_before_putdown": "Picked up another container before putting the previous one down.",
    "multiple_containers": "Picked up more than one container at once.",
    "repeated_container": "Picked up a container that was already completed.",
    "premature_button": "Pressed the button before completing the required actions.",
    "pickup_before_button": "Picked up a cube before pressing the start button.",
    "extra_pickup": "Picked up a cube after completing the required repetitions.",
    "extra_object_pickup": "Picked up an object after completing the required repetitions.",
    "too_many_placements": "Completed more pick-and-place cycles than requested.",
    "too_many_swings": "Completed more swings than requested.",
    "wrong_swing_order": "Visited the swing targets in the wrong order.",
    "wrong_color_or_too_many": "Put an unrequested or extra cube into the bin.",
    "distractor_in_bin": "Put an unrequested cube into the bin.",
    "incorrect_bin_count": "The bin contains the wrong number of cubes.",
    "extra_cube_in_bin": "Put an extra cube into the bin.",
    "insufficient_remaining_target_cubes": "Too many required cubes left the conveyor uncollected.",
    "stopped_wrong_pass": "Stopped the cube on the wrong pass.",
    "stopped_off_target": "Stopped the cube away from the target.",
    "missed_requested_pass": "Missed the requested stopping pass.",
    "repeated_button_press": "Pressed the same button more than once.",
    "button_deadline_missed": "Did not press all required buttons before the deadline.",
    "early_container_opening": "Exposed a cube before both buttons were pressed.",
    "wrong_placement": "Placed the cube outside the required target.",
    "incomplete_highlights": "Finished before picking up every highlighted cube.",
    "wrong_peg": "Picked up the wrong peg.",
    "wrong_peg_end": "Grasped the wrong end of the peg.",
    "wrong_insertion_end": "Inserted the wrong end of the peg.",
    "wrong_insertion_side": "Inserted the peg from the wrong side of the box.",
    "wrong_drawing_target": "Touched the wrong drawing target.",
    "wrong_endpoint_stop": "Stopped at the wrong drawing target.",
    "wrong_drawing_pattern": "The drawn pattern does not match the demonstration.",
    "wrong_route_direction": "Drew a path segment in the wrong direction.",
    "invalid_route_segment": "The drawn segment has no valid route direction.",
    "idle": "Stopped drawing for too long.",
    "discontinuous": "The drawn path is not continuous.",
    "trace_lateral_extent": "A white-trace curve extends too far from the line between its targets.",
    "trace_endpoint_overshoot": "The white trace extends too far past a curve's target endpoints.",
    "trace_reference_missing": "A curve has no valid demonstrated white trace.",
    "trace_incomplete": "A required white-trace curve was not drawn.",
    "obstacle_contact": "The robot or drawing stick touched an obstacle.",
    "robot_or_gripper_touched_cube": "The robot or gripper directly touched the cube.",
    "robot_grasped_cube": "Grasped the cube directly with the gripper.",
    "illegal_tool_surface_touched_cube": "An unpermitted part of the tool touched the cube.",
    "cube_lifted_for_non_scoop_method": "Lifted the cube during a method that requires it to stay on the table.",
    "target_moved": "Moved the target.",
    "fixed_fixture_moved": "Moved a fixed fixture.",
    "cube_state_discontinuity": "The cube moved discontinuously.",
    "another_tool_was_used": "Used a tool that does not match the demonstration.",
    "another_tool_was_moved": "Moved a tool that is not part of the demonstrated action.",
    "cube_moved_by_another_tool": "Moved the cube with a different tool from the demonstration.",
    "cue_exceeded_joint_limits": "Moved the cue beyond its allowed range.",
    "poke_shaft_contact_too_long": "The tool shaft touched the cube for too long.",
    "non_head_contact_too_long": "The tool shaft or handle touched the cube for more than 0.5 seconds without separating.",
    "pan_touched_target_area": "The pan touched the target area at table height.",
    "pan_pushed_cube_on_table_after_carry": "Pushed the cube along the table with the pan after carrying it.",
    "hook_final_approach_used_non_inner_face": "Used an unpermitted tool face for the final approach.",
    "ball_launched_instead_of_released": "Threw the ball instead of releasing it.",
    "held_ball_pushed_cube": "Pushed the cube while still holding the ball.",
    "robot_touched_ball_again_after_release": "Touched the ball again after releasing it.",
    "settled_shot_missed_target": "The shot settled without reaching the target.",
    "required_method_not_available": "The scene does not support the required action.",
    "failure_check_error": "The simulator could not complete a task-rule check.",
}


def record_failure(env, code):
    """Latch the first recognized cause, without changing task scoring."""
    if isinstance(code, str) and code in MESSAGES and getattr(env, "_failure_code", None) is None:
        env._failure_code = code


def clear_failure(env):
    env._failure_code = None


def failure_message(env):
    """Unknown causes get a truthful fallback, never internal strings or reprs."""
    code = getattr(env, "_failure_code", None)
    return MESSAGES.get(code if isinstance(code, str) else None, "A task rule was violated.")


def record_predicate_failure(env, reasons, result, triggered):
    """Select the reason for an already evaluated predicate; never rerun it."""
    if isinstance(reasons, str):
        if triggered(result):
            record_failure(env, reasons)
    elif isinstance(reasons, (list, tuple)) and isinstance(result, (list, tuple)):
        for reason, value in zip(reasons, result):
            if triggered(value):
                record_predicate_failure(env, reason, value, triggered)
                break

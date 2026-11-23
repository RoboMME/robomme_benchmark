"""Build evaluation episodes from supplied benchmark metadata."""

ACTION_SPACES = ("joint_angle", "ee_pose", "waypoint")
OBSERVATION_OPTIONS = frozenset(
    {
        "include_front_depth",
        "include_wrist_depth",
        "include_front_camera_extrinsic",
        "include_wrist_camera_extrinsic",
        "include_front_camera_intrinsic",
        "include_wrist_camera_intrinsic",
    }
)


def make_episode(
    task,
    *,
    seed,
    difficulty,
    variant=None,
    native_difficulty=None,
    episode_metadata=None,
    action_space="waypoint",
    gui_render=False,
    perceptual_variations=True,
    mechanism_variations=True,
    **observation_options,
):
    """Build a task lazily using its supplied execution budget."""
    if episode_metadata is None:
        raise ValueError("Evaluation requires supplied episode metadata")
    if type(perceptual_variations) is not bool:
        raise TypeError("perceptual_variations must be a boolean")
    if type(mechanism_variations) is not bool:
        raise TypeError("mechanism_variations must be a boolean")
    if action_space not in ACTION_SPACES:
        raise ValueError(f"action_space must be one of {ACTION_SPACES}")
    unknown = set(observation_options) - OBSERVATION_OPTIONS
    if unknown:
        raise TypeError(f"Unsupported observation options: {sorted(unknown)}")
    from robomme_plus.evaluation.metadata import normalize_episode

    index = episode_metadata.get("episode", 0) if isinstance(episode_metadata, dict) else 0
    episode_metadata = normalize_episode(task, episode_metadata, index)
    from robomme_plus.variation.mechanism import effective_metadata
    episode_metadata = effective_metadata(episode_metadata, mechanism_variations)
    for name, supplied in (("seed", seed), ("difficulty", difficulty),
                           ("variant", variant), ("native_difficulty", native_difficulty)):
        # Historical InsertPeg seed-only records denote the retained hard tier.
        if name == "difficulty" and task == "InsertPeg" and supplied is None:
            supplied = "hard"
        if name in ("variant", "native_difficulty") and supplied is None:
            continue
        if supplied != episode_metadata.get(name):
            raise ValueError(f"{name} differs from the supplied episode metadata")
    difficulty = episode_metadata["difficulty"]
    variant = episode_metadata.get("variant")
    native_difficulty = episode_metadata.get("native_difficulty")
    import gymnasium as gym

    from robomme_plus import register_envs
    from robomme_plus.registry import get_task
    from robomme_plus.evaluation.demonstration import DemonstrationWrapper
    from robomme_plus.control.ee_pose import EndeffectorDemonstrationWrapper
    from robomme_plus.evaluation.failure import FailAwareWrapper
    from robomme_plus.evaluation.reset import ResetRetryWrapper
    from robomme_plus.control.waypoint import MultiStepDemonstrationWrapper

    spec = get_task(task)
    register_envs(task)
    options = {"variant": variant} if spec.variants else {}
    if perceptual_variations and episode_metadata.get("perceptual_variation") is not None:
        options["_perceptual_variation"] = episode_metadata["perceptual_variation"]
    if task == 'ButtonUnmaskSwap':
        options['continuous_motion'] = episode_metadata.get('continuous_motion', False)
        options['motion_workspace'] = episode_metadata.get('motion_workspace', 'legacy')
        if options['continuous_motion']:
            options['continuous_motion_version'] = episode_metadata.get('continuous_motion_version', 1)
    if task == "InsertPeg":
        options["_oracle_wrist_visible_grasp"] = episode_metadata.get("oracle_wrist_visible_grasp", False)
    if task in ('PatternLock', 'RouteStick'):
        options['stick_change_version'] = episode_metadata.get('stick_change_version', 1)
    if task == 'RouteStick':
        if any(key in episode_metadata for key in ('execution_stick_angle_deg', 'execution_stick_tilt_deg')):
            raise ValueError('RouteStick only supports a straight stick; remove stick angle metadata')
        options['execution_route_layout'] = episode_metadata.get('execution_route_layout')
    if task == 'PatternLock':
        options['execution_stick_angle_deg'] = episode_metadata.get('execution_stick_angle_deg')
        options['execution_grid_scale'] = episode_metadata.get('execution_grid_scale')
    if task == "MoveCube" and episode_metadata is not None:
        if "scene_parameters" not in episode_metadata:
            raise ValueError("MoveCube evaluation requires scene_parameters in the supplied metadata")
        options["_scene_parameters"] = episode_metadata["scene_parameters"]
        from robomme_plus.evaluation.timing import episode_control_frequency
        options["sim_config"] = {"control_freq": episode_control_frequency(task, episode_metadata)}
        from robomme_plus.evaluation.demonstration_actions import load_demonstration_actions
        demonstration_actions = load_demonstration_actions(episode_metadata)
        if demonstration_actions is not None:
            options["_demonstration_actions"] = demonstration_actions
    if task == "PickHighlight":
        if native_difficulty is not None:
            options["_native_difficulty"] = native_difficulty
        options["highlight_mode"] = episode_metadata.get("highlight_mode", "staggered")
        options["layout_version"] = episode_metadata.get("layout_version", 1)
        options["local_adjustments"] = episode_metadata.get("local_adjustments", False)
        options["color_mode"] = episode_metadata.get("color_mode", "independent")
    control_mode = "pd_joint_pos"
    if (task == "MoveCube" and options.get("_scene_parameters", {}).get("schema")
            in ("robomme-ood-move-cube/2", "robomme-ood-move-cube/3")):
        control_mode = "pd_joint_pos_vel"
    if mechanism_variations and episode_metadata.get("mechanism_variation") is not None:
        from robomme_plus.variation.mechanism.runtime import make_env
        def construct(**kwargs):
            return make_env(spec, profile=episode_metadata["mechanism_variation"], **kwargs)
    else:
        def construct(**kwargs):
            return gym.make(spec.gym_id, **kwargs)
    def build():
        env = construct(
            seed=seed,
            difficulty=difficulty,
            **options,
            obs_mode="rgb+depth+segmentation",
            control_mode=control_mode,
            render_mode="rgb_array",
            reward_mode="dense",
        )
        try:
            env = DemonstrationWrapper(
                env,
                episode_metadata=episode_metadata,
                gui_render=gui_render,
                **observation_options,
            )
            if action_space == "ee_pose":
                env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
            elif action_space == "waypoint":
                env = MultiStepDemonstrationWrapper(
                    env, gui_render=gui_render, vis=gui_render
                )
            return env
        except Exception:
            env.close()
            raise

    return FailAwareWrapper(ResetRetryWrapper(build(), rebuild=build))

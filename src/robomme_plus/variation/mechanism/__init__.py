"""Opt-in task mechanics for marked test episodes and atomic training examples."""
import copy

TASKS = frozenset(('BinFill', 'ButtonUnmask', 'ButtonUnmaskSwap', 'PatternLock',
    'PickXtimes', 'StopCube', 'SwingXtimes', 'VideoPlaceButton', 'VideoPlaceOrder',
    'VideoRepick', 'VideoUnmask', 'VideoUnmaskSwap', 'RouteStick'))


def validate_profile(task, difficulty, profile, perceptual_profile=None):
    if profile is None:
        return
    if isinstance(profile,dict) and 'atomic' in profile:
        expected={'RouteStick':'longer_activation','ButtonUnmaskSwap':'container_speed_2x'}
        if (set(profile)!={'version','atomic'} or type(profile.get('version')) is not int
                or profile['version']!=1 or task not in expected
                or profile['atomic']!=expected[task] or difficulty not in ('easy','medium','hard')):
            raise ValueError('Invalid atomic mechanism variation for this task')
        if perceptual_profile is not None:
            raise ValueError('Perceptual and mechanism variation selections must be disjoint')
        return
    if (task not in TASKS or difficulty != 'hard' or not isinstance(profile, dict)
            or type(profile.get('version')) is not int or profile['version'] != 1):
        raise ValueError('mechanism_variation requires version 1 and a supported hard task')
    if perceptual_profile is not None:
        raise ValueError('Perceptual and mechanism variation selections must be disjoint')
    if task in ('VideoRepick', 'VideoUnmaskSwap'):
        allowed = {'VideoRepick': ('VideoUnmaskSwap', 'ButtonUnmaskSwap'),
                   'VideoUnmaskSwap': ('VideoRepick', 'ButtonUnmaskSwap')}
        if profile.get('source_motion_task') not in allowed[task]:
            raise ValueError('mechanism_variation requires a supported source_motion_task')
    if task == 'StopCube':
        layout = profile.get('layout')
        required = {'shape', 'axes', 'half_size_m', 'button_xy', 'route_center_xy',
                    'target_xy', 'rotation_degrees', 'original_layout',
                    'minimum_button_path_center_distance_m', 'expert_countable_views', 'expert_target_views'}
        if not isinstance(layout, dict) or not required <= layout.keys():
            raise ValueError('StopCube mechanism_variation requires its validated layout')


def effective_metadata(record, enabled):
    """Disabling mechanics also restores the original episode's time budget."""
    if type(enabled) is not bool:
        raise TypeError('mechanism_variations must be a boolean')
    result = copy.deepcopy(record)
    if result.get('mechanism_variation') is not None and not enabled:
        from robomme_plus.registry import episode_budget
        result['oracle_policy_step'] = result['base_oracle_policy_step']
        result['max_steps'] = episode_budget(result['oracle_policy_step'])
    return result

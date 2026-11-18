"""Explicit single-factor appearance profiles for selected training episodes."""
import copy

# Object recoloring is one factor; the board/container preset recolors only
# those two object roles as specified for VideoUnmaskSwap training.
PRESETS = {
    'light_dim': dict(lighting='dim'),
    'light_disco': dict(lighting='disco'),
    'light_ring_pattern': dict(lighting='ring_pattern', white_rings=True),
    'light_moving_patches': dict(lighting='moving_patches'),
    'light_static_color': dict(lighting='static_color'),
    'light_overexposure': dict(lighting='overexposure'),
    'table_marble': dict(table='marble'),
    'table_light_oak': dict(table='light_oak'),
    'table_steel': dict(table='steel'),
    'ring_colors': dict(object_groups=['rings']),
    # Dark navy / warm ivory is outside the saturated, medium-bright test pool.
    'button_colors': dict(object_groups=['button_body', 'button_cap'],
                          object_colors={'button_body': [.025, .055, .16],
                                         'button_cap': [.96, .92, .80]}),
    'board_light_gray_containers': dict(object_groups=['board_or_bin', 'container'],
                                       container_color=[.75, .75, .75]),
}
TASK_PRESETS = {
    'BinFill': 'light_dim', 'PickXtimes': 'table_marble', 'StopCube': 'ring_colors',
    'ButtonUnmask': 'button_colors', 'SwingXtimes': 'light_disco',
    'VideoUnmask': 'light_ring_pattern', 'VideoUnmaskSwap': 'board_light_gray_containers',
    'PickHighlight': 'table_light_oak', 'VideoRepick': 'light_moving_patches',
    'VideoPlaceButton': 'ring_colors', 'PatternLock': 'table_steel',
    'VideoPlaceOrder': 'light_static_color', 'InsertPeg': 'light_overexposure',
}


def validate(task, profile):
    if (not isinstance(profile, dict) or set(profile) != {'version', 'atomic'}
            or type(profile.get('version')) is not int or profile['version'] != 1
            or task not in TASK_PRESETS or profile.get('atomic') != TASK_PRESETS[task]):
        raise ValueError('Invalid atomic perceptual variation for this task')


def configuration(task, seed, profile):
    validate(task, profile)
    return dict(version=1, profile=copy.deepcopy(profile), seed=int(seed),
                **(dict(table='wood', lighting='normal', object_groups=[])
                   | copy.deepcopy(PRESETS[profile['atomic']])))

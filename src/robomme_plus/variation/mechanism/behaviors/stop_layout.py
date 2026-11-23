"""Apply the seed's collision-clear, expert-visible StopCube layout."""
import numpy as np


def relocate(route, button, half_size, record):
    if record['shape'] != route.shape or not np.allclose(record['axes'], route.axes):
        raise ValueError('Mechanism layout does not match the seeded StopCube route')
    if not np.isclose(record['half_size_m'], half_size):
        raise ValueError('Mechanism layout does not match the StopCube size')
    route.center = tuple(record['route_center_xy'])
    route.rotation_degrees = record['rotation_degrees']
    angle = np.deg2rad(route.rotation_degrees)
    route.rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if not np.allclose(route.target, record['target_xy']):
        raise ValueError('Mechanism layout does not match the StopCube target')
    audit = dict(record['original_layout'])
    audit.update(button_xy=record['button_xy'], target_xy=record['target_xy'],
        route_center_xy=record['route_center_xy'],
        minimum_button_path_center_distance_m=record['minimum_button_path_center_distance_m'],
        button_to_target_m=float(np.linalg.norm(np.asarray(record['target_xy']) - record['button_xy'])),
        button_moved=not np.allclose(button, record['button_xy']),
        button_target_y_offset_m=record.get('button_target_y_offset_m'),
        expert_preparation_height_m=.15, expert_countable_views=record['expert_countable_views'],
        expert_target_views=record['expert_target_views'], expert_visibility_selected=True)
    return np.asarray(record['button_xy']), audit

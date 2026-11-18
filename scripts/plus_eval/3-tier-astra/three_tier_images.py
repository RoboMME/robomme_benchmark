"""Count three-tier planner attachments, whose sheets contain 16 video frames."""
import math

from astra_image_sampling import MAX_DEMONSTRATION_IMAGES, MAX_OBSERVATION_IMAGES

FRAMES_PER_SHEET = 16


def demonstration_indices(count, *, initial_memory_frames=0):
    reserved = 2 + math.ceil(initial_memory_frames / FRAMES_PER_SHEET)
    capacity = (MAX_DEMONSTRATION_IMAGES - reserved) * FRAMES_PER_SHEET
    if count < 0 or initial_memory_frames < 0 or capacity < 2:
        raise ValueError('Invalid three-tier demonstration budget')
    selected = min(count, capacity)
    indices = (list(range(count)) if count <= capacity else
               [i * (count - 1) // (selected - 1) for i in range(selected)])
    return indices, {'source_frames': count, 'selected_frames': selected,
                     'source_demo_images': math.ceil(count / FRAMES_PER_SHEET),
                     'input_demo_images': math.ceil(selected / FRAMES_PER_SHEET),
                     'input_images': reserved + math.ceil(selected / FRAMES_PER_SHEET),
                     'max_initial_images': MAX_DEMONSTRATION_IMAGES,
                     'frames_per_sheet': FRAMES_PER_SHEET,
                     'applied': count > capacity, 'sampling': 'uniform_with_endpoints',
                     'source_frame_indices': indices}


def is_image_limit_error(detail):
    return isinstance(detail, dict) and 'max_images_per_request' in (detail.get('type'), detail.get('code'))


def request_image_budget():
    return {'max_request_images': MAX_OBSERVATION_IMAGES,
            'scope': 'independent_planner_request', 'submitted_requests': 0,
            'submitted_images': 0, 'largest_request_images': 0, 'stopped': False}

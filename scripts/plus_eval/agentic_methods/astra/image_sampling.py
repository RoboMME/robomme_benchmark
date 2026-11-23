"""Astra image selection; its demonstration budget counts actual API images."""
import math


MAX_OBSERVATION_IMAGES = 1500
MAX_DEMONSTRATION_IMAGES = 750
MAX_ONE_SHOT_CONTEXT_IMAGES = 1100
CONFIGURATION = {
    "max_initial_images": MAX_DEMONSTRATION_IMAGES,
    "includes_current_wrist": True,
    "sampling": "uniform_with_endpoints",
    "scope": "initial_demonstration_request",
}


def observation_indices(count, frame_stride, *, first=False, max_demo_images=MAX_DEMONSTRATION_IMAGES):
    if count < 1 or frame_stride < 1:
        raise ValueError("Observation count and frame stride must be positive")
    if not 3 <= max_demo_images <= MAX_OBSERVATION_IMAGES:
        raise ValueError("Demonstration image limit must be between 3 and 1500")
    # Preserve the existing per-observation guard for execution updates.
    stride = max(frame_stride, math.ceil((count - 1) / (MAX_OBSERVATION_IMAGES - 2)))
    indices = sorted(set(range(0, count, stride)) | {count - 1})
    source_images = len(indices) + 1  # Current wrist view accompanies front frames.
    applied = first and source_images > max_demo_images
    if applied:
        front_count = max_demo_images - 1
        indices = [i * (count - 1) // (front_count - 1) for i in range(front_count)]
    report = None
    if first:
        report = {
            **CONFIGURATION,
            "max_initial_images": max_demo_images,
            "source_observation_frames": count,
            "source_images": source_images,
            "input_images": len(indices) + 1,
            "applied": applied,
            "base_frame_stride": frame_stride,
            "base_effective_stride": stride,
            "front_frame_indices": indices,
        }
    return indices, stride, report


def image_count(value):
    """Count recorded API image blocks, including nested message content."""
    if isinstance(value, dict):
        return int(value.get("type") == "input_image") + sum(image_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(image_count(v) for v in value)
    return 0

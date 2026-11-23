"""Limit FrameSamp+modul demonstrations while retaining their full time span."""
import numpy as np


VIDEO_FPS = 20
CAPPED_METHOD = "framesamp-modul"
CAPPED_METHODS = frozenset((CAPPED_METHOD, "framesamp-modul-plus", "framesamp-modul-ood"))
MAX_DEMONSTRATION_SECONDS = 40
MAX_DEMONSTRATION_FRAMES = VIDEO_FPS * MAX_DEMONSTRATION_SECONDS
CONFIGURATION = {
    "video_fps": VIDEO_FPS,
    "max_demo_seconds": MAX_DEMONSTRATION_SECONDS,
    "max_demo_frames": MAX_DEMONSTRATION_FRAMES,
    "sampling": "uniform_with_endpoints",
    "preserve_current_execution_observation": True,
}


def configuration_for_method(method):
    return dict(CONFIGURATION) if method in CAPPED_METHODS else {"sampling": "none"}


def prepare_demonstration(obs, *, method):
    """Reset observations contain demo frames followed by one execution frame.

    The cap uses the evaluator's 20 FPS video time base. All temporal columns
    share the same indices; the final execution observation is kept separately.
    Only FrameSamp+modul uses this cap. Other adapters receive their original
    observations and retain their own demonstration handling.
    """
    if method not in CAPPED_METHODS:
        return obs, None
    count = len(obs.get("front_rgb_list", []))
    demo_count = max(0, count - 1)
    selected_count = min(demo_count, MAX_DEMONSTRATION_FRAMES)
    report = {
        **CONFIGURATION,
        "source_demo_frames": demo_count,
        "input_demo_frames": selected_count,
        "source_video_seconds": demo_count / VIDEO_FPS,
        "input_video_seconds": selected_count / VIDEO_FPS,
        "applied": demo_count > MAX_DEMONSTRATION_FRAMES,
    }
    if not report["applied"]:
        return obs, report
    demo_indices = np.linspace(0, demo_count - 1, selected_count, dtype=np.int64)
    indices = np.append(demo_indices, count - 1)
    sampled = dict(obs)
    for key, values in obs.items():
        if not key.endswith("_list"):
            continue
        if len(values) != count:
            raise ValueError(f"{key} has {len(values)} entries for {count} observation frames")
        sampled[key] = [values[int(index)] for index in indices]
    report["source_indices"] = indices.tolist()
    return sampled, report

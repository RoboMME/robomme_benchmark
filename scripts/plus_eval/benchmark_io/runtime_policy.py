"""Versioned operator limits; never supplied by generated policy code."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class RuntimePolicy:
    version: int = 2
    controller_seconds: int = 1200
    startup_seconds: int = 600
    initial_no_progress_seconds: int = 600
    no_progress_seconds: int = 180
    provider_seconds: int = 60
    simulator_rpc_seconds: int = 180
    worker_seconds: int = 2100
    development_idle_seconds: int = 1800
    development_seconds: int = 172800
    minimum_worker_memory_gib: int = 8
    memory_headroom_gib: int = 3
    conditioning_video_frames: int = 512

    def __post_init__(self):
        if self.version != 2 or any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('Invalid versioned runtime policy')
        if self.worker_seconds <= self.controller_seconds + self.startup_seconds:
            raise ValueError('Outer worker deadline must allow startup and cleanup')
        if self.conditioning_video_frames < 2:
            raise ValueError('Video sampling must preserve both endpoints')


def configured():
    path = os.environ.get('ROBOMME_AGENTIC_RUNTIME_POLICY')
    return RuntimePolicy(**json.loads(Path(path).read_text())) if path else None


class ProgressWatchdog:
    def __init__(self, policy, *, now=None):
        self.policy = policy
        self.started = self.changed = time.monotonic() if now is None else now
        self.step = 0

    def check(self, step, *, now=None):
        now = time.monotonic() if now is None else now
        if step > self.step:
            self.step, self.changed = step, now
        if now - self.started >= self.policy.controller_seconds:
            return 'controller_wall_timeout'
        limit = self.policy.no_progress_seconds if self.step > 0 else self.policy.initial_no_progress_seconds
        if now - self.changed >= limit:
            return 'controller_no_progress'
        return None


def observation_memory_gib(frames, policy):
    """Allow full mandatory RGB-D input, decoding copies and program overhead."""
    size = sum(camera.rgb.nbytes + camera.depth_m.nbytes
               for frame in frames for camera in frame.cameras.values())
    return max(policy.minimum_worker_memory_gib,
               math.ceil(2 * size / 1024**3 + policy.memory_headroom_gib))


def sample_reset_frames(frames, cap):
    """Sample conditioning timestamps only; always retain the live boundary."""
    video=[f for f in frames if f['phase']=='video_demo']
    execution=[f for f in frames if f['phase']!='video_demo']
    if cap < 2:raise ValueError('At least two video timestamps are required')
    count=len(video)
    indices=list(range(count)) if count<=cap else [i*(count-1)//(cap-1) for i in range(cap)]
    return [video[i] for i in indices]+execution,dict(
        strategy='uniform_endpoints',original_frames=count,retained_frames=len(indices),
        maximum_frames=cap,source_frame_indices=[video[i]['frame_index'] for i in indices],
        both_cameras_retained=True)

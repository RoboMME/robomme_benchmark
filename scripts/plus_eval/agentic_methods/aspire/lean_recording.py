"""Evaluation-only artifact reduction; leaves observations and control unchanged."""
import dataclasses
from pathlib import Path
import numpy as np
from cap_harness.artifacts import RunRecorder, Serializer, atomic_json, redact


class LogSerializer(Serializer):
    def encode(self, value, arrays):
        if isinstance(value, np.ndarray):
            return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype)}
        # Avoid the original serializer's special image/segmentation array writers.
        if dataclasses.is_dataclass(value):
            return {'type': type(value).__name__, **{field.name: self.encode(getattr(value, field.name), arrays)
                    for field in dataclasses.fields(value)}}
        if isinstance(value, (list, tuple)) and len(value) > 100:
            return {'type': type(value).__name__, 'length': len(value),
                    'first': self.encode(value[0], arrays), 'last': self.encode(value[-1], arrays)}
        return super().encode(value, arrays)


class LogRecorder(RunRecorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, capture_videos=False)
        self.serializer = LogSerializer(self.root)

    def keyframe(self, label):
        pass

    def segmentation_overlay(self, value):
        pass

    def _after_step(self, action, result):
        if result.ok:
            self.step_count += 1
            self.cumulative_reward += result.reward or 0.0
            self.terminated, self.truncated = result.terminated, result.truncated
            if result.observation is not None:
                self.last_observation = result.observation
        self._event('control_step', step=self.step_count, ok=result.ok,
                    terminated=result.terminated, truncated=result.truncated)

    def on_reset(self, observation, metadata):
        self.last_observation = observation
        atomic_json(self.root / 'episode/reset.json', {'metadata': metadata, 'artifact_profile': 'logs'})

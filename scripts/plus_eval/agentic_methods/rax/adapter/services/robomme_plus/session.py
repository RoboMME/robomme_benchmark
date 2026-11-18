"""RAX simulator RPC using the public RoboMME-Plus environment.

The upstream tools retain normalized world delta pose actions. This adapter
converts each row against the latest measured native TCP; +1 closes in RAX,
whereas -1 closes in RoboMME. No task-specific control logic is added.
"""
from pathlib import Path
import os
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_io.episode import Episode
from benchmark_io.protocol import Study
from services.simulator.io import RejectedBeforeAction, reject_unless, write_json
from . import tasks as T
from .reset_video import RESET_PAGE_FRAMES


def delta_to_native(action, eef):
    action, eef = np.asarray(action, float), np.asarray(eef, float)
    if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("RAX delta actions require seven finite values in [-1,1]")
    orientation = Rotation.from_rotvec(action[3:6] * T.ROTATION_SCALE_RAD) * Rotation.from_euler("xyz", eef[3:6])
    return np.r_[eef[:3] + action[:3] * T.POSITION_SCALE_M, orientation.as_euler("xyz"), -action[6]]


class RoboMMEPlusSession:
    def __init__(self, output, task, collection, *, include_depth):
        T.task_info(task)
        self.output, self.task, self.collection = Path(output), task, collection
        self.output.mkdir(parents=True, exist_ok=True)
        self.include_depth = include_depth
        self.episode_id, self.episode, self.video = None, None, None
        self.step_id = 0
        self.terminated = self.truncated = False
        self.finish_reason = self.invalid_reason = None
        self.metadata = dict(task=task, benchmark="robomme_plus", cameras=list(T.CAMERAS),
                             control_frequency_hz=20, action_dim=7, state_dim=8,
                             image_size=256, frame_stride=T.FRAME_STRIDE, depth=include_depth)

    def reset(self, layout_id, **_):
        if self.episode is not None:
            raise RuntimeError("One episode per simulator process")
        if self.collection == 1:
            from benchmark_io.evaluation import EvaluationStudy
            manifest_root = os.environ.get("ROBOMME_PLUS_FROZEN_EVALUATION")
            if not manifest_root:
                raise PermissionError("Test episodes require the operator-only frozen evaluation manifest")
            study = EvaluationStudy(manifest_root)
        elif self.collection == 0:
            study = Study.open(os.environ["ROBOMME_PLUS_STUDY"])
        else:
            raise PermissionError("Only the released training and test datasets are available")
        if study.config["method"] != "rax":
            raise ValueError("Not a RAX study")
        self.episode = Episode(self.task, int(layout_id), study=study, output=self.output / "recording",
                               action_space="ee_pose", purpose=os.environ.get("ROBOMME_PLUS_PURPOSE", "development"))
        self.episode_id = uuid.uuid4().hex
        self.initial_frames = [self._observation(frame) for frame in self.episode.frames]
        self.obs = self.initial_frames[-1]
        self._sync()
        return dict(episode_id=self.episode_id, step_id=self.step_id, native_step_limit=self.episode.limit,
                    video_sampling=getattr(self.episode,'video_sampling',None))

    def _observation(self, frame):
        eef = frame["eef_state"]
        obs = dict(state=np.r_[eef[:3], Rotation.from_euler("xyz", eef[3:6]).as_rotvec(),
                               frame["gripper_state"]].tolist(),
                   instruction=self.episode.instruction, step_id=frame["control_step"], phase=frame["phase"],
                   source_frame_index=frame['frame_index'],
                   timestamp_s=(frame['frame_index'] if frame['phase']=='video_demo' else frame['control_step']) / self.episode.frequency,
                   camera_intrinsics={}, camera_poses={})
        for camera in T.CAMERAS:
            data = frame["cameras"][camera]
            obs[camera] = data["rgb"]
            if self.include_depth:
                obs[f"{camera}_depth_m"] = data["depth_m"]
                obs["camera_intrinsics"][camera] = dict(matrix=data["intrinsics"].tolist(), width=256, height=256)
                # RAX's upstream geometry uses OpenGL cameras; public benchmark extrinsics use OpenCV.
                pose = data["camera_to_world"].copy()
                pose[:3, :3] = pose[:3, :3] @ np.diag([1., -1., -1.])
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()
                obs["camera_poses"][camera] = dict(position_m=pose[:3, 3].tolist(),
                    quaternion_wxyz=q[[3, 0, 1, 2]].tolist(), frame="world", convention="OpenGL")
        return obs

    def _sync(self):
        e = self.episode
        self.step_id, self.terminated, self.truncated = e.steps, e.terminated, e.truncated
        write_json(self.output / "operator_state.json", dict(episode_id=self.episode_id, step_id=self.step_id,
            benchmark="robomme_plus", task=self.task, reward=dict(native_success=e.success,
            native_reward=float(e.success), native_score_percent=100 * float(e.success),
            terminated=e.terminated, truncated=e.truncated)))

    def chunk_step(self, actions):
        from .tools import validate_actions
        validate_actions(np.asarray(actions).tolist())
        frames, steps = [], []
        for action in actions:
            self.episode.step(delta_to_native(action, self.episode.current["eef_state"]))
            self._sync()
            steps.append(dict(step_id=self.step_id, executed_action=list(action),
                              terminated=self.terminated, truncated=self.truncated))
            if self.step_id % T.FRAME_STRIDE == 0:
                frames.append(self._observation(self.episode.current))
            if self.terminated or self.truncated:
                break
        self.obs = self._observation(self.episode.current)
        if not frames or frames[-1]["step_id"] != self.step_id:
            frames.append(self.obs)
        if self.terminated or self.truncated:
            self._write_summary("terminal")
        return dict(episode_id=self.episode_id, step_id=self.step_id, frames=frames, steps=steps,
                    terminated=self.terminated, truncated=self.truncated, frame_stride=T.FRAME_STRIDE,
                    control_frequency_hz=20, observation_frequency_hz=20 / T.FRAME_STRIDE,
                    **({"success": self.episode.success} if self.terminated or self.truncated else {}))

    def _write_summary(self, reason):
        if self.episode is None:
            return
        if self.invalid_reason:
            self.episode.truncated = True
        self._sync()
        write_json(self.output / "evaluation_outcome.json", dict(complete=self.terminated or self.truncated,
            native_success=self.episode.success, native_score=float(self.episode.success),
            valid_for_success_rate=not bool(self.invalid_reason), valid_for_score=not bool(self.invalid_reason),
            native_control_steps=self.step_id, native_reference_step_limit=self.episode.limit,
            step_limit_enforced=True, episode_step_limit=self.episode.limit,
            score_source="robomme_plus", invalid_reason=self.invalid_reason, reason=reason))

    def finish(self, reason):
        self.finish_reason = reason
        self._write_summary(reason)
        self.episode.close(error=self.invalid_reason)
        return dict(episode_id=self.episode_id, step_id=self.step_id, reason=reason)

    def close(self):
        if self.episode:
            self.episode.close(error=self.invalid_reason)

    def dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op == "reset":
            return self.reset(**args)
        reject_unless(args.get("episode_id") == self.episode_id and self.episode_id is not None, "Stale episode")
        reject_unless(args.get("step_id") == self.step_id, "Stale step")
        if op == "observe":
            return self.obs
        if op == "reset_sequence":
            return dict(episode_id=self.episode_id, step_id=self.step_id, frames=self.initial_frames,
                        phase="conditioning_video_then_execution", frame_stride=1)
        if op == "reset_sequence_page":
            start, count = args.get("start"), args.get("count")
            reject_unless(type(start) is int and 0 <= start < len(self.initial_frames),
                          "Invalid reset-video start index")
            reject_unless(type(count) is int and 1 <= count <= RESET_PAGE_FRAMES,
                          "Invalid reset-video page size")
            stop = min(start + count, len(self.initial_frames))
            return dict(episode_id=self.episode_id, step_id=self.step_id,
                        start=start, next_start=stop, total_frames=len(self.initial_frames),
                        frames=self.initial_frames[start:stop])
        if op == "chunk_step":
            return self.chunk_step(args["actions"])
        if op == "finish_pilot":
            return self.finish(args.get("reason", "finish"))
        raise RejectedBeforeAction(f"Unsupported operation {op}")

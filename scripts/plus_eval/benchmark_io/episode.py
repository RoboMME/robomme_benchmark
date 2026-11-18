"""Trusted simulator connection for benchmark adapters. Never agent policy code.

Only public RGB-D, calibration, robot proprioception, goal and terminal outcome
cross the boundary. No task objects, subgoals, oracle actions or task implementation
are returned. The additional Panda base/hand transforms are robot calibration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def latest(value):
    return value[-1] if isinstance(value, (tuple, list)) else value


def scalar(value, default=False):
    if value is None:
        return default
    value = array(latest(value))
    return value.reshape(-1)[-1].item() if value.size else default


def camera_to_world(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=float).reshape(3, 4)
    transform = np.eye(4)
    transform[:3, :3] = extrinsic[:, :3].T
    transform[:3, 3] = -extrinsic[:, :3].T @ extrinsic[:, 3]
    return transform


def pose_matrix(pose):
    from scipy.spatial.transform import Rotation
    p = array(pose.p).reshape(-1, 3)[0]
    q = array(pose.q).reshape(-1, 4)[0]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = p
    return matrix


def public_frames(obs, info, *, first_step, reset=False, indices=None):
    """Translate every returned frame; reset's last frame begins execution."""
    n = len(obs["front_rgb_list"])
    frames = []
    for i in range(n) if indices is None else indices:
        frame = {key: array(obs[f"{key}_list"][i]).copy()
                 for key in ("joint_state", "eef_state", "gripper_state")}
        frame["cameras"] = {}
        for name in ("front", "wrist"):
            intrinsic = info[f"{name}_camera_intrinsic"]
            if isinstance(intrinsic, list):
                intrinsic = intrinsic[min(i, len(intrinsic) - 1)]
            frame["cameras"][name] = {
                "rgb": array(obs[f"{name}_rgb_list"][i]).astype(np.uint8),
                "depth_m": np.maximum(array(obs[f"{name}_depth_list"][i]).squeeze(), 0).astype(np.float32) / 1000,
                "intrinsics": array(intrinsic).reshape(3, 3),
                "camera_to_world": camera_to_world(obs[f"{name}_camera_extrinsic_list"][i]),
            }
        frame["phase"] = "video_demo" if reset and i < n - 1 else "execution"
        frame["frame_index"] = i
        frame["control_step"] = 0 if reset else first_step + i + 1
        frames.append(frame)
    return frames


class Episode:
    def __init__(self, task, episode, *, study, output, action_space="joint_angle", purpose="development",
                 builder_factory=None):
        if action_space not in ("joint_angle", "ee_pose"):
            raise ValueError("Use a one-step public action space")
        self.study, self.task = study, task
        self.action_space, self.env = action_space, None
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.video = None
        self.closed, self.terminated, self.truncated = False, False, False
        self.success, self.steps = False, 0
        self.frames, self.instruction = [], ""
        self.attempt = study.reserve(task, episode, purpose=purpose)
        (self.output / "attempt.json").write_text(json.dumps(self.attempt, indent=2))
        self.full_artifacts = study.config.get("artifact_profile", "full") == "full"
        self.actions = (self.output / "actions.jsonl" if self.full_artifacts else Path(os.devnull)).open("w")
        self.observations = (self.output / "observations.jsonl" if self.full_artifacts else Path(os.devnull)).open("w")
        try:
            if builder_factory is None:
                from robomme_plus import BenchmarkEnvBuilder
                import robomme_plus
                from .protocol import source_fingerprint
                pin = study.root / "benchmark_source.json"
                if pin.is_file() and json.loads(pin.read_text()) != source_fingerprint(Path(robomme_plus.__file__).parent):
                    raise ValueError("Benchmark code changed since preparation; refusing this attempt")
                builder_factory = BenchmarkEnvBuilder
            builder = builder_factory(task, dataset=getattr(study, "dataset", "train"), action_space=action_space)
            self.episode = self.attempt["episode"]
            metadata = builder.get_episode_metadata(self.episode)
            expected = study.config["records"][task][self.episode]
            # Pin the full normalized record, including variation settings.
            if metadata != expected:
                raise ValueError("Current training metadata differs from the prepared study")
            self.limit, self.frequency = metadata["max_steps"], metadata["control_freq"]
            self.env = builder.make_env_for_episode(self.episode, **{
                f"include_{camera}_{field}": True for camera in ("front", "wrist")
                for field in ("depth", "camera_intrinsic", "camera_extrinsic")})
            if getattr(study, "dataset", "train") == "train":
                # A failed setup still spends the reserved training start. A
                # subsequent try must go through Study.reserve again.
                try:
                    limit = self.env.get_wrapper_attr("limit_reset_attempts")
                except AttributeError:
                    limit = None
                if limit is not None:
                    limit(1)
            obs, info = self.env.reset()
            goal = latest(info["task_goal"])
            self.instruction = str(goal)
            from .runtime_policy import configured, sample_reset_frames
            runtime_policy = configured()
            self.video_sampling = None
            indices = None
            if runtime_policy:
                n=len(obs['front_rgb_list'])
                selected,self.video_sampling=sample_reset_frames([
                    dict(frame_index=i,phase='video_demo' if i<n-1 else 'execution') for i in range(n)],
                    runtime_policy.conditioning_video_frames)
                indices=[f['frame_index'] for f in selected]
                (self.output/'video_sampling.json').write_text(json.dumps(self.video_sampling,indent=2))
            self.frames = public_frames(obs, info, first_step=0, reset=True, indices=indices)
            self.current = self.frames[-1]
            self._robot_calibration()
            if self.full_artifacts:
                import imageio.v2 as imageio
                self.video = imageio.get_writer(str(self.output / "rollout.mp4"), fps=self.frequency,
                                               codec="libx264", pixelformat="yuv420p",
                                               output_params=["-movflags", "+faststart"])
            self._record_frames(self.frames)
            self.events = (self.output / "events.jsonl").open("w")
            self.events.write(json.dumps(dict(event="reset", video_demo_frames=len(self.frames) - 1,
                                             instruction=self.instruction, limit=self.limit)) + "\n")
            self.events.flush()
            self._consume_info(info)
        except BaseException as exc:
            self.close(error=f"{type(exc).__name__}: {exc}")
            raise

    def _robot_calibration(self):
        # Robot articulation only: no actors, object poses, task list or oracle.
        robot = self.env.unwrapped.agent.robot
        self.robot_base_world = pose_matrix(robot.pose)
        hand = next(link for link in robot.get_links() if link.name == "panda_hand")
        self.panda_hand_world = pose_matrix(hand.pose)
        from scipy.spatial.transform import Rotation
        native_tcp = np.eye(4)
        native_tcp[:3, :3] = Rotation.from_euler("xyz", self.current["eef_state"][3:6]).as_matrix()
        native_tcp[:3, 3] = self.current["eef_state"][:3]
        self.native_tcp_to_hand = np.linalg.inv(native_tcp) @ self.panda_hand_world

    def _consume_info(self, info):
        self.steps = int(scalar(info.get("execution_total_control_steps"), self.steps))
        self.success = bool(scalar(info.get("success")))
        self.failure_reason = str(latest(info.get("failure_reason", "")))
        self.current["control_step"] = self.steps

    def _record_frames(self, frames):
        if not self.full_artifacts:
            return
        for frame in frames:
            image = np.concatenate([frame["cameras"][c]["rgb"] for c in ("front", "wrist")], axis=1)
            if frame["phase"] == "video_demo":
                image = image.copy()
                image[:4, :, :] = (255, 40, 40)
            self.video.append_data(image)
            record = {key: frame[key].tolist() if isinstance(frame[key], np.ndarray) else frame[key]
                      for key in ("joint_state", "eef_state", "gripper_state", "phase", "frame_index", "control_step")}
            record["calibration"] = {name: {key: data[key].tolist() for key in ("intrinsics", "camera_to_world")}
                                     for name, data in frame["cameras"].items()}
            self.observations.write(json.dumps(record) + "\n")
        self.observations.flush()

    def step(self, action):
        if getattr(self, 'abort_requested', None):
            raise RuntimeError('Operator watchdog ended this episode; no further actions')
        if self.closed or self.terminated or self.truncated:
            raise RuntimeError("Episode is closed or terminal; no implicit reset")
        action = np.asarray(action, dtype=float)
        dimension = 8 if self.action_space == "joint_angle" else 7
        if action.shape != (dimension,) or not np.isfinite(action).all():
            raise ValueError(f"Expected {dimension} finite action values")
        if not -1 <= action[-1] <= 1:
            raise ValueError("Gripper must be in [-1,1], -1 closed, +1 open")
        self.actions.write(json.dumps(dict(step=self.steps, action=action.tolist(), at=time.time())) + "\n")
        self.actions.flush()
        obs, _, terminated, truncated, info = self.env.step(action)
        self.frames = public_frames(obs, info, first_step=self.steps)
        self.current = self.frames[-1]
        self.terminated, self.truncated = bool(scalar(terminated)), bool(scalar(truncated))
        self._consume_info(info)
        self._robot_calibration()
        self._record_frames(self.frames)
        self.events.write(json.dumps(dict(event="step", step=self.steps, terminated=self.terminated,
                                         truncated=self.truncated, action_error=bool(info.get("action_error")))) + "\n")
        self.events.flush()
        return self.current

    def close(self, *, error=None):
        if self.closed:
            return
        self.closed = True
        outcome = dict(success=self.success, terminated=self.terminated, truncated=self.truncated,
                       control_steps=self.steps, error=error, purpose=self.attempt["purpose"])
        cleanup_errors = []
        for resource in (self.video, getattr(self, "actions", None), getattr(self, "observations", None),
                         getattr(self, "events", None), self.env):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if cleanup_errors:
            outcome["cleanup_errors"] = cleanup_errors
        (self.output / "outcome.json").write_text(json.dumps(outcome, indent=2))
        self.study.finish(self.attempt["id"], outcome)

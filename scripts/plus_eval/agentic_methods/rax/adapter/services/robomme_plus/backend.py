# Derived from RAX LIBERO adapter d49c8ad; benchmark IO adaptation only.
"""Trusted RAX backend for released RoboMME-Plus training/test episodes.

Each episode runs in a separate simulator process. Only public observations
cross that boundary; long initial videos are transferred in bounded pages.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path

import numpy as np

from services.artifact_io import stream_artifacts
from services.robomme_plus import tasks as TK
from services.robomme_plus import tools as T
from services.robomme_plus.process import RoboMMEPlusProcess
from services.robomme_plus.reset_video import reset_frames
from services.mcp_reply import _agent_safe, _depth_png_data, _jsonable, _png_data
from services.simulator.io import action_execution_scope, exit_report, reject_unless


class RoboMMEPlusBackend:
    # evaluate() reads the simulator's operator_state.json, which outlives its process.
    evaluates_from_record = True

    def __init__(self, config):
        from services.mcp_contract import Contract

        self.config = config
        self.info = TK.task_info(config.task)
        root = Path(config.root) / "native"
        root.mkdir(mode=0o700, exist_ok=True)
        self.contract = Contract.from_config(config, isolated=True)
        self.observation_profile = self.contract.observation_profile
        self.frame_workspace = root / "workspace"
        self.frame_root = self.frame_workspace / "runtime" / "frames"
        self.sim = RoboMMEPlusProcess(output_root=root / "results")
        self._evaluation = None
        self._raw = None  # Last raw observation (host memory only), for pixel geometry.
        self._environment_spec = None
        self._instruction = None  # The live episode's instruction (the simulator's BDDL parse).
        self._initial_sequence = None

    # ------------------------------------------------------------------ tools
    def tools(self):
        return [t for t in self.contract.select(T.tool_definitions()) if t["name"] in self.contract.robot_names]

    @staticmethod
    def validate(name, arguments):
        T.validate_request(name, arguments)

    def _geometry(self):
        return self.contract.observation.geometry

    def call(self, name, arguments):
        # Each motion sends one mutating RPC, then assembles the packet from its frames.
        with action_execution_scope(self.sim, name):
            return self._call(name, arguments)

    def _call(self, name, arguments):
        self.contract.require_robot(name, arguments)
        T.validate_request(name, arguments)
        if name == T.STATUS:
            return self._status(), []
        sim = self.sim
        if not sim.active:
            raise RuntimeError("No active RoboMME-Plus episode; start an episode first")
        before = (sim.episode_id, sim.step_id)
        if name == T.OBSERVE:
            packet, images = self._observation_packet(sim.request("observe"))
            initial = self._initial_sequence
            if initial is not None and initial[0] == sim.episode_id and sim.step_id == 0:
                # The native frozen runner discards start()'s return packet, then
                # observes through Gateway. Publish the initial sequence on that
                # existing safe path before its frames directory is mounted.
                packet["frame_sequence"] = dict(initial[1])
            return packet, images
        if name == T.PIXEL_TO_POSITION:
            if not self._geometry():
                raise PermissionError("Pixel-to-position is unavailable with RGB-only observations")
            return self._pixel_to_position(arguments), []
        try:
            if name == T.STEP:
                result = sim.request("chunk_step", actions=np.asarray(arguments["actions"], dtype=np.float64))
            else:
                raise ValueError(f"Unknown RoboMME-Plus tool: {name}")
        finally:
            if (sim.episode_id, sim.step_id) != before:
                self._evaluation = None
        return self._motion_packet(result)

    # ---------------------------------------------------------------- packets
    def _status(self):
        sim = self.sim
        instruction = self._instruction if sim.episode_id else TK.instruction(self.config.task)
        return {"active": sim.active, "finished": sim.finished, "task": self.config.task,
                "episode_id": sim.episode_id, "step_id": sim.step_id if sim.episode_id else None,
                "environment_spec": self._environment_spec, "instruction": instruction,
                "robot": "Franka Panda arm with a parallel-jaw gripper",
                "action_layout": T.action_layout(), "state_layout": T.state_layout(),
                "cameras": list(T.CAMERAS), "image_size": [TK.IMAGE_SIZE, TK.IMAGE_SIZE],
                "max_action_rows": T.MAX_ACTION_ROWS, "control_frequency_hz": TK.CONTROL_HZ}

    def _attachments(self, observation):
        content, attachments, depth = [], [], {}
        for camera in T.CAMERAS:
            if camera in observation:
                content.append({"type": "image", "data": _png_data(np.asarray(observation[camera], dtype=np.uint8)),
                                "mimeType": "image/png"})
                attachments.append({"kind": "rgb", "camera": camera, "content_index": len(content)})
        if self._geometry():
            for camera in T.CAMERAS:
                key = f"{camera}_depth_m"
                if key in observation:
                    values = np.asarray(observation[key], dtype=np.float32)
                    encoded, stats = _depth_png_data(np.where(np.isfinite(values), values, 0.0))
                    content.append({"type": "image", "data": encoded, "mimeType": "image/png"})
                    depth[camera] = dict(stats, content_index=len(content))
                    attachments.append({"kind": "depth", "camera": camera, "content_index": len(content)})
        return content, attachments, depth

    def _observation_fields(self, observation, attachments, depth):
        fields = {"phase": observation.get("phase", "execution"), "state": observation.get("state"),
                  "images": [a["camera"] for a in attachments if a["kind"] == "rgb"],
                  "attachments": attachments}
        for key in ('source_frame_index','timestamp_s'):
            if key in observation:fields[key]=observation[key]
        if self._geometry() and depth:
            fields.update(depth=depth, camera_intrinsics=observation.get("camera_intrinsics"),
                          camera_poses=observation.get("camera_poses"))
        return fields

    def _observation_packet(self, observation, *, transition=None):
        self._raw = (self.sim.episode_id, self.sim.step_id, observation)
        content, attachments, depth = self._attachments(observation)
        packet = {"episode_id": self.sim.episode_id, "step_id": self.sim.step_id, "task": self.config.task,
                  "instruction": observation.get("instruction"), "environment_spec": self._environment_spec,
                  **self._observation_fields(observation, attachments, depth)}
        if transition is not None:
            packet["transition"] = _jsonable(transition)
        return self.contract.observation.clean(_agent_safe(packet)), content

    def _motion_packet(self, result):
        frames = result.pop("frames", []) or []
        transition = dict(result)
        if not frames:
            return self._observation_packet(self.sim.request("observe"), transition=transition)
        return self._sequence_packet(iter(frames), transition)

    def _sequence_packet(self, frames, transition, *, kind="robomme_plus"):
        """Save images as they arrive, retaining only metadata and the final frame."""
        run_id = self.sim.run_id or "unscoped"
        sequence_id = f"seq_{self.sim.step_id:06d}_{uuid.uuid4().hex[:8]}"
        logical = Path("runtime") / "frames" / kind / run_id / sequence_id
        saved = []
        with stream_artifacts(self.frame_root / kind / run_id / sequence_id) as write:
            for index, observation in enumerate(frames):
                content, attachments, depth = self._attachments(observation)
                record = {"frame_index": index, "step_id": int(observation["step_id"]),
                          **self._observation_fields(observation, attachments, depth)}
                artifacts, files = {}, []
                for attachment in attachments:
                    item = content[attachment["content_index"] - 1]
                    filename = f"frame_{index:06d}_{attachment['camera']}_{attachment['kind']}.png"
                    artifacts[filename] = base64.b64decode(item["data"])
                    files.append({"kind": attachment["kind"], "camera": attachment["camera"],
                                  "path": (logical / filename).as_posix()})
                write(artifacts)
                entry = {k: v for k, v in record.items() if k != "attachments"}
                if "depth" in entry:
                    entry["depth"] = {c: {k: v for k, v in d.items() if k != "content_index"}
                                      for c, d in entry["depth"].items()}
                saved.append({**entry, "files": files})
                final, final_content, final_observation = record, content, observation
            if not saved:
                raise ValueError("Cannot publish an empty frame sequence")
            manifest = {"sequence_id": sequence_id, "source": kind, "artifact_id": run_id,
                        "frequency_hz": TK.CONTROL_HZ / transition.get("frame_stride", T.FRAME_STRIDE),
                        "frame_count": len(saved), "transition": _agent_safe(transition), "frames": saved}
            write({"manifest.json": (json.dumps(self.contract.observation.clean(_jsonable(manifest)),
                                               ensure_ascii=False, indent=2) + "\n").encode()})
        manifest_path = (logical / "manifest.json").as_posix()
        self._raw = (self.sim.episode_id, self.sim.step_id, final_observation)
        directory = Path(manifest_path).parent.as_posix()
        packet = {"episode_id": self.sim.episode_id, "step_id": self.sim.step_id, "task": self.config.task,
                  "instruction": final_observation.get("instruction"), "environment_spec": self._environment_spec,
                  **{k: v for k, v in final.items() if k != "frame_index"},
                  "transition": _jsonable(transition),
                  "frame_sequence": {
                      "sequence_id": sequence_id, "frequency_hz": TK.CONTROL_HZ / transition.get("frame_stride", T.FRAME_STRIDE),
                      "frame_stride_steps": transition.get("frame_stride", T.FRAME_STRIDE), "frame_count": len(saved),
                      "manifest_path": manifest_path,
                      "frame_path_pattern": f"{directory}/frame_{{frame_index:06d}}_{{camera}}_{{kind}}.png",
                      "frame_index_range": [0, len(saved) - 1],
                      "streams": [{"camera": a["camera"], "kind": a["kind"]} for a in final["attachments"]],
                      "intermediate_frames": "see each frame's files; RGB and depth are retained when available",
                      "returned_images": "final_frame_only"}}
        return self.contract.observation.clean(_agent_safe(packet)), final_content

    def _pixel_to_position(self, arguments):
        if self._raw is None or self._raw[0] != self.sim.episode_id:
            raise ValueError("Observe first; no current observation")
        episode, step, observation = self._raw
        reject_unless(arguments["observation_step_id"] == self.sim.step_id and step == self.sim.step_id,
                      f"Stale pixel selection: observation_step_id={arguments['observation_step_id']}, "
                      f"current step_id={self.sim.step_id}; observe again and reselect")
        results = []
        for item in arguments["selections"]:
            camera = item["camera"]
            depth = observation.get(f"{camera}_depth_m")
            if depth is None:
                results.append({"camera": camera, "valid": False, "reason": "no depth for this camera"})
                continue
            point = T.pixel_to_world(depth, observation["camera_intrinsics"][camera], observation["camera_poses"][camera],
                                     float(item["u"]), float(item["v"]), int(arguments.get("depth_window_radius", 1)))
            results.append({"camera": camera, "u": item["u"], "v": item["v"], **point})
        return {"episode_id": episode, "observation_step_id": step, "selections": results}

    # -------------------------------------------------------------- lifecycle
    def start(self, *, formal, seed):
        self._evaluation = None
        self._raw = None
        self._initial_sequence = None
        c = self.config
        collection = c.formal_collection if formal else c.eval_seed
        TK.layout(c.task, collection, int(seed))
        if self.sim.process is not None:
            try:
                if self.sim.active:
                    self.sim.finish("formal_episode" if formal else "reset")
            finally:
                self.sim.stop()
        mode = "formal" if formal else "exploration"
        run_id = f"{c.task}_{mode}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        timeouts = TK.task_timeouts(c.task)
        reset = self.sim.start(task=c.task, collection=collection, layout_id=int(seed), run_id=run_id,
                               sim_gpu=c.sim_gpu, startup_timeout=timeouts["startup_seconds"],
                               include_depth=self._geometry(),
                               # The supervisor's own wall limit also bounds the episode.
                               episode_timeout_seconds=int(c.formal.wall_seconds or 0))
        limit = int(reset["native_step_limit"])
        self._environment_spec = {"benchmark": "robomme_plus", "episode_mode": mode, "episode_step_limit": limit,
                                  "native_step_limit": limit, "control_frequency_hz": TK.CONTROL_HZ,
                                  "startup_seconds": round(self.sim.startup_seconds or 0.0, 1)}
        observation = self.sim.request("observe")
        self._instruction = observation.get("instruction")
        transition = dict(episode_id=self.sim.episode_id, step_id=self.sim.step_id,
                          phase="conditioning_video_then_execution", frame_stride=1)
        if reset.get('video_sampling'):
            transition['video_sampling']=reset['video_sampling']
        packet, images = self._sequence_packet(reset_frames(self.sim), transition)
        self._initial_sequence = (self.sim.episode_id, dict(packet["frame_sequence"]))
        packet.update(reset=_agent_safe({"episode_id": reset["episode_id"], "step_id": reset["step_id"]}),
                      environment_spec=self._environment_spec, artifact_id=self.sim.run_id,
                      artifact_storage="service_managed_not_agent_accessible")
        return packet, images

    def evaluate(self, agent_assessment=None, confidence=None):
        sim = self.sim
        identity = (sim.episode_id, sim.step_id)
        if self._evaluation is not None and self._evaluation[0] == identity:
            return dict(self._evaluation[1])
        if sim.output_dir is None or sim.episode_id is None:
            raise RuntimeError("No RoboMME-Plus episode to evaluate")
        state = json.loads((sim.output_dir / "sim" / "operator_state.json").read_text())
        if state.get("episode_id") != sim.episode_id or state.get("step_id") != sim.step_id:
            raise RuntimeError(
                f"Native evaluation identity mismatch: the simulator last recorded episode "
                f"{state.get('episode_id')!r} at step {state.get('step_id')!r}, but this episode is "
                f"{sim.episode_id!r} at step {sim.step_id!r}; "
                + ("the simulator process was already stopped" if sim.process is None
                   else exit_report(sim.process, sim.log_path, since=sim.log_mark)))
        success = (state.get("reward") or {}).get("native_success")
        if type(success) is not bool:
            raise RuntimeError("Native evaluation unavailable")
        evaluation = {"status": "completed", "task_complete": success, "episode_id": sim.episode_id,
                      "step_id": sim.step_id, "task": self.config.task, "evaluation_kind": "binary_task_completion",
                      "decided_at_unix_s": time.time()}
        if agent_assessment:
            evaluation["agent_assessment"] = str(agent_assessment)[:2000]
        self._evaluation = (identity, dict(evaluation))
        return dict(evaluation)

    def close(self):
        try:
            if self.sim.active:
                self.sim.finish("close")
        finally:
            self.sim.stop()

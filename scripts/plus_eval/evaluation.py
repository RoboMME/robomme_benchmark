#!/usr/bin/env python3
"""Shared RoboMME-Plus rollout, artifacts, timing and comparison protocol.

Adapters subclass Evaluation and implement setup/start_episode/act/observe/end_episode.
Only this module steps the environment or writes rollout videos and final results.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr, ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import logging
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from privacy import RedactingWriter, refresh_secrets, safe_items, scrub, write_text
from demonstration_sampling import configuration_for_method, prepare_demonstration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA_VERSION = 1
TERMINAL = {"success", "fail", "timeout", "error"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in safe_items(value)}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return scrub(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if hasattr(value, "detach"):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str):
        return scrub(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"Unsupported artifact type: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(json_value(value), allow_nan=False) + "\n")


def stats(values):
    values = list(values)
    return {"count": len(values), "total_seconds": sum(values),
            "mean_seconds": float(np.mean(values)) if values else None,
            "p50_seconds": float(np.percentile(values, 50)) if values else None,
            "p95_seconds": float(np.percentile(values, 95)) if values else None}


def api_usage(response):
    u = (response or {}).get("usage") or {}
    details = u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}
    out_details = u.get("output_tokens_details") or u.get("completion_tokens_details") or {}
    return {"input_tokens": u.get("input_tokens", u.get("prompt_tokens")),
            "cached_input_tokens": details.get("cached_tokens", 0) or 0,
            "cache_write_tokens": details.get("cache_write_tokens", 0) or 0,
            "output_tokens": u.get("output_tokens", u.get("completion_tokens")),
            "reasoning_tokens": out_details.get("reasoning_tokens", 0) or 0}


def estimate_cost(usage, model, pricing):
    """USD estimate; reasoning is already included in output, never billed twice."""
    rate = pricing.get("models", {}).get(model)
    if not rate or usage["input_tokens"] is None or usage["output_tokens"] is None:
        return None
    quantities = {"input": usage["input_tokens"] - usage["cached_input_tokens"] - usage["cache_write_tokens"],
                  "cached_input": usage["cached_input_tokens"], "cache_write": usage["cache_write_tokens"],
                  "output": usage["output_tokens"]}
    if any(n < 0 or (n and key not in rate) for key, n in quantities.items()):
        return None
    return sum(n * rate.get(key, 0) for key, n in quantities.items()) / 1e6


def validate_pricing(pricing):
    if not pricing:
        return
    if not pricing.get("source") or not pricing.get("as_of") or not isinstance(pricing.get("models"), dict):
        raise ValueError("Pricing JSON needs source, as_of and models (USD per million tokens).")
    for rates in pricing["models"].values():
        for name, value in rates.items():
            if name not in {"input", "cached_input", "cache_write", "output"} or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid API price; use nonnegative USD per million tokens.")


@contextmanager
def capture_log(path):
    refresh_secrets()
    with Path(path).open("a", buffering=1) as stream:
        log_stream = RedactingWriter(stream)
        handler = logging.StreamHandler(log_stream)
        # Format first: exception tracebacks contain paths as well as messages.
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        stdout, stderr = RedactingWriter(sys.stdout, stream), RedactingWriter(sys.stderr, stream)
        logger = logging.getLogger()
        previous_level = logger.level
        logger.setLevel(min(previous_level, logging.INFO))
        logger.addHandler(handler)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            stdout.finish()
            stderr.finish()
            log_stream.finish()
            handler.close()


class VideoRecorder:
    """Streaming front | wrist, at the simulator's 20 Hz; no text overlays."""
    def __init__(self, path, fps=20):
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1)
        self.count = 0

    def add(self, front, wrist, demo=False):
        frame = np.concatenate([np.asarray(front), np.asarray(wrist)], axis=1).astype(np.uint8)
        if demo:
            border = min(5, max(1, min(frame.shape[:2]) // 32))
            frame[:border] = frame[-border:] = (255, 0, 0)
            frame[:, :border] = frame[:, -border:] = (255, 0, 0)
        self.writer.append_data(frame)
        self.count += 1

    def close(self):
        self.writer.close()


class EpisodeRecorder:
    def __init__(self, directory, pricing=None, api_applicable=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "calls").mkdir()
        (self.directory / "assets").mkdir()
        for name in ("events.jsonl", "steps.jsonl", "annotations.jsonl", "frames.jsonl"):
            (self.directory / name).touch()
        self.pricing = pricing or {}
        self.api_applicable = api_applicable
        self.events = []
        self.step = self.frame_count = self.call_count = 0
        self.last_call_id = None
        self.video = None
        self.disk_seconds = 0.0
        self.image_assets = {}

    def artifact(self, value):
        """Externalize media/tensors by content hash; JSON paths are episode-relative."""
        if isinstance(value, dict):
            return {k: self.artifact(v) for k, v in safe_items(value)}
        if isinstance(value, (list, tuple)):
            return [self.artifact(v) for v in value]
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            header, data = value.split(",", 1)
            blob = base64.b64decode(data, validate=True)
            extension = {"data:image/png;base64": ".png", "data:image/jpeg;base64": ".jpg"}.get(header, ".bin")
            return {"path": self._blob(blob, extension), "mime_type": header[5:].split(";")[0]}
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            # NumPy cannot represent bfloat16: preserve its original dtype in metadata.
            dtype = str(tensor.dtype)
            array = tensor.float().numpy() if dtype == "torch.bfloat16" else tensor.numpy()
            result = self.artifact(array)
            return ({**result, "source_dtype": dtype} if isinstance(result, dict)
                    else {"values": result, "shape": list(tensor.shape), "source_dtype": dtype})
        if hasattr(value, "__array__") or hasattr(value, "__array_interface__"):
            value = np.asarray(value)
        if isinstance(value, np.ndarray):
            # Text/object arrays must go through redaction, never opaque .npy files.
            if value.dtype.kind in "OUSV":
                return json_value(value)
            if value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] in (3, 4):
                from PIL import Image
                key = (value.shape, hashlib.sha256(value.tobytes()).digest())
                if key in self.image_assets:
                    return self.image_assets[key].copy()
                out = io.BytesIO()
                Image.fromarray(value).save(out, format="PNG")
                artifact = {"path": self._blob(out.getvalue(), ".png"), "shape": list(value.shape)}
                self.image_assets[key] = artifact
                return artifact.copy()
            if value.size <= 128:
                return json_value(value)
            out = io.BytesIO()
            np.save(out, value, allow_pickle=False)
            return {"path": self._blob(out.getvalue(), ".npy"), "shape": list(value.shape), "dtype": str(value.dtype)}
        return json_value(value)

    def _blob(self, blob, extension):
        path = self.directory / "assets" / (hashlib.sha256(blob).hexdigest() + extension)
        if not path.exists():
            path.write_bytes(blob)
        return str(path.relative_to(self.directory))

    def event(self, component, kind, seconds, **extra):
        event = {"schema_version": SCHEMA_VERSION, "event_id": len(self.events), "step": self.step,
                 "recorded_utc": utc_now(), "component": component, "kind": kind, "seconds": seconds, **extra}
        self.events.append(event)
        append_json(self.directory / "events.jsonl", event)
        return event

    def call(self, component, function, *, inputs=None, prompt=None, kind="model", model=None,
             output_transform=None):
        """Time only function(), keeping own serialization outside the timer.

        Nested recording time is subtracted from the measured wall time. API retry
        waits are explicit events, included in their enclosing model call only.
        """
        call_id = f"{self.call_count:06d}_{component}"
        self.call_count += 1
        directory = self.directory / "calls" / call_id
        directory.mkdir()
        disk_start = time.perf_counter()
        write_json(directory / "input.json", self.artifact(inputs))
        if prompt is not None:
            write_text(directory / "prompt.txt", prompt)
        self.disk_seconds += time.perf_counter() - disk_start
        started = time.perf_counter()
        disk_before = self.disk_seconds
        output = None
        error = None
        try:
            output = function()
            return output
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": scrub(exc)}
            raise
        finally:
            wall = time.perf_counter() - started
            seconds = max(0.0, wall - (self.disk_seconds - disk_before))
            disk_start = time.perf_counter()
            try:
                saved_output = output_transform(output) if output_transform and output is not None else output
                write_json(directory / "output.json", self.artifact(saved_output))
                extra = {"call_id": call_id, "directory": str(directory.relative_to(self.directory)),
                         "ok": error is None, "error": error, "wall_seconds": wall}
                if isinstance(saved_output, dict):
                    extra["reported_timing"] = {k: saved_output[k] for k in ("policy_timing", "server_timing") if k in saved_output}
                if kind == "api":
                    response = saved_output if isinstance(saved_output, dict) else {}
                    reported_model = response.get("model") or model
                    usage = api_usage(response)
                    extra.update(model=reported_model, requested_model=model, usage=usage,
                                 estimated_cost_usd=estimate_cost(usage, reported_model, self.pricing))
                self.event(component, kind, seconds, **extra)
                self.last_call_id = call_id
            finally:
                self.disk_seconds += time.perf_counter() - disk_start

    def annotation(self, **value):
        append_json(self.directory / "annotations.jsonl", {
            "step": self.step, "next_video_frame": self.frame_count,
            "observation_video_frame": self.frame_count - 1 if self.frame_count else None,
            "call_id": self.last_call_id, **self.artifact(value),
        })

    def frames(self, obs, initial=False, previous_step=0):
        if not obs:
            return
        front, wrist = obs.get("front_rgb_list", []), obs.get("wrist_rgb_list", [])
        if len(front) != len(wrist):
            raise ValueError("Front and wrist frame counts differ.")
        if front and self.video is None:
            self.video = VideoRecorder(self.directory / "episode.mp4")
        for index, (f, w) in enumerate(zip(front, wrist)):
            demo = initial and index < len(front) - 1
            self.video.add(f, w, demo)
            # RoboMME-Plus returns every control frame for waypoint actions. Record the
            # batch boundaries as well, so gaps can never masquerade as alignment.
            aligned = len(front) == self.step - previous_step
            control_step = (None if demo else 0) if initial else (previous_step + index + 1 if aligned else None)
            append_json(self.directory / "frames.jsonl", {
                "frame": self.frame_count, "video_time_seconds": self.frame_count / 20,
                "phase": "demonstration" if demo else "execution", "control_step": control_step,
                "batch_index": index, "batch_size": len(front), "batch_end_step": self.step,
            })
            self.frame_count += 1

    def metrics(self):
        by_kind, by_component = defaultdict(list), defaultdict(list)
        for event in self.events:
            by_kind[event["kind"]].append(event["seconds"])
            by_component[event["component"]].append(event["seconds"])
        api = [e for e in self.events if e["kind"] == "api"]
        known = [e["estimated_cost_usd"] for e in api if e["estimated_cost_usd"] is not None]
        return {
            "latency": {k: stats(by_kind[k]) for k in ("model", "api", "simulator_step", "simulator_reset", "simulator_build", "preprocess", "transport", "retry_wait")},
            "components": {k: stats(v) for k, v in sorted(by_component.items())},
            "api": {"applicable": self.api_applicable, "attempts": len(api),
                    "successful_requests": sum(e["ok"] for e in api),
                    "usage": {k: sum(e["usage"][k] or 0 for e in api) for k in api_usage({})},
                    "requests_with_usage": sum(e["usage"]["input_tokens"] is not None for e in api),
                    "estimated_cost_usd": sum(known) if len(known) == len(api) else None,
                    "known_cost_usd": sum(known), "priced_attempts": len(known),
                    "unpriced_attempts": len(api) - len(known), "pricing": self.pricing or None},
            "video_frames": self.frame_count,
        }


class PolicyStop(Exception):
    """A policy stopping before environment success is a scored failure."""


class EpisodeStop(PolicyStop):
    """A policy failure that ends the episode, including any remaining trials."""


class PolicyUnavailable(RuntimeError):
    """Stop this queue; a broken policy service must not spoil later episodes."""


async def maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def select_cases(builder_class, tasks=None, episodes=None, action_space="joint_angle"):
    available = builder_class.get_task_list()
    tasks = available if tasks is None else tasks
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(available):
        raise ValueError(f"Choose distinct tasks from: {', '.join(available)}")
    if episodes is not None and (not episodes or len(set(episodes)) != len(episodes)):
        raise ValueError("Episode indices must be distinct and nonempty.")
    builders, cases = {}, []
    for task in tasks:
        builder = builder_class(task, dataset="test", action_space=action_space, gui_render=False)
        builders[task] = builder
        for episode in range(builder.get_episode_num()) if episodes is None else episodes:
            if not 0 <= episode < builder.get_episode_num():
                raise ValueError(f"{task}: episode index {episode} out of range.")
            cases.append({"task": task, "episode": episode})
    return builders, {"benchmark": "robomme_plus", "dataset": "test", "cases": cases}


def aggregate_metrics(results):
    measured = [r for r in results if "metrics" in r]
    latency = {}
    for kind in ("model", "api", "simulator_step", "simulator_reset", "preprocess", "transport", "retry_wait"):
        values = [r["metrics"]["latency"][kind] for r in measured]
        count = sum(v["count"] for v in values)
        seconds = sum(v["total_seconds"] for v in values)
        latency[kind] = {"count": count, "total_seconds": seconds, "mean_seconds": seconds / count if count else None}
    api = [r["metrics"]["api"] for r in measured]
    costs = [a["estimated_cost_usd"] for a in api]
    return {"latency": latency, "episode_wall_seconds": sum(r["elapsed_seconds"] for r in measured),
            "api": {"attempts": sum(a["attempts"] for a in api),
                    "estimated_cost_usd": sum(costs) if costs and all(v is not None for v in costs) else None,
                    "known_cost_usd": sum(a["known_cost_usd"] for a in api),
                    "priced_attempts": sum(a["priced_attempts"] for a in api),
                    "unpriced_attempts": sum(a["unpriced_attempts"] for a in api),
                    "usage": {k: sum(a["usage"][k] for a in api) for k in api_usage({})}}}


def summarize(cases, results, method=None):
    # Disk records are authoritative; never count a replaced episode twice.
    expected = {(c['task'], c['episode']) for c in cases}
    rows = {(r['task'], r['episode']): r for r in results if (r['task'], r['episode']) in expected}
    results = list(rows.values())
    finished = [r for r in results if not r.get('interrupted')]
    counts = Counter(r["status"] for r in finished)
    report = {"schema_version": SCHEMA_VERSION, "benchmark": "robomme_plus", "dataset": "test",
              "method": method, "requested": len(cases), "completed": len(finished),
              "pending": len(cases) - len(finished), "complete": len(cases) == len(finished),
              "interrupted": len(results) - len(finished),
              "counts": {s: counts[s] for s in sorted(TERMINAL)},
              "success_rate": counts["success"] / len(cases) if cases else None,
              "metrics": aggregate_metrics(finished), "results": results}
    report["per_task"] = {}
    for task in dict.fromkeys(c["task"] for c in cases):
        selected = [c for c in cases if c["task"] == task]
        rows = [r for r in finished if r["task"] == task]
        report["per_task"][task] = {"requested": len(selected), "completed": len(rows),
                                   "success_rate": sum(r["success"] for r in rows) / len(selected),
                                   "metrics": aggregate_metrics(rows)}
    return report


class Evaluation:
    method = "unspecified"
    action_space = "joint_angle"
    api_applicable = False
    observation_options = {}

    def __init__(self, args):
        self.args = args

    def setup(self):
        pass

    def configuration(self):
        """Resolved inference settings beyond the command-line arguments."""
        return {}

    def start_episode(self, obs, info):
        pass

    def act(self, obs, info):
        raise NotImplementedError

    def observe(self, obs, info):
        pass

    def end_episode(self, result):
        pass

    async def run_episode(self, builder, task, episode, directory, pricing=None):
        self.recorder = rec = EpisodeRecorder(directory, pricing, self.api_applicable)
        self.task, self.episode = task, episode
        self.goal = None
        result = {"schema_version": SCHEMA_VERSION, "benchmark": "robomme_plus", "dataset": "test",
                  "method": self.method, "task": task, "episode": episode, "status": "error",
                  "success": False, "steps": 0, "max_steps": builder.episode_max_steps(episode),
                  "action_space": self.action_space, "started_utc": utc_now()}
        env = None
        actions = []
        started = time.perf_counter()
        with capture_log(rec.directory / "episode.log"):
            try:
                if hasattr(builder, "get_episode_metadata"):
                    write_json(rec.directory / "metadata.json", builder.get_episode_metadata(episode))
                t = time.perf_counter()
                try:
                    env = builder.make_env_for_episode(episode, **self.observation_options)
                    result["control_freq"] = int(getattr(getattr(env, "unwrapped", env), "control_freq", 20))
                finally:
                    rec.event("environment_build", "simulator_build", time.perf_counter() - t)
                t = time.perf_counter()
                try:
                    obs, info = env.reset()
                finally:
                    rec.event("environment_reset", "simulator_reset", time.perf_counter() - t)
                if not obs or info.get("status") == "error":
                    raise RuntimeError(f"Environment reset failed: {info}")
                goal = info["task_goal"]
                self.goal = goal[0] if isinstance(goal, (tuple, list)) else goal
                result["task_goal"] = self.goal
                rec.frames(obs, initial=True)
                obs, sampling = prepare_demonstration(obs, method=self.method)
                if sampling is not None:
                    result["demonstration_preprocessing"] = sampling
                    write_json(rec.directory / "demonstration_sampling.json", sampling)
                print(f"{self.method}: {task} episode {episode}: {self.goal} (budget {result['max_steps']})", flush=True)
                await maybe_await(self.start_episode(obs, info))
                trial_step = int(info.get("execution_control_steps", 0))
                while trial_step < result["max_steps"]:
                    try:
                        proposed_action = await maybe_await(self.act(obs, info))
                    except PolicyStop as stop:
                        if isinstance(stop, EpisodeStop):
                            raise
                        # A voluntary policy stop ends this trial. Only tasks
                        # explicitly granting a retry may continue internally.
                        try:
                            retry = env.get_wrapper_attr("try_execution_retry")
                        except AttributeError:
                            retry = None
                        retried = retry("fail") if callable(retry) else None
                        if retried is None:
                            raise
                        previous_step = rec.step
                        obs, _, _, _, info = retried
                        trial_step = int(info["execution_control_steps"])
                        rec.step = int(info["execution_total_control_steps"])
                        result.update(steps=rec.step, trial_steps=trial_step,
                                      trial_index=info["trial_index"])
                        append_json(rec.directory / "steps.jsonl", {
                            "event": "trial_reset", "policy_stop": str(stop), "info": info,
                            "step_before": previous_step, "step_after": rec.step,
                        })
                        rec.frames(obs, previous_step=previous_step)
                        await maybe_await(self.observe(obs, info))
                        continue
                    action = np.asarray(proposed_action)
                    expected = 7 if self.action_space == "waypoint" else 8
                    if action.shape != (expected,) or not np.isfinite(action).all():
                        raise ValueError(f"Expected one finite {expected}D action, got {action.shape}.")
                    previous_step = rec.step
                    actions.append(action.copy())
                    t = time.perf_counter()
                    try:
                        obs, reward, terminated, truncated, info = env.step(action)
                    except BaseException as exc:
                        append_json(rec.directory / "steps.jsonl", {"env_call": len(actions) - 1,
                            "step_before": previous_step, "action": action, "error": scrub(exc)})
                        raise
                    finally:
                        rec.event("environment_step", "simulator_step", time.perf_counter() - t)
                    info = info or {}
                    # Error recovery may omit the counter; don't invent completed steps.
                    fallback = previous_step + (1 if self.action_space == "joint_angle" and info.get("status") != "error" else 0)
                    trial_step = int(info.get("execution_control_steps", fallback))
                    rec.step = int(info.get("execution_total_control_steps", trial_step))
                    result["steps"] = rec.step
                    if "trial_index" in info:
                        result["trial_index"] = info["trial_index"]
                        result["trial_steps"] = trial_step
                    append_json(rec.directory / "steps.jsonl", {
                        "env_call": len(actions) - 1, "step_before": previous_step, "step_after": rec.step,
                        "action": action, "reward": reward, "terminated": bool(terminated), "truncated": bool(truncated),
                        "info": info, "first_video_frame": rec.frame_count,
                        "observation": {k: v for k, v in (obs or {}).items() if k.endswith("state_list")},
                    })
                    rec.frames(obs, previous_step=previous_step)
                    status = info.get("status", "unknown")
                    if status == "error" or obs is None:
                        raise RuntimeError(f"Environment error: {info}")
                    if terminated or truncated or status in TERMINAL:
                        if status not in TERMINAL:
                            raise RuntimeError(f"Environment stopped with unexpected status: {status}")
                        result.update(status=status, success=status == "success", end_reason=f"environment:{status}")
                        break
                    await maybe_await(self.observe(obs, info))
                    if len(actions) % 100 == 0:
                        print(f"{task} episode {episode}: {rec.step}/{result['max_steps']} control steps", flush=True)
                else:
                    result.update(status="timeout", end_reason="ood_control_step_budget")
            except PolicyStop as exc:
                result.update(status="fail", end_reason=str(exc))
            except BaseException as exc:
                result.update(status="error", success=False, error=scrub(exc), error_type=type(exc).__name__)
                if (isinstance(exc, (PolicyUnavailable, ConnectionError)) or
                        type(exc).__name__ in ('ConnectionClosedError', 'ConnectionClosedOK') or
                        'out of memory' in str(exc).lower()):
                    result['fatal_error'] = True
                traceback.print_exc()
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    result["interrupted"] = True
            finally:
                for component, callback in (("policy_cleanup", lambda: self.end_episode(result)),
                                             ("environment_close", lambda: env.close() if env is not None else None),
                                             ("video_close", lambda: rec.video.close() if rec.video is not None else None)):
                    t = time.perf_counter()
                    try:
                        await maybe_await(callback())
                    except BaseException as exc:
                        result.setdefault("artifact_errors", []).append({"component": component, "error": scrub(exc)})
                        traceback.print_exc()
                    rec.event(component, "cleanup", time.perf_counter() - t)
                np.save(rec.directory / "actions.npy", np.asarray(actions).reshape(-1, 7 if self.action_space == "waypoint" else 8))
                result.update(metrics=rec.metrics(), env_step_calls=len(actions), finished_utc=utc_now(),
                              elapsed_seconds=time.perf_counter() - started,
                              artifacts_complete=result["status"] != "error" and not bool(result.get("artifact_errors")))
                if result.get('artifact_errors') and result['status'] != 'error':
                    result.update(status='error', success=False, error='Rollout cleanup or artifact writing failed')
                result["metrics"]["simulated_seconds"] = result["steps"] / result.get("control_freq", 20)
                result["metrics"]["simulator_seconds_per_control_step"] = (
                    result["metrics"]["latency"]["simulator_step"]["total_seconds"] / result["steps"]
                    if result["steps"] else None)
                result["metrics"]["model_seconds_per_control_step"] = (
                    result["metrics"]["latency"]["model"]["total_seconds"] / result["steps"]
                    if result["steps"] else None)
                write_json(rec.directory / "result.json", result)
                print(f"{task} episode {episode}: {result['status']}, {result['steps']} control steps", flush=True)
        return result

    async def run(self):
        refresh_secrets()
        from robomme_plus.env_record_wrapper import BenchmarkEnvBuilder
        from run_storage import run_lock, read, rebuild_batch
        directory = self.args.output_dir
        parent_batch = directory.parent / 'batch.json'
        managed = not os.environ.get('OOD_QUEUE_MANAGED') and parent_batch.exists()
        with ExitStack() as stack:
            if managed:
                stack.enter_context(run_lock(directory.parent, '.rerun.lock'))
            stack.enter_context(run_lock(directory))
            try:
                return await self.run_locked(BenchmarkEnvBuilder)
            finally:
                if managed and (directory / 'config.json').exists():
                    batch = read(parent_batch)
                    if directory.name in batch['tasks']:
                        row = batch['tasks'][directory.name]
                        row['episodes'] = sorted({c['episode'] for c in read(directory / 'config.json')['cases'] if c['task'] == directory.name})
                        row['requested'] = len(row['episodes'])
                    rebuild_batch(directory.parent, batch)

    async def run_locked(self, builder_class):
        from run_storage import RunSession, benchmark_sources, mode_from_args, read, rebuild_task
        args = self.args
        mode = mode_from_args(args)
        old_path = args.output_dir / 'config.json'
        old = read(old_path) if old_path.exists() else None
        tasks = args.tasks
        if old and tasks is None:
            tasks = list(dict.fromkeys(c['task'] for c in old['cases']))
        builders, document = select_cases(builder_class, tasks, args.episodes, self.action_space)
        if old and args.tasks is None and args.episodes is None:
            available = {(c['task'], c['episode']) for c in document['cases']}
            if any((c['task'], c['episode']) not in available for c in old['cases']):
                raise ValueError('Saved episode selection is invalid for the current benchmark metadata')
            document['cases'] = old['cases']
        pricing = json.loads(args.api_pricing_json.read_text()) if args.api_pricing_json else {}
        if old and not args.api_pricing_json:
            pricing = old.get('api_pricing', {})
        validate_pricing(pricing)
        directory = args.output_dir
        config = {"schema_version": SCHEMA_VERSION, "method": self.method, **document,
                  "arguments": vars(args), "api_pricing": pricing, "started_utc": utc_now(),
                  "video_fps": 20, "demonstration_preprocessing": configuration_for_method(self.method),
                  "budget_rule": "ceil(1.5 * oracle_policy_step) + 200",
                  "runtime": {"python": platform.python_version(), "executable": sys.executable,
                              "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
        if os.environ.get('OOD_QUEUE_MANAGED'):
            config['runtime']['queue_workers'] = int(os.environ.get('OOD_TASK_WORKERS', '1'))
        source_files = [Path(__file__), Path(__file__).with_name("privacy.py"),
                        Path(__file__).with_name('run_storage.py'),
                        Path(__file__).with_name('demonstration_sampling.py'), Path(inspect.getfile(type(self)))]
        if self.method == 'embodied-astra':
            source_files.append(Path(__file__).with_name('astra_image_sampling.py'))
        source_files.extend(getattr(self, 'source_dependencies', ()))
        config["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        config["benchmark_source_sha256"] = benchmark_sources()
        metadata = ROOT / "src/robomme_plus/env_metadata/test"
        config["metadata_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(metadata.glob("*.json"))}
        config["episode_metadata"] = {
            f"{case['task']}_ep{case['episode']}": builders[case['task']].get_episode_metadata(case['episode'])
            for case in document['cases']}
        config["git_revisions"] = {str(path.relative_to(ROOT)): subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
                                   for path in [ROOT, *[p for p in Path(inspect.getfile(type(self))).parent.iterdir() if (p / ".git").exists()]]}
        session = RunSession(directory, config, mode)
        with capture_log(directory / "run.log"):
            try:
                print(f'{self.method}: {mode}, {len(session.todo)} selected episodes need evaluation', flush=True)
                if not session.todo:
                    (directory / 'run_error.json').unlink(missing_ok=True)
                    write_json(directory / 'run_state.json', {'status': 'no_work', 'attempt_id': session.id,
                                                             'pending': session.pending, 'finished_utc': utc_now()})
                    return rebuild_task(directory)['results']
                write_json(directory / 'run_state.json', {'status': 'running', 'attempt_id': session.id,
                                                         'pending': session.pending, 'started_utc': utc_now()})
                t = time.perf_counter()
                await maybe_await(self.setup())
                session.setup_complete(time.perf_counter() - t, self.configuration())
                for case in session.todo:
                    task, episode = case["task"], case["episode"]
                    write_json(directory / 'run_state.json', {'status': 'running', 'attempt_id': session.id,
                                                             'current_case': case, 'pending': session.pending, 'updated_utc': utc_now()})
                    path = session.episode_directory(case)
                    result = await self.run_episode(builders[task], task, episode, path, pricing)
                    session.record(case, result, path)
                    write_json(directory / 'run_state.json', {'status': 'running', 'attempt_id': session.id,
                                                             'pending': session.pending, 'updated_utc': utc_now()})
                    if result.get('interrupted'):
                        raise KeyboardInterrupt()
                    if result.get('fatal_error') or (result['status'] == 'error' and args.fail_fast):
                        raise RuntimeError("Stopped after evaluation error; see episode.log and result.json.")
                (directory / 'run_error.json').unlink(missing_ok=True)
                write_json(directory / 'run_state.json', {'status': 'completed', 'attempt_id': session.id,
                                                         'pending': session.pending, 'finished_utc': utc_now()})
            except BaseException as exc:
                state = {'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)) else 'error',
                         'error': scrub(exc), 'type': type(exc).__name__, 'attempt_id': session.id,
                         'pending': session.pending, 'finished_utc': utc_now()}
                write_json(directory / 'run_state.json', state)
                write_json(directory / "run_error.json", state)
                raise
            finally:
                report = rebuild_task(directory)
                write_json(session.attempt / 'summary.json', report)
                if (directory / 'run_state.json').exists():
                    write_json(session.attempt / 'run_state.json', read(directory / 'run_state.json'))
        return report['results']


def common_parser(description):
    from run_storage import add_run_modes
    parser = argparse.ArgumentParser(description=description)
    # asyncio.run's default SIGINT handler only cancels at an async suspension.
    # Policy/API and simulator calls can be synchronous, so interrupt them now
    # and let run_episode save an interrupted result before normal cleanup.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser.add_argument("--tasks", nargs="+", help="Default: all RoboMME-Plus test tasks.")
    parser.add_argument("--episodes", nargs="+", type=int, help="Default: every available RoboMME-Plus test episode.")
    parser.add_argument("--output-dir", "--run-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--api-pricing-json", type=Path, help="Optional model rate card; absent/unmatched pricing means unknown cost.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first episode error; default: record it and continue sequentially.")
    add_run_modes(parser)
    return parser


def compare_runs(paths, output):
    rows = []
    for path in paths:
        report = json.loads((path / "summary.json").read_text())
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: not a unified protocol v{SCHEMA_VERSION} run")
        for result in report["results"]:
            m = result["metrics"]
            row = {"run": path.name, "method": report["method"], "task": result["task"], "episode": result["episode"],
                   "status": 'interrupted' if result.get('interrupted') else result["status"],
                   "interrupted": bool(result.get('interrupted')), "success": result["success"], "control_steps": result["steps"],
                   "wall_seconds": result["elapsed_seconds"], "estimated_api_cost_usd": m["api"]["estimated_cost_usd"],
                   "known_api_cost_usd": m["api"]["known_cost_usd"],
                   "api_attempts": m["api"]["attempts"], "unpriced_attempts": m["api"]["unpriced_attempts"],
                   "model_seconds_per_control_step": m["latency"]["model"]["total_seconds"] / result["steps"] if result["steps"] else None,
                   "simulator_seconds_per_control_step": m["latency"]["simulator_step"]["total_seconds"] / result["steps"] if result["steps"] else None,
                   "run_requested": report["requested"], "run_pending": report["pending"],
                   "run_errors": report["counts"]["error"], "run_success_rate": report["success_rate"]}
            for kind in ("model", "simulator_step", "api"):
                row.update({f"{kind}_{key}": value for key, value in m["latency"][kind].items()})
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", rows)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["run", "method", "task", "episode", "status", "success"])
        writer.writeheader()
        writer.writerows(json_value(rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare", help="Export comparable per-episode metrics from unified runs.")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--output-dir", type=Path, required=True)
    summary = sub.add_parser("summarize", help="Rebuild a run summary from its saved episodes, without inference.")
    summary.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        compare_runs(args.runs, args.output_dir)
    else:
        from run_storage import rebuild_batch, rebuild_task, recover_replacement, run_lock
        batch = (args.run / 'batch.json').exists()
        with run_lock(args.run, '.rerun.lock' if batch else '.evaluation.lock'):
            if batch:
                for task in json.loads((args.run / 'batch.json').read_text())['tasks']:
                    with run_lock(args.run / task):
                        recover_replacement(args.run / task)
                rebuild_batch(args.run)
            else:
                recover_replacement(args.run)
                rebuild_task(args.run)


if __name__ == "__main__":
    main()

# Derived from RAX LIBERO adapter d49c8ad; benchmark IO adaptation only.
"""Host-side owner of one RoboMME-Plus simulator subprocess (stdlib + msgpack + numpy only).

The subprocess runs ``python -m services.robomme_plus.server`` in ``runtime/envs/robomme_plus`` with the
pinned RoboMME-Plus checkout on PYTHONPATH (the package is not pip-installed, so the checkout stays
clean) and ``LIBERO_CONFIG_PATH`` at the runtime-owned config, so LIBERO never prompts for paths.
This module never imports libero, robosuite or mujoco.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from services.robomme_plus.tasks import COLLECTION_SUFFIX, EXPLORATION_COLLECTIONS, runtime_root, source_root, task_info
from services.simulator.io import exit_report, log_size, simulator_request

RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_STARTUP_SECONDS = 300


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def robomme_plus_python() -> str:
    return os.environ.get("ROBOMME_PLUS_PYTHON") or str(source_root() / ".venv/bin/python")


def simulator_environment(sim_gpu, base=None):
    env = dict(os.environ if base is None else base)
    for key in ("PYTHONHOME", "CONDA_PREFIX", "RAX_AGENT_TRIAL_ID"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(sim_gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join([str(project_root()), str(source_root() / "src"),
                                          str(Path(__file__).resolve().parents[3])]),
               XDG_RUNTIME_DIR=str(runtime_root() / "xdg-runtime"))
    return env


class RoboMMEPlusProcess:
    """One simulator process, its RPC client and its identity-checked requests."""

    def __init__(self, *, output_root: Path, python: str | None = None):
        self.output_root = Path(output_root)
        self.python = python or robomme_plus_python()
        self.process: subprocess.Popen[str] | None = None
        self.rpc = None
        self.ready = threading.Event()
        self.bound_port: int | None = None
        self.log_path: Path | None = None
        self.output_dir: Path | None = None
        self.run_id: str | None = None
        self.task: str | None = None
        self.episode_id: str | None = None
        self.step_id = 0
        self.metadata: dict[str, Any] | None = None
        self.finished = False
        self.log_mark = 0
        self.startup_seconds: float | None = None

    @property
    def active(self) -> bool:
        return self.process is not None and self.rpc is not None and not self.finished

    def _read_log(self, stream, log_file) -> None:
        try:
            for line in iter(stream.readline, ""):
                log_file.write(line)
                log_file.flush()
                if '"event": "ready"' not in line:
                    continue
                with suppress(json.JSONDecodeError):
                    packet = json.loads(line)
                    if packet.get("event") == "ready":
                        self.bound_port = int(packet["port"])
                        self.ready.set()
        finally:
            with suppress(OSError):
                stream.close()
            with suppress(OSError):
                log_file.close()

    def start(self, *, task: str, collection: int, layout_id: int, run_id: str, sim_gpu,
              startup_timeout: float = DEFAULT_STARTUP_SECONDS, include_depth: bool = True,
              episode_timeout_seconds: int = 0) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("A RoboMME-Plus episode is already active; finish it before starting another")
        task_info(task)
        if collection not in EXPLORATION_COLLECTIONS and collection not in COLLECTION_SUFFIX:
            raise ValueError("collection must be 0 (training) or 1 (frozen test)")
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, numbers, dots, underscores and hyphens")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = (self.output_root / run_id).resolve()
        if self.output_dir.exists():
            raise ValueError(f"Artifact identifier already exists: {run_id}")
        self.output_dir.mkdir(parents=True)
        (self.output_dir / "sim").mkdir()
        (self.output_dir / "native_runtime").mkdir()
        self.run_id, self.log_path = run_id, self.output_dir / "sim.log"
        env = simulator_environment(sim_gpu)
        Path(env["XDG_RUNTIME_DIR"]).mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [self.python, "-m", "services.robomme_plus.server", "--task", task,
                   "--output", str(self.output_dir / "sim"), "--port", "0",
                   "--collection", str(int(collection)),
                   "--episode-timeout-seconds", str(int(episode_timeout_seconds)),
                   "--depth" if include_depth else "--no-depth"]
        started = time.monotonic()
        log_handle = self.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, cwd=self.output_dir / "native_runtime", env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1, start_new_session=True)
        except Exception:
            log_handle.close()
            raise
        self.ready.clear()
        threading.Thread(target=self._read_log, args=(self.process.stdout, log_handle),
                         name="libero-pro-sim-log", daemon=True).start()
        deadline = started + startup_timeout
        try:
            while not self.ready.wait(timeout=min(1.0, max(0.0, deadline - time.monotonic()))):
                if self.process.poll() is not None:
                    time.sleep(1)  # Let the log reader drain the pipe.
                    raise RuntimeError(f"RoboMME-Plus simulator exited during startup: {exit_report(self.process, self.log_path)}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"RoboMME-Plus simulator did not become ready within {startup_timeout:g}s")
            from services.simulator.protocol import RPCClient
            timeout = max(900.0, startup_timeout)
            if os.environ.get('ROBOMME_AGENTIC_RUNTIME_POLICY'):
                from benchmark_io.runtime_policy import configured
                timeout = configured().startup_seconds
            self.rpc = RPCClient("127.0.0.1", self.bound_port, timeout=timeout)
            self.metadata = self.rpc.request("metadata")
            if self.metadata.get("task") != task:
                raise RuntimeError(f"Simulator task mismatch: expected {task!r}, got {self.metadata.get('task')!r}")
            reset = self.rpc.request("reset", layout_id=int(layout_id), source="agent", policy_version="libero-pro-mcp")
            if os.environ.get('ROBOMME_AGENTIC_RUNTIME_POLICY'):
                self.rpc.sock.settimeout(configured().simulator_rpc_seconds)
            self.episode_id, self.step_id = reset["episode_id"], int(reset["step_id"])
            self.task, self.finished = task, False
            self.log_mark = log_size(self.log_path)  # Startup output is not an episode error.
            self.startup_seconds = time.monotonic() - started
            return reset
        except Exception:
            self.stop()
            raise

    def request(self, op: str, **kwargs: Any) -> Any:
        if self.rpc is None or self.episode_id is None or self.finished:
            raise RuntimeError("No active RoboMME-Plus episode")
        result = simulator_request(self, op, episode_id=self.episode_id, step_id=self.step_id, **kwargs)
        if isinstance(result, dict) and result.get("step_id") is not None and result.get("episode_id") == self.episode_id:
            self.step_id = int(result["step_id"])
        return result

    def finish(self, reason: str) -> Any:
        if self.rpc is None or self.episode_id is None:
            raise RuntimeError("No active RoboMME-Plus episode")
        if self.finished:
            return {"episode_id": self.episode_id, "step_id": self.step_id, "reason": reason}
        result = self.rpc.request("finish_pilot", episode_id=self.episode_id, step_id=self.step_id, reason=reason)
        self.finished = True
        return result

    def stop(self) -> None:
        rpc, process = self.rpc, self.process
        self.rpc = self.process = None
        self.finished = True
        if rpc is not None:
            with suppress(Exception):
                rpc.close()
        if process is not None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    with suppress(OSError):
                        os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

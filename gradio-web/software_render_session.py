from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import traceback
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("robomme.software_render")
DEFAULT_SOFTWARE_VULKAN_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
SOFTWARE_RENDER_MODE_ENV = "ROBOMME_FORCE_SOFTWARE_RENDER_MODE"
RENDER_BACKEND_AUTO_ENV = "ROBOMME_RENDER_BACKEND_AUTO"
SKIP_APP_BOOTSTRAP_ENV = "ROBOMME_SKIP_APP_BOOTSTRAP"
SOFTWARE_RENDER_CANDIDATES = ["pci:0000:00:00.0", "cpu"]


class SoftwareRenderUnsupportedError(RuntimeError):
    """Raised when the current environment cannot run SAPIEN software rendering."""


class RemoteSessionError(RuntimeError):
    """Structured error returned from the software render subprocess."""

    def __init__(self, error_type: str, message: str, traceback_text: str | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.traceback_text = traceback_text


def _ensure_repo_paths() -> None:
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    src_dir = parent_dir / "src"
    for path in (parent_dir, current_dir, src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _configure_child_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format=(
                "%(asctime)s | %(levelname)s | %(name)s | "
                "pid=%(process)d tid=%(threadName)s | %(message)s"
            ),
            stream=sys.stdout,
            force=True,
        )
    root.setLevel(level)


def _sanitize_options(options):
    clean_opts = []
    if not options:
        return clean_opts
    for opt in options:
        clean_opt = opt.copy()
        if "solve" in clean_opt:
            del clean_opt["solve"]
        if "available" in clean_opt:
            clean_opt["available"] = bool(clean_opt["available"])
        clean_opts.append(clean_opt)
    return clean_opts


def _build_snapshot(session, *, last_execution_frames=None):
    return {
        "env_id": session.env_id,
        "episode_idx": session.episode_idx,
        "language_goal": session.language_goal,
        "difficulty": session.difficulty,
        "seed": session.seed,
        "demonstration_frames": list(session.demonstration_frames or []),
        "base_frames": list(session.base_frames or []),
        "wrist_frames": list(session.wrist_frames or []),
        "available_options": list(session.available_options or []),
        "raw_solve_options": _sanitize_options(session.raw_solve_options),
        "seg_vis": session.seg_vis,
        "is_demonstration": bool(
            getattr(session.env, "current_task_demonstration", False) if session.env else False
        ),
        "non_demonstration_task_length": session.non_demonstration_task_length,
        "last_execution_frames": list(last_execution_frames or []),
    }


def _configure_software_render_env() -> None:
    os.environ["VK_ICD_FILENAMES"] = DEFAULT_SOFTWARE_VULKAN_ICD
    os.environ[SOFTWARE_RENDER_MODE_ENV] = "1"
    os.environ[RENDER_BACKEND_AUTO_ENV] = "1"
    os.environ.pop("ROBOMME_RENDER_BACKEND", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)


def _probe_software_render_backend():
    import sapien

    attempts = []
    for backend in SOFTWARE_RENDER_CANDIDATES:
        try:
            device = sapien.Device(backend)
            render_system = sapien.render.RenderSystem(device)
            del render_system
            try:
                device_summary = sapien.render.get_device_summary()
            except Exception as exc:
                device_summary = f"<unavailable: {exc}>"
            return {
                "backend": backend,
                "device_name": getattr(device, "name", None),
                "pci_string": getattr(device, "pci_string", None),
                "device_summary": device_summary,
            }
        except Exception as exc:
            attempts.append(f"{backend}: {exc}")

    attempts_text = "; ".join(attempts) if attempts else "no candidates tried"
    vk_icd = os.environ.get("VK_ICD_FILENAMES")
    vk_icd_exists = Path(vk_icd).exists() if vk_icd else False
    raise SoftwareRenderUnsupportedError(
        "Current Hugging Face ZeroGPU Space only provides compute access, and "
        "SAPIEN software Vulkan rendering is unavailable. Please use a standard GPU Space. "
        f"Details: attempts={attempts_text}; VK_ICD_FILENAMES={vk_icd}; icd_exists={vk_icd_exists}"
    )


def _software_render_worker_main(conn, dataset_root, gui_render):
    _ensure_repo_paths()
    _configure_child_logging()
    session = None
    try:
        _configure_software_render_env()
        LOGGER.info(
            "Software render subprocess starting with VK_ICD_FILENAMES=%s candidates=%s",
            os.environ.get("VK_ICD_FILENAMES"),
            SOFTWARE_RENDER_CANDIDATES,
        )
        probe = _probe_software_render_backend()
        LOGGER.info(
            "Software render self-test succeeded backend=%s device_name=%s pci=%s",
            probe.get("backend"),
            probe.get("device_name"),
            probe.get("pci_string"),
        )
        LOGGER.debug("Software render device summary:\n%s", probe.get("device_summary"))

        from oracle_logic import OracleSession

        session = OracleSession(dataset_root=dataset_root, gui_render=gui_render)
        conn.send(
            {
                "ok": True,
                "event": "ready",
                "startup": probe,
                "snapshot": _build_snapshot(session),
            }
        )
    except Exception as exc:
        conn.send(
            {
                "ok": False,
                "event": "startup_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        conn.close()
        return

    while True:
        try:
            payload = conn.recv()
        except EOFError:
            break

        method = payload.get("method")
        args = payload.get("args", ())
        kwargs = payload.get("kwargs", {})

        if method == "close":
            try:
                if session is not None:
                    session.close()
            finally:
                conn.send({"ok": True, "result": None, "snapshot": _build_snapshot(session)})
                conn.close()
                return

        try:
            if method == "load_episode":
                result = session.load_episode(*args, **kwargs)
                snapshot = _build_snapshot(session)
            elif method == "update_observation":
                result = session.update_observation(*args, **kwargs)
                snapshot = _build_snapshot(session)
            elif method == "execute_action":
                before = len(session.base_frames or [])
                result = session.execute_action(*args, **kwargs)
                snapshot = _build_snapshot(
                    session,
                    last_execution_frames=list((session.base_frames or [])[before:]),
                )
            elif method == "get_reference_action":
                result = session.get_reference_action(*args, **kwargs)
                snapshot = _build_snapshot(session)
            else:
                raise RuntimeError(f"Unsupported software render RPC method: {method}")

            conn.send({"ok": True, "result": result, "snapshot": snapshot})
        except Exception as exc:
            conn.send(
                {
                    "ok": False,
                    "event": "call_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


class SoftwareRenderSessionClient:
    def __init__(self, dataset_root=None, gui_render=False):
        self.dataset_root = str(dataset_root) if dataset_root is not None else None
        self.gui_render = gui_render
        self._ctx = multiprocessing.get_context("spawn")
        self._conn = None
        self._proc = None
        self.startup_info = None
        self.startup_error = None

    def _start(self):
        parent_conn, child_conn = self._ctx.Pipe()
        previous_skip_bootstrap = os.environ.get(SKIP_APP_BOOTSTRAP_ENV)
        os.environ[SKIP_APP_BOOTSTRAP_ENV] = "1"
        proc = self._ctx.Process(
            target=_software_render_worker_main,
            args=(child_conn, self.dataset_root, self.gui_render),
            daemon=True,
        )
        try:
            proc.start()
        finally:
            if previous_skip_bootstrap is None:
                os.environ.pop(SKIP_APP_BOOTSTRAP_ENV, None)
            else:
                os.environ[SKIP_APP_BOOTSTRAP_ENV] = previous_skip_bootstrap
        child_conn.close()

        if not parent_conn.poll(60):
            proc.terminate()
            proc.join(timeout=5)
            raise RuntimeError("Software render subprocess startup timed out")

        try:
            startup = parent_conn.recv()
        except EOFError as exc:
            proc.join(timeout=5)
            raise RuntimeError("Software render subprocess exited before completing startup") from exc
        if not startup.get("ok", False):
            proc.join(timeout=5)
            self.startup_error = SoftwareRenderUnsupportedError(startup.get("message", "startup error"))
            LOGGER.warning(
                "Software render subprocess startup failed: %s\n%s",
                startup.get("message"),
                startup.get("traceback"),
            )
            raise self.startup_error

        self._conn = parent_conn
        self._proc = proc
        self.startup_info = startup.get("startup")
        return startup

    def _ensure_started(self):
        if self.startup_error is not None:
            raise self.startup_error
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            return
        self._start()

    def call(self, method: str, *args, **kwargs):
        self._ensure_started()
        assert self._conn is not None
        assert self._proc is not None

        self._conn.send({"method": method, "args": args, "kwargs": kwargs})
        if not self._conn.poll(300):
            raise RuntimeError(f"Software render subprocess timed out while handling {method}")

        payload = self._conn.recv()
        if payload.get("ok", False):
            return payload

        raise RemoteSessionError(
            payload.get("error_type", "RuntimeError"),
            payload.get("message", "software render subprocess error"),
            payload.get("traceback"),
        )

    def close(self):
        if self._conn is None or self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self._conn.send({"method": "close", "args": (), "kwargs": {}})
                if self._conn.poll(10):
                    self._conn.recv()
        except Exception:
            pass
        finally:
            try:
                self._conn.close()
            except Exception:
                pass
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=5)
            self._conn = None
            self._proc = None

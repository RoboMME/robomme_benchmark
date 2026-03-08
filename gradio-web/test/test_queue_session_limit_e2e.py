from __future__ import annotations

import contextlib
import importlib
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from PIL import Image


gr = pytest.importorskip("gradio")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("playwright.sync_api")

import uvicorn
from fastapi import FastAPI
from playwright.sync_api import sync_playwright


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http_ready(url: str, timeout_s: float = 20.0) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urlopen(url, timeout=1.0) as resp:  # noqa: S310 - local test URL only
                if int(getattr(resp, "status", 200)) < 500:
                    return
        except URLError:
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Server did not become ready: {url}")


def _wait_until(predicate, timeout_s: float = 10.0, interval_s: float = 0.1) -> None:
    end = time.time() + timeout_s
    while time.time() < end:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError("Condition was not met before timeout")


def _minimal_load_result(uid: str, log_text: str = "ready"):
    obs = Image.new("RGB", (32, 32), color=(12, 24, 36))
    return (
        uid,
        gr.update(visible=True),
        obs,
        log_text,
        gr.update(choices=[("pick", 0)], value=None),
        "goal",
        "No need for coordinates",
        gr.update(value=None, visible=False),
        gr.update(visible=False, interactive=False),
        "BinFill (Episode 1)",
        "Completed: 0",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value="hint"),
        gr.update(visible=False),
        gr.update(interactive=True),
    )


def _mount_demo(demo):
    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="queue-session-limit-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)
    return root_url, demo, server, thread


def test_gradio_queue_limits_init_loads_to_four(monkeypatch):
    importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    def fake_init_app(request):
        uid = str(getattr(request, "session_hash", "missing"))
        time.sleep(6.0)
        return _minimal_load_result(uid, log_text=f"ready:{uid}")

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            pages = []
            for _ in range(5):
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(root_url, wait_until="domcontentloaded")
                pages.append(page)
                time.sleep(0.25)

            snapshots = {}

            def _queue_snapshot_ready():
                texts = [page.evaluate("() => document.body.innerText") for page in pages]
                snapshots["texts"] = texts
                first_four_ready = all("processing" in text.lower() for text in texts[:4])
                fifth_queued = "queue:" in texts[4].lower()
                return first_four_ready and fifth_queued

            _wait_until(_queue_snapshot_ready, timeout_s=10.0)

            first_four = snapshots["texts"][:4]
            fifth_text = snapshots["texts"][4]

            assert all("processing" in text.lower() for text in first_four)
            assert "queue:" in fifth_text.lower()

            pages[0].wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            pages[4].wait_for_selector("#main_interface_root", state="visible", timeout=25000)

            loaded_text = pages[4].evaluate("() => document.body.innerText")
            assert "queue:" not in loaded_text.lower()
            assert "processing" not in loaded_text.lower()

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_gradio_state_ttl_cleans_up_idle_session(monkeypatch):
    state_manager = importlib.reload(importlib.import_module("state_manager"))
    user_manager_mod = importlib.reload(importlib.import_module("user_manager"))
    importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    monkeypatch.setattr(ui_layout, "SESSION_TIMEOUT", 2)

    closed = []

    class _FakeProxy:
        def __init__(self, uid):
            self.uid = uid

        def close(self):
            closed.append(self.uid)

    def fake_init_app(request):
        uid = str(getattr(request, "session_hash", "missing"))
        state_manager.GLOBAL_SESSIONS[uid] = _FakeProxy(uid)
        user_manager_mod.user_manager.session_progress[uid] = {
            "completed_count": 0,
            "current_env_id": "BinFill",
            "current_episode_idx": 1,
        }
        return _minimal_load_result(uid)

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)

            uid = next(iter(state_manager.GLOBAL_SESSIONS))
            assert uid in user_manager_mod.user_manager.session_progress

            _wait_until(
                lambda: uid in closed
                and uid not in state_manager.GLOBAL_SESSIONS
                and uid not in user_manager_mod.user_manager.session_progress,
                timeout_s=8.0,
            )

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()

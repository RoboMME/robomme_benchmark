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
        gr.update(interactive=True),
    )


def _read_progress_text(page) -> str | None:
    return page.evaluate(
        """() => {
            const node = document.querySelector('.progress-text');
            if (!node) return null;
            const text = (node.textContent || '').trim();
            return text || null;
        }"""
    )


def _read_unified_overlay_text(page) -> str | None:
    progress_text = _read_progress_text(page)
    if progress_text:
        return progress_text
    snapshot = _read_session_wait_overlay_snapshot(page)
    content = snapshot.get("content")
    return str(content).strip() if content else None


def _read_progress_overlay_snapshot(page) -> dict[str, float | bool | None]:
    return page.evaluate(
        """() => {
            const node = document.querySelector('#native_progress_host .wrap');
            if (!node) {
                return { present: false, width: null, height: null, background: null };
            }
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return {
                present: true,
                width: rect.width,
                height: rect.height,
                background: style.backgroundColor || null,
            };
        }"""
    )


def _read_session_wait_overlay_snapshot(page) -> dict[str, float | bool | None]:
    return page.evaluate(
        """() => {
            const host = document.getElementById('native_progress_host');
            if (!host) {
                return {
                    present: false,
                    visible: false,
                    width: null,
                    height: null,
                    background: null,
                    content: null,
                    proseBackground: null,
                    proseBorderRadius: null,
                    proseBoxShadow: null,
                };
            }
            const markdown = host.querySelector('[data-testid="markdown"]');
            const prose = markdown ? markdown.querySelector('.prose, .md') || markdown : null;
            const text = prose ? ((prose.innerText || prose.textContent || '').trim()) : '';
            const rect = host.getBoundingClientRect();
            const style = getComputedStyle(host);
            const proseStyle = prose ? getComputedStyle(prose) : null;
            return {
                present: true,
                visible: style.display !== 'none' && rect.width > 0 && rect.height > 0 && text.length > 0,
                width: rect.width,
                height: rect.height,
                background: style.backgroundColor || null,
                content: text || null,
                proseBackground: proseStyle ? proseStyle.backgroundColor || null : null,
                proseBorderRadius: proseStyle ? proseStyle.borderRadius || null : null,
                proseBoxShadow: proseStyle ? proseStyle.boxShadow || null : null,
            };
        }"""
    )


def _read_log_output_value(page) -> str | None:
    return page.evaluate(
        """() => {
            const root = document.getElementById('log_output');
            if (!root) return null;
            const field = root.querySelector('textarea, input');
            if (field && typeof field.value === 'string') {
                const value = field.value.trim();
                return value || null;
            }
            const value = (root.textContent || '').trim();
            return value || null;
        }"""
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


def test_gradio_queue_respects_configured_limit_on_init_load(monkeypatch):
    config = importlib.reload(importlib.import_module("config"))
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
            total_pages = int(config.SESSION_INIT_CONCURRENCY_LIMIT) + 2
            for _ in range(total_pages):
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(root_url, wait_until="domcontentloaded")
                pages.append(page)
                time.sleep(0.25)

            def _queue_snapshot_ready():
                progress_texts = [_read_unified_overlay_text(page) for page in pages]
                processing_ready = all(
                    text and config.UI_TEXT["progress"]["episode_loading"] in text
                    for text in progress_texts[: config.SESSION_INIT_CONCURRENCY_LIMIT]
                )
                queued_ready = all(
                    text
                    and config.UI_TEXT["progress"]["queue_wait"] in text
                    and "queue:" in text.lower()
                    for text in progress_texts[config.SESSION_INIT_CONCURRENCY_LIMIT :]
                )
                return processing_ready and queued_ready

            _wait_until(_queue_snapshot_ready, timeout_s=10.0)
            processing_pages = [
                _read_unified_overlay_text(page) or ""
                for page in pages[: config.SESSION_INIT_CONCURRENCY_LIMIT]
            ]
            queued_pages = [
                _read_unified_overlay_text(page) or ""
                for page in pages[config.SESSION_INIT_CONCURRENCY_LIMIT :]
            ]

            assert all(config.UI_TEXT["progress"]["episode_loading"] in text for text in processing_pages)
            assert all(
                config.UI_TEXT["progress"]["queue_wait"] in text and "queue:" in text.lower()
                for text in queued_pages
            )
            assert pages[0].evaluate("() => !!document.getElementById('loading_overlay_group')") is False
            overlay_snapshot = _read_progress_overlay_snapshot(pages[-1])
            assert overlay_snapshot["present"] is True
            assert overlay_snapshot["width"] and overlay_snapshot["width"] > 0
            assert overlay_snapshot["height"] and overlay_snapshot["height"] >= 400
            assert overlay_snapshot["background"] == "rgba(255, 255, 255, 0.92)"
            session_wait_snapshot = _read_session_wait_overlay_snapshot(pages[-1])
            assert session_wait_snapshot["content"] in (None, "")

            _wait_until(lambda: _read_progress_text(pages[0]) is None, timeout_s=15.0)
            _wait_until(lambda: _read_progress_text(pages[-1]) is None, timeout_s=25.0)

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

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return Image.new("RGB", (32, 32), color=(10, 20, 30))

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


def test_single_load_uses_native_episode_loading_copy(monkeypatch):
    config = importlib.reload(importlib.import_module("config"))
    importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    def fake_init_app(request):
        uid = str(getattr(request, "session_hash", "missing"))
        time.sleep(2.5)
        return _minimal_load_result(uid)

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")

            _wait_until(
                lambda: (_read_progress_text(page) or "").startswith(config.UI_TEXT["progress"]["episode_loading"]),
                timeout_s=8.0,
            )
            assert page.evaluate("() => !!document.getElementById('loading_overlay_group')") is False
            overlay_snapshot = _read_progress_overlay_snapshot(page)
            assert overlay_snapshot["present"] is True
            assert overlay_snapshot["width"] and overlay_snapshot["width"] > 0
            assert overlay_snapshot["height"] and overlay_snapshot["height"] >= 400
            assert overlay_snapshot["background"] == "rgba(255, 255, 255, 0.92)"

            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            _wait_until(lambda: _read_progress_text(page) is None, timeout_s=8.0)

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_execute_does_not_use_episode_loading_copy(monkeypatch):
    importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    obs = Image.new("RGB", (32, 32), color=(10, 20, 30))

    def fake_init_app(request):
        uid = str(getattr(request, "session_hash", "missing"))
        return _minimal_load_result(uid, log_text="ready")

    def fake_precheck_execute_inputs(uid, option_idx, coords_str):
        return None

    def fake_switch_to_execute_phase(uid):
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def fake_execute_step(uid, option_idx, coords_str):
        time.sleep(1.5)
        return (
            gr.update(value=obs, interactive=False),
            "executed",
            "BinFill (Episode 1)",
            "Completed: 0",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def fake_switch_to_action_phase(uid=None):
        return (
            gr.update(interactive=True),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)
    monkeypatch.setattr(ui_layout, "precheck_execute_inputs", fake_precheck_execute_inputs)
    monkeypatch.setattr(ui_layout, "switch_to_execute_phase", fake_switch_to_execute_phase)
    monkeypatch.setattr(ui_layout, "execute_step", fake_execute_step)
    monkeypatch.setattr(ui_layout, "switch_to_action_phase", fake_switch_to_action_phase)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            page.locator("#exec_btn button, button#exec_btn").first.click()
            page.wait_for_timeout(500)

            body_text = page.evaluate("() => document.body.innerText")
            assert "The episode is loading..." not in body_text
            assert "Lots of people are playing! Please wait..." not in body_text

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_late_user_waits_for_active_session_slot_release(monkeypatch):
    config = importlib.reload(importlib.import_module("config"))
    state_manager = importlib.reload(importlib.import_module("state_manager"))
    callbacks = importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    closed = []

    class _FakeProxy:
        def __init__(self):
            self.env_id = None
            self.episode_idx = None
            self.language_goal = "goal"
            self.available_options = [("pick", 0)]
            self.raw_solve_options = [{"label": "a", "action": "pick", "available": False}]
            self.demonstration_frames = []

        def load_episode(self, env_id, episode_idx):
            self.env_id = env_id
            self.episode_idx = episode_idx
            return Image.new("RGB", (32, 32), color=(10, 20, 30)), "loaded"

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return Image.new("RGB", (32, 32), color=(10, 20, 30))

        def close(self):
            closed.append((self.env_id, self.episode_idx))

    monkeypatch.setattr(state_manager, "ProcessSessionProxy", _FakeProxy)
    monkeypatch.setattr(
        callbacks.user_manager,
        "init_session",
        lambda uid: (
            True,
            "ok",
            {"current_task": {"env_id": "BinFill", "episode_idx": 1}, "completed_count": 0},
        ),
    )
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: False)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            page1 = browser.new_page(viewport={"width": 1280, "height": 900})
            page1.goto(root_url, wait_until="domcontentloaded")
            _wait_until(lambda: len(state_manager.GLOBAL_SESSIONS) == 1, timeout_s=15.0)
            _wait_until(lambda: _read_progress_text(page1) is None, timeout_s=15.0)

            page2 = browser.new_page(viewport={"width": 1280, "height": 900})
            page2.goto(root_url, wait_until="domcontentloaded")
            _wait_until(lambda: len(state_manager.GLOBAL_SESSIONS) == 2, timeout_s=15.0)
            _wait_until(lambda: _read_progress_text(page2) is None, timeout_s=15.0)

            assert len(state_manager.GLOBAL_SESSIONS) == config.SESSION_CONCURRENCY_LIMIT
            assert len(state_manager.ACTIVE_SESSION_SLOTS) == config.SESSION_CONCURRENCY_LIMIT

            page3 = browser.new_page(viewport={"width": 1280, "height": 900})
            page3.goto(root_url, wait_until="domcontentloaded")

            _wait_until(
                lambda: _read_session_wait_overlay_snapshot(page3)["visible"] is True,
                timeout_s=10.0,
            )
            time.sleep(1.0)
            assert len(state_manager.GLOBAL_SESSIONS) == config.SESSION_CONCURRENCY_LIMIT
            assert len(state_manager.ACTIVE_SESSION_SLOTS) == config.SESSION_CONCURRENCY_LIMIT
            assert _read_session_wait_overlay_snapshot(page3)["visible"] is True
            wait_snapshot = _read_session_wait_overlay_snapshot(page3)
            assert wait_snapshot["content"] == f'{config.UI_TEXT["progress"]["queue_wait"]} | queue: 1'
            assert wait_snapshot["proseBackground"] == "rgba(0, 0, 0, 0)"
            assert wait_snapshot["proseBorderRadius"] == "0px"
            assert wait_snapshot["proseBoxShadow"] == "none"

            page1.close()

            _wait_until(lambda: len(closed) >= 1, timeout_s=10.0)
            _wait_until(lambda: _read_session_wait_overlay_snapshot(page3)["visible"] is False, timeout_s=15.0)
            _wait_until(lambda: len(state_manager.GLOBAL_SESSIONS) == config.SESSION_CONCURRENCY_LIMIT, timeout_s=10.0)
            _wait_until(lambda: len(state_manager.ACTIVE_SESSION_SLOTS) == config.SESSION_CONCURRENCY_LIMIT, timeout_s=10.0)

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_second_and_later_late_users_show_queue_overlay(monkeypatch):
    config = importlib.reload(importlib.import_module("config"))
    state_manager = importlib.reload(importlib.import_module("state_manager"))
    callbacks = importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    class _FakeProxy:
        def __init__(self):
            self.env_id = None
            self.episode_idx = None
            self.language_goal = "goal"
            self.available_options = [("pick", 0)]
            self.raw_solve_options = [{"label": "a", "action": "pick", "available": False}]
            self.demonstration_frames = []

        def load_episode(self, env_id, episode_idx):
            self.env_id = env_id
            self.episode_idx = episode_idx
            return Image.new("RGB", (32, 32), color=(10, 20, 30)), "loaded"

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return Image.new("RGB", (32, 32), color=(10, 20, 30))

        def close(self):
            return None

    monkeypatch.setattr(state_manager, "ProcessSessionProxy", _FakeProxy)
    monkeypatch.setattr(
        callbacks.user_manager,
        "init_session",
        lambda uid: (
            True,
            "ok",
            {"current_task": {"env_id": "BinFill", "episode_idx": 1}, "completed_count": 0},
        ),
    )
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: False)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            active_pages = []
            for _ in range(config.SESSION_CONCURRENCY_LIMIT):
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(root_url, wait_until="domcontentloaded")
                active_pages.append(page)
                time.sleep(0.25)

            for page in active_pages:
                _wait_until(lambda page=page: _read_progress_text(page) is None, timeout_s=15.0)

            waiting_page = browser.new_page(viewport={"width": 1280, "height": 900})
            waiting_page.goto(root_url, wait_until="domcontentloaded")

            queued_pages = []
            for _ in range(2):
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(root_url, wait_until="domcontentloaded")
                queued_pages.append(page)
                time.sleep(0.25)

            pending_pages = [waiting_page, *queued_pages]

            def _pending_pages_ready():
                snapshots = [_read_session_wait_overlay_snapshot(page) for page in pending_pages]
                return all(snapshot["visible"] is True for snapshot in snapshots)

            _wait_until(_pending_pages_ready, timeout_s=10.0)

            for page in pending_pages:
                overlay_snapshot = _read_session_wait_overlay_snapshot(page)
                assert overlay_snapshot["present"] is True
                assert overlay_snapshot["width"] and overlay_snapshot["width"] > 0
                assert overlay_snapshot["height"] and overlay_snapshot["height"] >= 400
                assert config.UI_TEXT["progress"]["queue_wait"] in str(overlay_snapshot["content"] or "")
                assert "queue:" in str(overlay_snapshot["content"] or "").lower()
                assert overlay_snapshot["proseBackground"] == "rgba(0, 0, 0, 0)"
                assert overlay_snapshot["proseBorderRadius"] == "0px"
                assert overlay_snapshot["proseBoxShadow"] == "none"

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_active_user_execute_is_not_blocked_by_waiting_init_loads(monkeypatch):
    config = importlib.reload(importlib.import_module("config"))
    state_manager = importlib.reload(importlib.import_module("state_manager"))
    callbacks = importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    class _FakeProxy:
        def __init__(self):
            self.env_id = None
            self.episode_idx = None
            self.language_goal = "goal"
            self.available_options = [("pick", 0)]
            self.raw_solve_options = [{"label": "a", "action": "pick", "available": False}]
            self.demonstration_frames = []
            self.base_frames = [Image.new("RGB", (8, 8), color=(1, 2, 3))]
            self.difficulty = None
            self.seed = None

        def load_episode(self, env_id, episode_idx):
            self.env_id = env_id
            self.episode_idx = episode_idx
            return Image.new("RGB", (32, 32), color=(10, 20, 30)), "loaded"

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return Image.new("RGB", (32, 32), color=(10, 20, 30))

        def execute_action(self, option_idx, click_coords):
            _ = option_idx, click_coords
            time.sleep(0.5)
            return Image.new("RGB", (32, 32), color=(30, 40, 50)), "SUCCESS", False

        def update_observation(self, use_segmentation=False):
            _ = use_segmentation
            return self.get_pil_image(), ""

        def close(self):
            return None

    monkeypatch.setattr(state_manager, "ProcessSessionProxy", _FakeProxy)
    monkeypatch.setattr(
        callbacks.user_manager,
        "init_session",
        lambda uid: (
            True,
            "ok",
            {"current_task": {"env_id": "BinFill", "episode_idx": 1}, "completed_count": 0},
        ),
    )
    monkeypatch.setattr(callbacks, "get_task_hint", lambda env_id: "")
    monkeypatch.setattr(callbacks, "should_show_demo_video", lambda env_id: False)

    demo = ui_layout.create_ui_blocks()
    root_url, demo, server, thread = _mount_demo(demo)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            page1 = browser.new_page(viewport={"width": 1280, "height": 900})
            page1.goto(root_url, wait_until="domcontentloaded")
            _wait_until(lambda: _read_progress_text(page1) is None, timeout_s=15.0)

            page2 = browser.new_page(viewport={"width": 1280, "height": 900})
            page2.goto(root_url, wait_until="domcontentloaded")
            _wait_until(lambda: _read_progress_text(page2) is None, timeout_s=15.0)

            waiting_page = browser.new_page(viewport={"width": 1280, "height": 900})
            waiting_page.goto(root_url, wait_until="domcontentloaded")

            queued_page = browser.new_page(viewport={"width": 1280, "height": 900})
            queued_page.goto(root_url, wait_until="domcontentloaded")

            _wait_until(lambda: _read_session_wait_overlay_snapshot(waiting_page)["visible"] is True, timeout_s=10.0)
            _wait_until(lambda: _read_session_wait_overlay_snapshot(queued_page)["visible"] is True, timeout_s=10.0)

            page1.locator("#action_radio input[type='radio']").first.check()
            page1.locator("#exec_btn button, button#exec_btn").first.click()

            _wait_until(
                lambda: "SUCCESS" in (_read_log_output_value(page1) or ""),
                timeout_s=6.0,
            )
            page1_progress = _read_progress_text(page1) or ""
            assert "queue:" not in page1_progress.lower()
            assert config.UI_TEXT["progress"]["queue_wait"] not in page1_progress
            assert _read_session_wait_overlay_snapshot(waiting_page)["visible"] is True
            assert _read_session_wait_overlay_snapshot(queued_page)["visible"] is True

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()

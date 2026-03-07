from __future__ import annotations

import contextlib
import importlib
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
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


def _resolve_button_snapshot(page, elem_id: str) -> dict[str, str | bool | None]:
    return page.evaluate(
        """(elemId) => {
            const button = document.querySelector(`#${elemId} button`) || document.querySelector(`button#${elemId}`);
            if (!button) {
                return {
                    found: false,
                    disabled: null,
                    backgroundColor: null,
                    borderColor: null,
                    color: null,
                };
            }
            const style = getComputedStyle(button);
            return {
                found: true,
                disabled: button.disabled,
                backgroundColor: style.backgroundColor,
                borderColor: style.borderColor,
                color: style.color,
            };
        }""",
        elem_id,
    )


def _read_header_task_value(page) -> str | None:
    return page.evaluate(
        """() => {
            const root = document.getElementById('header_task');
            if (!root) return null;
            const input = root.querySelector('input');
            if (input && typeof input.value === 'string') {
                const value = input.value.trim();
                return value || null;
            }
            const selected = root.querySelector('.single-select');
            if (!selected) return null;
            const text = (selected.textContent || '').trim();
            return text || null;
        }"""
    )


def _read_coords_box_value(page) -> str | None:
    return page.evaluate(
        """() => {
            const root = document.getElementById('coords_box');
            if (!root) return null;
            const field = root.querySelector('textarea, input');
            if (!field) return null;
            const value = typeof field.value === 'string' ? field.value.trim() : '';
            return value || null;
        }"""
    )


def _read_live_obs_geometry(page) -> dict[str, dict[str, float] | None]:
    return page.evaluate(
        """() => {
            const root = document.getElementById('live_obs');
            const container = root?.querySelector('.image-container');
            const uploadContainer = root?.querySelector('.upload-container');
            const frame = root?.querySelector('.image-frame');
            const img = root?.querySelector('img');
            const measure = (node) => {
                if (!node) return null;
                const rect = node.getBoundingClientRect();
                return { width: rect.width, height: rect.height };
            };
            return {
                root: measure(root),
                container: measure(container),
                uploadContainer: measure(uploadContainer),
                frame: measure(frame),
                img: measure(img),
            };
        }"""
    )


def _read_font_probe_snapshot(page) -> dict[str, str | None]:
    return page.evaluate(
        """() => {
            const heading = document.querySelector('#header_title h2');
            const field = document.querySelector('#font_probe textarea, #font_probe input');
            const prose = document.querySelector('#body_probe p');
            const readSize = (node) => (node ? getComputedStyle(node).fontSize : null);
            return {
                header: readSize(heading),
                field: readSize(field),
                body: readSize(prose),
            };
        }"""
    )


@pytest.fixture
def font_size_probe_ui_url(monkeypatch):
    config_module = importlib.reload(importlib.import_module("config"))
    monkeypatch.setattr(config_module, "UI_GLOBAL_FONT_SIZE", "32px")
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    with gr.Blocks(title="Native font size probe test") as demo:
        gr.Markdown("## RoboMME Human Evaluation", elem_id="header_title")
        gr.Textbox(value="font probe", label="Probe", elem_id="font_probe")
        gr.Markdown("Probe body text", elem_id="body_probe")

    _app, root_url, _share_url = demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
        show_error=True,
        ssr_mode=False,
        css=ui_layout.CSS,
    )
    _wait_http_ready(root_url)

    try:
        yield root_url
    finally:
        demo.close()


@pytest.fixture
def phase_machine_ui_url():
    state = {"precheck_calls": 0}
    demo_video_url = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    with gr.Blocks(title="Native phase machine test") as demo:
        gr.HTML(f"<style>{ui_layout.CSS}</style>")
        phase_state = gr.State("init")

        with gr.Column(visible=True, elem_id="login_group") as login_group:
            login_btn = gr.Button("Login", elem_id="login_btn")

        with gr.Column(visible=False, elem_id="main_interface") as main_interface:
            with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                video_display = gr.Video(value=None, elem_id="demo_video", autoplay=True)

            with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
                img_display = gr.Image(value=np.zeros((24, 24, 3), dtype=np.uint8), elem_id="live_obs")

            with gr.Column(visible=False, elem_id="control_panel_group") as control_panel_group:
                options_radio = gr.Radio(choices=[("pick", 0)], value=0, elem_id="action_radio")
                coords_box = gr.Textbox(value="please click the keypoint selection image", elem_id="coords_box")
                with gr.Column(visible=False, elem_id="action_buttons_row") as action_buttons_row:
                    exec_btn = gr.Button("EXECUTE", elem_id="exec_btn")
                    reference_action_btn = gr.Button(
                        "Ground Truth Action",
                        elem_id="reference_action_btn",
                        interactive=False,
                    )
                    next_task_btn = gr.Button("Next Task", elem_id="next_task_btn")

        log_output = gr.Markdown("", elem_id="log_output")

        def login_fn():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value=demo_video_url, visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(value="please click the keypoint selection image"),
                "demo_video",
            )

        def on_video_end_fn():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(interactive=True),
                "action_keypoint",
            )

        def precheck_fn(_option_idx, _coords):
            state["precheck_calls"] += 1
            if state["precheck_calls"] == 1:
                raise gr.Error("please click the keypoint selection image before execute!")

        def to_execute_fn():
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "execution_playback",
            )

        def execute_fn():
            time.sleep(0.8)
            return (
                "executed",
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        def to_action_fn():
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                "action_keypoint",
            )

        login_btn.click(
            fn=login_fn,
            outputs=[
                login_group,
                main_interface,
                video_phase_group,
                video_display,
                action_phase_group,
                control_panel_group,
                action_buttons_row,
                reference_action_btn,
                coords_box,
                phase_state,
            ],
            queue=False,
        )

        video_display.end(
            fn=on_video_end_fn,
            outputs=[
                video_phase_group,
                action_phase_group,
                control_panel_group,
                action_buttons_row,
                reference_action_btn,
                phase_state,
            ],
            queue=False,
        )

        exec_btn.click(
            fn=precheck_fn,
            inputs=[options_radio, coords_box],
            outputs=[],
            queue=False,
        ).then(
            fn=to_execute_fn,
            outputs=[
                options_radio,
                exec_btn,
                next_task_btn,
                img_display,
                reference_action_btn,
                phase_state,
            ],
            queue=False,
        ).then(
            fn=execute_fn,
            outputs=[log_output, next_task_btn, exec_btn],
            queue=False,
        ).then(
            fn=to_action_fn,
            outputs=[options_radio, exec_btn, next_task_btn, img_display, reference_action_btn, phase_state],
            queue=False,
        )

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="native-phase-machine-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        yield root_url, state
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_global_font_size_applies_except_header_title(font_size_probe_ui_url):
    root_url = font_size_probe_ui_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(root_url, wait_until="domcontentloaded")

        page.wait_for_selector("#header_title h2", timeout=10000)
        page.wait_for_selector("#font_probe textarea, #font_probe input", timeout=10000)
        page.wait_for_selector("#body_probe p", timeout=10000)
        page.wait_for_function(
            """() => {
                const heading = document.querySelector('#header_title h2');
                const field = document.querySelector('#font_probe textarea, #font_probe input');
                const prose = document.querySelector('#body_probe p');
                if (!heading || !field || !prose) return false;
                return (
                    getComputedStyle(heading).fontSize === '26px' &&
                    getComputedStyle(field).fontSize === '32px' &&
                    getComputedStyle(prose).fontSize === '32px'
                );
            }""",
            timeout=10000,
        )

        snapshot = _read_font_probe_snapshot(page)
        assert snapshot["header"] == "26px"
        assert snapshot["field"] == "32px"
        assert snapshot["body"] == "32px"
        assert snapshot["header"] != snapshot["field"]

        browser.close()


def test_phase_machine_runtime_flow_and_execute_precheck(phase_machine_ui_url):
    root_url, state = phase_machine_ui_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(root_url, wait_until="domcontentloaded")

        page.wait_for_timeout(2500)
        page.wait_for_selector("#login_btn", timeout=20000)
        page.click("#login_btn")

        page.wait_for_function(
            """() => {
                const el = document.getElementById('demo_video');
                return !!el && getComputedStyle(el).display !== 'none';
            }"""
        )

        phase_after_login = page.evaluate(
            """() => {
                const visible = (id) => {
                    const el = document.getElementById(id);
                    if (!el) return false;
                    const st = getComputedStyle(el);
                    return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                };
                return {
                    video: visible('demo_video'),
                    action: visible('live_obs'),
                    control: visible('action_radio'),
                };
            }"""
        )
        assert phase_after_login == {
            "video": True,
            "action": False,
            "control": False,
        }

        page.wait_for_selector("#demo_video video", timeout=5000)
        did_dispatch_end = page.evaluate(
            """() => {
                const videoEl = document.querySelector('#demo_video video');
                if (!videoEl) return false;
                videoEl.dispatchEvent(new Event('ended', { bubbles: true }));
                return true;
            }"""
        )
        assert did_dispatch_end

        page.wait_for_function(
            """() => {
                const action = document.getElementById('live_obs');
                const control = document.getElementById('action_radio');
                if (!action || !control) return false;
                return getComputedStyle(action).display !== 'none' && getComputedStyle(control).display !== 'none';
            }"""
        )

        did_click_exec = page.evaluate(
            """() => {
                const btn = document.getElementById('exec_btn');
                if (!btn) return false;
                btn.click();
                return true;
            }"""
        )
        assert did_click_exec
        page.wait_for_timeout(300)

        phase_after_failed_precheck = page.evaluate(
            """() => {
                const visible = (id) => {
                    const el = document.getElementById(id);
                    if (!el) return false;
                    return getComputedStyle(el).display !== 'none';
                };
                return {
                    action: visible('live_obs'),
                };
            }"""
        )
        assert phase_after_failed_precheck == {"action": True}

        did_click_exec = page.evaluate(
            """() => {
                const btn = document.getElementById('exec_btn');
                if (!btn) return false;
                btn.click();
                return true;
            }"""
        )
        assert did_click_exec

        page.wait_for_function(
            """() => {
                const resolveButton = (id) => {
                    return document.querySelector(`#${id} button`) || document.querySelector(`button#${id}`);
                };
                const execBtn = resolveButton('exec_btn');
                const nextBtn = resolveButton('next_task_btn');
                return !!execBtn && !!nextBtn && execBtn.disabled === true && nextBtn.disabled === true;
            }"""
        )

        interactive_snapshot = page.evaluate(
            """() => {
                const resolveButton = (id) => {
                    return document.querySelector(`#${id} button`) || document.querySelector(`button#${id}`);
                };
                const execBtn = resolveButton('exec_btn');
                const nextBtn = resolveButton('next_task_btn');
                return {
                    execDisabled: execBtn ? execBtn.disabled : null,
                    nextDisabled: nextBtn ? nextBtn.disabled : null,
                };
            }"""
        )
        assert interactive_snapshot["execDisabled"] is True
        assert interactive_snapshot["nextDisabled"] is True

        page.wait_for_function(
            """() => {
                const execBtn = document.querySelector('button#exec_btn') || document.querySelector('#exec_btn button');
                const action = document.getElementById('live_obs');
                if (!execBtn || !action) return false;
                return execBtn.disabled === false && getComputedStyle(action).display !== 'none';
            }""",
            timeout=6000,
        )

        final_interactive_snapshot = page.evaluate(
            """() => {
                const resolveButton = (id) => {
                    return document.querySelector(`#${id} button`) || document.querySelector(`button#${id}`);
                };
                const execBtn = resolveButton('exec_btn');
                const nextBtn = resolveButton('next_task_btn');
                return {
                    execDisabled: execBtn ? execBtn.disabled : null,
                    nextDisabled: nextBtn ? nextBtn.disabled : null,
                };
            }"""
        )
        assert final_interactive_snapshot["execDisabled"] is False
        assert final_interactive_snapshot["nextDisabled"] is False

        browser.close()

    assert state["precheck_calls"] >= 2


def test_reference_action_button_is_green_only_when_interactive(phase_machine_ui_url):
    root_url, _state = phase_machine_ui_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(root_url, wait_until="domcontentloaded")

        page.wait_for_timeout(2500)
        page.wait_for_selector("#login_btn", timeout=20000)
        page.click("#login_btn")

        disabled_snapshot = _resolve_button_snapshot(page, "reference_action_btn")
        if disabled_snapshot["found"]:
            assert disabled_snapshot["disabled"] is True
            assert disabled_snapshot["backgroundColor"] != "rgb(31, 139, 76)"

        page.wait_for_selector("#demo_video video", timeout=5000)
        did_dispatch_end = page.evaluate(
            """() => {
                const videoEl = document.querySelector('#demo_video video');
                if (!videoEl) return false;
                videoEl.dispatchEvent(new Event('ended', { bubbles: true }));
                return true;
            }"""
        )
        assert did_dispatch_end

        page.wait_for_function(
            """() => {
                const button = document.querySelector('#reference_action_btn button') || document.querySelector('button#reference_action_btn');
                return !!button && button.disabled === false;
            }""",
            timeout=6000,
        )

        enabled_snapshot = _resolve_button_snapshot(page, "reference_action_btn")
        assert enabled_snapshot["found"] is True
        assert enabled_snapshot["disabled"] is False
        assert enabled_snapshot["backgroundColor"] == "rgb(31, 139, 76)"
        assert enabled_snapshot["borderColor"] == "rgb(31, 139, 76)"
        assert enabled_snapshot["color"] == "rgb(255, 255, 255)"

        browser.close()


def test_unified_loading_overlay_init_flow(monkeypatch):
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    canonical_copy = "Logging in and setting up environment... Please wait."
    legacy_copy = "Loading environment, please wait..."
    fake_obs = np.zeros((24, 24, 3), dtype=np.uint8)
    fake_obs_img = Image.fromarray(fake_obs)
    calls = {"init": 0}

    def fake_show_loading_info():
        return gr.update(visible=True)

    def fake_init_app(_request=None):
        calls["init"] += 1
        time.sleep(0.8)
        return (
            "uid-init",
            gr.update(visible=True),  # main_interface
            gr.update(value=fake_obs_img, interactive=False),  # img_display
            "ready",  # log_output
            gr.update(choices=[("pick", 0)], value=None),  # options_radio
            "goal",  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=None, visible=False),  # video_display
            "PickXtimes (Episode 1)",  # task_info_box
            "Completed: 0",  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=True),   # action_phase_group
            gr.update(visible=True),   # control_panel_group
            gr.update(value="hint"),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
            gr.update(interactive=True),  # reference_action_btn
        )

    monkeypatch.setattr(ui_layout, "show_loading_info", fake_show_loading_info)
    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)

    demo = ui_layout.create_ui_blocks()

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="native-unified-loading-overlay-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")

            page.wait_for_selector("#loading_overlay_group", state="visible", timeout=2500)

            overlay_text = page.evaluate(
                """() => {
                    const el = document.getElementById('loading_overlay_group');
                    return el ? (el.textContent || '') : '';
                }"""
            )
            assert canonical_copy in overlay_text
            assert legacy_copy not in page.content()

            page.wait_for_selector("#loading_overlay_group", state="hidden", timeout=15000)
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            page.wait_for_function(
                """() => {
                    const root = document.getElementById('header_task');
                    const input = root ? root.querySelector('input') : null;
                    return !!input && input.value.trim() === 'PickXtimes';
                }""",
                timeout=5000,
            )
            assert _read_header_task_value(page) == "PickXtimes"

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()

    assert calls["init"] >= 1


def test_live_obs_client_resize_fills_width_and_keeps_click_mapping(monkeypatch):
    callbacks = importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    fake_obs = np.zeros((24, 48, 3), dtype=np.uint8)
    fake_obs_img = Image.fromarray(fake_obs)

    class FakeSession:
        raw_solve_options = [{"available": True}]

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return fake_obs_img.copy()

    def fake_init_app(_request=None):
        return (
            "uid-live-obs-resize",
            gr.update(visible=True),  # main_interface
            gr.update(value=fake_obs_img.copy(), interactive=False),  # img_display
            "ready",  # log_output
            gr.update(choices=[("pick", 0)], value=0),  # options_radio
            "goal",  # goal_box
            gr.update(
                value="please click the keypoint selection image",
                visible=True,
                interactive=False,
            ),  # coords_box
            gr.update(value=None, visible=False),  # video_display
            "ResizeEnv (Episode 1)",  # task_info_box
            "Completed: 0",  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=True),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(value="hint"),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
            gr.update(interactive=True),  # reference_action_btn
        )

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)
    monkeypatch.setattr(callbacks, "get_session", lambda uid: FakeSession())
    monkeypatch.setattr(callbacks, "update_session_activity", lambda uid: None)

    demo = ui_layout.create_ui_blocks()

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="live-obs-client-resize-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            page.wait_for_selector("#live_obs img", timeout=15000)
            page.wait_for_selector("#coords_box textarea, #coords_box input", timeout=15000)
            page.wait_for_function(
                """() => {
                    const container = document.querySelector('#live_obs .image-container');
                    const img = document.querySelector('#live_obs img');
                    if (!container || !img) return false;
                    const containerRect = container.getBoundingClientRect();
                    const imgRect = img.getBoundingClientRect();
                    return imgRect.width > 200 && Math.abs(containerRect.width - imgRect.width) <= 2;
                }""",
                timeout=10000,
            )

            initial_geometry = _read_live_obs_geometry(page)
            assert initial_geometry["container"] is not None
            assert initial_geometry["img"] is not None
            assert initial_geometry["uploadContainer"] is not None
            assert initial_geometry["frame"] is not None
            assert initial_geometry["img"]["width"] > 200
            assert abs(initial_geometry["container"]["width"] - initial_geometry["img"]["width"]) <= 2
            assert abs(initial_geometry["uploadContainer"]["width"] - initial_geometry["img"]["width"]) <= 2
            assert abs(initial_geometry["frame"]["width"] - initial_geometry["img"]["width"]) <= 2
            assert initial_geometry["img"]["width"] / initial_geometry["img"]["height"] == pytest.approx(2.0, rel=0.02)

            page.set_viewport_size({"width": 1024, "height": 900})
            page.wait_for_function(
                """(prevWidth) => {
                    const container = document.querySelector('#live_obs .image-container');
                    const img = document.querySelector('#live_obs img');
                    if (!container || !img) return false;
                    const containerRect = container.getBoundingClientRect();
                    const imgRect = img.getBoundingClientRect();
                    return imgRect.width < prevWidth - 20 && Math.abs(containerRect.width - imgRect.width) <= 2;
                }""",
                arg=initial_geometry["img"]["width"],
                timeout=10000,
            )

            resized_geometry = _read_live_obs_geometry(page)
            assert resized_geometry["img"] is not None
            assert resized_geometry["container"] is not None
            assert resized_geometry["img"]["width"] < initial_geometry["img"]["width"] - 20
            assert abs(resized_geometry["container"]["width"] - resized_geometry["img"]["width"]) <= 2
            assert resized_geometry["img"]["width"] / resized_geometry["img"]["height"] == pytest.approx(2.0, rel=0.02)

            box = page.locator("#live_obs img").bounding_box()
            assert box is not None
            target_x = box["x"] + ((36.5) / 48.0) * box["width"]
            target_y = box["y"] + ((12.5) / 24.0) * box["height"]
            page.mouse.click(target_x, target_y)
            page.wait_for_function(
                """() => {
                    const root = document.getElementById('coords_box');
                    const field = root?.querySelector('textarea, input');
                    return !!field && /^\\d+\\s*,\\s*\\d+$/.test(field.value.trim());
                }""",
                timeout=5000,
            )
            coords_value = _read_coords_box_value(page)
            assert coords_value is not None
            coord_x, coord_y = [int(part.strip()) for part in coords_value.split(",", 1)]
            assert abs(coord_x - 36) <= 1
            assert abs(coord_y - 12) <= 1

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_live_obs_client_resize_after_hidden_phase_becomes_visible(tmp_path):
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    full_red = np.zeros((256, 256, 3), dtype=np.uint8)
    full_red[:, :] = [255, 0, 0]

    with gr.Blocks() as demo:
        demo.css = ui_layout.CSS

        show_btn = gr.Button("Show", elem_id="show_btn")

        with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
            gr.Image(
                value=full_red,
                elem_id="live_obs",
                elem_classes=["live-obs-resizable"],
                buttons=[],
                sources=[],
            )

        demo.load(
            fn=None,
            js=ui_layout.LIVE_OBS_CLIENT_RESIZE_JS,
            queue=False,
        )

        show_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[action_phase_group],
            queue=False,
        )

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="live-obs-hidden-phase-resize-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)
    screenshot_path = tmp_path / "live_obs_hidden_phase.png"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")
            page.wait_for_function(
                "() => !!window.__robommeLiveObsResizerInstalled",
                timeout=5000,
            )

            page.click("#show_btn")
            page.wait_for_selector("#live_obs img", timeout=10000)
            page.wait_for_function(
                """() => {
                    const container = document.querySelector('#live_obs .image-container');
                    const img = document.querySelector('#live_obs img');
                    if (!container || !img) return false;
                    const containerRect = container.getBoundingClientRect();
                    const imgRect = img.getBoundingClientRect();
                    return imgRect.width > 300 && Math.abs(containerRect.width - imgRect.width) <= 2;
                }""",
                timeout=10000,
            )

            geometry = _read_live_obs_geometry(page)
            assert geometry["container"] is not None
            assert geometry["img"] is not None
            assert geometry["img"]["width"] > 300
            assert abs(geometry["container"]["width"] - geometry["img"]["width"]) <= 2
            object_fit = page.evaluate(
                """() => getComputedStyle(document.querySelector('#live_obs img')).objectFit"""
            )
            assert object_fit == "contain"

            page.locator("#live_obs img").screenshot(path=str(screenshot_path))

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()

    screenshot = Image.open(screenshot_path).convert("RGB")
    width, height = screenshot.size
    samples = [
        screenshot.getpixel((width // 2, height // 2)),
        screenshot.getpixel((max(1, width // 10), height // 2)),
        screenshot.getpixel((min(width - 2, (width * 9) // 10), height // 2)),
    ]
    for pixel in samples:
        assert pixel[0] > 200
        assert pixel[1] < 30
        assert pixel[2] < 30


def test_header_task_shows_env_after_init(monkeypatch):
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    fake_obs = np.zeros((24, 24, 3), dtype=np.uint8)
    fake_obs_img = Image.fromarray(fake_obs)

    def fake_init_app(request=None):
        _ = request
        return (
            "uid-auto",
            gr.update(visible=True),  # main_interface
            gr.update(value=fake_obs_img, interactive=False),  # img_display
            "ready",  # log_output
            gr.update(choices=[("pick", 0)], value=None),  # options_radio
            "goal",  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=None, visible=False),  # video_display
            "PickXtimes (Episode 1)",  # task_info_box
            "Completed: 0",  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=True),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(value="hint"),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
            gr.update(interactive=True),  # reference_action_btn
        )

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)

    demo = ui_layout.create_ui_blocks()

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="header-task-url-auto-login-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(f"{root_url}?user=user1", wait_until="domcontentloaded")
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            page.wait_for_function(
                """() => {
                    const root = document.getElementById('header_task');
                    const input = root ? root.querySelector('input') : null;
                    return !!input && input.value.trim() === 'PickXtimes';
                }""",
                timeout=5000,
            )
            assert _read_header_task_value(page) == "PickXtimes"
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


@pytest.mark.parametrize(
    "task_info_text,expected_header_value",
    [
        ("pickxtimes (Episode 1)", "PickXtimes"),
        ("EnvFromSessionOnly (Episode 1)", "EnvFromSessionOnly"),
    ],
)
def test_header_task_env_normalization_and_fallback(monkeypatch, task_info_text, expected_header_value):
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    fake_obs = np.zeros((24, 24, 3), dtype=np.uint8)
    fake_obs_img = Image.fromarray(fake_obs)

    def fake_init_app(_request=None):
        return (
            "uid-auto",
            gr.update(visible=True),  # main_interface
            gr.update(value=fake_obs_img, interactive=False),  # img_display
            "ready",  # log_output
            gr.update(choices=[("pick", 0)], value=None),  # options_radio
            "goal",  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=None, visible=False),  # video_display
            task_info_text,  # task_info_box
            "Completed: 0",  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=False),  # video_phase_group
            gr.update(visible=True),  # action_phase_group
            gr.update(visible=True),  # control_panel_group
            gr.update(value="hint"),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
            gr.update(interactive=True),  # reference_action_btn
        )

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)

    demo = ui_layout.create_ui_blocks()

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="header-task-normalization-fallback-test")
    app = gr.mount_gradio_app(app, demo, path="/")

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_http_ready(root_url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(root_url, wait_until="domcontentloaded")
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            page.wait_for_function(
                """(expectedValue) => {
                    const root = document.getElementById('header_task');
                    const input = root ? root.querySelector('input') : null;
                    return !!input && input.value.trim() === expectedValue;
                }""",
                arg=expected_header_value,
                timeout=5000,
            )
            assert _read_header_task_value(page) == expected_header_value
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_phase_machine_runtime_local_video_path_end_transition():
    import gradio_callbacks as cb

    demo_video_path = gr.get_video("world.mp4")
    fake_obs = np.zeros((24, 24, 3), dtype=np.uint8)

    class FakeSession:
        def __init__(self):
            self.env_id = "VideoUnmask"
            self.language_goal = "place cube on target"
            self.available_options = [("pick", 0)]
            self.raw_solve_options = [{"available": False}]
            self.demonstration_frames = [fake_obs.copy() for _ in range(4)]

        def load_episode(self, env_id, episode_idx):
            self.env_id = env_id
            return fake_obs.copy(), f"loaded {env_id}:{episode_idx}"

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return fake_obs.copy()

    originals = {
        "get_session": cb.get_session,
        "reset_play_button_clicked": cb.reset_play_button_clicked,
        "reset_execute_count": cb.reset_execute_count,
        "set_task_start_time": cb.set_task_start_time,
        "set_ui_phase": cb.set_ui_phase,
        "save_video": cb.save_video,
    }

    fake_session = FakeSession()

    cb.get_session = lambda uid: fake_session
    cb.reset_play_button_clicked = lambda uid: None
    cb.reset_execute_count = lambda uid, env_id, ep_num: None
    cb.set_task_start_time = lambda uid, env_id, ep_num, start_time: None
    cb.set_ui_phase = lambda uid, phase: None
    cb.save_video = lambda frames, suffix="": demo_video_path

    try:
        with gr.Blocks(title="Native phase machine local video test") as demo:
            uid_state = gr.State(value="uid-local-video")
            with gr.Column(visible=False, elem_id="main_interface") as main_interface:
                with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                    video_display = gr.Video(value=None, elem_id="demo_video", autoplay=False)

                with gr.Column(visible=True, elem_id="action_phase_group") as action_phase_group:
                    img_display = gr.Image(value=fake_obs.copy(), elem_id="live_obs")

                with gr.Column(visible=True, elem_id="control_panel_group") as control_panel_group:
                    options_radio = gr.Radio(choices=[("pick", 0)], value=None, elem_id="action_radio")

            log_output = gr.Markdown("", elem_id="log_output")
            goal_box = gr.Textbox("")
            coords_box = gr.Textbox("No need for coordinates")
            task_info_box = gr.Textbox("")
            progress_info_box = gr.Textbox("")
            task_hint_display = gr.Textbox("")
            with gr.Column(visible=False) as loading_overlay:
                gr.Markdown("Loading...")

            restart_episode_btn = gr.Button("restart", interactive=False)
            next_task_btn = gr.Button("next", interactive=False)
            exec_btn = gr.Button("execute", interactive=False)
            reference_action_btn = gr.Button("reference", interactive=False)

            def load_fn():
                status = {
                    "current_task": {"env_id": "VideoUnmask", "episode_idx": 1},
                    "completed_count": 0,
                }
                return cb._load_status_task("uid-local-video", status)

            demo.load(
                fn=load_fn,
                outputs=[
                    uid_state,
                    main_interface,
                    img_display,
                    log_output,
                    options_radio,
                    goal_box,
                    coords_box,
                    video_display,
                    task_info_box,
                    progress_info_box,
                    restart_episode_btn,
                    next_task_btn,
                    exec_btn,
                    video_phase_group,
                    action_phase_group,
                    control_panel_group,
                    task_hint_display,
                    loading_overlay,
                    reference_action_btn,
                ],
                queue=False,
            )

            video_display.end(
                fn=cb.on_video_end_transition,
                inputs=[uid_state],
                outputs=[video_phase_group, action_phase_group, control_panel_group, log_output],
                queue=False,
            )

        port = _free_port()
        host = "127.0.0.1"
        root_url = f"http://{host}:{port}/"

        app = FastAPI(title="native-phase-machine-local-video-test")
        app = gr.mount_gradio_app(app, demo, path="/")

        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _wait_http_ready(root_url)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(root_url, wait_until="domcontentloaded")
                page.wait_for_selector("#main_interface", state="visible", timeout=20000)

                page.wait_for_selector("#demo_video video", timeout=5000)
                phase_after_login = page.evaluate(
                    """() => {
                        const visible = (id) => {
                            const el = document.getElementById(id);
                            if (!el) return false;
                            const st = getComputedStyle(el);
                            return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                        };
                        return {
                            video: visible('demo_video'),
                            action: visible('live_obs'),
                            control: visible('action_radio'),
                        };
                    }"""
                )
                assert phase_after_login == {
                    "video": True,
                    "action": False,
                    "control": False,
                }

                did_dispatch_end = page.evaluate(
                    """() => {
                        const videoEl = document.querySelector('#demo_video video');
                        if (!videoEl) return false;
                        videoEl.dispatchEvent(new Event('ended', { bubbles: true }));
                        return true;
                    }"""
                )
                assert did_dispatch_end

                page.wait_for_function(
                    """() => {
                        const visible = (id) => {
                            const el = document.getElementById(id);
                            if (!el) return false;
                            const st = getComputedStyle(el);
                            return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                        };
                        return visible('live_obs') && visible('action_radio') && !visible('demo_video');
                    }""",
                    timeout=2000,
                )

                browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            demo.close()
    finally:
        for name, value in originals.items():
            setattr(cb, name, value)

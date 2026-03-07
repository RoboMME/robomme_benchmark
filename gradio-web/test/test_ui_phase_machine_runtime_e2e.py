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


def _read_elem_classes(page, elem_id: str) -> list[str] | None:
    return page.evaluate(
        """(elemId) => {
            const root = document.getElementById(elemId);
            return root ? Array.from(root.classList) : null;
        }""",
        elem_id,
    )


def _read_media_card_wait_snapshot(page) -> dict[str, str | float | None]:
    return page.evaluate(
        """() => {
            const card = document.getElementById('media_card');
            if (!card) {
                return {
                    opacity: null,
                    borderColor: null,
                    boxShadow: null,
                    animationName: null,
                };
            }
            const style = getComputedStyle(card, '::after');
            return {
                opacity: Number.parseFloat(style.opacity || '0'),
                borderColor: style.borderColor || null,
                boxShadow: style.boxShadow || null,
                animationName: style.animationName || null,
            };
        }"""
    )


def _read_live_obs_transform_snapshot(page) -> dict[str, str | None]:
    return page.evaluate(
        """() => {
            const img = document.querySelector('#live_obs img');
            const frame = document.querySelector('#live_obs .image-frame');
            return {
                imgTransform: img ? getComputedStyle(img).transform : null,
                frameTransform: frame ? getComputedStyle(frame).transform : null,
            };
        }"""
    )


def _read_phase_visibility(page) -> dict[str, bool | str | None]:
    return page.evaluate(
        """() => {
            const visible = (id) => {
                const el = document.getElementById(id);
                if (!el) return false;
                const st = getComputedStyle(el);
                return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
            };
            const videoEl = document.querySelector('#demo_video video');
            return {
                videoPhase: visible('video_phase_group'),
                video: visible('demo_video'),
                watchButton: visible('watch_demo_video_btn'),
                actionPhase: visible('action_phase_group'),
                action: visible('live_obs'),
                controlPhase: visible('control_panel_group'),
                control: visible('action_radio'),
                currentSrc: videoEl ? videoEl.currentSrc : null,
            };
        }"""
    )


def _read_demo_video_controls(page) -> dict[str, bool | None]:
    return page.evaluate(
        """() => {
            const visible = (id) => {
                const el = document.getElementById(id);
                if (!el) return false;
                const st = getComputedStyle(el);
                return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
            };
            const videoEl = document.querySelector('#demo_video video');
            const button =
                document.querySelector('#watch_demo_video_btn button') ||
                document.querySelector('button#watch_demo_video_btn');
            return {
                videoVisible: visible('demo_video'),
                buttonVisible: visible('watch_demo_video_btn'),
                buttonDisabled: button ? button.disabled : null,
                autoplay: videoEl ? videoEl.autoplay : null,
                paused: videoEl ? videoEl.paused : null,
            };
        }"""
    )


def _click_demo_video_button(page) -> None:
    page.locator("#watch_demo_video_btn button, button#watch_demo_video_btn").first.click()


def _dispatch_video_event(page, event_name: str) -> bool:
    return page.evaluate(
        """(eventName) => {
            const targets = [
                document.querySelector('#demo_video video'),
                document.getElementById('demo_video'),
            ].filter(Boolean);
            if (!targets.length) return false;
            for (const target of targets) {
                target.dispatchEvent(new Event(eventName, { bubbles: true, composed: true }));
            }
            return true;
        }""",
        event_name,
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
    state = {"precheck_calls": 0, "play_clicks": 0}
    demo_video_url = "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    with gr.Blocks(title="Native phase machine test") as demo:
        gr.HTML(f"<style>{ui_layout.CSS}</style>")
        phase_state = gr.State("init")

        with gr.Column(visible=True, elem_id="login_group") as login_group:
            login_btn = gr.Button("Login", elem_id="login_btn")

        with gr.Column(visible=False, elem_id="main_interface") as main_interface:
            with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                video_display = gr.Video(value=None, elem_id="demo_video", autoplay=False)
                watch_demo_video_btn = gr.Button(
                    "Watch Video Input🎬",
                    elem_id="watch_demo_video_btn",
                    interactive=False,
                    visible=False,
                )

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
        simulate_stop_btn = gr.Button("Simulate Stop", elem_id="simulate_stop_btn")

        demo.load(
            fn=None,
            js=ui_layout.DEMO_VIDEO_PLAY_BINDING_JS,
            queue=False,
        )

        def login_fn():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value=demo_video_url, visible=True),
                gr.update(visible=True, interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(value="please click the keypoint selection image"),
                "demo_video",
            )

        def on_play_demo_fn():
            state["play_clicks"] += 1
            return gr.update(visible=True, interactive=False)

        def on_simulate_stop_fn():
            return "stopped"

        def on_video_end_fn():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(interactive=True),
                gr.update(visible=False, interactive=False),
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
                watch_demo_video_btn,
                action_phase_group,
                control_panel_group,
                action_buttons_row,
                reference_action_btn,
                coords_box,
                phase_state,
            ],
            queue=False,
        )

        watch_demo_video_btn.click(
            fn=on_play_demo_fn,
            outputs=[watch_demo_video_btn],
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
                watch_demo_video_btn,
                phase_state,
            ],
            queue=False,
        )
        video_display.stop(
            fn=on_video_end_fn,
            outputs=[
                video_phase_group,
                action_phase_group,
                control_panel_group,
                action_buttons_row,
                reference_action_btn,
                watch_demo_video_btn,
                phase_state,
            ],
            queue=False,
        )
        simulate_stop_btn.click(
            fn=on_simulate_stop_fn,
            outputs=[log_output],
            js="""() => {
                const show = (id, visible) => {
                    const el = document.getElementById(id);
                    if (!el) return;
                    el.style.display = visible ? '' : 'none';
                };
                show('video_phase_group', false);
                show('demo_video', false);
                show('action_phase_group', true);
                show('live_obs', true);
                show('control_panel_group', true);
                show('action_radio', true);
                show('action_buttons_row', true);
                show('watch_demo_video_btn', false);
                const refBtn =
                    document.querySelector('#reference_action_btn button') ||
                    document.querySelector('button#reference_action_btn');
                if (refBtn) {
                    refBtn.disabled = false;
                }
                return [];
            }""",
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
                    watchButton: visible('watch_demo_video_btn'),
                    action: visible('live_obs'),
                    control: visible('action_radio'),
                };
            }"""
        )
        assert phase_after_login == {
            "video": True,
            "watchButton": True,
            "action": False,
            "control": False,
        }

        page.wait_for_selector("#demo_video video", timeout=5000)
        page.wait_for_function(
            """() => {
                const videoEl = document.querySelector('#demo_video video');
                const button =
                    document.querySelector('#watch_demo_video_btn button') ||
                    document.querySelector('button#watch_demo_video_btn');
                return !!videoEl && !!videoEl.currentSrc && !!button && button.disabled === false && videoEl.paused === true;
            }""",
            timeout=10000,
        )
        controls_after_login = _read_demo_video_controls(page)
        assert controls_after_login["videoVisible"] is True
        assert controls_after_login["buttonVisible"] is True
        assert controls_after_login["buttonDisabled"] is False
        assert controls_after_login["autoplay"] is False
        assert controls_after_login["paused"] is True

        _click_demo_video_button(page)
        page.wait_for_function(
            """() => {
                const videoEl = document.querySelector('#demo_video video');
                const button =
                    document.querySelector('#watch_demo_video_btn button') ||
                    document.querySelector('button#watch_demo_video_btn');
                if (!videoEl || !button) return false;
                return button.disabled === true && (videoEl.paused === false || videoEl.currentTime > 0);
            }""",
            timeout=10000,
        )
        controls_after_click = _read_demo_video_controls(page)
        assert controls_after_click["buttonDisabled"] is True
        assert controls_after_click["paused"] is False

        did_dispatch_end = _dispatch_video_event(page, "ended")
        assert did_dispatch_end

        page.wait_for_function(
            """() => {
                const action = document.getElementById('live_obs');
                const control = document.getElementById('action_radio');
                const watchButton = document.getElementById('watch_demo_video_btn');
                if (!action || !control || !watchButton) return false;
                return (
                    getComputedStyle(action).display !== 'none' &&
                    getComputedStyle(control).display !== 'none' &&
                    getComputedStyle(watchButton).display === 'none'
                );
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
    assert state["play_clicks"] == 1


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
        _click_demo_video_button(page)
        page.wait_for_function(
            """() => {
                const button =
                    document.querySelector('#watch_demo_video_btn button') ||
                    document.querySelector('button#watch_demo_video_btn');
                return !!button && button.disabled === true;
            }""",
            timeout=5000,
        )
        did_dispatch_end = _dispatch_video_event(page, "ended")
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


@pytest.mark.xfail(
    reason="Gradio 6.9.0 output video stop path is not reliably triggerable in headless Chromium; transition contract is covered by unit tests.",
    strict=False,
)
def test_demo_video_stop_event_transitions_and_hides_button(phase_machine_ui_url):
    root_url, state = phase_machine_ui_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(root_url, wait_until="domcontentloaded")

        page.wait_for_timeout(2500)
        page.wait_for_selector("#login_btn", timeout=20000)
        page.click("#login_btn")
        page.wait_for_selector("#demo_video video", timeout=5000)

        _click_demo_video_button(page)
        page.wait_for_function(
            """() => {
                const button =
                    document.querySelector('#watch_demo_video_btn button') ||
                    document.querySelector('button#watch_demo_video_btn');
                return !!button && button.disabled === true;
            }""",
            timeout=5000,
        )

        page.locator("#simulate_stop_btn button, button#simulate_stop_btn").first.click()

        page.wait_for_function(
            """() => {
                const visible = (id) => {
                    const el = document.getElementById(id);
                    if (!el) return false;
                    const st = getComputedStyle(el);
                    return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                };
                return (
                    !visible('watch_demo_video_btn') &&
                    !visible('demo_video') &&
                    visible('live_obs') &&
                    visible('action_radio')
                );
            }""",
            timeout=5000,
        )

        browser.close()

    assert state["play_clicks"] == 1


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
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
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


def test_no_video_task_hides_manual_demo_button(monkeypatch):
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    fake_obs = np.zeros((24, 24, 3), dtype=np.uint8)
    fake_obs_img = Image.fromarray(fake_obs)

    def fake_init_app(_request=None):
        return (
            "uid-no-video",
            gr.update(visible=True),  # main_interface
            gr.update(value=fake_obs_img.copy(), interactive=False),  # img_display
            "ready",  # log_output
            gr.update(choices=[("pick", 0)], value=None),  # options_radio
            "goal",  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=None, visible=False),  # video_display
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
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

    app = FastAPI(title="native-no-video-test")
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
                """() => {
                    const visible = (id) => {
                        const el = document.getElementById(id);
                        if (!el) return false;
                        const st = getComputedStyle(el);
                        return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                    };
                    return (
                        !visible('video_phase_group') &&
                        !visible('demo_video') &&
                        !visible('watch_demo_video_btn') &&
                        visible('action_phase_group') &&
                        visible('control_panel_group')
                    );
                }""",
                timeout=5000,
            )

            phase_snapshot = _read_phase_visibility(page)
            controls_snapshot = _read_demo_video_controls(page)
            assert phase_snapshot["videoPhase"] is False
            assert phase_snapshot["video"] is False
            assert phase_snapshot["watchButton"] is False
            assert phase_snapshot["actionPhase"] is True
            assert phase_snapshot["controlPhase"] is True
            assert controls_snapshot["buttonVisible"] is False

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_keypoint_wait_state_pulses_live_obs_and_updates_system_log(monkeypatch):
    config_module = importlib.reload(importlib.import_module("config"))
    callbacks = importlib.reload(importlib.import_module("gradio_callbacks"))
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    fake_obs = np.zeros((24, 48, 3), dtype=np.uint8)
    fake_obs[:, :] = [15, 20, 25]
    fake_obs_img = Image.fromarray(fake_obs)

    class FakeSession:
        raw_solve_options = [{"available": [object()]}, {"available": False}]

        def get_pil_image(self, use_segmented=False):
            _ = use_segmented
            return fake_obs_img.copy()

    def fake_init_app(_request=None):
        return (
            "uid-keypoint-wait",
            gr.update(visible=True),  # main_interface
            gr.update(
                value=fake_obs_img.copy(),
                interactive=False,
                elem_classes=config_module.get_live_obs_elem_classes(),
            ),  # img_display
            config_module.UI_TEXT["log"]["action_selection_prompt"],  # log_output
            gr.update(choices=[("pick", 0), ("skip", 1)], value=None),  # options_radio
            "goal",  # goal_box
            gr.update(
                value=config_module.UI_TEXT["coords"]["not_needed"],
                visible=True,
                interactive=False,
            ),  # coords_box
            gr.update(value=None, visible=False),  # video_display
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
            "KeypointEnv (Episode 1)",  # task_info_box
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

    app = FastAPI(title="keypoint-wait-state-test")
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
            page.add_style_tag(content=ui_layout.CSS)
            page.wait_for_selector("#main_interface_root", state="visible", timeout=15000)
            page.wait_for_selector("#live_obs img", timeout=15000)

            initial_classes = _read_elem_classes(page, "live_obs")
            assert initial_classes is not None
            assert config_module.LIVE_OBS_KEYPOINT_WAIT_CLASS not in initial_classes
            assert _read_log_output_value(page) == config_module.UI_TEXT["log"]["action_selection_prompt"]
            initial_card_wait = _read_media_card_wait_snapshot(page)
            initial_transforms = _read_live_obs_transform_snapshot(page)
            initial_img_box = page.locator("#live_obs img").bounding_box()
            initial_frame_box = page.locator("#live_obs .image-frame").bounding_box()
            assert initial_card_wait["opacity"] == 0
            assert initial_card_wait["animationName"] == "none"
            assert initial_transforms["imgTransform"] == "none"
            assert initial_transforms["frameTransform"] == "none"
            assert initial_img_box is not None
            assert initial_frame_box is not None

            page.locator("#action_radio input[type='radio']").first.check(force=True)

            page.wait_for_function(
                """(state) => {
                    const liveObs = document.getElementById('live_obs');
                    const coordsRoot = document.getElementById('coords_box');
                    const coordsField = coordsRoot?.querySelector('textarea, input');
                    const logRoot = document.getElementById('log_output');
                    const logField = logRoot?.querySelector('textarea, input');
                    const mediaCard = document.getElementById('media_card');
                    const mediaAfter = mediaCard ? getComputedStyle(mediaCard, '::after') : null;
                    const coordsValue = coordsField ? coordsField.value.trim() : '';
                    const logValue = logField ? logField.value.trim() : (logRoot?.textContent || '').trim();
                    return (
                        !!liveObs &&
                        liveObs.classList.contains(state.waitClass) &&
                        !!mediaAfter &&
                        Number.parseFloat(mediaAfter.opacity || '0') > 0.5 &&
                        mediaAfter.animationName === state.cardAnimation &&
                        coordsValue === state.coordsPrompt &&
                        logValue === state.waitLog
                    );
                }""",
                arg={
                    "cardAnimation": "media-card-keypoint-ring",
                    "waitClass": config_module.LIVE_OBS_KEYPOINT_WAIT_CLASS,
                    "coordsPrompt": config_module.UI_TEXT["coords"]["select_keypoint"],
                    "waitLog": config_module.UI_TEXT["log"]["keypoint_selection_prompt"],
                },
                timeout=5000,
            )

            wait_classes = _read_elem_classes(page, "live_obs")
            assert wait_classes is not None
            assert config_module.LIVE_OBS_KEYPOINT_WAIT_CLASS in wait_classes
            assert _read_coords_box_value(page) == config_module.UI_TEXT["coords"]["select_keypoint"]
            assert _read_log_output_value(page) == config_module.UI_TEXT["log"]["keypoint_selection_prompt"]
            wait_card = _read_media_card_wait_snapshot(page)
            wait_transforms = _read_live_obs_transform_snapshot(page)
            wait_img_box = page.locator("#live_obs img").bounding_box()
            wait_frame_box = page.locator("#live_obs .image-frame").bounding_box()
            assert wait_card["opacity"] is not None and wait_card["opacity"] > 0.5
            assert wait_card["animationName"] == "media-card-keypoint-ring"
            assert wait_card["borderColor"] != "rgba(225, 29, 72, 0)"
            assert wait_transforms["imgTransform"] == "none"
            assert wait_transforms["frameTransform"] == "none"
            assert wait_img_box is not None
            assert wait_frame_box is not None
            assert wait_img_box["x"] == pytest.approx(initial_img_box["x"], abs=1.0)
            assert wait_img_box["y"] == pytest.approx(initial_img_box["y"], abs=1.0)
            assert wait_img_box["width"] == pytest.approx(initial_img_box["width"], abs=1.0)
            assert wait_img_box["height"] == pytest.approx(initial_img_box["height"], abs=1.0)
            assert wait_frame_box["x"] == pytest.approx(initial_frame_box["x"], abs=1.0)
            assert wait_frame_box["y"] == pytest.approx(initial_frame_box["y"], abs=1.0)
            assert wait_frame_box["width"] == pytest.approx(initial_frame_box["width"], abs=1.0)
            assert wait_frame_box["height"] == pytest.approx(initial_frame_box["height"], abs=1.0)

            box = page.locator("#live_obs img").bounding_box()
            assert box is not None
            target_x = box["x"] + ((24.5) / 48.0) * box["width"]
            target_y = box["y"] + ((8.5) / 24.0) * box["height"]
            page.mouse.click(target_x, target_y)

            page.wait_for_function(
                """(state) => {
                    const liveObs = document.getElementById('live_obs');
                    const coordsRoot = document.getElementById('coords_box');
                    const coordsField = coordsRoot?.querySelector('textarea, input');
                    const logRoot = document.getElementById('log_output');
                    const logField = logRoot?.querySelector('textarea, input');
                    const coordsValue = coordsField ? coordsField.value.trim() : '';
                    const logValue = logField ? logField.value.trim() : (logRoot?.textContent || '').trim();
                    return (
                        !!liveObs &&
                        !liveObs.classList.contains(state.waitClass) &&
                        /^\\d+\\s*,\\s*\\d+$/.test(coordsValue) &&
                        logValue === state.actionLog
                    );
                }""",
                arg={
                    "waitClass": config_module.LIVE_OBS_KEYPOINT_WAIT_CLASS,
                    "actionLog": config_module.UI_TEXT["log"]["action_selection_prompt"],
                },
                timeout=5000,
            )

            coords_value = _read_coords_box_value(page)
            assert coords_value is not None
            coord_x, coord_y = [int(part.strip()) for part in coords_value.split(",", 1)]
            assert abs(coord_x - 24) <= 1
            assert abs(coord_y - 8) <= 1
            final_classes = _read_elem_classes(page, "live_obs")
            assert final_classes is not None
            assert config_module.LIVE_OBS_KEYPOINT_WAIT_CLASS not in final_classes
            assert config_module.LIVE_OBS_BASE_CLASS in final_classes
            assert _read_log_output_value(page) == config_module.UI_TEXT["log"]["action_selection_prompt"]
            final_card_wait = _read_media_card_wait_snapshot(page)
            final_transforms = _read_live_obs_transform_snapshot(page)
            assert final_card_wait["opacity"] == 0
            assert final_card_wait["animationName"] == "none"
            assert final_transforms["imgTransform"] == "none"
            assert final_transforms["frameTransform"] == "none"

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


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
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
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
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
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
            gr.update(visible=False, interactive=False),  # watch_demo_video_btn
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


def test_header_task_switch_to_video_task_shows_demo_phase(monkeypatch):
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

    fake_obs = np.zeros((24, 24, 3), dtype=np.uint8)
    fake_obs_img = Image.fromarray(fake_obs)
    demo_video_path = gr.get_video("world.mp4")
    switch_calls = []

    def _pick_task_response(uid, task_name, show_video):
        return (
            uid,
            gr.update(visible=True),  # main_interface
            gr.update(value=fake_obs_img, interactive=False),  # img_display
            "demo prompt" if show_video else "ready",  # log_output
            gr.update(choices=[("pick", 0)], value=None),  # options_radio
            "video goal" if show_video else "goal",  # goal_box
            "No need for coordinates",  # coords_box
            gr.update(value=demo_video_path if show_video else None, visible=show_video),  # video_display
            gr.update(visible=show_video, interactive=show_video),  # watch_demo_video_btn
            f"{task_name} (Episode 1)",  # task_info_box
            "Completed: 0",  # progress_info_box
            gr.update(interactive=True),  # restart_episode_btn
            gr.update(interactive=True),  # next_task_btn
            gr.update(interactive=True),  # exec_btn
            gr.update(visible=show_video),  # video_phase_group
            gr.update(visible=not show_video),  # action_phase_group
            gr.update(visible=not show_video),  # control_panel_group
            gr.update(value="video hint" if show_video else "hint"),  # task_hint_display
            gr.update(visible=False),  # loading_overlay
            gr.update(interactive=True),  # reference_action_btn
        )

    def fake_init_app(request=None):
        _ = request
        return _pick_task_response("uid-header-video", "PickXtimes", show_video=False)

    def fake_switch_env_wrapper(uid, selected_env):
        switch_calls.append((uid, selected_env))
        return _pick_task_response(
            uid,
            selected_env,
            show_video=selected_env == "VideoPlaceButton",
        )

    monkeypatch.setattr(ui_layout, "init_app", fake_init_app)
    monkeypatch.setattr(ui_layout, "switch_env_wrapper", fake_switch_env_wrapper)
    monkeypatch.setattr(ui_layout.user_manager, "env_choices", ["PickXtimes", "VideoPlaceButton"])

    demo = ui_layout.create_ui_blocks()

    port = _free_port()
    host = "127.0.0.1"
    root_url = f"http://{host}:{port}/"

    app = FastAPI(title="header-task-switch-video-phase-test")
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
                """() => {
                    const root = document.getElementById('header_task');
                    const input = root ? root.querySelector('input') : null;
                    return !!input && input.value.trim() === 'PickXtimes';
                }""",
                timeout=5000,
            )
            assert switch_calls == []

            page.click("#header_task input")
            page.get_by_role("option", name="VideoPlaceButton").click()

            page.wait_for_function(
                """() => {
                    const visible = (id) => {
                        const el = document.getElementById(id);
                        if (!el) return false;
                        const st = getComputedStyle(el);
                        return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                    };
                    const videoEl = document.querySelector('#demo_video video');
                    const button =
                        document.querySelector('#watch_demo_video_btn button') ||
                        document.querySelector('button#watch_demo_video_btn');
                    return (
                        visible('video_phase_group') &&
                        visible('demo_video') &&
                        visible('watch_demo_video_btn') &&
                        !visible('action_phase_group') &&
                        !visible('control_panel_group') &&
                        !!(videoEl && videoEl.currentSrc) &&
                        !!button &&
                        button.disabled === false &&
                        videoEl.paused === true &&
                        videoEl.autoplay === false
                    );
                }""",
                timeout=10000,
            )

            phase_after_switch = _read_phase_visibility(page)
            assert phase_after_switch["videoPhase"] is True
            assert phase_after_switch["video"] is True
            assert phase_after_switch["watchButton"] is True
            assert phase_after_switch["actionPhase"] is False
            assert phase_after_switch["controlPhase"] is False
            assert phase_after_switch["currentSrc"]
            assert switch_calls == [("uid-header-video", "VideoPlaceButton")]

            page.wait_for_timeout(1500)
            assert switch_calls == [("uid-header-video", "VideoPlaceButton")]
            assert _read_header_task_value(page) == "VideoPlaceButton"

            _click_demo_video_button(page)
            page.wait_for_function(
                """() => {
                    const button =
                        document.querySelector('#watch_demo_video_btn button') ||
                        document.querySelector('button#watch_demo_video_btn');
                    return !!button && button.disabled === true;
                }""",
                timeout=5000,
            )

            did_dispatch_end = _dispatch_video_event(page, "ended")
            assert did_dispatch_end

            page.wait_for_function(
                """() => {
                    const visible = (id) => {
                        const el = document.getElementById(id);
                        if (!el) return false;
                        const st = getComputedStyle(el);
                        return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                    };
                    return (
                        !visible('video_phase_group') &&
                        !visible('demo_video') &&
                        !visible('watch_demo_video_btn') &&
                        visible('action_phase_group') &&
                        visible('control_panel_group') &&
                        visible('live_obs') &&
                        visible('action_radio')
                    );
                }""",
                timeout=5000,
            )

            phase_after_end = _read_phase_visibility(page)
            assert phase_after_end["videoPhase"] is False
            assert phase_after_end["video"] is False
            assert phase_after_end["watchButton"] is False
            assert phase_after_end["actionPhase"] is True
            assert phase_after_end["action"] is True
            assert phase_after_end["controlPhase"] is True
            assert phase_after_end["control"] is True

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        demo.close()


def test_phase_machine_runtime_local_video_path_end_transition():
    import gradio_callbacks as cb
    ui_layout = importlib.reload(importlib.import_module("ui_layout"))

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
            demo.load(
                fn=None,
                js=ui_layout.DEMO_VIDEO_PLAY_BINDING_JS,
                queue=False,
            )
            with gr.Column(visible=False, elem_id="main_interface") as main_interface:
                with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                    video_display = gr.Video(value=None, elem_id="demo_video", autoplay=False)
                    watch_demo_video_btn = gr.Button(
                        "Watch Video Input🎬",
                        elem_id="watch_demo_video_btn",
                        interactive=False,
                        visible=False,
                    )

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
                    watch_demo_video_btn,
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

            watch_demo_video_btn.click(
                fn=cb.on_demo_video_play,
                inputs=[uid_state],
                outputs=[watch_demo_video_btn],
                queue=False,
            )

            video_display.end(
                fn=cb.on_video_end_transition,
                inputs=[uid_state],
                outputs=[
                    video_phase_group,
                    action_phase_group,
                    control_panel_group,
                    log_output,
                    watch_demo_video_btn,
                ],
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
                            watchButton: visible('watch_demo_video_btn'),
                            action: visible('live_obs'),
                            control: visible('action_radio'),
                        };
                    }"""
                )
                assert phase_after_login == {
                    "video": True,
                    "watchButton": True,
                    "action": False,
                    "control": False,
                }

                controls_after_login = _read_demo_video_controls(page)
                assert controls_after_login["buttonVisible"] is True
                assert controls_after_login["buttonDisabled"] is False
                assert controls_after_login["autoplay"] is False
                assert controls_after_login["paused"] is True

                _click_demo_video_button(page)
                page.wait_for_function(
                    """() => {
                        const button =
                            document.querySelector('#watch_demo_video_btn button') ||
                            document.querySelector('button#watch_demo_video_btn');
                        return !!button && button.disabled === true;
                    }""",
                    timeout=5000,
                )

                did_dispatch_end = _dispatch_video_event(page, "ended")
                assert did_dispatch_end

                page.wait_for_function(
                    """() => {
                        const visible = (id) => {
                            const el = document.getElementById(id);
                            if (!el) return false;
                            const st = getComputedStyle(el);
                            return st.display !== 'none' && st.visibility !== 'hidden' && el.getClientRects().length > 0;
                        };
                        return (
                            visible('live_obs') &&
                            visible('action_radio') &&
                            !visible('demo_video') &&
                            !visible('watch_demo_video_btn')
                        );
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

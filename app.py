"""Dummy Gradio entrypoint with layout similar to original RoboMME UI."""

import logging
import os
import sys

# Disable SSR for HF Spaces compatibility (avoids gradio_api heartbeat 404).
os.environ["GRADIO_SSR_MODE"] = "false"

import gradio as gr
import numpy as np


PHASE_DEMO_VIDEO = "demo_video"
PHASE_ACTION_KEYPOINT = "action_keypoint"

DUMMY_TASKS = [
    "Peg Insertion Side",
    "Pick Cube and Place",
    "Stack Green Block",
    "Open Cabinet Door",
]

CSS = """
#loading_overlay_group {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    background: rgba(255, 255, 255, 0.92) !important;
    text-align: center !important;
}

#loading_overlay_group > div {
    min-height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}
"""


def _setup_logging() -> logging.Logger:
    """Configure terminal logging for runtime debugging."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Keep noisy dependency logs down unless explicitly requested.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logger = logging.getLogger("robomme.app")
    logger.info("Logging initialized with LOG_LEVEL=%s", level_name)
    return logger


LOGGER = _setup_logging()


def _task_goal(task_name: str) -> str:
    return f"Dummy goal for {task_name}"


def _task_hint(task_name: str) -> str:
    return (
        f"[DUMMY] Current task: {task_name}\n"
        "1) Select an action option.\n"
        "2) Click image when action needs coordinates.\n"
        "3) Press EXECUTE to simulate one step."
    )


def _task_actions(task_name: str) -> list[str]:
    return [
        f"{task_name}: Move",
        f"{task_name}: Grasp",
        f"{task_name}: Place",
        "Reset Arm Pose",
    ]


def _dummy_frame(task_name: str, step: int) -> np.ndarray:
    seed = sum(ord(ch) for ch in task_name) + step * 31
    r = 80 + (seed % 120)
    g = 80 + ((seed * 3) % 120)
    b = 80 + ((seed * 7) % 120)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = [r, g, b]
    frame[20:340, 20:620] = [min(r + 25, 255), min(g + 25, 255), min(b + 25, 255)]
    frame[170:190, 40:600] = [245, 245, 245]
    return frame


def _build_task_updates(task_name: str, step: int, phase: str) -> tuple:
    in_video_phase = phase == PHASE_DEMO_VIDEO
    log_text = (
        f"[DUMMY] Loaded task: {task_name}\n"
        + ("[DUMMY] Demo video phase. Click 'Skip Demo Video' to continue." if in_video_phase else "[DUMMY] Action phase ready.")
    )
    return (
        _task_goal(task_name),
        gr.update(choices=_task_actions(task_name), value=None),
        "",
        log_text,
        _task_hint(task_name),
        f"Episode progress: {step}/5 (dummy)",
        _dummy_frame(task_name, step),
        gr.update(visible=in_video_phase),
        gr.update(visible=not in_video_phase),
        gr.update(visible=not in_video_phase),
        phase,
    )


def create_dummy_demo() -> gr.Blocks:
    """Build a dummy app that mimics original layout without ManiSkill."""

    def on_task_change(task_name: str):
        LOGGER.debug("on_task_change(task_name=%s)", task_name)
        task = task_name if task_name in DUMMY_TASKS else DUMMY_TASKS[0]
        step = 0
        return (step,) + _build_task_updates(task, step, PHASE_DEMO_VIDEO)

    def skip_video(task_name: str, step: int):
        LOGGER.debug("skip_video(task_name=%s, step=%s)", task_name, step)
        task = task_name if task_name in DUMMY_TASKS else DUMMY_TASKS[0]
        return _build_task_updates(task, step, PHASE_ACTION_KEYPOINT)

    def on_reference_action(task_name: str):
        LOGGER.debug("on_reference_action(task_name=%s)", task_name)
        actions = _task_actions(task_name)
        return (
            gr.update(value=actions[0]),
            "[DUMMY] Filled with reference action.",
        )

    def on_map_click(evt: gr.SelectData):
        if evt is None or evt.index is None:
            LOGGER.debug("on_map_click received empty event")
            return ""
        x, y = evt.index
        LOGGER.debug("on_map_click(x=%s, y=%s)", x, y)
        return f"({x}, {y})"

    def execute_step(task_name: str, action_name: str, coords_text: str, step: int):
        LOGGER.info(
            "execute_step(task_name=%s, action=%s, coords=%s, step=%s)",
            task_name,
            action_name,
            coords_text,
            step,
        )
        task = task_name if task_name in DUMMY_TASKS else DUMMY_TASKS[0]
        next_step = min(step + 1, 5)
        action = action_name or "No action selected"
        coords = coords_text.strip() if coords_text else "N/A"
        log = (
            f"[DUMMY] Execute step {next_step}/5\n"
            f"- task: {task}\n"
            f"- action: {action}\n"
            f"- coords: {coords}"
        )
        progress = f"Episode progress: {next_step}/5 (dummy)"
        return next_step, _dummy_frame(task, next_step), log, progress

    def restart_episode(task_name: str):
        LOGGER.info("restart_episode(task_name=%s)", task_name)
        task = task_name if task_name in DUMMY_TASKS else DUMMY_TASKS[0]
        step = 0
        return (step,) + _build_task_updates(task, step, PHASE_DEMO_VIDEO)

    def next_task(current_task: str):
        LOGGER.info("next_task(current_task=%s)", current_task)
        try:
            idx = DUMMY_TASKS.index(current_task)
        except ValueError:
            idx = 0
        nxt = DUMMY_TASKS[(idx + 1) % len(DUMMY_TASKS)]
        step = 0
        return (nxt, step) + _build_task_updates(nxt, step, PHASE_DEMO_VIDEO)

    with gr.Blocks(title="Oracle Planner Interface") as demo:
        demo.theme = gr.themes.Soft()
        demo.css = CSS

        step_state = gr.State(0)
        ui_phase_state = gr.State(PHASE_DEMO_VIDEO)

        gr.Markdown("## RoboMME Human Evaluation", elem_id="header_title")
        with gr.Row():
            with gr.Column(scale=1):
                header_task_box = gr.Dropdown(
                    choices=DUMMY_TASKS,
                    value=DUMMY_TASKS[0],
                    label="Current Task",
                    show_label=True,
                    interactive=True,
                    elem_id="header_task",
                )
            with gr.Column(scale=2):
                header_goal_box = gr.Textbox(
                    value=_task_goal(DUMMY_TASKS[0]),
                    label="Goal",
                    show_label=True,
                    interactive=False,
                    lines=1,
                    elem_id="header_goal",
                )

        with gr.Column(visible=False, elem_id="loading_overlay_group") as loading_overlay:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        with gr.Column(visible=True, elem_id="main_interface_root") as main_interface:
            with gr.Row(elem_id="main_layout_row"):
                with gr.Column(scale=5):
                    with gr.Column(elem_classes=["native-card"], elem_id="media_card"):
                        with gr.Column(visible=True, elem_id="video_phase_group") as video_phase_group:
                            video_display = gr.Video(
                                label="Demonstration Video",
                                interactive=False,
                                elem_id="demo_video",
                                autoplay=False,
                                show_label=True,
                                value=None,
                                visible=True,
                            )
                            skip_video_btn = gr.Button("Skip Demo Video", variant="secondary")

                        with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
                            img_display = gr.Image(
                                label="Keypoint Selection",
                                interactive=False,
                                type="numpy",
                                elem_id="live_obs",
                                show_label=True,
                                buttons=[],
                                sources=[],
                                value=_dummy_frame(DUMMY_TASKS[0], 0),
                            )

                with gr.Column(scale=4):
                    with gr.Column(visible=False, elem_id="control_panel_group") as control_panel_group:
                        with gr.Row(elem_id="right_top_row", equal_height=False):
                            with gr.Column(scale=3, elem_id="right_action_col"):
                                with gr.Column(elem_classes=["native-card"], elem_id="action_selection_card"):
                                    options_radio = gr.Radio(
                                        choices=_task_actions(DUMMY_TASKS[0]),
                                        label=" Action Selection",
                                        type="value",
                                        show_label=True,
                                        elem_id="action_radio",
                                    )
                                    coords_box = gr.Textbox(
                                        label="Coords",
                                        value="",
                                        interactive=False,
                                        show_label=False,
                                        visible=False,
                                        elem_id="coords_box",
                                    )

                            with gr.Column(scale=2, elem_id="right_log_col"):
                                with gr.Column(elem_classes=["native-card"], elem_id="log_card"):
                                    log_output = gr.Textbox(
                                        value="[DUMMY] Demo video phase. Click 'Skip Demo Video' to continue.",
                                        lines=4,
                                        max_lines=None,
                                        show_label=True,
                                        interactive=False,
                                        elem_id="log_output",
                                        label="System Log",
                                    )

                        with gr.Row(elem_id="action_buttons_row"):
                            with gr.Column(elem_classes=["native-card", "native-button-card"], elem_id="exec_btn_card"):
                                exec_btn = gr.Button("EXECUTE", variant="stop", size="lg", elem_id="exec_btn")

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="reference_btn_card",
                            ):
                                reference_action_btn = gr.Button(
                                    "Ground Truth Action",
                                    variant="secondary",
                                    interactive=True,
                                    elem_id="reference_action_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="restart_episode_btn_card",
                            ):
                                restart_episode_btn = gr.Button(
                                    "restart episode",
                                    variant="secondary",
                                    interactive=True,
                                    elem_id="restart_episode_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="next_task_btn_card",
                            ):
                                next_task_btn = gr.Button(
                                    "change episode",
                                    variant="primary",
                                    interactive=True,
                                    elem_id="next_task_btn",
                                )

                        with gr.Accordion(
                            "Task Hint",
                            open=False,
                            visible=True,
                            elem_classes=["native-card"],
                            elem_id="task_hint_card",
                        ):
                            task_hint_display = gr.Textbox(
                                value=_task_hint(DUMMY_TASKS[0]),
                                lines=8,
                                max_lines=16,
                                show_label=False,
                                interactive=True,
                                elem_id="task_hint_display",
                            )

        progress_info_box = gr.Textbox(value="Episode progress: 0/5 (dummy)", visible=False)

        header_task_box.change(
            fn=on_task_change,
            inputs=[header_task_box],
            outputs=[
                step_state,
                header_goal_box,
                options_radio,
                coords_box,
                log_output,
                task_hint_display,
                progress_info_box,
                img_display,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                ui_phase_state,
            ],
        )

        skip_video_btn.click(
            fn=skip_video,
            inputs=[header_task_box, step_state],
            outputs=[
                header_goal_box,
                options_radio,
                coords_box,
                log_output,
                task_hint_display,
                progress_info_box,
                img_display,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                ui_phase_state,
            ],
        )

        img_display.select(
            fn=on_map_click,
            outputs=[coords_box],
        )

        reference_action_btn.click(
            fn=on_reference_action,
            inputs=[header_task_box],
            outputs=[options_radio, log_output],
        )

        exec_btn.click(
            fn=execute_step,
            inputs=[header_task_box, options_radio, coords_box, step_state],
            outputs=[step_state, img_display, log_output, progress_info_box],
        )

        restart_episode_btn.click(
            fn=restart_episode,
            inputs=[header_task_box],
            outputs=[
                step_state,
                header_goal_box,
                options_radio,
                coords_box,
                log_output,
                task_hint_display,
                progress_info_box,
                img_display,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                ui_phase_state,
            ],
        )

        next_task_btn.click(
            fn=next_task,
            inputs=[header_task_box],
            outputs=[
                header_task_box,
                step_state,
                header_goal_box,
                options_radio,
                coords_box,
                log_output,
                task_hint_display,
                progress_info_box,
                img_display,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                ui_phase_state,
            ],
        )

        demo.load(
            fn=on_task_change,
            inputs=[header_task_box],
            outputs=[
                step_state,
                header_goal_box,
                options_radio,
                coords_box,
                log_output,
                task_hint_display,
                progress_info_box,
                img_display,
                video_phase_group,
                action_phase_group,
                control_panel_group,
                ui_phase_state,
            ],
        )

    return demo


demo = create_dummy_demo()

# Ensure external launch() callers (e.g., Spaces runtime) also keep SSR disabled.
_original_launch = demo.launch


def _patched_launch(**kwargs):
    kwargs.setdefault("ssr_mode", False)
    kwargs.setdefault("show_error", True)
    kwargs.setdefault("debug", True)
    kwargs.setdefault("quiet", False)
    LOGGER.info("Launching app with kwargs=%s", kwargs)
    return _original_launch(**kwargs)


demo.launch = _patched_launch


if __name__ == "__main__":
    LOGGER.info("Starting app.py entrypoint")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
        show_error=True,
        debug=True,
        quiet=False,
    )

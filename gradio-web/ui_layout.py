"""
Native Gradio UI layout.
Sequential media phases: Demo Video -> Action+Keypoint.
Two-column layout: Keypoint Selection | Right Panel.
"""

import ast

import gradio as gr

from config import (
    CONTROL_PANEL_SCALE,
    LIVE_OBS_REFRESH_HZ,
    KEYPOINT_SELECTION_SCALE,
    RIGHT_TOP_ACTION_SCALE,
    RIGHT_TOP_LOG_SCALE,
)
from gradio_callbacks import (
    execute_step,
    init_app,
    load_next_task_wrapper,
    on_map_click,
    on_option_select,
    on_reference_action,
    on_video_end_transition,
    precheck_execute_inputs,
    refresh_live_obs,
    restart_episode_wrapper,
    show_loading_info,
    switch_env_wrapper,
    switch_to_action_phase,
    switch_to_execute_phase,
)
from user_manager import user_manager


PHASE_INIT = "init"
PHASE_DEMO_VIDEO = "demo_video"
PHASE_ACTION_KEYPOINT = "action_keypoint"
PHASE_EXECUTION_PLAYBACK = "execution_playback"


# Deprecated: no legacy runtime JS logic in native Gradio mode.
SYNC_JS = ""


LIVE_OBS_CLIENT_RESIZE_JS = r"""
() => {
    if (window.__robommeLiveObsResizerInstalled) {
        if (typeof window.__robommeLiveObsSchedule === "function") {
            window.__robommeLiveObsSchedule();
        }
        return;
    }

    const state = {
        rafId: null,
        intervalId: null,
        lastAppliedWidth: null,
        lastWrapperNode: null,
        lastFrameNode: null,
        lastImageNode: null,
        rootObserver: null,
        layoutObserver: null,
        phaseObserver: null,
        bodyObserver: null,
    };

    const getTargets = () => {
        const root = document.getElementById("live_obs");
        if (!root) {
            return null;
        }
        return {
            root,
            container: root.querySelector(".image-container"),
            frame: root.querySelector(".image-frame"),
            image: root.querySelector("img"),
            mediaCard: document.getElementById("media_card"),
            actionPhaseGroup: document.getElementById("action_phase_group"),
        };
    };

    const applyResize = () => {
        state.rafId = null;
        const targets = getTargets();
        const wrapper = targets?.root?.querySelector(".upload-container") || targets?.frame?.parentElement;
        if (!targets || !targets.container || !wrapper || !targets.frame || !targets.image) {
            return;
        }

        const containerWidth = Math.floor(targets.container.getBoundingClientRect().width);
        if (!Number.isFinite(containerWidth) || containerWidth < 2) {
            return;
        }

        if (
            state.lastAppliedWidth === containerWidth &&
            state.lastWrapperNode === wrapper &&
            state.lastFrameNode === targets.frame &&
            state.lastImageNode === targets.image
        ) {
            return;
        }

        wrapper.style.width = `${containerWidth}px`;
        wrapper.style.maxWidth = "none";
        wrapper.style.display = "block";

        targets.frame.style.width = `${containerWidth}px`;
        targets.frame.style.maxWidth = "none";
        targets.frame.style.display = "block";

        targets.image.style.width = `${containerWidth}px`;
        targets.image.style.maxWidth = "none";
        targets.image.style.height = "auto";
        targets.image.style.display = "block";
        targets.image.style.objectFit = "contain";
        targets.image.style.objectPosition = "center center";

        state.lastAppliedWidth = containerWidth;
        state.lastWrapperNode = wrapper;
        state.lastFrameNode = targets.frame;
        state.lastImageNode = targets.image;
    };

    const scheduleResize = () => {
        if (state.rafId !== null) {
            return;
        }
        state.rafId = window.requestAnimationFrame(applyResize);
    };

    const observeLiveObs = () => {
        const targets = getTargets();
        if (!targets) {
            return false;
        }

        state.rootObserver?.disconnect();
        state.rootObserver = new MutationObserver(scheduleResize);
        state.rootObserver.observe(targets.root, {
            childList: true,
            subtree: true,
            attributes: true,
        });

        state.layoutObserver?.disconnect();
        if (window.ResizeObserver) {
            state.layoutObserver = new ResizeObserver(scheduleResize);
            [targets.root, targets.container, targets.mediaCard, targets.actionPhaseGroup]
                .filter(Boolean)
                .forEach((node) => state.layoutObserver.observe(node));
        }

        state.phaseObserver?.disconnect();
        state.phaseObserver = new MutationObserver(scheduleResize);
        [targets.root, targets.actionPhaseGroup, targets.root.parentElement, targets.root.parentElement?.parentElement]
            .filter(Boolean)
            .forEach((node) =>
                state.phaseObserver.observe(node, {
                    attributes: true,
                    attributeFilter: ["class", "style", "hidden"],
                })
            );

        scheduleResize();
        return true;
    };

    window.__robommeLiveObsSchedule = scheduleResize;
    window.addEventListener("resize", scheduleResize, { passive: true });
    document.addEventListener("visibilitychange", scheduleResize);

    if (!observeLiveObs()) {
        state.bodyObserver = new MutationObserver(() => {
            if (observeLiveObs()) {
                state.bodyObserver?.disconnect();
                state.bodyObserver = null;
            }
            scheduleResize();
        });
        state.bodyObserver.observe(document.body, {
            childList: true,
            subtree: true,
        });
    }

    state.intervalId = window.setInterval(scheduleResize, 250);

    window.__robommeLiveObsResizerInstalled = true;
}
"""


CSS = f"""
.native-card {{
}}

#loading_overlay_group {{
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    background: rgba(255, 255, 255, 0.92) !important;
    text-align: center !important;
}}

#loading_overlay_group > div {{
    min-height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}}

#loading_overlay_group h3 {{
    margin: 0 !important;
}}

button#reference_action_btn:not(:disabled),
#reference_action_btn:not(:disabled),
#reference_action_btn button:not(:disabled) {{
    background: #1f8b4c !important;
    border-color: #1f8b4c !important;
    color: #ffffff !important;
}}

button#reference_action_btn:not(:disabled):hover,
#reference_action_btn:not(:disabled):hover,
#reference_action_btn button:not(:disabled):hover {{
    background: #19713d !important;
    border-color: #19713d !important;
}}

#live_obs.live-obs-resizable .image-container {{
    width: 100%;
}}

#live_obs.live-obs-resizable .upload-container {{
    width: 100%;
}}
"""


def extract_last_goal(goal_text):
    """Extract last goal from goal text that may be a list representation."""
    if not goal_text:
        return ""
    text = goal_text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            goals = ast.literal_eval(text)
            if isinstance(goals, list) and goals:
                for goal in reversed(goals):
                    goal_text = str(goal).strip()
                    if goal_text:
                        return goal_text
        except Exception:
            pass
    return text.split("\n")[0].strip()


def _phase_from_updates(main_interface_update, video_phase_update):
    if isinstance(main_interface_update, dict) and main_interface_update.get("visible") is False:
        return PHASE_INIT
    if isinstance(video_phase_update, dict) and video_phase_update.get("visible") is True:
        return PHASE_DEMO_VIDEO
    return PHASE_ACTION_KEYPOINT


def _with_phase_from_load(load_result):
    phase = _phase_from_updates(load_result[1], load_result[13])
    return (*load_result, phase)


def create_ui_blocks():
    """构建 Gradio Blocks，并完成页面阶段状态（phase）的联动绑定。"""

    def render_header_task(task_text):
        clean_task = str(task_text or "").strip()
        if not clean_task:
            return None
        if clean_task.lower().startswith("current task:"):
            clean_task = clean_task.split(":", 1)[1].strip()
        marker = " (Episode "
        if marker in clean_task:
            clean_task = clean_task.split(marker, 1)[0].strip()
        return " ".join(clean_task.splitlines()).strip() or None

    def render_header_goal(goal_text):
        last_goal = extract_last_goal(goal_text or "")
        return last_goal if last_goal else "—"

    with gr.Blocks(title="Oracle Planner Interface") as demo:
        demo.theme = gr.themes.Soft()
        demo.css = CSS

        gr.Markdown("## RoboMME Human Evaluation", elem_id="header_title")
        with gr.Row():
            with gr.Column(scale=1):
                header_task_box = gr.Dropdown(
                    choices=list(user_manager.env_choices),
                    value=render_header_task(""),
                    label="Current Task",
                    show_label=True,
                    interactive=True,
                    elem_id="header_task",
                )
            with gr.Column(scale=2):
                header_goal_box = gr.Textbox(
                    value=render_header_goal(""),
                    label="Goal",
                    show_label=True,
                    interactive=False,
                    lines=1,
                    elem_id="header_goal",
                )

        with gr.Column(visible=True, elem_id="loading_overlay_group") as loading_overlay:
            gr.Markdown("### Logging in and setting up environment... Please wait.")

        uid_state = gr.State(value=None)
        ui_phase_state = gr.State(value=PHASE_INIT)
        live_obs_timer = gr.Timer(value=1.0 / LIVE_OBS_REFRESH_HZ, active=True)

        task_info_box = gr.Textbox(visible=False, elem_id="task_info_box")
        progress_info_box = gr.Textbox(visible=False)
        goal_box = gr.Textbox(visible=False)

        with gr.Column(visible=False, elem_id="main_interface_root") as main_interface:
            with gr.Row(elem_id="main_layout_row"):
                with gr.Column(scale=KEYPOINT_SELECTION_SCALE):
                    with gr.Column(elem_classes=["native-card"], elem_id="media_card"):
                        with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                            video_display = gr.Video(
                                label="Demonstration Video",
                                interactive=False,
                                elem_id="demo_video",
                                autoplay=True,
                                show_label=True,
                                visible=True,
                            )

                        with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
                            img_display = gr.Image(
                                label="Keypoint Selection",
                                interactive=False,
                                type="pil",
                                elem_id="live_obs",
                                elem_classes=["live-obs-resizable"],
                                show_label=True,
                                buttons=[],
                                sources=[],
                            )

                with gr.Column(scale=CONTROL_PANEL_SCALE):
                    with gr.Column(visible=False, elem_id="control_panel_group") as control_panel_group:
                        with gr.Row(elem_id="right_top_row", equal_height=False):
                            with gr.Column(scale=RIGHT_TOP_ACTION_SCALE, elem_id="right_action_col"):
                                with gr.Column(elem_classes=["native-card"], elem_id="action_selection_card"):
                                    options_radio = gr.Radio(
                                        choices=[],
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

                            with gr.Column(scale=RIGHT_TOP_LOG_SCALE, elem_id="right_log_col"):
                                with gr.Column(elem_classes=["native-card"], elem_id="log_card"):
                                    log_output = gr.Textbox(
                                        value="",
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
                                    interactive=False,
                                    elem_id="reference_action_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="restart_episode_btn_card",
                            ):
                                restart_episode_btn = gr.Button(
                                    "restart episode",
                                    variant="secondary",
                                    interactive=False,
                                    elem_id="restart_episode_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="next_task_btn_card",
                            ):
                                next_task_btn = gr.Button(
                                    "change episode",
                                    variant="primary",
                                    interactive=False,
                                    elem_id="next_task_btn",
                                )

                        with gr.Column(visible=True, elem_classes=["native-card"], elem_id="task_hint_card"):
                            task_hint_display = gr.Textbox(
                                value="",
                                lines=8,
                                max_lines=16,
                                show_label=True,
                                label="Task Hint",
                                interactive=True,
                                elem_id="task_hint_display",
                            )

        def _normalize_env_choice(env_value, choices):
            if env_value is None:
                return None
            env_text = str(env_value).strip()
            if not env_text:
                return None
            lower_map = {}
            for choice in choices:
                choice_text = str(choice).strip()
                if choice_text:
                    lower_map.setdefault(choice_text.lower(), choice_text)
            return lower_map.get(env_text.lower(), env_text)

        def _build_header_task_update(task_text, fallback_env=None):
            base_choices = list(user_manager.env_choices)
            parsed_env = render_header_task(task_text)
            selected_env = _normalize_env_choice(parsed_env, base_choices)
            if selected_env is None:
                selected_env = _normalize_env_choice(fallback_env, base_choices)

            choices = list(base_choices)
            if selected_env and selected_env not in choices:
                choices.append(selected_env)
            return gr.update(choices=choices, value=selected_env)

        def sync_header_from_task(task_text, goal_text):
            return _build_header_task_update(task_text), render_header_goal(goal_text)

        def sync_header_from_goal(goal_text, task_text, current_header_task):
            return _build_header_task_update(task_text, fallback_env=current_header_task), render_header_goal(goal_text)

        def init_app_with_phase(request: gr.Request):
            return _with_phase_from_load(init_app(request))

        def load_next_task_with_phase(uid):
            return _with_phase_from_load(load_next_task_wrapper(uid))

        def restart_episode_with_phase(uid):
            return _with_phase_from_load(restart_episode_wrapper(uid))

        def switch_env_with_phase(uid, selected_env):
            return _with_phase_from_load(switch_env_wrapper(uid, selected_env))

        task_info_box.change(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )
        goal_box.change(
            fn=sync_header_from_goal,
            inputs=[goal_box, task_info_box, header_task_box],
            outputs=[header_task_box, header_goal_box],
        )

        header_task_box.input(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=switch_env_with_phase,
            inputs=[uid_state, header_task_box],
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
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        next_task_btn.click(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=load_next_task_with_phase,
            inputs=[uid_state],
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
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        restart_episode_btn.click(fn=show_loading_info, outputs=[loading_overlay]).then(
            fn=restart_episode_with_phase,
            inputs=[uid_state],
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
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

        video_display.end(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[video_phase_group, action_phase_group, control_panel_group, log_output],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        )
        video_display.stop(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[video_phase_group, action_phase_group, control_panel_group, log_output],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        )

        img_display.select(
            fn=on_map_click,
            inputs=[uid_state, options_radio],
            outputs=[img_display, coords_box],
        )

        options_radio.change(
            fn=on_option_select,
            inputs=[uid_state, options_radio, coords_box],
            outputs=[coords_box, img_display],
        )

        reference_action_btn.click(
            fn=on_reference_action,
            inputs=[uid_state],
            outputs=[img_display, options_radio, coords_box, log_output],
        )

        exec_btn.click(
            fn=precheck_execute_inputs,
            inputs=[uid_state, options_radio, coords_box],
            outputs=[],
            show_progress="hidden",
        ).then(
            fn=switch_to_execute_phase,
            inputs=[uid_state],
            outputs=[
                options_radio,
                exec_btn,
                restart_episode_btn,
                next_task_btn,
                img_display,
                reference_action_btn,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_EXECUTION_PLAYBACK,
            outputs=[ui_phase_state],
            show_progress="hidden",
        ).then(
            fn=execute_step,
            inputs=[uid_state, options_radio, coords_box],
            outputs=[img_display, log_output, task_info_box, progress_info_box, restart_episode_btn, next_task_btn, exec_btn],
            show_progress="hidden",
        ).then(
            fn=switch_to_action_phase,
            inputs=[uid_state],
            outputs=[
                options_radio,
                exec_btn,
                restart_episode_btn,
                next_task_btn,
                img_display,
                reference_action_btn,
            ],
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_KEYPOINT,
            outputs=[ui_phase_state],
            show_progress="hidden",
        )

        live_obs_timer.tick(
            fn=refresh_live_obs,
            inputs=[uid_state, ui_phase_state],
            outputs=[img_display],
            queue=False,
            show_progress="hidden",
        )

        demo.load(
            fn=None,
            js=LIVE_OBS_CLIENT_RESIZE_JS,
            queue=False,
        )

        demo.load(
            fn=init_app_with_phase,
            inputs=[],
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
                ui_phase_state,
            ],
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box],
        )

    return demo

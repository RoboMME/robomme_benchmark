"""
Native Gradio UI layout.
Sequential media phases: Demo Video -> Action+Point.
Two-column layout: Point Selection | Right Panel.
"""

import ast
import json

import gradio as gr

from config import (
    CONTROL_PANEL_SCALE,
    LIVE_OBS_BASE_CLASS,
    LIVE_OBS_POINT_WAIT_CLASS,
    SESSION_CONCURRENCY_ID,
    SESSION_CONCURRENCY_LIMIT,
    SESSION_INIT_CONCURRENCY_ID,
    SESSION_INIT_CONCURRENCY_LIMIT,
    SESSION_TIMEOUT,
    LIVE_OBS_REFRESH_HZ,
    POINT_SELECTION_SCALE,
    RIGHT_TOP_ACTION_SCALE,
    RIGHT_TOP_LOG_SCALE,
    UI_TEXT,
    UI_GLOBAL_FONT_SIZE,
    get_live_obs_elem_classes,
)
from gradio_callbacks import (
    cleanup_current_request_session,
    cleanup_user_session,
    execute_step,
    init_app,
    load_next_task_wrapper,
    on_map_click,
    on_demo_video_play,
    on_option_select,
    on_reference_action,
    on_video_end_transition,
    precheck_execute_inputs,
    refresh_live_obs,
    resume_pending_init,
    restart_episode_wrapper,
    switch_env_wrapper,
    switch_to_action_phase,
    switch_to_execute_phase,
    touch_session,
    touch_session_or_preserve_pending,
)
from user_manager import user_manager


PHASE_INIT = "init"
PHASE_DEMO_VIDEO = "demo_video"
PHASE_ACTION_POINT = "action_point"
PHASE_EXECUTION_PLAYBACK = "execution_playback"
LOAD_STATUS_MODE_IDLE = "idle"
LOAD_STATUS_MODE_EPISODE_LOAD = "episode_load"

APP_THEME = gr.themes.Default()


# Deprecated: no legacy runtime JS logic in native Gradio mode.
SYNC_JS = ""


DEMO_VIDEO_PLAY_BINDING_JS = r"""
() => {
    const bindPlayButton = () => {
        const button =
            document.querySelector("#watch_demo_video_btn button") ||
            document.querySelector("button#watch_demo_video_btn");
        if (!button || button.dataset.robommeDemoPlayBound === "1") {
            return;
        }
        button.dataset.robommeDemoPlayBound = "1";
        button.addEventListener("click", () => {
            const videoEl = document.querySelector("#demo_video video");
            if (!videoEl) {
                return;
            }
            const playPromise = videoEl.play();
            if (playPromise && typeof playPromise.catch === "function") {
                playPromise.catch(() => {});
            }
        });
    };

    if (!window.__robommeDemoPlayBindingInstalled) {
        const observer = new MutationObserver(() => bindPlayButton());
        observer.observe(document.body, {
            childList: true,
            subtree: true,
        });
        window.__robommeDemoPlayBindingInstalled = true;
    }

    bindPlayButton();
}
"""


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


THEME_LOCK_HEAD = r"""
<script>
(() => {
    const applyLightTheme = () => {
        const normalizeThemeState = (store) => {
            try {
                store.setItem("theme", "light");
                store.setItem("color-scheme", "light");
                store.setItem("gradio-theme", "light");
                for (const key of Object.keys(store)) {
                    if (!/theme|color-scheme/i.test(key)) {
                        continue;
                    }
                    const value = store.getItem(key);
                    if (typeof value === "string" && /dark/i.test(value)) {
                        store.setItem(key, value.replace(/dark/gi, "light"));
                    }
                }
            } catch (error) {
                console.debug("Failed to normalize theme state", error);
            }
        };
        const normalizeNode = (node) => {
            if (!node) {
                return;
            }
            if (node.classList.contains("dark")) {
                node.classList.remove("dark");
            }
            if (node.dataset.theme !== "light") {
                node.dataset.theme = "light";
            }
            if (node.getAttribute("data-color-scheme") !== "light") {
                node.setAttribute("data-color-scheme", "light");
            }
            if (node.style.colorScheme !== "light") {
                node.style.colorScheme = "light";
            }
        };

        normalizeThemeState(window.localStorage);
        normalizeThemeState(window.sessionStorage);
        normalizeNode(document.documentElement);
        normalizeNode(document.body);
    };

    applyLightTheme();
    window.__robommeForceLightTheme = applyLightTheme;

    if (window.__robommeThemeLockInstalled) {
        return;
    }

    const observer = new MutationObserver(() => applyLightTheme());
    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class", "data-theme", "style"],
    });

    const attachBodyObserver = () => {
        if (!document.body || document.body.dataset.robommeThemeObserved === "1") {
            return;
        }
        document.body.dataset.robommeThemeObserved = "1";
        observer.observe(document.body, {
            attributes: true,
            attributeFilter: ["class", "data-theme", "style"],
        });
    };

    attachBodyObserver();
    document.addEventListener("DOMContentLoaded", attachBodyObserver, { once: true });
    window.addEventListener("load", applyLightTheme, { once: true });
    window.setTimeout(applyLightTheme, 0);
    window.setTimeout(applyLightTheme, 100);
    window.__robommeThemeLockInstalled = true;
})();
</script>
"""


THEME_LOCK_JS = r"""
() => {
    if (typeof window.__robommeForceLightTheme === "function") {
        window.__robommeForceLightTheme();
        window.setTimeout(window.__robommeForceLightTheme, 0);
    }
}
"""


SET_EPISODE_LOAD_MODE_JS = f"""
() => {{
    window.__robommeLoadStatusMode = {json.dumps(LOAD_STATUS_MODE_EPISODE_LOAD)};
}}
"""


SET_EPISODE_LOAD_MODE_IF_SWITCH_JS = f"""
(_uid, selectedEnv, currentTaskEnv) => {{
    const normalize = (value) => (value == null ? "" : String(value).trim().toLowerCase());
    const nextEnv = normalize(selectedEnv);
    const currentEnv = normalize(currentTaskEnv);
    window.__robommeLoadStatusMode =
        nextEnv && nextEnv !== currentEnv
            ? {json.dumps(LOAD_STATUS_MODE_EPISODE_LOAD)}
            : {json.dumps(LOAD_STATUS_MODE_IDLE)};
}}
"""


RESET_EPISODE_LOAD_MODE_JS = f"""
() => {{
    window.__robommeLoadStatusMode = {json.dumps(LOAD_STATUS_MODE_IDLE)};
}}
"""


PROGRESS_TEXT_REWRITE_JS = f"""
() => {{
    const modeEpisodeLoad = {json.dumps(LOAD_STATUS_MODE_EPISODE_LOAD)};
    const modeIdle = {json.dumps(LOAD_STATUS_MODE_IDLE)};
    const episodeLoadingText = {json.dumps(UI_TEXT["progress"]["episode_loading"])};
    const queueWaitText = {json.dumps(UI_TEXT["progress"]["queue_wait"])};

    window.__robommeLoadStatusMode = window.__robommeLoadStatusMode || modeIdle;
    const getMode = () => window.__robommeLoadStatusMode || modeIdle;

    const ensureOverlayStyles = () => {{
        const host = document.getElementById("native_progress_host");
        if (!(host instanceof HTMLElement)) {{
            return;
        }}
        host.style.setProperty("position", "fixed", "important");
        host.style.setProperty("inset", "0", "important");
        host.style.setProperty("z-index", "9999", "important");
        host.style.setProperty("pointer-events", "none", "important");
        host.style.setProperty("width", "100vw", "important");
        host.style.setProperty("height", "100vh", "important");
        host.style.setProperty("min-height", "100vh", "important");
        host.style.setProperty("overflow", "visible", "important");

        const wrap = host.querySelector(".wrap");
        if (wrap instanceof HTMLElement) {{
            wrap.style.setProperty("position", "fixed", "important");
            wrap.style.setProperty("inset", "0", "important");
            wrap.style.setProperty("width", "100vw", "important");
            wrap.style.setProperty("height", "100vh", "important");
            wrap.style.setProperty("min-height", "100vh", "important");
            wrap.style.setProperty("padding", "0", "important");
            wrap.style.setProperty("display", "flex", "important");
            wrap.style.setProperty("align-items", "center", "important");
            wrap.style.setProperty("justify-content", "center", "important");
            wrap.style.setProperty("background", "rgba(255, 255, 255, 0.92)", "important");
            wrap.style.setProperty("backdrop-filter", "blur(2px)", "important");
        }}
        const markdown = host.querySelector('[data-testid="markdown"]');
        const prose =
            markdown instanceof HTMLElement
                ? markdown.querySelector(".prose, .md") || markdown
                : null;
        if (markdown instanceof HTMLElement) {{
            markdown.style.setProperty("position", "fixed", "important");
            markdown.style.setProperty("inset", "0", "important");
            markdown.style.setProperty("display", "flex", "important");
            markdown.style.setProperty("align-items", "center", "important");
            markdown.style.setProperty("justify-content", "center", "important");
            markdown.style.setProperty("padding", "24px", "important");
        }}
        if (prose instanceof HTMLElement) {{
            prose.style.setProperty("width", "min(720px, calc(100vw - 48px))", "important");
            prose.style.setProperty("max-width", "calc(100vw - 48px)", "important");
            prose.style.setProperty("margin", "0", "important");
            prose.style.setProperty("padding", "0", "important");
            prose.style.setProperty("text-align", "center", "important");
            prose.style.setProperty("color", "#0f172a", "important");
            prose.style.setProperty("font-size", "var(--text-lg)", "important");
            prose.style.setProperty("font-weight", "600", "important");
            prose.style.setProperty("line-height", "1.5", "important");
            prose.style.setProperty("white-space", "pre-line", "important");
        }}
    }};

    const splitSegments = (text) =>
        text
            .split("|")
            .map((part) => part.trim())
            .filter(Boolean);

    const rewriteNode = (node) => {{
        if (!(node instanceof HTMLElement)) {{
            return;
        }}

        const displayed = (node.innerText || node.textContent || "").trim();
        const previousCustom = node.dataset.robommeProgressCustom || "";
        const raw =
            node.dataset.robommeProgressCustomized === "1" && displayed === previousCustom
                ? node.dataset.robommeProgressRaw || displayed
                : displayed;

        if (getMode() !== modeEpisodeLoad) {{
            if (
                node.dataset.robommeProgressCustomized === "1" &&
                displayed === previousCustom &&
                node.dataset.robommeProgressRaw
            ) {{
                node.textContent = node.dataset.robommeProgressRaw;
            }}
            delete node.dataset.robommeProgressCustomized;
            delete node.dataset.robommeProgressRaw;
            delete node.dataset.robommeProgressCustom;
            return;
        }}

        const normalized = raw.toLowerCase();
        let custom = null;

        if (normalized.startsWith("processing")) {{
            const segments = splitSegments(raw);
            const suffix = segments.length > 1 ? ` | ${{segments.slice(1).join(" | ")}}` : "";
            custom = `${{episodeLoadingText}}${{suffix}}`;
        }} else if (normalized.startsWith("queue:")) {{
            custom = `${{queueWaitText}} | ${{raw}}`;
        }}

        if (!custom) {{
            return;
        }}

        node.dataset.robommeProgressCustomized = "1";
        node.dataset.robommeProgressRaw = raw;
        node.dataset.robommeProgressCustom = custom;
        if (displayed !== custom) {{
            node.textContent = custom;
        }}
    }};

    const rewriteAll = () => {{
        ensureOverlayStyles();
        document.querySelectorAll(".progress-text").forEach(rewriteNode);
    }};

    const scheduleRewrite = () => {{
        if (window.__robommeProgressRewriteRaf) {{
            return;
        }}
        window.__robommeProgressRewriteRaf = window.requestAnimationFrame(() => {{
            window.__robommeProgressRewriteRaf = null;
            rewriteAll();
        }});
    }};

    if (!window.__robommeProgressRewriteInstalled) {{
        const observer = new MutationObserver(scheduleRewrite);
        observer.observe(document.body, {{
            childList: true,
            subtree: true,
            characterData: true,
        }});
        window.setInterval(scheduleRewrite, 200);
        window.__robommeProgressRewriteInstalled = true;
    }}

    scheduleRewrite();
}}
"""


CSS = f"""
:root {{
    --body-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --prose-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --input-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --block-label-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --block-title-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --block-info-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --checkbox-label-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --button-large-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --button-medium-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --button-small-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --section-header-text-size: {UI_GLOBAL_FONT_SIZE} !important;
    --text-md: {UI_GLOBAL_FONT_SIZE} !important;
    color-scheme: light !important;
}}

.native-card {{
}}

#header_title h2 {{
    font-size: var(--text-xxl) !important;
}}

#native_progress_host {{
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    pointer-events: none !important;
}}

#native_progress_host .wrap {{
    width: 100vw !important;
    min-height: 100vh !important;
}}

#native_progress_host .wrap.translucent {{
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(2px);
}}

#native_progress_host .pending {{
    min-height: 100vh !important;
}}

#reference_action_btn button:not(:disabled),
button#reference_action_btn:not(:disabled) {{
    background: #1f8b4c !important;
    border-color: #1f8b4c !important;
    color: #ffffff !important;
}}

#reference_action_btn button:not(:disabled):hover,
button#reference_action_btn:not(:disabled):hover {{
    background: #19713d !important;
    border-color: #19713d !important;
}}

#live_obs.live-obs-resizable .image-container {{
    width: 100%;
}}

#live_obs.live-obs-resizable .upload-container {{
    width: 100%;
}}

#watch_demo_video_btn,
#watch_demo_video_btn button,
button#watch_demo_video_btn {{
    width: 100%;
}}

#media_card {{
    --media-card-radius: 8px;
    position: relative;
    border-radius: var(--media-card-radius);
    overflow: visible;
}}

#media_card > div,
#media_card #action_phase_group,
#media_card #video_phase_group,
#media_card #live_obs,
#media_card #live_obs button,
#media_card #live_obs .image-frame,
#media_card #live_obs img,
#media_card #demo_video,
#media_card #demo_video video {{
    border-radius: var(--media-card-radius);
}}

#media_card::after {{
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    border: 3px solid rgba(225, 29, 72, 0.00);
    box-sizing: border-box;
    box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.00);
    opacity: 0;
    pointer-events: none;
    transition: opacity 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    z-index: 0;
}}

@keyframes media-card-point-ring {{
    0% {{
        box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.00);
        border-color: rgba(225, 29, 72, 0.72);
        opacity: 0.72;
    }}
    18% {{
        box-shadow: 0 0 0 4px rgba(225, 29, 72, 0.28);
        border-color: rgba(225, 29, 72, 0.94);
        opacity: 1;
    }}
    36% {{
        box-shadow: 0 0 0 10px rgba(225, 29, 72, 0.12);
        border-color: rgba(225, 29, 72, 0.88);
        opacity: 0.94;
    }}
    62% {{
        box-shadow: 0 0 0 2px rgba(225, 29, 72, 0.18);
        border-color: rgba(225, 29, 72, 0.96);
        opacity: 1;
    }}
    100% {{
        box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.00);
        border-color: rgba(225, 29, 72, 0.72);
        opacity: 0.72;
    }}
}}

#media_card:has(#live_obs.{LIVE_OBS_POINT_WAIT_CLASS})::after {{
    border-color: rgba(225, 29, 72, 0.94);
    box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.00);
    opacity: 1;
    animation: media-card-point-ring 1.2s ease-in-out infinite;
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


def capitalize_first_letter(text):
    """Uppercase only the first character for display."""
    if not text:
        return text
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:]


def render_header_goal(goal_text):
    """Render header goal from raw goal text using display-only normalization."""
    last_goal = extract_last_goal(goal_text or "")
    if not last_goal:
        return "—"
    return capitalize_first_letter(last_goal)


def _phase_from_updates(main_interface_update, video_phase_update):
    if isinstance(main_interface_update, dict) and main_interface_update.get("visible") is False:
        return PHASE_INIT
    if isinstance(video_phase_update, dict) and video_phase_update.get("visible") is True:
        return PHASE_DEMO_VIDEO
    return PHASE_ACTION_POINT


def _with_phase_from_load(load_result):
    phase = _phase_from_updates(load_result[1], load_result[14])
    return (
        *load_result,
        phase,
        False,
        gr.update(value=""),
        gr.update(active=False),
    )


def _phase_visibility_updates(phase):
    if phase == PHASE_DEMO_VIDEO:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    if phase in {PHASE_ACTION_POINT, PHASE_EXECUTION_PLAYBACK}:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


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

    with gr.Blocks(title="Oracle Planner Interface") as demo:
        demo.css = CSS
        demo.theme = APP_THEME
        demo.head = THEME_LOCK_HEAD

        gr.Markdown("## 🔥 RoboMME Interactive Demo 🚀🚀🚀", elem_id="header_title")
        with gr.Row():
            with gr.Column(scale=1):
                header_task_box = gr.Dropdown(
                    choices=list(user_manager.env_choices),
                    value=render_header_task(""),
                    label="Current Task 🏆",
                    show_label=True,
                    interactive=True,
                    elem_id="header_task",
                )
            with gr.Column(scale=2):
                header_goal_box = gr.Textbox(
                    value=render_header_goal(""),
                    label="Task Goal ✅",
                    show_label=True,
                    interactive=False,
                    lines=1,
                    elem_id="header_goal",
                )

        uid_state = gr.State(
            value=None,
            time_to_live=SESSION_TIMEOUT,
            delete_callback=cleanup_user_session,
        )
        ui_phase_state = gr.State(value=PHASE_INIT)
        session_boot_pending_state = gr.State(value=False)
        current_task_env_state = gr.State(value=None)
        suppress_next_option_change_state = gr.State(value=False)
        live_obs_timer = gr.Timer(value=1.0 / LIVE_OBS_REFRESH_HZ, active=True)
        session_init_retry_timer = gr.Timer(value=0.5, active=False)

        task_info_box = gr.Textbox(visible=False, elem_id="task_info_box")
        progress_info_box = gr.Textbox(visible=False)
        goal_box = gr.Textbox(visible=False)

        with gr.Column(visible=True, elem_id="main_interface_root") as main_interface:
            native_progress_host = gr.Markdown(
                value="",
                visible=True,
                container=False,
                elem_id="native_progress_host",
            )
            with gr.Row(elem_id="main_layout_row"):
                with gr.Column(scale=POINT_SELECTION_SCALE):
                    with gr.Column(elem_classes=["native-card"], elem_id="media_card"):
                        with gr.Column(visible=False, elem_id="video_phase_group") as video_phase_group:
                            video_display = gr.Video(
                                label="Demonstration Video 🎬",
                                interactive=False,
                                elem_id="demo_video",
                                autoplay=False,
                                show_label=True,
                                visible=True,
                            )
                            watch_demo_video_btn = gr.Button(
                                "Watch Video Input 🎬",
                                variant="primary",
                                size="lg",
                                interactive=False,
                                visible=False,
                                elem_id="watch_demo_video_btn",
                            )

                        with gr.Column(visible=False, elem_id="action_phase_group") as action_phase_group:
                            img_display = gr.Image(
                                label="Point Selection",
                                interactive=False,
                                type="pil",
                                elem_id="live_obs",
                                elem_classes=get_live_obs_elem_classes(),
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
                                        label="Action Selection 🦾",
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
                                        label="System Log 📝",
                                    )

                        with gr.Row(elem_id="action_buttons_row"):
                            with gr.Column(elem_classes=["native-card", "native-button-card"], elem_id="exec_btn_card"):
                                exec_btn = gr.Button("EXECUTE 🤖", variant="stop", size="lg", elem_id="exec_btn")

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="reference_btn_card",
                            ):
                                reference_action_btn = gr.Button(
                                    "Ground Truth Action 🙋‍♀️",
                                    variant="secondary",
                                    interactive=False,
                                    elem_id="reference_action_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="restart_episode_btn_card",
                            ):
                                restart_episode_btn = gr.Button(
                                    "Restart Episode 🔄",
                                    variant="secondary",
                                    interactive=False,
                                    elem_id="restart_episode_btn",
                                )

                            with gr.Column(
                                elem_classes=["native-card", "native-button-card"],
                                elem_id="next_task_btn_card",
                            ):
                                next_task_btn = gr.Button(
                                    "Change Episode ⏩️",
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
                                label="Task Hint💡",
                                interactive=True,
                                elem_id="task_hint_display",
                            )

        load_flow_outputs = [
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
            reference_action_btn,
            ui_phase_state,
            session_boot_pending_state,
            native_progress_host,
            session_init_retry_timer,
        ]
        phase_visibility_outputs = [
            video_phase_group,
            action_phase_group,
            control_panel_group,
        ]
        action_queue_kwargs = {
            "concurrency_id": SESSION_CONCURRENCY_ID,
            "concurrency_limit": SESSION_CONCURRENCY_LIMIT,
        }
        init_queue_kwargs = {
            "concurrency_id": SESSION_INIT_CONCURRENCY_ID,
            "concurrency_limit": SESSION_INIT_CONCURRENCY_LIMIT,
        }

        def _skip_load_flow():
            return tuple(gr.skip() for _ in range(len(load_flow_outputs)))

        def _pending_init_flow(uid, queue_position):
            queue_label = f'queue: {max(1, int(queue_position or 1))}'
            return (
                uid,
                gr.update(visible=True),
                gr.update(interactive=False),
                "",
                gr.update(choices=[], value=None),
                "",
                "",
                gr.update(value=None, visible=False),
                gr.update(visible=False, interactive=False),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(interactive=False),
                PHASE_INIT,
                True,
                gr.update(value=f'{UI_TEXT["progress"]["queue_wait"]} | {queue_label}'),
                gr.update(active=True),
            )

        def _coerce_init_load_result(result):
            if isinstance(result, dict):
                status = result.get("status")
                if status == "pending":
                    return _pending_init_flow(result.get("uid"), result.get("queue_position"))
                if status == "skip":
                    return _skip_load_flow()
                if status == "ready":
                    return _with_phase_from_load(result.get("load_result"))
            return _with_phase_from_load(result)

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

        def _resolve_header_task_state(task_text, fallback_env=None):
            base_choices = list(user_manager.env_choices)
            parsed_env = render_header_task(task_text)
            selected_env = _normalize_env_choice(parsed_env, base_choices)
            if selected_env is None:
                selected_env = _normalize_env_choice(fallback_env, base_choices)

            choices = list(base_choices)
            if selected_env and selected_env not in choices:
                choices.append(selected_env)
            return choices, selected_env

        def _build_header_task_update(task_text, fallback_env=None):
            choices, selected_env = _resolve_header_task_state(task_text, fallback_env=fallback_env)
            return gr.update(choices=choices, value=selected_env)

        def sync_header_from_task(task_text, goal_text):
            _, selected_env = _resolve_header_task_state(task_text)
            return _build_header_task_update(task_text), render_header_goal(goal_text), selected_env

        def sync_header_from_goal(goal_text, task_text, current_header_task):
            _, selected_env = _resolve_header_task_state(task_text, fallback_env=current_header_task)
            return (
                _build_header_task_update(task_text, fallback_env=current_header_task),
                render_header_goal(goal_text),
                selected_env,
            )

        def init_app_with_phase(request: gr.Request):
            return _coerce_init_load_result(init_app(request))

        def resume_pending_init_with_phase(uid, init_pending, request: gr.Request):
            return _coerce_init_load_result(resume_pending_init(uid, init_pending, request))

        def load_next_task_with_phase(uid):
            return _with_phase_from_load(load_next_task_wrapper(uid))

        def restart_episode_with_phase(uid):
            return _with_phase_from_load(restart_episode_wrapper(uid))

        def switch_env_with_phase(uid, selected_env):
            return _with_phase_from_load(switch_env_wrapper(uid, selected_env))

        def _normalize_selected_env(selected_env, current_task_env):
            base_choices = list(user_manager.env_choices)
            normalized_selected_env = _normalize_env_choice(selected_env, base_choices)
            normalized_current_env = _normalize_env_choice(current_task_env, base_choices)
            return normalized_selected_env, normalized_current_env

        def maybe_switch_env_with_phase(uid, selected_env, current_task_env):
            normalized_selected_env, normalized_current_env = _normalize_selected_env(
                selected_env,
                current_task_env,
            )
            if not normalized_selected_env or normalized_selected_env == normalized_current_env:
                return _skip_load_flow()
            return switch_env_with_phase(uid, normalized_selected_env)

        task_info_box.change(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        )
        goal_box.change(
            fn=sync_header_from_goal,
            inputs=[goal_box, task_info_box, header_task_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        )

        header_task_switch = header_task_box.select(
            fn=maybe_switch_env_with_phase,
            inputs=[uid_state, header_task_box, current_task_env_state],
            outputs=load_flow_outputs,
            show_progress="full",
            js=SET_EPISODE_LOAD_MODE_IF_SWITCH_JS,
            show_progress_on=[native_progress_host],
            **action_queue_kwargs,
        ).then(
            fn=_phase_visibility_updates,
            inputs=[ui_phase_state],
            outputs=phase_visibility_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )
        header_task_switch.success(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )
        header_task_switch.failure(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )

        next_task_click = next_task_btn.click(
            fn=load_next_task_with_phase,
            inputs=[uid_state],
            outputs=load_flow_outputs,
            show_progress="full",
            js=SET_EPISODE_LOAD_MODE_JS,
            show_progress_on=[native_progress_host],
            **action_queue_kwargs,
        ).then(
            fn=_phase_visibility_updates,
            inputs=[ui_phase_state],
            outputs=phase_visibility_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )
        next_task_click.success(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )
        next_task_click.failure(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )

        restart_episode_click = restart_episode_btn.click(
            fn=restart_episode_with_phase,
            inputs=[uid_state],
            outputs=load_flow_outputs,
            show_progress="full",
            js=SET_EPISODE_LOAD_MODE_JS,
            show_progress_on=[native_progress_host],
            **action_queue_kwargs,
        ).then(
            fn=_phase_visibility_updates,
            inputs=[ui_phase_state],
            outputs=phase_visibility_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )
        restart_episode_click.success(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )
        restart_episode_click.failure(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )

        video_display.end(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[
                video_phase_group,
                action_phase_group,
                control_panel_group,
                log_output,
                watch_demo_video_btn,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_POINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )
        video_display.stop(
            fn=on_video_end_transition,
            inputs=[uid_state],
            outputs=[
                video_phase_group,
                action_phase_group,
                control_panel_group,
                log_output,
                watch_demo_video_btn,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_POINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )

        img_display.select(
            fn=on_map_click,
            inputs=[uid_state, options_radio],
            outputs=[img_display, coords_box, log_output],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )

        options_radio.change(
            fn=on_option_select,
            inputs=[uid_state, options_radio, coords_box, suppress_next_option_change_state],
            outputs=[coords_box, img_display, log_output, suppress_next_option_change_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )

        watch_demo_video_btn.click(
            fn=on_demo_video_play,
            inputs=[uid_state],
            outputs=[watch_demo_video_btn],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )

        reference_action_btn.click(
            fn=on_reference_action,
            inputs=[uid_state, options_radio],
            outputs=[img_display, options_radio, coords_box, log_output, suppress_next_option_change_state],
            **action_queue_kwargs,
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )

        exec_btn.click(
            fn=precheck_execute_inputs,
            inputs=[uid_state, options_radio, coords_box],
            outputs=[],
            queue=False,
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
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_EXECUTION_PLAYBACK,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=execute_step,
            inputs=[uid_state, options_radio, coords_box],
            outputs=[img_display, log_output, task_info_box, progress_info_box, restart_episode_btn, next_task_btn, exec_btn],
            show_progress="hidden",
            **action_queue_kwargs,
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
            queue=False,
            show_progress="hidden",
        ).then(
            fn=lambda: PHASE_ACTION_POINT,
            outputs=[ui_phase_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session,
            inputs=[uid_state],
            outputs=[uid_state],
            queue=False,
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
            js=THEME_LOCK_JS,
            queue=False,
        )

        demo.load(
            fn=None,
            js=LIVE_OBS_CLIENT_RESIZE_JS,
            queue=False,
        )

        demo.load(
            fn=None,
            js=DEMO_VIDEO_PLAY_BINDING_JS,
            queue=False,
        )

        demo.load(
            fn=None,
            js=PROGRESS_TEXT_REWRITE_JS,
            queue=False,
        )

        init_load = demo.load(
            fn=init_app_with_phase,
            inputs=[],
            outputs=load_flow_outputs,
            show_progress="full",
            js=SET_EPISODE_LOAD_MODE_JS,
            show_progress_on=[native_progress_host],
            **init_queue_kwargs,
        ).then(
            fn=_phase_visibility_updates,
            inputs=[ui_phase_state],
            outputs=phase_visibility_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session_or_preserve_pending,
            inputs=[uid_state, session_boot_pending_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )
        init_load.success(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )
        init_load.failure(
            fn=None,
            js=RESET_EPISODE_LOAD_MODE_JS,
            queue=False,
            show_progress="hidden",
        )

        session_init_retry_timer.tick(
            fn=resume_pending_init_with_phase,
            inputs=[uid_state, session_boot_pending_state],
            outputs=load_flow_outputs,
            show_progress="hidden",
            **init_queue_kwargs,
        ).then(
            fn=_phase_visibility_updates,
            inputs=[ui_phase_state],
            outputs=phase_visibility_outputs,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=sync_header_from_task,
            inputs=[task_info_box, goal_box],
            outputs=[header_task_box, header_goal_box, current_task_env_state],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=touch_session_or_preserve_pending,
            inputs=[uid_state, session_boot_pending_state],
            outputs=[uid_state],
            queue=False,
            show_progress="hidden",
        )

        demo.unload(fn=cleanup_current_request_session)
        demo.queue(max_size=None, default_concurrency_limit=None)

    return demo

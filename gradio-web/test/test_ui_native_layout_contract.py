from __future__ import annotations


def test_native_ui_has_no_legacy_runtime_js_or_card_shell_tokens(reload_module):
    ui_layout = reload_module("ui_layout")

    assert ui_layout.SYNC_JS.strip() == ""

    css = ui_layout.CSS
    assert ".native-card" in css

    forbidden_tokens = [
        "card-shell-hit",
        "card-shell-button",
        "floating-card",
        "applyCardShellOnce",
        "media_card_anchor",
        "action_selection_card_anchor",
        "next_task_btn_card_anchor",
        "MutationObserver",
    ]
    for token in forbidden_tokens:
        assert token not in css


def test_native_ui_css_uses_configured_global_font_size_variables(reload_module):
    config = reload_module("config")
    ui_layout = reload_module("ui_layout")

    css = ui_layout.CSS

    assert f"--body-text-size: {config.UI_GLOBAL_FONT_SIZE} !important;" in css
    assert f"--prose-text-size: {config.UI_GLOBAL_FONT_SIZE} !important;" in css
    assert f"--input-text-size: {config.UI_GLOBAL_FONT_SIZE} !important;" in css
    assert f"--block-label-text-size: {config.UI_GLOBAL_FONT_SIZE} !important;" in css
    assert f"--button-large-text-size: {config.UI_GLOBAL_FONT_SIZE} !important;" in css
    assert f"--section-header-text-size: {config.UI_GLOBAL_FONT_SIZE} !important;" in css
    assert f"--text-md: {config.UI_GLOBAL_FONT_SIZE} !important;" in css


def test_native_ui_css_excludes_header_title_from_global_font_size(reload_module):
    ui_layout = reload_module("ui_layout")

    assert "#header_title h2" in ui_layout.CSS
    assert "font-size: var(--text-xxl) !important;" in ui_layout.CSS


def test_extract_last_goal_prefers_last_list_item(reload_module):
    ui_layout = reload_module("ui_layout")

    assert ui_layout.extract_last_goal("['goal a', 'goal b']") == "goal b"


def test_native_ui_config_contains_phase_machine_and_precheck_chain(reload_module):
    ui_layout = reload_module("ui_layout")
    demo = ui_layout.create_ui_blocks()

    try:
        cfg = demo.get_config_file()

        elem_ids = {
            comp.get("props", {}).get("elem_id")
            for comp in cfg.get("components", [])
            if comp.get("props", {}).get("elem_id")
        }

        required_ids = {
            "header_task",
            "loading_overlay_group",
            "main_layout_row",
            "media_card",
            "log_card",
            "right_top_row",
            "right_action_col",
            "right_log_col",
            "control_panel_group",
            "video_phase_group",
            "action_phase_group",
            "demo_video",
            "watch_demo_video_btn",
            "live_obs",
            "action_radio",
            "coords_box",
            "exec_btn",
            "reference_action_btn",
            "restart_episode_btn",
            "next_task_btn",
        }
        missing = required_ids - elem_ids
        assert not missing, f"missing required elem_ids: {sorted(missing)}"

        values = [
            comp.get("props", {}).get("value")
            for comp in cfg.get("components", [])
            if "value" in comp.get("props", {})
        ]
        assert all("_anchor" not in str(v) for v in values)
        assert any(
            "Logging in and setting up environment... Please wait." in str(v)
            for v in values
        )
        assert all("Loading environment, please wait..." not in str(v) for v in values)

        log_output_comp = next(
            comp
            for comp in cfg.get("components", [])
            if comp.get("props", {}).get("elem_id") == "log_output"
        )
        assert log_output_comp.get("props", {}).get("max_lines") is None

        demo_video_comp = next(
            comp
            for comp in cfg.get("components", [])
            if comp.get("props", {}).get("elem_id") == "demo_video"
        )
        assert demo_video_comp.get("props", {}).get("autoplay") is False

        api_names = [dep.get("api_name") for dep in cfg.get("dependencies", [])]
        assert "on_demo_video_play" in api_names
        assert "precheck_execute_inputs" in api_names
        assert "switch_to_execute_phase" in api_names
        assert "execute_step" in api_names
        assert "switch_to_action_phase" in api_names
    finally:
        demo.close()

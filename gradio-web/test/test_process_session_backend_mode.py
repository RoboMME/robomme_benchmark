from __future__ import annotations


class _FakeSoftwareRenderClient:
    def __init__(self, dataset_root=None, gui_render=False):
        self.dataset_root = dataset_root
        self.gui_render = gui_render
        self.closed = False
        self.calls = []

    def call(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if method == "load_episode":
            return {
                "result": ("IMG", "Ready"),
                "snapshot": {
                    "env_id": "BinFill",
                    "episode_idx": 7,
                    "language_goal": "goal",
                    "difficulty": "hard",
                    "seed": 123,
                    "demonstration_frames": [],
                    "base_frames": [],
                    "wrist_frames": [],
                    "available_options": [("A. pick", 0)],
                    "raw_solve_options": [{"label": "A", "action": "pick", "available": False}],
                    "seg_vis": None,
                    "is_demonstration": False,
                    "non_demonstration_task_length": 9,
                    "last_execution_frames": [],
                },
            }
        if method == "get_reference_action":
            return {
                "result": {"ok": True, "message": "resolved"},
                "snapshot": {
                    "env_id": "BinFill",
                    "episode_idx": 7,
                    "language_goal": "goal",
                    "difficulty": "hard",
                    "seed": 123,
                    "demonstration_frames": [],
                    "base_frames": [],
                    "wrist_frames": [],
                    "available_options": [],
                    "raw_solve_options": [],
                    "seg_vis": None,
                    "is_demonstration": False,
                    "non_demonstration_task_length": 9,
                    "last_execution_frames": [],
                },
            }
        return {"result": None, "snapshot": {}}

    def close(self):
        self.closed = True


def test_process_session_proxy_uses_software_render_backend_on_compute_only_spaces(
    monkeypatch, reload_module
):
    monkeypatch.setenv("SPACE_ID", "user/demo")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    process_session = reload_module("process_session")
    monkeypatch.setattr(process_session, "SoftwareRenderSessionClient", _FakeSoftwareRenderClient)

    proxy = process_session.ProcessSessionProxy(dataset_root=None, gui_render=False)

    assert proxy._backend_mode == "software_render_subprocess"
    img, message = proxy.load_episode("BinFill", 7)

    assert img == "IMG"
    assert message == "Ready"
    assert proxy.env_id == "BinFill"
    assert proxy.episode_idx == 7
    assert proxy.seed == 123
    assert proxy.non_demonstration_task_length == 9
    assert proxy.available_options == [("A. pick", 0)]

    reference = proxy.get_reference_action()

    assert reference["ok"] is True

    proxy.close()
    assert proxy._software_session.closed is True


def test_process_session_proxy_load_episode_returns_explicit_software_render_error(
    monkeypatch, reload_module
):
    monkeypatch.setenv("SPACE_ID", "user/demo")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    process_session = reload_module("process_session")

    class _FailingSoftwareRenderClient:
        def __init__(self, dataset_root=None, gui_render=False):
            self.dataset_root = dataset_root
            self.gui_render = gui_render

        def call(self, method, *args, **kwargs):
            raise process_session.SoftwareRenderUnsupportedError("llvmpipe unavailable")

        def close(self):
            return None

    monkeypatch.setattr(process_session, "SoftwareRenderSessionClient", _FailingSoftwareRenderClient)

    proxy = process_session.ProcessSessionProxy(dataset_root=None, gui_render=False)
    img, message = proxy.load_episode("BinFill", 0)

    assert img is None
    assert "ZeroGPU Space only provides compute access" in message
    assert "llvmpipe unavailable" in message

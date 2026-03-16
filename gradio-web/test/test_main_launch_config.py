from __future__ import annotations

import os
import sys
import types
from pathlib import Path


DEFAULT_LLVMPipe_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"


class _FakeDemo:
    def __init__(self):
        self.launch_kwargs = None
        self.theme = "default"
        self.head = "<script>window.__robommeForceLightTheme=()=>{};</script>"

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return None


def test_main_launch_passes_ui_css_and_forces_cpu_runtime(monkeypatch, reload_module):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "compute,utility,graphics")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/tmp/nvidia_icd.json")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main = reload_module("main")
    fake_demo = _FakeDemo()
    fake_ui_layout = types.SimpleNamespace(
        CSS="#reference_action_btn button:not(:disabled){background:#1f8b4c;}",
        create_ui_blocks=lambda: fake_demo,
    )

    monkeypatch.setitem(sys.modules, "ui_layout", fake_ui_layout)
    monkeypatch.setenv("PORT", "7861")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/tmp/another_nvidia_icd.json")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main.main()

    assert fake_demo.launch_kwargs is not None
    assert fake_demo.launch_kwargs["server_name"] == "0.0.0.0"
    assert fake_demo.launch_kwargs["server_port"] == 7861
    assert fake_demo.launch_kwargs["theme"] == fake_demo.theme
    assert fake_demo.launch_kwargs["css"] == fake_ui_layout.CSS
    assert fake_demo.launch_kwargs["head"] == fake_demo.head
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert os.environ["VK_ICD_FILENAMES"] == "/tmp/another_nvidia_icd.json"
    assert "NVIDIA_DRIVER_CAPABILITIES" not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert "MUJOCO_GL" not in os.environ


def test_configure_cpu_only_runtime_autosets_llvmpipe_icd(monkeypatch, reload_module):
    original_exists = Path.exists

    def fake_exists(self):
        if str(self) == DEFAULT_LLVMPipe_ICD:
            return True
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main = reload_module("main")
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)

    main.configure_cpu_only_runtime()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert os.environ["VK_ICD_FILENAMES"] == DEFAULT_LLVMPipe_ICD
    assert "NVIDIA_DRIVER_CAPABILITIES" not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert "MUJOCO_GL" not in os.environ


def test_configure_cpu_only_runtime_preserves_existing_vk_icd(monkeypatch, reload_module):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/tmp/custom_icd.json")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main = reload_module("main")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/tmp/preserved_icd.json")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main.configure_cpu_only_runtime()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert os.environ["VK_ICD_FILENAMES"] == "/tmp/preserved_icd.json"
    assert "NVIDIA_DRIVER_CAPABILITIES" not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert "MUJOCO_GL" not in os.environ


def test_configure_cpu_only_runtime_clears_stale_sapien_render_device(monkeypatch, reload_module):
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")

    main = reload_module("main")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda:0")

    main.configure_cpu_only_runtime()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert "SAPIEN_RENDER_DEVICE" not in os.environ

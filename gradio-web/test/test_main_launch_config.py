from __future__ import annotations

import os
import sys
import types
from pathlib import Path


DEFAULT_LLVMPipe_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
DEFAULT_CPU_RENDER_BACKEND = "cpu"
DEFAULT_ZEROGPU_RENDER_BACKEND = "cuda"
RENDER_BACKEND_AUTO_ENV = "ROBOMME_RENDER_BACKEND_AUTO"


class _FakeDemo:
    def __init__(self):
        self.launch_kwargs = None
        self.theme = "default"
        self.css = "#app{}"
        self.head = "<script>window.__robommeForceLightTheme=()=>{};</script>"

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return None


def test_main_launch_passes_ui_css_and_uses_local_cpu_fallback(monkeypatch, reload_module):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("ROBOMME_RENDER_BACKEND", raising=False)
    monkeypatch.delenv(RENDER_BACKEND_AUTO_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/tmp/another_nvidia_icd.json")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main = reload_module("main")
    fake_demo = _FakeDemo()
    fake_ui_layout = types.SimpleNamespace(
        CSS="#reference_action_btn button:not(:disabled){background:#1f8b4c;}",
        create_ui_blocks=lambda: fake_demo,
    )

    monkeypatch.setitem(sys.modules, "ui_layout", fake_ui_layout)
    monkeypatch.setenv("PORT", "7861")

    built_demo = main.build_app()
    main.main(demo=built_demo)

    assert built_demo is fake_demo
    assert fake_demo.launch_kwargs is not None
    assert fake_demo.launch_kwargs["server_name"] == "0.0.0.0"
    assert fake_demo.launch_kwargs["server_port"] == 7861
    assert fake_demo.launch_kwargs["theme"] == fake_demo.theme
    assert fake_demo.launch_kwargs["css"] == fake_demo.css
    assert fake_demo.launch_kwargs["head"] == fake_demo.head
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert os.environ["ROBOMME_RENDER_BACKEND"] == DEFAULT_CPU_RENDER_BACKEND
    assert os.environ[RENDER_BACKEND_AUTO_ENV] == "1"
    assert os.environ["VK_ICD_FILENAMES"] == "/tmp/another_nvidia_icd.json"
    assert "NVIDIA_DRIVER_CAPABILITIES" not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert "MUJOCO_GL" not in os.environ


def test_configure_runtime_autosets_llvmpipe_icd(monkeypatch, reload_module):
    original_exists = Path.exists

    def fake_exists(self):
        if str(self) == DEFAULT_LLVMPipe_ICD:
            return True
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("ROBOMME_RENDER_BACKEND", raising=False)
    monkeypatch.delenv(RENDER_BACKEND_AUTO_ENV, raising=False)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main = reload_module("main")
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)

    main.configure_runtime()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert os.environ["ROBOMME_RENDER_BACKEND"] == DEFAULT_CPU_RENDER_BACKEND
    assert os.environ[RENDER_BACKEND_AUTO_ENV] == "1"
    assert os.environ["VK_ICD_FILENAMES"] == DEFAULT_LLVMPipe_ICD
    assert "NVIDIA_DRIVER_CAPABILITIES" not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert "MUJOCO_GL" not in os.environ


def test_configure_runtime_preserves_gpu_env_on_spaces(monkeypatch, reload_module):
    monkeypatch.setenv("SPACE_ID", "user/demo")
    monkeypatch.delenv(RENDER_BACKEND_AUTO_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("ROBOMME_RENDER_BACKEND", "pci:7")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setenv("VK_ICD_FILENAMES", "/tmp/preserved_icd.json")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    main = reload_module("main")

    result = main.configure_runtime()

    assert result["mode"] == "spaces"
    assert result["cpu_only"] is False
    assert result["render_backend"] == "pci:7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert os.environ["ROBOMME_RENDER_BACKEND"] == "pci:7"
    assert RENDER_BACKEND_AUTO_ENV not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert os.environ["NVIDIA_DRIVER_CAPABILITIES"] == "graphics"
    assert os.environ["VK_ICD_FILENAMES"] == "/tmp/preserved_icd.json"
    assert "MUJOCO_GL" not in os.environ


def test_configure_runtime_spaces_defaults_to_gpu_render_backend(monkeypatch, reload_module):
    monkeypatch.setenv("SPACE_ID", "user/demo")
    monkeypatch.delenv(RENDER_BACKEND_AUTO_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("SAPIEN_RENDER_DEVICE", "cuda")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    monkeypatch.delenv("ROBOMME_RENDER_BACKEND", raising=False)

    main = reload_module("main")

    result = main.configure_runtime()

    assert result["mode"] == "spaces"
    assert result["cpu_only"] is False
    assert result["render_backend"] == DEFAULT_ZEROGPU_RENDER_BACKEND
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert os.environ["ROBOMME_RENDER_BACKEND"] == DEFAULT_ZEROGPU_RENDER_BACKEND
    assert os.environ[RENDER_BACKEND_AUTO_ENV] == "1"
    assert "VK_ICD_FILENAMES" not in os.environ
    assert "SAPIEN_RENDER_DEVICE" not in os.environ
    assert "MUJOCO_GL" not in os.environ

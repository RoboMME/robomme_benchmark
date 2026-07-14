"""No-Patch 报告调试环境快照的轻量单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "data-generation-v2-noPatch"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import write_generation_report as writer


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Mock RTX"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (9, 0)

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(total_memory=24 * 1024**3)


class _FakeTorch:
    __version__ = "9.9.9+mock"
    version = SimpleNamespace(cuda="12.9")
    backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 99999))
    cuda = _FakeCuda()


class _FakeDistribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.name = name
        self.version = version


def _nvidia_row() -> str:
    values = {source: f"value-{index}" for index, (source, _) in enumerate(writer.GPU_QUERY_FIELDS)}
    values.update(
        {
            "index": "0",
            "uuid": "GPU-mock-uuid",
            "serial": "mock-serial",
            "name": "Mock RTX",
            "pci.bus_id": "00000000:AA:00.0",
            "driver_version": "999.99",
            "vbios_version": "99.99",
            "compute_cap": "9.0",
            "memory.total": "24576",
            "memory.used": "1024",
            "memory.free": "23552",
            "power.limit": "450.00",
            "power.draw": "123.45",
            "temperature.gpu": "42",
            "utilization.gpu": "12",
            "utilization.memory": "3",
            "pstate": "P2",
            "clocks.current.graphics": "2100",
            "clocks.current.memory": "10000",
            "clocks.max.graphics": "2500",
            "clocks.max.memory": "12000",
            "pcie.link.gen.current": "4",
            "pcie.link.gen.max": "5",
            "pcie.link.width.current": "16",
            "pcie.link.width.max": "16",
            "persistence_mode": "Enabled",
            "addressing_mode": "None",
            "fan.speed": "35",
        }
    )
    return ",".join(values[source] for source, _ in writer.GPU_QUERY_FIELDS)


def _install_successful_probes(monkeypatch) -> None:
    lscpu = {
        "lscpu": [
            {"field": "Architecture:", "data": "x86_64"},
            {"field": "Model name:", "data": "Mock CPU"},
            {"field": "Socket(s):", "data": "2"},
            {"field": "Thread(s) per core:", "data": "2"},
        ]
    }

    def fake_run(command):
        command = tuple(command)
        if command == ("lscpu", "--json"):
            return json.dumps(lscpu), None
        if command == ("nvidia-smi",):
            return "NVIDIA-SMI 999.99 Driver Version: 999.99 CUDA Version: 12.9", None
        if command[0] == "nvidia-smi":
            return _nvidia_row(), None
        if command == ("uv", "--version"):
            return "uv 9.9.9", None
        if command == ("git", "--version"):
            return "git version 9.9.9", None
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(writer, "_run_command", fake_run)
    monkeypatch.setattr(writer.importlib, "import_module", lambda name: _FakeTorch)
    monkeypatch.setattr(
        writer.importlib_metadata,
        "distributions",
        lambda: [
            _FakeDistribution("zeta", "1.0"),
            _FakeDistribution("Torch", "9.9.9+mock"),
            _FakeDistribution("h5py", "3.0"),
            _FakeDistribution("mani_skill", "3.0"),
        ],
    )
    monkeypatch.setattr(writer.shutil, "which", lambda name: f"/mock/bin/{name}")


def test_snapshot_schema_markdown_and_full_package_list(monkeypatch, tmp_path) -> None:
    _install_successful_probes(monkeypatch)
    monkeypatch.setattr(writer, "REPORTS_ROOT", tmp_path / "reports")

    report = {"status": "passed", "parameters": {}, "validation": {}}
    paths = writer.write_generation_report(tmp_path, report)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    snapshot = payload["debug_environment"]

    assert payload["schema_version"] == 3
    assert snapshot["schema_version"] == 1
    assert snapshot["cpu"]["lscpu"]["raw"]["lscpu"][1]["data"] == "Mock CPU"
    assert snapshot["gpu"]["nvidia_smi"]["cuda_version"] == "12.9"
    device = snapshot["gpu"]["devices"][0]
    assert device["uuid"] == "GPU-mock-uuid"
    assert device["fan_speed_percent"] == "35"
    for _, field in writer.GPU_QUERY_FIELDS:
        assert field in device
    assert snapshot["gpu"]["torch"]["visible_devices"][0]["name"] == "Mock RTX"
    assert snapshot["packages"]["distributions"] == [
        {"name": "h5py", "version": "3.0"},
        {"name": "mani_skill", "version": "3.0"},
        {"name": "Torch", "version": "9.9.9+mock"},
        {"name": "zeta", "version": "1.0"},
    ]

    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "## 调试环境" in markdown
    assert markdown.index("## 调试环境") < markdown.index("## 参数")
    assert "### GPU（nvidia-smi）" in markdown
    assert "fan_speed_percent" in markdown
    assert "### Python、工具与运行时" in markdown
    assert "全部依赖版本数：4" in markdown
    assert "| mani-skill | 3.0 |" in markdown


def test_failed_gpu_torch_and_package_probes_still_write_report(monkeypatch, tmp_path) -> None:
    def failed_run(command):
        command = tuple(command)
        if command == ("lscpu", "--json"):
            return None, "FileNotFoundError: lscpu"
        if command[0] == "nvidia-smi":
            return None, "FileNotFoundError: nvidia-smi"
        return None, f"unavailable: {command[0]}"

    def failed_import(name: str):
        assert name == "torch"
        raise ImportError("mock torch unavailable")

    def failed_distributions():
        raise RuntimeError("mock distribution lookup failure")

    monkeypatch.setattr(writer, "_run_command", failed_run)
    monkeypatch.setattr(writer.importlib, "import_module", failed_import)
    monkeypatch.setattr(writer.importlib_metadata, "distributions", failed_distributions)
    monkeypatch.setattr(writer, "REPORTS_ROOT", tmp_path / "reports")

    paths = writer.write_generation_report(
        tmp_path,
        {"status": "failed", "parameters": {}, "validation": {}},
    )
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    assert Path(paths["json"]).is_file()
    assert Path(paths["markdown"]).is_file()
    snapshot = payload["debug_environment"]
    assert "FileNotFoundError" in snapshot["gpu"]["nvidia_smi"]["error"]
    assert "import" in snapshot["gpu"]["torch"]["errors"]
    assert "distributions" in snapshot["packages"]["errors"]

    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "### 探测错误" in markdown
    assert "nvidia-smi" in markdown
    assert "Torch/import" in markdown

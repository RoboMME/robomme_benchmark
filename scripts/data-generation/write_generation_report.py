#!/usr/bin/env python3
"""Assemble and write the No-Patch generation, validation, and joint_action comparison report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from compare_joint_actions import JointActionComparisonError, compare_joint_actions
from validate_generated_dataset_contract import (
    MAX_EPISODES,
    METADATA_ROOT,
    REFERENCE_ROOT,
    DatasetContractError,
    parse_tasks,
    read_train_metadata,
    validate_generated_dataset_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_ABS_DIFF = 1e-8
REPORT_SCHEMA_VERSION = 4
DEBUG_ENVIRONMENT_SCHEMA_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_ROOT = SCRIPT_DIR / "reports"
REFERENCE_REVISION = "a5e4e25ffe8af34f64944f9533d06455ce5f8337"
GPU_ID = "0"


class ReportGenerationError(RuntimeError):
    """Report input or writing does not satisfy the No-Patch report contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_section(paths: Sequence[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ReportGenerationError(f"Manifest file does not exist: {path}")
        size_bytes = path.stat().st_size
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": size_bytes,
                "sha256": _sha256(path),
            }
        )
    return {
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }


def build_artifact_manifest(
    output_dir: str | Path,
    tasks: Sequence[str],
    reference_root: str | Path,
) -> dict[str, Any]:
    """Hash the complete generated deliverable and the official HDF5 files used for comparison."""
    output = Path(output_dir).expanduser().resolve()
    reference = Path(reference_root).expanduser().resolve()
    generated_paths: list[Path] = []
    reference_paths: list[Path] = []
    for task in tasks:
        generated_paths.extend(
            (
                output / f"record_dataset_{task}.h5",
                output / f"record_dataset_{task}_metadata.json",
            )
        )
        reference_paths.append(reference / f"record_dataset_{task}.h5")
    return {
        "status": "collected",
        "reference_revision": REFERENCE_REVISION,
        "generated": _manifest_section(generated_paths),
        "official_reference_hdf5": _manifest_section(reference_paths),
    }


def _head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_text_atomic(path: Path, value: str) -> None:
    """Atomically replace with a temporary file in the same directory to avoid leaving a partial report after an interruption."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def new_generation_report(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Create the base report provenance shared by the generator and read-only revalidation."""
    lock_path = REPO_ROOT / "uv.lock"
    if not lock_path.is_file():
        raise ReportGenerationError(f"Missing uv.lock: {lock_path}")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_head": _head(),
        "uv_lock_sha256": _sha256(lock_path),
        "parameters": dict(parameters),
    }


def build_validation_report(
    output_dir: str | Path,
    tasks: Sequence[str],
    episodes: Sequence[int],
    *,
    records_by_task: Mapping[str, Mapping[int, Mapping[str, Any]]] | None = None,
    reference_root: str | Path = REFERENCE_ROOT,
    metadata_root: str | Path = METADATA_ROOT,
    max_abs_diff: float = MAX_ABS_DIFF,
) -> dict[str, Any]:
    """Combine independent contract and joint_action comparison results for the same scope."""
    contract = validate_generated_dataset_contract(
        output_dir,
        tasks,
        episodes,
        records_by_task=records_by_task,
        reference_root=reference_root,
        metadata_root=metadata_root,
    )
    comparison = compare_joint_actions(
        output_dir,
        reference_root,
        tasks,
        episodes,
        max_abs_diff=max_abs_diff,
    )
    artifact_manifest = (
        build_artifact_manifest(output_dir, tasks, reference_root)
        if contract["scope"].get("full_16x100") is True
        else {
            "status": "not_collected",
            "reason": "SHA-256 manifests are collected only for the complete 16x100 scope",
            "reference_revision": REFERENCE_REVISION,
        }
    )
    acceptance = {
        **dict(contract["acceptance"]),
        "max_allowed_abs_diff": float(max_abs_diff),
        "max_abs_diff": comparison["max_abs_diff"],
    }
    passed = bool(contract["passed"] and comparison["passed"])
    return {
        "passed": passed,
        "scope": contract["scope"],
        "metadata": contract["metadata"],
        "generated": contract["generated"],
        "official": contract["official"],
        "joint_action_comparison": comparison,
        "artifact_manifest": artifact_manifest,
        "acceptance": acceptance,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any) -> str:
    return "Unknown" if value is None else str(value)


def _collect_errors(report: Mapping[str, Any], limit: int = 12) -> list[str]:
    """Extract a small error summary from the structured report; JSON retains the complete error list."""
    messages: list[str] = []
    validation = _mapping(report.get("validation"))
    metadata = _mapping(validation.get("metadata"))
    for audit in metadata.get("audits", []):
        if isinstance(audit, Mapping):
            messages.extend(str(item) for item in audit.get("errors", []))
    for section_name in ("generated", "official"):
        section = _mapping(validation.get(section_name))
        for audit in section.get("audits", []):
            if isinstance(audit, Mapping):
                messages.extend(str(item) for item in audit.get("errors", []))
    comparison = _mapping(validation.get("joint_action_comparison"))
    messages.extend(str(item) for item in comparison.get("errors", []))
    error = _mapping(report.get("error"))
    if error:
        messages.append(f"{error.get('type', 'Error')}: {error.get('message', '')}")
    return messages[:limit]


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-readable complete summary; per-trajectory details remain in JSON."""
    validation = _mapping(report.get("validation"))
    scope = _mapping(validation.get("scope"))
    acceptance = _mapping(validation.get("acceptance"))
    comparison = _mapping(validation.get("joint_action_comparison"))
    generation = _mapping(report.get("generation"))
    metadata = _mapping(validation.get("metadata"))
    generated = _mapping(validation.get("generated"))
    official = _mapping(validation.get("official"))
    parameters = _mapping(report.get("parameters"))
    artifact_manifest = _mapping(validation.get("artifact_manifest"))
    generated_manifest = _mapping(artifact_manifest.get("generated"))
    reference_manifest = _mapping(artifact_manifest.get("official_reference_hdf5"))
    debug_environment = _mapping(report.get("debug_environment"))
    lines = [
        "# No-Patch Dataset Generation Report",
        "",
        "## Run Provenance",
        "",
        f"- Status: {report.get('status')}",
        f"- Current HEAD: {report.get('current_head')}",
        f"- uv.lock SHA-256: {report.get('uv_lock_sha256')}",
        f"- Report time (UTC): {report.get('generated_at_utc')}",
        f"- Report mode: {report.get('report_mode', 'generation')}",
        "",
    ]
    lines.extend(_render_debug_environment(debug_environment))
    lines.extend(("", "## Parameters", ""))
    lines.extend(
        f"    {line}"
        for line in json.dumps(parameters, ensure_ascii=False, indent=2).splitlines()
    )
    lines.extend(
        (
            "",
            "## Scope",
            "",
            f"- Task count: {len(scope.get('tasks', []))}",
            f"- Episodes: {scope.get('episode_indices', [])}",
            f"- Expected trajectory count: {scope.get('expected_episode_count', 0)}",
            f"- Complete 16x100: {scope.get('full_16x100', False)}",
            "",
            "## Generation and Contract",
            "",
            f"- Successful workers: {_value(generation.get('success_count'))}",
            f"- Failed workers: {_value(generation.get('failure_count'))}",
            f"- Metadata errors: {metadata.get('error_count', 0)}",
            f"- Generated HDF5 errors: {generated.get('error_count', 0)}",
            f"- Official HDF5 errors: {official.get('error_count', 0)}",
            f"- Official final completions: {acceptance.get('official_final_completed', 0)}/{acceptance.get('expected_final_completed', 0)}",
            f"- Generated final completions: {acceptance.get('generated_final_completed', 0)}/{acceptance.get('expected_final_completed', 0)}",
            "",
            "## Element-wise joint_action Comparison",
            "",
            f"- Vectors: {comparison.get('joint_vector_count', 0)}",
            f"- Elements: {comparison.get('joint_element_count', 0)}",
            f"- Different elements: {comparison.get('different_element_count', 0)}",
            f"- Comparison errors: {comparison.get('error_count', 0)}",
            f"- Maximum absolute difference: {comparison.get('max_abs_diff')}",
            f"- Maximum-difference location: {comparison.get('max_abs_diff_location')}",
            f"- Maximum allowed absolute difference: {acceptance.get('max_allowed_abs_diff')}",
            f"- Within tolerance: {comparison.get('within_max_abs_diff', False)}",
            "",
            "## File Manifest",
            "",
            f"- Status: {artifact_manifest.get('status', 'not collected')}",
            f"- Official reference revision: {artifact_manifest.get('reference_revision', REFERENCE_REVISION)}",
            f"- Generated files: {generated_manifest.get('file_count', 0)}",
            f"- Generated bytes: {generated_manifest.get('total_size_bytes', 0)}",
            f"- Official reference HDF5 files: {reference_manifest.get('file_count', 0)}",
            f"- Official reference HDF5 bytes: {reference_manifest.get('total_size_bytes', 0)}",
            "",
            "## Conclusion",
            "",
            f"- Full acceptance passed: {validation.get('passed', False)}",
        )
    )
    generated_files = generated_manifest.get("files", [])
    if generated_files:
        lines.extend(
            (
                "",
                "### Generated Artifact SHA-256",
                "",
                "| File | Bytes | SHA-256 |",
                "| --- | ---: | --- |",
            )
        )
        for item in generated_files:
            if isinstance(item, Mapping):
                lines.append(
                    f"| {item.get('name')} | {item.get('size_bytes')} | {item.get('sha256')} |"
                )
    errors = _collect_errors(report)
    if errors:
        lines.extend(("", "## Error Summary", ""))
        lines.extend(f"- {message}" for message in errors)
        lines.append("- See the co-located JSON for complete errors and the per-trajectory audit.")
    report_paths = _mapping(report.get("report_paths"))
    if report_paths:
        lines.extend(
            (
                "",
                "## Artifacts",
                "",
                f"- JSON: {report_paths.get('json')}",
                f"- Markdown: {report_paths.get('markdown')}",
            )
        )
    return "\n".join(lines) + "\n"


def write_generation_report(
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, str]:
    """Write the unified JSON/Markdown to the fixed reports/ directory under the script directory."""
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise ReportGenerationError(f"Output directory does not exist: {output}")
    report_dir = REPORTS_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "generation_report.json"
    markdown_path = report_dir / "generation_report.md"
    paths = {"json": str(json_path), "markdown": str(markdown_path)}
    report["schema_version"] = REPORT_SCHEMA_VERSION
    try:
        report["debug_environment"] = collect_debug_environment()
    except Exception as exc:
        report["debug_environment"] = _fallback_debug_environment(exc)
    report["report_paths"] = paths
    write_text_atomic(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(markdown_path, render_markdown(report))
    return paths


def _load_prior_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportGenerationError(f"Unable to read prior report {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReportGenerationError(f"prior report must be a JSON object: {path}")
    return dict(payload)


def build_revalidation_report(
    output_dir: str | Path,
    env: str,
    tasks: Sequence[str],
    episodes: Sequence[int],
    *,
    workers: int | None = None,
    gpus: str | None = None,
    prior_report: Mapping[str, Any] | None = None,
    prior_report_path: str | None = None,
    reference_root: str | Path = REFERENCE_ROOT,
    metadata_root: str | Path = METADATA_ROOT,
    max_abs_diff: float = MAX_ABS_DIFF,
) -> dict[str, Any]:
    """Revalidate existing output without calling gym, starting generation workers, or modifying HDF5/metadata."""
    output = Path(output_dir).expanduser().resolve()
    prior_parameters = _mapping(prior_report.get("parameters")) if prior_report else {}
    parameters = dict(prior_parameters)
    parameters.update(
        {
            "output_dir": str(output),
            "env": env,
            "tasks": list(tasks),
            "episodes": len(episodes),
            "metadata_root": str(Path(metadata_root).expanduser().resolve()),
            "reference_root": str(Path(reference_root).expanduser().resolve()),
            "max_abs_diff": float(max_abs_diff),
        }
    )
    if workers is not None:
        parameters["workers"] = workers
    if gpus is not None:
        parsed_gpus = [item.strip() for item in gpus.split(",") if item.strip()]
        if parsed_gpus != [GPU_ID]:
            raise ReportGenerationError("--gpus must be exactly 0")
        parameters["gpus"] = parsed_gpus

    report = new_generation_report(parameters)
    report["report_mode"] = "existing_output_revalidation"
    if prior_report_path is not None:
        report["prior_report"] = prior_report_path
    if prior_report and prior_report.get("current_head"):
        report["prior_generation_head"] = prior_report["current_head"]
    generation = _mapping(prior_report.get("generation")) if prior_report else {}
    report["generation"] = (
        dict(generation)
        if generation
        else {
            "mode": "existing_output_revalidation",
            "requested_count": len(tasks) * len(episodes),
            "success_count": None,
            "failure_count": None,
            "results": [],
        }
    )
    records_by_task = read_train_metadata(metadata_root)
    report["validation"] = build_validation_report(
        output,
        tasks,
        episodes,
        records_by_task=records_by_task,
        reference_root=reference_root,
        metadata_root=metadata_root,
        max_abs_diff=max_abs_diff,
    )
    report["status"] = "passed" if report["validation"]["passed"] else "failed"
    return report


GPU_QUERY_FIELDS = (
    ("index", "index"),
    ("uuid", "uuid"),
    ("serial", "serial"),
    ("name", "name"),
    ("pci.bus_id", "pci_bus_id"),
    ("pci.device_id", "pci_device_id"),
    ("pci.sub_device_id", "pci_sub_device_id"),
    ("driver_version", "driver_version"),
    ("vbios_version", "vbios_version"),
    ("compute_cap", "compute_capability"),
    ("memory.total", "memory_total_mib"),
    ("memory.used", "memory_used_mib"),
    ("memory.free", "memory_free_mib"),
    ("power.limit", "power_limit_w"),
    ("power.draw", "power_draw_w"),
    ("temperature.gpu", "temperature_c"),
    ("utilization.gpu", "utilization_gpu_percent"),
    ("utilization.memory", "utilization_memory_percent"),
    ("pstate", "pstate"),
    ("clocks.current.graphics", "graphics_clock_mhz"),
    ("clocks.current.memory", "memory_clock_mhz"),
    ("clocks.max.graphics", "max_graphics_clock_mhz"),
    ("clocks.max.memory", "max_memory_clock_mhz"),
    ("pcie.link.gen.current", "pcie_link_gen_current"),
    ("pcie.link.gen.max", "pcie_link_gen_max"),
    ("pcie.link.width.current", "pcie_link_width_current"),
    ("pcie.link.width.max", "pcie_link_width_max"),
    ("persistence_mode", "persistence_mode"),
    ("addressing_mode", "addressing_mode"),
    ("fan.speed", "fan_speed_percent"),
)
GPU_STATIC_FIELDS = frozenset(
    {
        "index",
        "uuid",
        "serial",
        "name",
        "pci_bus_id",
        "pci_device_id",
        "pci_sub_device_id",
        "driver_version",
        "vbios_version",
        "compute_capability",
        "memory_total_mib",
        "power_limit_w",
        "max_graphics_clock_mhz",
        "max_memory_clock_mhz",
        "pcie_link_gen_max",
        "pcie_link_width_max",
        "persistence_mode",
        "addressing_mode",
    }
)
CORE_PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "mani-skill",
    "sapien",
    "mplib",
    "gymnasium",
    "numpy",
    "h5py",
    "opencv-python",
    "setuptools",
    "robomme",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_HOME",
    "PYTHONPATH",
)
SLURM_ALLOCATION_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_NNODES",
    "SLURM_NODELIST",
)


def _exception_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _probe(section: dict[str, Any], field: str, callback: Any) -> None:
    """Collect a single field on a best-effort basis; failures are recorded only for that field."""
    try:
        section[field] = callback()
    except Exception as exc:
        section[field] = None
        section.setdefault("errors", {})[field] = _exception_text(exc)


def _run_command(command: Sequence[str]) -> tuple[str | None, str | None]:
    """Run an allowlisted command and return stdout or a serializable error."""
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, _exception_text(exc)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return None, detail or f"exit code {result.returncode}"
    return result.stdout, None


def _normalized_nvidia_value(value: str) -> str | None:
    normalized = value.strip()
    if normalized.casefold() in {"", "n/a", "none", "not supported", "unknown"}:
        return None
    return normalized


def _header_component(text: str, marker: str) -> str | None:
    if marker not in text:
        return None
    remainder = text.split(marker, 1)[1].split("|", 1)[0].strip()
    return remainder.split()[0] if remainder else None


def _collect_host_environment() -> dict[str, Any]:
    host: dict[str, Any] = {"errors": {}}
    _probe(host, "hostname", platform.node)
    _probe(host, "machine", platform.machine)
    _probe(host, "architecture", lambda: platform.architecture()[0])
    _probe(host, "processor", platform.processor)
    return host


def _collect_operating_system() -> dict[str, Any]:
    operating_system: dict[str, Any] = {"errors": {}, "libc": {"errors": {}}}
    _probe(operating_system, "system", platform.system)
    _probe(operating_system, "release", platform.release)
    _probe(operating_system, "kernel", platform.release)
    _probe(operating_system, "version", platform.version)
    _probe(operating_system, "platform", platform.platform)
    libc = operating_system["libc"]
    _probe(libc, "library", lambda: platform.libc_ver()[0])
    _probe(libc, "version", lambda: platform.libc_ver()[1])
    return operating_system


def _memory_total_bytes() -> int:
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key == "MemTotal":
                return int(value.strip().split()[0]) * 1024
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(pages) * int(page_size)


def _collect_cpu_environment() -> dict[str, Any]:
    cpu: dict[str, Any] = {
        "os_cpu_count": None,
        "affinity_cpu_ids": None,
        "affinity_cpu_count": None,
        "lscpu": {"raw": None, "error": None},
        "errors": {},
    }
    _probe(cpu, "os_cpu_count", os.cpu_count)
    try:
        affinity = sorted(os.sched_getaffinity(0))
        cpu["affinity_cpu_ids"] = affinity
        cpu["affinity_cpu_count"] = len(affinity)
    except Exception as exc:
        cpu["errors"]["affinity_cpu_ids"] = _exception_text(exc)

    stdout, error = _run_command(("lscpu", "--json"))
    if error is not None:
        cpu["lscpu"]["error"] = error
        return cpu
    try:
        raw = json.loads(stdout or "")
        if not isinstance(raw, Mapping):
            raise TypeError("lscpu --json output is not an object")
        cpu["lscpu"]["raw"] = dict(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        cpu["lscpu"]["error"] = _exception_text(exc)
    return cpu


def _collect_memory_environment() -> dict[str, Any]:
    memory: dict[str, Any] = {"total_bytes": None, "errors": {}}
    _probe(memory, "total_bytes", _memory_total_bytes)
    return memory


def _collect_storage_environment() -> dict[str, Any]:
    storage: dict[str, Any] = {
        "path": str(REPO_ROOT),
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
        "errors": {},
    }
    try:
        usage = shutil.disk_usage(REPO_ROOT)
        storage["total_bytes"] = int(usage.total)
        storage["used_bytes"] = int(usage.used)
        storage["free_bytes"] = int(usage.free)
    except OSError as exc:
        storage["errors"]["repository_filesystem"] = _exception_text(exc)
    return storage


def _collect_torch_environment() -> dict[str, Any]:
    torch_environment: dict[str, Any] = {
        "module_available": False,
        "version": None,
        "compiled_cuda_version": None,
        "cudnn_version": None,
        "cuda_available": None,
        "visible_device_count": None,
        "visible_devices": [],
        "errors": {},
    }
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        torch_environment["errors"]["import"] = _exception_text(exc)
        return torch_environment

    torch_environment["module_available"] = True
    _probe(torch_environment, "version", lambda: str(torch.__version__))
    _probe(
        torch_environment,
        "compiled_cuda_version",
        lambda: getattr(torch.version, "cuda", None),
    )
    _probe(
        torch_environment,
        "cudnn_version",
        lambda: torch.backends.cudnn.version(),
    )
    _probe(torch_environment, "cuda_available", lambda: bool(torch.cuda.is_available()))
    _probe(torch_environment, "visible_device_count", lambda: int(torch.cuda.device_count()))

    count = torch_environment.get("visible_device_count")
    if not torch_environment.get("cuda_available") or not isinstance(count, int):
        return torch_environment
    for index in range(count):
        device: dict[str, Any] = {"index": index, "errors": {}}
        _probe(device, "name", lambda index=index: str(torch.cuda.get_device_name(index)))
        _probe(
            device,
            "compute_capability",
            lambda index=index: ".".join(
                str(item) for item in torch.cuda.get_device_capability(index)
            ),
        )
        _probe(
            device,
            "total_memory_bytes",
            lambda index=index: int(
                torch.cuda.get_device_properties(index).total_memory
            ),
        )
        torch_environment["visible_devices"].append(device)
    return torch_environment


def _collect_gpu_environment() -> dict[str, Any]:
    nvidia_smi: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_version": None,
        "error": None,
        "query_error": None,
        "row_errors": [],
    }
    environment: dict[str, Any] = {
        "nvidia_smi": nvidia_smi,
        "devices": [],
        "torch": _collect_torch_environment(),
    }
    header, header_error = _run_command(("nvidia-smi",))
    if header_error is not None:
        nvidia_smi["error"] = header_error
        return environment

    nvidia_smi["available"] = True
    nvidia_smi["version"] = _header_component(header or "", "NVIDIA-SMI ")
    nvidia_smi["cuda_version"] = _header_component(header or "", "CUDA Version:")
    query_fields = ",".join(source for source, _ in GPU_QUERY_FIELDS)
    stdout, query_error = _run_command(
        ("nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits")
    )
    if query_error is not None:
        nvidia_smi["query_error"] = query_error
        return environment

    for row_index, row in enumerate(csv.reader((stdout or "").splitlines())):
        if len(row) != len(GPU_QUERY_FIELDS):
            nvidia_smi["row_errors"].append(
                f"row {row_index}: expected {len(GPU_QUERY_FIELDS)} columns; got {len(row)} columns"
            )
            continue
        device = {
            field: _normalized_nvidia_value(value)
            for (_, field), value in zip(GPU_QUERY_FIELDS, row, strict=True)
        }
        environment["devices"].append(device)
    return environment


def _collect_python_environment() -> dict[str, Any]:
    python_environment: dict[str, Any] = {
        "implementation": None,
        "version": None,
        "version_info": None,
        "abi_flags": None,
        "soabi": None,
        "cache_tag": None,
        "compiler": None,
        "build": None,
        "executable": None,
        "prefix": None,
        "base_prefix": None,
        "virtual_environment": None,
        "errors": {},
    }
    probes = (
        ("implementation", platform.python_implementation),
        ("version", lambda: sys.version),
        ("version_info", lambda: list(sys.version_info[:3])),
        ("abi_flags", lambda: getattr(sys, "abiflags", "")),
        ("soabi", lambda: sysconfig.get_config_var("SOABI")),
        ("cache_tag", lambda: getattr(sys.implementation, "cache_tag", None)),
        ("compiler", platform.python_compiler),
        ("build", lambda: list(platform.python_build())),
        ("executable", lambda: sys.executable),
        ("prefix", lambda: sys.prefix),
        ("base_prefix", lambda: sys.base_prefix),
        ("virtual_environment", lambda: os.environ.get("VIRTUAL_ENV")),
    )
    for field, callback in probes:
        _probe(python_environment, field, callback)
    return python_environment


def _collect_tool(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    snapshot: dict[str, Any] = {"path": path, "version": None, "error": None}
    if path is None:
        snapshot["error"] = f"{name} is not on PATH"
        return snapshot
    stdout, error = _run_command((name, "--version"))
    if error is not None:
        snapshot["error"] = error
    else:
        snapshot["version"] = (stdout or "").strip()
    return snapshot


def _collect_tools_environment() -> dict[str, Any]:
    return {"uv": _collect_tool("uv"), "git": _collect_tool("git")}


def _collect_packages_environment() -> dict[str, Any]:
    packages: dict[str, Any] = {
        "distribution_count": 0,
        "distributions": [],
        "errors": {},
    }
    try:
        distributions = list(importlib_metadata.distributions())
    except Exception as exc:
        packages["errors"]["distributions"] = _exception_text(exc)
        return packages

    for index, distribution in enumerate(distributions):
        try:
            name = distribution.metadata.get("Name") or distribution.name
            packages["distributions"].append(
                {"name": str(name), "version": str(distribution.version)}
            )
        except Exception as exc:
            packages["errors"][f"distribution_{index}"] = _exception_text(exc)
    packages["distributions"].sort(
        key=lambda item: (item["name"].casefold(), item["version"])
    )
    packages["distribution_count"] = len(packages["distributions"])
    return packages


def _allowlisted_environment(keys: Sequence[str]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def collect_debug_environment() -> dict[str, Any]:
    """Collect allowlisted debug provenance; an individual probe failure never interrupts the report."""
    snapshot: dict[str, Any] = {
        "schema_version": DEBUG_ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": {},
    }
    collectors = {
        "host": _collect_host_environment,
        "operating_system": _collect_operating_system,
        "cpu": _collect_cpu_environment,
        "memory": _collect_memory_environment,
        "storage": _collect_storage_environment,
        "gpu": _collect_gpu_environment,
        "python": _collect_python_environment,
        "tools": _collect_tools_environment,
        "runtime_environment": lambda: _allowlisted_environment(
            RUNTIME_ENVIRONMENT_KEYS
        ),
        "scheduler_environment": lambda: _allowlisted_environment(
            SLURM_ALLOCATION_KEYS
        ),
        "packages": _collect_packages_environment,
    }
    for name, collector in collectors.items():
        try:
            snapshot[name] = collector()
        except Exception as exc:
            message = _exception_text(exc)
            snapshot[name] = {"errors": {"collection": message}}
            snapshot["errors"][name] = message
    return snapshot


def _fallback_debug_environment(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": DEBUG_ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": {"collect_debug_environment": _exception_text(error)},
    }


def _markdown_value(value: Any, missing: str = "Not set") -> str:
    if value is None:
        return missing
    if isinstance(value, (list, tuple)):
        return ", ".join(_markdown_value(item) for item in value)
    return " ".join(str(value).splitlines()).replace("|", "&#124;")


def _human_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return _markdown_value(value)
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if amount < 1024 or unit == "PiB":
            return f"{int(value)} bytes ({amount:.2f} {unit})"
        amount /= 1024
    return str(value)


def _lscpu_value(cpu: Mapping[str, Any], field_name: str) -> Any:
    lscpu = _mapping(cpu.get("lscpu"))
    raw = _mapping(lscpu.get("raw"))
    rows = raw.get("lscpu", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        field = str(row.get("field", "")).strip().rstrip(":")
        if field == field_name:
            return row.get("data")
    return None


def _error_items(prefix: str, value: Any) -> list[str]:
    if isinstance(value, Mapping):
        items: list[str] = []
        for key, nested in value.items():
            items.extend(_error_items(f"{prefix}/{key}", nested))
        return items
    if isinstance(value, list):
        return [
            f"{prefix}: {_markdown_value(item, 'Unknown error')}" for item in value
        ]
    return [f"{prefix}: {_markdown_value(value, 'Unknown error')}"]


def _debug_probe_errors(debug_environment: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_error_items("overall collection", debug_environment.get("errors", {})))
    for section_name in (
        "host",
        "operating_system",
        "cpu",
        "memory",
        "storage",
        "python",
        "tools",
        "packages",
    ):
        section = _mapping(debug_environment.get(section_name))
        errors.extend(_error_items(section_name, section.get("errors", {})))
    cpu = _mapping(debug_environment.get("cpu"))
    lscpu = _mapping(cpu.get("lscpu"))
    if lscpu.get("error"):
        errors.append(f"lscpu: {_markdown_value(lscpu.get('error'))}")
    gpu = _mapping(debug_environment.get("gpu"))
    nvidia_smi = _mapping(gpu.get("nvidia_smi"))
    if nvidia_smi.get("error"):
        errors.append(f"nvidia-smi: {_markdown_value(nvidia_smi.get('error'))}")
    if nvidia_smi.get("query_error"):
        errors.append(
            f"nvidia-smi query: {_markdown_value(nvidia_smi.get('query_error'))}"
        )
    errors.extend(_error_items("nvidia-smi row", nvidia_smi.get("row_errors", [])))
    torch = _mapping(gpu.get("torch"))
    errors.extend(_error_items("Torch", torch.get("errors", {})))
    return errors


def _render_debug_environment(debug_environment: Mapping[str, Any]) -> list[str]:
    """Display complete hardware fields and a core software summary in Markdown."""
    lines = ["## Debug Environment", ""]
    if not debug_environment:
        lines.extend(("- No debug environment snapshot was available.",))
        return lines

    host = _mapping(debug_environment.get("host"))
    operating_system = _mapping(debug_environment.get("operating_system"))
    libc = _mapping(operating_system.get("libc"))
    cpu = _mapping(debug_environment.get("cpu"))
    memory = _mapping(debug_environment.get("memory"))
    storage = _mapping(debug_environment.get("storage"))
    gpu = _mapping(debug_environment.get("gpu"))
    nvidia_smi = _mapping(gpu.get("nvidia_smi"))
    torch = _mapping(gpu.get("torch"))
    python_environment = _mapping(debug_environment.get("python"))
    tools = _mapping(debug_environment.get("tools"))
    uv = _mapping(tools.get("uv"))
    git = _mapping(tools.get("git"))
    packages = _mapping(debug_environment.get("packages"))

    lines.extend(
        (
            f"- Snapshot time (UTC): {_markdown_value(debug_environment.get('captured_at_utc'))}",
            f"- Host: {_markdown_value(host.get('hostname'))}",
            f"- OS: {_markdown_value(operating_system.get('system'))} "
            f"{_markdown_value(operating_system.get('release'))}",
            f"- Kernel: {_markdown_value(operating_system.get('kernel'))}",
            f"- Architecture: {_markdown_value(host.get('architecture'))}; "
            f"Machine: {_markdown_value(host.get('machine'))}",
            f"- libc: {_markdown_value(libc.get('library'))} "
            f"{_markdown_value(libc.get('version'))}",
            "",
            "### CPU, Memory, and Storage",
            "",
            "| Item | Value |",
            "| --- | --- |",
            f"| OS CPU count | {_markdown_value(cpu.get('os_cpu_count'))} |",
            f"| CPU affinity count | {_markdown_value(cpu.get('affinity_cpu_count'))} |",
            f"| CPU affinity IDs | {_markdown_value(cpu.get('affinity_cpu_ids'))} |",
            f"| CPU model | {_markdown_value(_lscpu_value(cpu, 'Model name'))} |",
            f"| CPU sockets | {_markdown_value(_lscpu_value(cpu, 'Socket(s)'))} |",
            f"| Threads per core | {_markdown_value(_lscpu_value(cpu, 'Thread(s) per core'))} |",
            f"| Total memory | {_human_bytes(memory.get('total_bytes'))} |",
            f"| Repository filesystem | {_markdown_value(storage.get('path'))} |",
            f"| Filesystem capacity | {_human_bytes(storage.get('total_bytes'))} |",
            f"| Filesystem used | {_human_bytes(storage.get('used_bytes'))} |",
            f"| Filesystem available | {_human_bytes(storage.get('free_bytes'))} |",
            "- The complete lscpu --json raw fields are available in JSON at "
            "debug_environment.cpu.lscpu.raw.",
            "",
            "### GPU (nvidia-smi)",
            "",
            f"- nvidia-smi available: {_markdown_value(nvidia_smi.get('available'))}",
            f"- nvidia-smi version: {_markdown_value(nvidia_smi.get('version'))}",
            f"- nvidia-smi CUDA version: {_markdown_value(nvidia_smi.get('cuda_version'))}",
        )
    )
    devices = gpu.get("devices", [])
    if isinstance(devices, list) and devices:
        lines.extend(("", "| GPU | Category | Field | Value |", "| --- | --- | --- | --- |"))
        for device in devices:
            if not isinstance(device, Mapping):
                continue
            index = _markdown_value(device.get("index"))
            for _, field in GPU_QUERY_FIELDS:
                category = "Static" if field in GPU_STATIC_FIELDS else "Dynamic"
                lines.append(
                    f"| {index} | {category} | {field} | "
                    f"{_markdown_value(device.get(field))} |"
                )
    else:
        lines.extend(("", "- No nvidia-smi query rows were obtained for visible GPUs."))

    lines.extend(
        (
            "",
            "### Python, Tools, and Runtime",
            "",
            "| Item | Value |",
            "| --- | --- |",
            f"| Python implementation | {_markdown_value(python_environment.get('implementation'))} |",
            f"| Full Python version | {_markdown_value(python_environment.get('version'))} |",
            f"| Python ABI / cache tag | {_markdown_value(python_environment.get('soabi'))} / "
            f"{_markdown_value(python_environment.get('cache_tag'))} |",
            f"| Python executable | {_markdown_value(python_environment.get('executable'))} |",
            f"| venv / prefix | {_markdown_value(python_environment.get('virtual_environment'))} / "
            f"{_markdown_value(python_environment.get('prefix'))} |",
            f"| base prefix | {_markdown_value(python_environment.get('base_prefix'))} |",
            f"| uv | {_markdown_value(uv.get('version'))} ({_markdown_value(uv.get('path'))}) |",
            f"| git | {_markdown_value(git.get('version'))} ({_markdown_value(git.get('path'))}) |",
            f"| Torch | {_markdown_value(torch.get('version'))} |",
            f"| Torch compiled CUDA | {_markdown_value(torch.get('compiled_cuda_version'))} |",
            f"| cuDNN | {_markdown_value(torch.get('cudnn_version'))} |",
            f"| Torch CUDA available / visible count | {_markdown_value(torch.get('cuda_available'))} / "
            f"{_markdown_value(torch.get('visible_device_count'))} |",
        )
    )
    visible_devices = torch.get("visible_devices", [])
    if isinstance(visible_devices, list) and visible_devices:
        lines.extend(("", "| Torch CUDA index | Name | Compute capability | Total memory |", "| --- | --- | --- | --- |"))
        for device in visible_devices:
            if isinstance(device, Mapping):
                lines.append(
                    f"| {_markdown_value(device.get('index'))} | "
                    f"{_markdown_value(device.get('name'))} | "
                    f"{_markdown_value(device.get('compute_capability'))} | "
                    f"{_human_bytes(device.get('total_memory_bytes'))} |"
                )

    lines.extend(("", "### Restricted Runtime Environment", "", "| Category | Variable | Value |", "| --- | --- | --- |"))
    for category, values in (
        ("Runtime", _mapping(debug_environment.get("runtime_environment"))),
        ("Slurm allocation", _mapping(debug_environment.get("scheduler_environment"))),
    ):
        for key, value in values.items():
            lines.append(f"| {category} | {key} | {_markdown_value(value)} |")

    distributions = packages.get("distributions", [])
    package_versions: dict[str, str] = {}
    if isinstance(distributions, list):
        for distribution in distributions:
            if isinstance(distribution, Mapping):
                name = distribution.get("name")
                version = distribution.get("version")
                if isinstance(name, str) and isinstance(version, str):
                    package_versions.setdefault(
                        name.casefold().replace("_", "-").replace(".", "-"),
                        version,
                    )
    lines.extend(
        (
            "",
            "### Dependencies",
            "",
            f"- Total distributions: {_markdown_value(packages.get('distribution_count'), '0')}.",
            "- The complete distribution list is available in the same JSON at "
            "debug_environment.packages.distributions.",
            "",
            "| Core dependency | Version |",
            "| --- | --- |",
        )
    )
    for name in CORE_PACKAGE_NAMES:
        lines.append(f"| {name} | {_markdown_value(package_versions.get(name))} |")

    probe_errors = _debug_probe_errors(debug_environment)
    if probe_errors:
        lines.extend(("", "### Probe Errors", ""))
        lines.extend(f"- {message}" for message in probe_errors[:30])
    return lines


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revalidate existing No-Patch output and write the latest complete report to scripts/data-generation/reports/generation_report.json and generation_report.md")
    parser.add_argument("--output-dir", required=True, help="Existing generated-data output directory")
    parser.add_argument("--env", "--environment", default="all", help="all or comma-separated environment names")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES, help="Number of episodes to revalidate per environment starting at episode 0")
    parser.add_argument("--workers", type=int, help="Optional: record in report parameters")
    parser.add_argument("--gpus", default=GPU_ID, help="must be 0; record the fixed physical GPU used for generation")
    parser.add_argument("--prior-report", help="Optional: retain generation provenance from a prior generation report")
    parser.add_argument("--metadata-root", default=str(METADATA_ROOT), help="Current train metadata directory")
    parser.add_argument("--reference-root", default=str(REFERENCE_ROOT), help="Official HDF5 directory")
    parser.add_argument("--max-abs-diff", type=float, default=MAX_ABS_DIFF)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    tasks: list[str] = []
    episodes: list[int] = []
    report: dict[str, Any] | None = None
    try:
        if not 1 <= args.episodes <= MAX_EPISODES:
            raise ReportGenerationError(f"--episodes must be in 1..{MAX_EPISODES}")
        tasks = parse_tasks(args.env)
        episodes = list(range(args.episodes))
        prior_path = (
            Path(args.prior_report).expanduser().resolve()
            if args.prior_report
            else None
        )
        prior = _load_prior_report(prior_path) if prior_path else None
        report = build_revalidation_report(
            output,
            args.env,
            tasks,
            episodes,
            workers=args.workers,
            gpus=args.gpus,
            prior_report=prior,
            prior_report_path=str(prior_path) if prior_path else None,
            metadata_root=args.metadata_root,
            reference_root=args.reference_root,
            max_abs_diff=args.max_abs_diff,
        )
        paths = write_generation_report(output, report)
    except (
        DatasetContractError,
        JointActionComparisonError,
        ReportGenerationError,
        ValueError,
    ) as exc:
        parameters = {
            "output_dir": str(output),
            "env": args.env,
            "tasks": tasks,
            "episodes": args.episodes,
            "metadata_root": str(args.metadata_root),
            "reference_root": str(args.reference_root),
            "max_abs_diff": args.max_abs_diff,
        }
        try:
            report = new_generation_report(parameters)
            report["report_mode"] = "existing_output_revalidation"
            report["status"] = "failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if output.is_dir():
                paths = write_generation_report(output, report)
            else:
                paths = {}
        except Exception as report_error:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": f"{exc}; additionally failed to write the report: {report_error}",
                    },
                    ensure_ascii=False,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "report_paths": paths,
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_paths": paths,
                "max_abs_diff": report["validation"]["joint_action_comparison"][
                    "max_abs_diff"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

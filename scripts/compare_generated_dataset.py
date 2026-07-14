"""严格比较重新生成的 RoboMME HDF5 与官方参考数据。

比较范围只包含 ``--episodes`` 和 ``--tasks`` 指定的对象；范围外的合法官方
任务/episode 会被记录但不会被判为差异，而生成侧的额外 episode 会被拒绝，
避免旧输出冒充本次生成。所有 dataset 内容都按有限大小的 hyperslab 分块读取，
因此不会为了比较图像等大数组而一次性载入内存。

默认只接受严格一致。显式启用 ``--allow-joint-action-allclose`` 时，报告仍保留
``strict_equal=false``，但可单独将仅发生在 ``action/joint_action`` 且通过
allclose 的浮点差异标记为 ``accepted=true``，不会混淆两种结论。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_TASK_IDS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
DEFAULT_EPISODES = tuple(range(9))
READ_BLOCK_BYTES = 8 * 1024 * 1024
MAX_OBJECT_BLOCK_ELEMENTS = 65_536
DEFAULT_JOINT_ACTION_MAX_ABS_DIFF = 1e-12


class UnsafePathError(ValueError):
    """输入或输出路径违反仓库边界/符号链接约束。"""


class UnsafeHDF5LinkError(ValueError):
    """HDF5 内含会跳转到其他位置的软链接或外部链接。"""


class _JsonlWriter:
    """逐条写 JSONL，同时保留已写记录数。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self._file = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(
            json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n"
        )
        self.count += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> _JsonlWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _json_safe(value: Any) -> Any:
    """把 NumPy/HDF5 元数据转换成严格 JSON 可表示的数据。"""

    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return {"real": _json_safe(value.real), "imag": _json_safe(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _update_frame(digest: Any, label: str, payload: bytes) -> None:
    """以带长度的字段更新摘要，避免不同字段拼接产生歧义。"""

    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _new_digest(domain: str) -> Any:
    digest = hashlib.sha256()
    _update_frame(digest, "domain", domain.encode("utf-8"))
    return digest


def _dtype_info(dtype: np.dtype[Any]) -> dict[str, Any]:
    dtype = np.dtype(dtype)
    string_info = h5py.check_string_dtype(dtype)
    vlen_info = h5py.check_vlen_dtype(dtype)
    enum_info = h5py.check_enum_dtype(dtype)
    return {
        "display": str(dtype),
        "str": dtype.str,
        "descr": dtype.descr if dtype.fields is not None else None,
        "string": (
            {"encoding": string_info.encoding, "length": string_info.length}
            if string_info is not None
            else None
        ),
        "vlen": str(vlen_info) if vlen_info is not None else None,
        "enum": enum_info,
    }


def _canonical_scalar_bytes(value: Any) -> bytes:
    """编码 object/vlen/compound 数组的单个元素。"""

    if isinstance(value, np.generic):
        if isinstance(value, np.void) and value.dtype.fields is not None:
            fields = []
            for name in value.dtype.names or ():
                fields.append((name, _canonical_scalar_bytes(value[name]).hex()))
            return b"structured:" + _canonical_json(fields)
        return _canonical_scalar_bytes(value.item())
    if isinstance(value, np.ndarray):
        digest = _new_digest("canonical-array-value-v1")
        _update_frame(digest, "dtype", _canonical_json(_dtype_info(value.dtype)))
        _update_frame(digest, "shape", _canonical_json(value.shape))
        _update_array_values(digest, value)
        return b"array:" + digest.digest()
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if value is None:
        return b"none"
    if isinstance(value, (bool, np.bool_)):
        return b"bool:1" if bool(value) else b"bool:0"
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        if math.isnan(value):
            return b"float:nan"
        if value == 0:
            return b"float:0"
        return b"float:" + value.hex().encode("ascii")
    if isinstance(value, complex):
        return (
            b"complex:"
            + _canonical_scalar_bytes(float(value.real))
            + b":"
            + _canonical_scalar_bytes(float(value.imag))
        )
    if isinstance(value, (list, tuple)):
        digest = _new_digest("canonical-sequence-v1")
        for item in value:
            _update_frame(digest, "item", _canonical_scalar_bytes(item))
        return b"sequence:" + digest.digest()
    return b"repr:" + repr(value).encode("utf-8")


def _update_array_values(digest: Any, array: np.ndarray[Any, Any]) -> None:
    """按元素顺序加入规范值；统一 NaN payload 与正负零。"""

    array = np.asarray(array)
    if array.dtype.hasobject or array.dtype.fields is not None:
        for value in array.reshape(-1):
            _update_frame(digest, "element", _canonical_scalar_bytes(value))
        return

    canonical = np.ascontiguousarray(array)
    if canonical.dtype.kind in "fc":
        canonical = canonical.copy()
        if canonical.dtype.kind == "f":
            canonical[np.isnan(canonical)] = np.nan
            canonical[canonical == 0] = 0
        else:
            real = canonical.real
            imag = canonical.imag
            real[np.isnan(real)] = np.nan
            imag[np.isnan(imag)] = np.nan
            real[real == 0] = 0
            imag[imag == 0] = 0
    digest.update(canonical.tobytes(order="C"))


def _iter_slices(
    shape: tuple[int, ...] | None, dtype: np.dtype[Any]
) -> Iterable[tuple[slice, ...] | tuple[()]]:
    """生成不超过目标字节数的 C 顺序 hyperslab。"""

    if shape is None:
        return
    if not shape:
        yield ()
        return
    if any(size == 0 for size in shape):
        return

    itemsize = max(int(np.dtype(dtype).itemsize), 1)
    budget = max(1, READ_BLOCK_BYTES // itemsize)
    if np.dtype(dtype).hasobject:
        budget = min(budget, MAX_OBJECT_BLOCK_ELEMENTS)

    block_shape = [1] * len(shape)
    remaining = budget
    for axis in range(len(shape) - 1, -1, -1):
        block_shape[axis] = min(shape[axis], max(1, remaining))
        remaining = max(1, remaining // block_shape[axis])

    ranges = [range(0, size, block) for size, block in zip(shape, block_shape)]
    for starts in itertools.product(*ranges):
        yield tuple(
            slice(start, min(start + block, size))
            for start, block, size in zip(starts, block_shape, shape)
        )


def _normalized_path(path: str | Path, workspace_root: Path) -> Path:
    raw = Path(path).expanduser()
    lexical = raw if raw.is_absolute() else workspace_root / raw
    return Path(os.path.abspath(lexical))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path, workspace_root: Path) -> None:
    relative = path.relative_to(workspace_root)
    current = workspace_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise UnsafePathError(f"路径包含符号链接：{current}")


def _reject_nested_symlinks(path: Path) -> None:
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        root = Path(directory)
        for name in [*directory_names, *file_names]:
            candidate = root / name
            if candidate.is_symlink():
                raise UnsafePathError(f"目录树包含符号链接：{candidate}")


def _safe_workspace_root(workspace_root: str | Path) -> tuple[Path, Path]:
    lexical = Path(os.path.abspath(Path(workspace_root).expanduser()))
    if lexical.is_symlink():
        raise UnsafePathError(f"workspace-root 不能是符号链接：{lexical}")
    if not lexical.is_dir():
        raise UnsafePathError(f"workspace-root 不存在或不是目录：{lexical}")
    return lexical, lexical.resolve(strict=True)


def _safe_path(
    path: str | Path,
    *,
    workspace_lexical: Path,
    workspace_resolved: Path,
    must_exist: bool,
    label: str,
) -> Path:
    lexical = _normalized_path(path, workspace_lexical)
    if not _is_relative_to(lexical, workspace_lexical):
        raise UnsafePathError(f"{label} 位于仓库外：{lexical}")
    _reject_symlink_components(lexical, workspace_lexical)
    if must_exist and not lexical.is_dir():
        raise UnsafePathError(f"{label} 不存在或不是目录：{lexical}")

    resolved = lexical.resolve(strict=must_exist)
    if not _is_relative_to(resolved, workspace_resolved):
        raise UnsafePathError(f"{label} 解析后位于仓库外：{resolved}")
    if must_exist:
        _reject_nested_symlinks(lexical)
    elif lexical.exists() and not lexical.is_dir():
        raise UnsafePathError(f"{label} 已存在但不是目录：{lexical}")
    elif lexical.exists():
        _reject_nested_symlinks(lexical)
    return lexical


def _validate_paths(
    reference_dir: str | Path,
    generated_dir: str | Path,
    report_dir: str | Path,
    workspace_root: str | Path,
) -> tuple[Path, Path, Path, Path]:
    workspace_lexical, workspace_resolved = _safe_workspace_root(workspace_root)
    reference = _safe_path(
        reference_dir,
        workspace_lexical=workspace_lexical,
        workspace_resolved=workspace_resolved,
        must_exist=True,
        label="reference-dir",
    )
    generated = _safe_path(
        generated_dir,
        workspace_lexical=workspace_lexical,
        workspace_resolved=workspace_resolved,
        must_exist=True,
        label="generated-dir",
    )
    report = _safe_path(
        report_dir,
        workspace_lexical=workspace_lexical,
        workspace_resolved=workspace_resolved,
        must_exist=False,
        label="report-dir",
    )

    if _is_relative_to(generated, reference) or _is_relative_to(reference, generated):
        raise UnsafePathError("reference-dir 与 generated-dir 不能相同或互相包含")
    for data_dir, label in ((reference, "reference-dir"), (generated, "generated-dir")):
        if _is_relative_to(report, data_dir) or _is_relative_to(data_dir, report):
            raise UnsafePathError(f"report-dir 不能与 {label} 相同或互相包含")
    return workspace_lexical, reference, generated, report


def _episode_numbers(data: h5py.File) -> list[int]:
    episodes: list[int] = []
    for key in data.keys():
        if key.startswith("episode_") and key.removeprefix("episode_").isdigit():
            episodes.append(int(key.removeprefix("episode_")))
    return sorted(episodes)


def _collect_object_paths(
    root: h5py.File | h5py.Group, *, exclude_episode_roots: bool = False
) -> dict[str, str]:
    """收集硬链接对象；拒绝 HDF5 软链接和外部链接。"""

    objects = {"": "group"}
    visited_groups: set[int] = set()

    def visit(group: h5py.Group | h5py.File, prefix: str) -> None:
        group_token = hash(group.id)
        if group_token in visited_groups:
            return
        visited_groups.add(group_token)
        for name in group.keys():
            if not prefix and exclude_episode_roots:
                suffix = name.removeprefix("episode_")
                if name.startswith("episode_") and suffix.isdigit():
                    continue
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                full_name = f"{group.name.rstrip('/')}/{name}"
                raise UnsafeHDF5LinkError(f"拒绝 HDF5 软链接/外部链接：{full_name}")
            obj = group[name]
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(obj, h5py.Group):
                objects[path] = "group"
                visit(obj, path)
            elif isinstance(obj, h5py.Dataset):
                objects[path] = "dataset"
            else:
                raise TypeError(f"不支持的 HDF5 对象类型：{obj.name} ({type(obj)})")

    visit(root, "")
    return objects


def _attribute_snapshot(owner: h5py.Group | h5py.Dataset, name: str) -> dict[str, Any]:
    attr_id = owner.attrs.get_id(name)
    value = owner.attrs[name]
    metadata = {
        "dtype": _dtype_info(attr_id.dtype),
        "shape": list(attr_id.shape),
    }
    digest = _new_digest("hdf5-attribute-v1")
    _update_frame(digest, "metadata", _canonical_json(metadata))
    _update_array_values(digest, np.asarray(value))
    return {
        "metadata": metadata,
        "value": np.asarray(value),
        "canonical_sha256": digest.hexdigest(),
    }


def _digest_attributes(parts: list[tuple[str, str]]) -> str:
    digest = _new_digest("hdf5-attributes-v1")
    for name, value_digest in sorted(parts):
        _update_frame(digest, "name", name.encode("utf-8"))
        _update_frame(digest, "value", bytes.fromhex(value_digest))
    return digest.hexdigest()


class _Metrics:
    """跨多个数据块累计严格相等、allclose 与误差统计。"""

    def __init__(self, reference_dtype: np.dtype[Any], generated_dtype: np.dtype[Any]) -> None:
        self.numeric = bool(
            np.issubdtype(reference_dtype, np.inexact)
            and np.issubdtype(generated_dtype, np.inexact)
        )
        self.strict_equal = True
        self.allclose = True
        self.nan_positions_equal = True
        self.diff_count = 0
        self.element_count = 0
        self.max_abs_diff: float | None = 0.0 if self.numeric else None
        self.max_rel_diff: float | None = 0.0 if self.numeric else None

    def update(self, reference: np.ndarray[Any, Any], generated: np.ndarray[Any, Any], *, rtol: float, atol: float) -> None:
        reference = np.asarray(reference)
        generated = np.asarray(generated)
        count = int(reference.size)
        self.element_count += count
        if count == 0:
            return

        if self.numeric:
            with np.errstate(invalid="ignore", over="ignore"):
                equal = np.equal(reference, generated)
                reference_nan = np.isnan(reference)
                generated_nan = np.isnan(generated)
                equal = np.asarray(equal) | (reference_nan & generated_nan)
                close = np.isclose(reference, generated, rtol=rtol, atol=atol, equal_nan=True)
            self.nan_positions_equal &= bool(np.array_equal(reference_nan, generated_nan))
            self.strict_equal &= bool(np.all(equal))
            self.allclose &= bool(np.all(close))
            self.diff_count += count - int(np.count_nonzero(equal))
            self._update_numeric_error(reference, generated, equal)
            return

        try:
            equal = np.asarray(np.equal(reference, generated), dtype=bool)
        except (TypeError, ValueError):
            equal = np.fromiter(
                (
                    _canonical_scalar_bytes(left) == _canonical_scalar_bytes(right)
                    for left, right in zip(reference.reshape(-1), generated.reshape(-1))
                ),
                dtype=bool,
                count=count,
            ).reshape(reference.shape)
        self.strict_equal &= bool(np.all(equal))
        self.allclose &= bool(np.all(equal))
        self.diff_count += count - int(np.count_nonzero(equal))

    def _update_numeric_error(
        self,
        reference: np.ndarray[Any, Any],
        generated: np.ndarray[Any, Any],
        equal: np.ndarray[Any, Any],
    ) -> None:
        calc_dtype = np.complex128 if (
            np.iscomplexobj(reference) or np.iscomplexobj(generated)
        ) else np.float64
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            left = reference.astype(calc_dtype, copy=False)
            right = generated.astype(calc_dtype, copy=False)
            finite_pair = np.isfinite(left) & np.isfinite(right)
            absolute = np.zeros(reference.shape, dtype=np.float64)
            absolute[finite_pair] = np.abs(left[finite_pair] - right[finite_pair])
            absolute[(~finite_pair) & (~equal)] = np.inf
            denominator = np.abs(left)
            relative = np.zeros(reference.shape, dtype=np.float64)
            nonzero = denominator != 0
            relative[nonzero] = absolute[nonzero] / denominator[nonzero]
            relative[(~nonzero) & (absolute != 0)] = np.inf
            relative[(~finite_pair) & (~equal)] = np.inf
        self.max_abs_diff = max(float(self.max_abs_diff or 0.0), float(np.max(absolute)))
        self.max_rel_diff = max(float(self.max_rel_diff or 0.0), float(np.max(relative)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "strict_equal": self.strict_equal,
            "allclose": self.allclose,
            "nan_positions_equal": self.nan_positions_equal,
            "diff_count": self.diff_count,
            "element_count": self.element_count,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
        }


def _shape_mismatch_metrics() -> dict[str, Any]:
    return {
        "strict_equal": False,
        "allclose": False,
        "nan_positions_equal": None,
        "diff_count": None,
        "element_count": 0,
        "max_abs_diff": None,
        "max_rel_diff": None,
    }


def _compare_loaded_values(
    reference: np.ndarray[Any, Any],
    generated: np.ndarray[Any, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if reference.shape != generated.shape:
        return _shape_mismatch_metrics()
    metrics = _Metrics(reference.dtype, generated.dtype)
    metrics.update(reference, generated, rtol=rtol, atol=atol)
    return metrics.as_dict()


def _compare_attributes(
    reference: h5py.Group | h5py.Dataset,
    generated: h5py.Group | h5py.Dataset,
    *,
    task_id: str,
    episode: int | None,
    display_path: str,
    emit_leaf: Callable[[dict[str, Any]], None],
    emit_difference: Callable[[dict[str, Any]], None],
    rtol: float,
    atol: float,
) -> tuple[str, str, bool, int, int]:
    reference_names = set(reference.attrs.keys())
    generated_names = set(generated.attrs.keys())
    reference_parts: list[tuple[str, str]] = []
    generated_parts: list[tuple[str, str]] = []
    strict_equal = True

    for name in sorted(reference_names | generated_names):
        attribute_path = f"{display_path}@{name}"
        reference_snapshot = (
            _attribute_snapshot(reference, name) if name in reference_names else None
        )
        generated_snapshot = (
            _attribute_snapshot(generated, name) if name in generated_names else None
        )
        if reference_snapshot is not None:
            reference_parts.append((name, reference_snapshot["canonical_sha256"]))
        if generated_snapshot is not None:
            generated_parts.append((name, generated_snapshot["canonical_sha256"]))

        if reference_snapshot is None or generated_snapshot is None:
            strict_equal = False
            record = {
                "task_id": task_id,
                "episode": episode,
                "kind": "attribute",
                "path": attribute_path,
                "strict_equal": False,
                "reference": (
                    None
                    if reference_snapshot is None
                    else {key: value for key, value in reference_snapshot.items() if key != "value"}
                ),
                "generated": (
                    None
                    if generated_snapshot is None
                    else {key: value for key, value in generated_snapshot.items() if key != "value"}
                ),
            }
            emit_leaf(record)
            emit_difference({**record, "category": "missing_attribute"})
            continue

        metadata_equal = reference_snapshot["metadata"] == generated_snapshot["metadata"]
        metrics = _compare_loaded_values(
            reference_snapshot["value"],
            generated_snapshot["value"],
            rtol=rtol,
            atol=atol,
        )
        attribute_equal = bool(metadata_equal and metrics["strict_equal"])
        strict_equal &= attribute_equal
        record = {
            "task_id": task_id,
            "episode": episode,
            "kind": "attribute",
            "path": attribute_path,
            "strict_equal": attribute_equal,
            "metadata_equal": metadata_equal,
            "metrics": metrics,
            "reference": {
                "metadata": reference_snapshot["metadata"],
                "canonical_sha256": reference_snapshot["canonical_sha256"],
            },
            "generated": {
                "metadata": generated_snapshot["metadata"],
                "canonical_sha256": generated_snapshot["canonical_sha256"],
            },
        }
        emit_leaf(record)
        if not attribute_equal:
            emit_difference({**record, "category": "attribute_mismatch"})

    return (
        _digest_attributes(reference_parts),
        _digest_attributes(generated_parts),
        strict_equal,
        len(reference_names),
        len(generated_names),
    )


def _digest_attributes_only(owner: h5py.Group | h5py.Dataset) -> tuple[str, int]:
    parts = []
    for name in sorted(owner.attrs.keys()):
        snapshot = _attribute_snapshot(owner, name)
        parts.append((name, snapshot["canonical_sha256"]))
    return _digest_attributes(parts), len(parts)


def _dataset_metadata(dataset: h5py.Dataset) -> dict[str, Any]:
    try:
        fillvalue: Any = dataset.fillvalue
    except (TypeError, ValueError):
        fillvalue = None
    return {
        "dtype": _dtype_info(dataset.dtype),
        "shape": list(dataset.shape) if dataset.shape is not None else None,
        "chunks": list(dataset.chunks) if dataset.chunks is not None else None,
        "compression": dataset.compression,
        "compression_opts": dataset.compression_opts,
        "shuffle": dataset.shuffle,
        "fletcher32": dataset.fletcher32,
        "scaleoffset": dataset.scaleoffset,
        "maxshape": list(dataset.maxshape) if dataset.maxshape is not None else None,
        "fillvalue": fillvalue,
    }


def _start_dataset_digest(metadata: dict[str, Any], attributes_digest: str) -> Any:
    digest = _new_digest("hdf5-dataset-v1")
    _update_frame(digest, "metadata", _canonical_json(metadata))
    _update_frame(digest, "attributes", bytes.fromhex(attributes_digest))
    return digest


def _consume_dataset(dataset: h5py.Dataset, digest: Any) -> None:
    for selection in _iter_slices(dataset.shape, dataset.dtype):
        _update_array_values(digest, np.asarray(dataset[selection]))


def _digest_dataset(dataset: h5py.Dataset) -> tuple[str, dict[str, Any], int]:
    metadata = _dataset_metadata(dataset)
    attributes_digest, attribute_count = _digest_attributes_only(dataset)
    digest = _start_dataset_digest(metadata, attributes_digest)
    _consume_dataset(dataset, digest)
    return digest.hexdigest(), metadata, attribute_count


def _compare_dataset(
    reference: h5py.Dataset,
    generated: h5py.Dataset,
    *,
    task_id: str,
    episode: int | None,
    display_path: str,
    emit_leaf: Callable[[dict[str, Any]], None],
    emit_difference: Callable[[dict[str, Any]], None],
    rtol: float,
    atol: float,
) -> tuple[str, str, bool, int, int]:
    reference_metadata = _dataset_metadata(reference)
    generated_metadata = _dataset_metadata(generated)
    (
        reference_attributes_digest,
        generated_attributes_digest,
        attributes_equal,
        reference_attribute_count,
        generated_attribute_count,
    ) = _compare_attributes(
        reference,
        generated,
        task_id=task_id,
        episode=episode,
        display_path=display_path,
        emit_leaf=emit_leaf,
        emit_difference=emit_difference,
        rtol=rtol,
        atol=atol,
    )
    reference_digest = _start_dataset_digest(
        reference_metadata, reference_attributes_digest
    )
    generated_digest = _start_dataset_digest(
        generated_metadata, generated_attributes_digest
    )

    if reference.shape != generated.shape:
        _consume_dataset(reference, reference_digest)
        _consume_dataset(generated, generated_digest)
        metrics = _shape_mismatch_metrics()
    else:
        metrics_accumulator = _Metrics(reference.dtype, generated.dtype)
        comparison_dtype = np.dtype(
            f"V{max(int(reference.dtype.itemsize), int(generated.dtype.itemsize), 1)}"
        )
        for selection in _iter_slices(reference.shape, comparison_dtype):
            reference_values = np.asarray(reference[selection])
            generated_values = np.asarray(generated[selection])
            _update_array_values(reference_digest, reference_values)
            _update_array_values(generated_digest, generated_values)
            metrics_accumulator.update(
                reference_values, generated_values, rtol=rtol, atol=atol
            )
        metrics = metrics_accumulator.as_dict()

    reference_sha = reference_digest.hexdigest()
    generated_sha = generated_digest.hexdigest()
    metadata_equal = reference_metadata == generated_metadata
    strict_equal = bool(metadata_equal and attributes_equal and metrics["strict_equal"])
    record = {
        "task_id": task_id,
        "episode": episode,
        "kind": "dataset",
        "path": display_path,
        "strict_equal": strict_equal,
        "metadata_equal": metadata_equal,
        "attributes_equal": attributes_equal,
        "metrics": metrics,
        "reference": {
            "metadata": reference_metadata,
            "canonical_sha256": reference_sha,
        },
        "generated": {
            "metadata": generated_metadata,
            "canonical_sha256": generated_sha,
        },
        "canonical_sha256_equal": reference_sha == generated_sha,
    }
    emit_leaf(record)
    if not metadata_equal:
        for field in reference_metadata.keys():
            if reference_metadata[field] != generated_metadata[field]:
                emit_difference(
                    {
                        "task_id": task_id,
                        "episode": episode,
                        "kind": "dataset",
                        "path": display_path,
                        "category": "dataset_metadata",
                        "field": field,
                        "reference": reference_metadata[field],
                        "generated": generated_metadata[field],
                    }
                )
    if not metrics["strict_equal"]:
        emit_difference({**record, "category": "dataset_content"})
    return (
        reference_sha,
        generated_sha,
        strict_equal,
        reference_attribute_count,
        generated_attribute_count,
    )


def _group_digest(group: h5py.Group | h5py.File) -> tuple[str, int]:
    attributes_digest, attribute_count = _digest_attributes_only(group)
    digest = _new_digest("hdf5-group-v1")
    _update_frame(digest, "attributes", bytes.fromhex(attributes_digest))
    return digest.hexdigest(), attribute_count


def _display_path(scope: str, relative_path: str) -> str:
    if not relative_path:
        return f"/{scope}" if scope else "/"
    return f"/{scope}/{relative_path}" if scope else f"/{relative_path}"


def _digest_existing_object(
    obj: h5py.Group | h5py.Dataset,
) -> tuple[str, int]:
    if isinstance(obj, h5py.Dataset):
        digest, _, attribute_count = _digest_dataset(obj)
        return digest, attribute_count
    return _group_digest(obj)


def _scope_digest(parts: list[tuple[str, str, str]]) -> str:
    digest = _new_digest("hdf5-scope-v1")
    for path, kind, object_digest in sorted(parts):
        _update_frame(digest, "path", path.encode("utf-8"))
        _update_frame(digest, "kind", kind.encode("ascii"))
        _update_frame(digest, "object", bytes.fromhex(object_digest))
    return digest.hexdigest()


def _compare_scope(
    reference_root: h5py.File | h5py.Group,
    generated_root: h5py.File | h5py.Group,
    *,
    task_id: str,
    episode: int | None,
    scope: str,
    exclude_episode_roots: bool,
    emit_leaf: Callable[[dict[str, Any]], None],
    emit_difference: Callable[[dict[str, Any]], None],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_objects = _collect_object_paths(
        reference_root, exclude_episode_roots=exclude_episode_roots
    )
    generated_objects = _collect_object_paths(
        generated_root, exclude_episode_roots=exclude_episode_roots
    )
    reference_parts: list[tuple[str, str, str]] = []
    generated_parts: list[tuple[str, str, str]] = []
    reference_attribute_count = 0
    generated_attribute_count = 0
    strict_equal = True
    dataset_count = 0
    group_count = 0

    for path in sorted(set(reference_objects) | set(generated_objects)):
        reference_kind = reference_objects.get(path)
        generated_kind = generated_objects.get(path)
        display_path = _display_path(scope, path)
        reference_obj = (
            reference_root[path]
            if path and reference_kind is not None
            else reference_root
        )
        generated_obj = (
            generated_root[path]
            if path and generated_kind is not None
            else generated_root
        )

        if reference_kind is None or generated_kind is None or reference_kind != generated_kind:
            strict_equal = False
            reference_digest = None
            generated_digest = None
            if reference_kind is not None:
                reference_digest, count = _digest_existing_object(reference_obj)
                reference_attribute_count += count
                reference_parts.append((path, reference_kind, reference_digest))
            if generated_kind is not None:
                generated_digest, count = _digest_existing_object(generated_obj)
                generated_attribute_count += count
                generated_parts.append((path, generated_kind, generated_digest))
            record = {
                "task_id": task_id,
                "episode": episode,
                "path": display_path,
                "kind": reference_kind or generated_kind,
                "category": "missing_object" if None in (reference_kind, generated_kind) else "object_kind",
                "reference": (
                    None
                    if reference_kind is None
                    else {"kind": reference_kind, "canonical_sha256": reference_digest}
                ),
                "generated": (
                    None
                    if generated_kind is None
                    else {"kind": generated_kind, "canonical_sha256": generated_digest}
                ),
                "strict_equal": False,
            }
            emit_leaf(record)
            emit_difference(record)
            continue

        if reference_kind == "dataset":
            dataset_count += 1
            (
                reference_digest,
                generated_digest,
                object_equal,
                reference_attrs,
                generated_attrs,
            ) = _compare_dataset(
                reference_obj,
                generated_obj,
                task_id=task_id,
                episode=episode,
                display_path=display_path,
                emit_leaf=emit_leaf,
                emit_difference=emit_difference,
                rtol=rtol,
                atol=atol,
            )
        else:
            group_count += 1
            (
                reference_attributes_digest,
                generated_attributes_digest,
                object_equal,
                reference_attrs,
                generated_attrs,
            ) = _compare_attributes(
                reference_obj,
                generated_obj,
                task_id=task_id,
                episode=episode,
                display_path=display_path,
                emit_leaf=emit_leaf,
                emit_difference=emit_difference,
                rtol=rtol,
                atol=atol,
            )
            reference_group_digest = _new_digest("hdf5-group-v1")
            generated_group_digest = _new_digest("hdf5-group-v1")
            _update_frame(
                reference_group_digest,
                "attributes",
                bytes.fromhex(reference_attributes_digest),
            )
            _update_frame(
                generated_group_digest,
                "attributes",
                bytes.fromhex(generated_attributes_digest),
            )
            reference_digest = reference_group_digest.hexdigest()
            generated_digest = generated_group_digest.hexdigest()
        reference_attribute_count += reference_attrs
        generated_attribute_count += generated_attrs
        reference_parts.append((path, reference_kind, reference_digest))
        generated_parts.append((path, generated_kind, generated_digest))
        strict_equal &= object_equal

    reference_sha = _scope_digest(reference_parts)
    generated_sha = _scope_digest(generated_parts)
    strict_equal &= reference_sha == generated_sha
    return {
        "strict_equal": strict_equal,
        "reference_canonical_sha256": reference_sha,
        "generated_canonical_sha256": generated_sha,
        "canonical_sha256_equal": reference_sha == generated_sha,
        "reference_object_count": len(reference_objects),
        "generated_object_count": len(generated_objects),
        "common_dataset_count": dataset_count,
        "common_group_count": group_count,
        "reference_attribute_count": reference_attribute_count,
        "generated_attribute_count": generated_attribute_count,
    }


def _missing_scope_digest() -> str:
    digest = _new_digest("hdf5-missing-scope-v1")
    return digest.hexdigest()


def _episode_root_digest(obj: h5py.Group | h5py.Dataset) -> str:
    """计算 episode 根对象及其后代的 scope digest。"""

    if isinstance(obj, h5py.Dataset):
        object_digest, _ = _digest_existing_object(obj)
        return _scope_digest([("", "dataset", object_digest)])
    return _scope_digest(
        [
            (
                path,
                kind,
                _digest_existing_object(obj[path] if path else obj)[0],
            )
            for path, kind in _collect_object_paths(obj).items()
        ]
    )


def _task_digest(shared_digest: str, episode_digests: list[tuple[int, str]]) -> str:
    digest = _new_digest("robomme-selected-task-v1")
    _update_frame(digest, "shared", bytes.fromhex(shared_digest))
    for episode, episode_digest in episode_digests:
        _update_frame(digest, "episode", str(episode).encode("ascii"))
        _update_frame(digest, "scope", bytes.fromhex(episode_digest))
    return digest.hexdigest()


def _validate_episodes(episodes: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(episode) for episode in episodes)))
    if not normalized:
        raise ValueError("episodes 不能为空")
    if any(episode < 0 for episode in normalized):
        raise ValueError(f"episodes 必须为非负整数：{normalized}")
    return normalized


def _file_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.h5") if path.is_file()}


def compare_generated_dataset(
    reference_dir: str | Path,
    generated_dir: str | Path,
    report_dir: str | Path,
    *,
    episodes: Sequence[int] = DEFAULT_EPISODES,
    workspace_root: str | Path = ".",
    task_ids: Sequence[str] = EXPECTED_TASK_IDS,
    rtol: float = 1e-7,
    atol: float = 0.0,
    allow_joint_action_allclose: bool = False,
    joint_action_max_abs_diff: float = DEFAULT_JOINT_ACTION_MAX_ABS_DIFF,
) -> dict[str, Any]:
    """比较选定 episode，写出摘要、完整叶子记录和差异记录。"""

    tolerances = (rtol, atol, joint_action_max_abs_diff)
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("rtol、atol 与 joint_action_max_abs_diff 必须为有限非负数")
    selected_episodes = _validate_episodes(episodes)
    normalized_task_ids = tuple(task_ids)
    if not normalized_task_ids:
        raise ValueError("task_ids 不能为空")
    if len(normalized_task_ids) != len(set(normalized_task_ids)):
        raise ValueError("task_ids 不能重复")
    unknown_task_ids = sorted(set(normalized_task_ids) - set(EXPECTED_TASK_IDS))
    if unknown_task_ids:
        raise ValueError(f"未知 task_ids：{unknown_task_ids}")
    workspace, reference, generated, report = _validate_paths(
        reference_dir, generated_dir, report_dir, workspace_root
    )
    report.mkdir(parents=True, exist_ok=True)

    differences_path = report / "differences.jsonl"
    leaves_path = report / "leaf_comparisons.jsonl"
    comparison_path = report / "comparison.json"
    expected_names = {
        f"record_dataset_{task_id}.h5" for task_id in normalized_task_ids
    }
    known_names = {
        f"record_dataset_{task_id}.h5" for task_id in EXPECTED_TASK_IDS
    }
    reference_names = _file_names(reference)
    generated_names = _file_names(generated)
    inventory = {
        "expected": sorted(expected_names),
        "reference": sorted(reference_names),
        "generated": sorted(generated_names),
        "reference_missing": sorted(expected_names - reference_names),
        "reference_unexpected": sorted(reference_names - known_names),
        "reference_out_of_scope": sorted((reference_names & known_names) - expected_names),
        "generated_missing": sorted(expected_names - generated_names),
        "generated_unexpected": sorted(generated_names - known_names),
        "generated_out_of_scope": sorted((generated_names & known_names) - expected_names),
    }

    task_reports: list[dict[str, Any]] = []
    with _JsonlWriter(differences_path) as differences, _JsonlWriter(
        leaves_path
    ) as leaves:
        for side in ("reference", "generated"):
            for name in inventory[f"{side}_missing"]:
                differences.write(
                    {
                        "category": "missing_file",
                        "side": side,
                        "path": name,
                    }
                )
            for name in inventory[f"{side}_unexpected"]:
                differences.write(
                    {
                        "category": "unexpected_file",
                        "side": side,
                        "path": name,
                    }
                )

        for task_id in normalized_task_ids:
            file_name = f"record_dataset_{task_id}.h5"
            reference_path = reference / file_name
            generated_path = generated / file_name
            task_report: dict[str, Any] = {
                "task_id": task_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "reference_exists": reference_path.is_file(),
                "generated_exists": generated_path.is_file(),
                "episodes": [],
                "strict_equal": False,
            }
            if not reference_path.is_file() or not generated_path.is_file():
                task_reports.append(task_report)
                continue

            task_report["reference_bytes"] = reference_path.stat().st_size
            task_report["generated_bytes"] = generated_path.stat().st_size
            with h5py.File(reference_path, "r") as reference_h5, h5py.File(
                generated_path, "r"
            ) as generated_h5:
                reference_available = _episode_numbers(reference_h5)
                generated_available = _episode_numbers(generated_h5)
                task_report["reference_available_episodes"] = reference_available
                task_report["generated_available_episodes"] = generated_available
                task_report["reference_extra_episodes_ignored"] = sorted(
                    set(reference_available) - set(selected_episodes)
                )
                generated_extra_episodes = sorted(
                    set(generated_available) - set(selected_episodes)
                )
                task_report["generated_extra_episodes"] = generated_extra_episodes
                for episode in generated_extra_episodes:
                    differences.write(
                        {
                            "category": "unexpected_generated_episode",
                            "task_id": task_id,
                            "episode": episode,
                            "path": f"/episode_{episode}",
                            "reference_exists": episode in reference_available,
                            "generated_exists": True,
                        }
                    )

                shared = _compare_scope(
                    reference_h5,
                    generated_h5,
                    task_id=task_id,
                    episode=None,
                    scope="",
                    exclude_episode_roots=True,
                    emit_leaf=leaves.write,
                    emit_difference=differences.write,
                    rtol=rtol,
                    atol=atol,
                )
                task_report["shared_file_scope"] = shared
                reference_episode_digests: list[tuple[int, str]] = []
                generated_episode_digests: list[tuple[int, str]] = []
                selected_equal = bool(
                    shared["strict_equal"] and not generated_extra_episodes
                )

                for episode in selected_episodes:
                    episode_key = f"episode_{episode}"
                    reference_exists = episode_key in reference_h5
                    generated_exists = episode_key in generated_h5
                    reference_obj = (
                        reference_h5[episode_key] if reference_exists else None
                    )
                    generated_obj = (
                        generated_h5[episode_key] if generated_exists else None
                    )
                    reference_kind = (
                        "group"
                        if isinstance(reference_obj, h5py.Group)
                        else "dataset" if isinstance(reference_obj, h5py.Dataset) else None
                    )
                    generated_kind = (
                        "group"
                        if isinstance(generated_obj, h5py.Group)
                        else "dataset" if isinstance(generated_obj, h5py.Dataset) else None
                    )
                    episode_report: dict[str, Any] = {
                        "episode": episode,
                        "reference_exists": reference_exists,
                        "generated_exists": generated_exists,
                        "reference_kind": reference_kind,
                        "generated_kind": generated_kind,
                        "strict_equal": False,
                    }
                    if (
                        not reference_exists
                        or not generated_exists
                        or reference_kind != "group"
                        or generated_kind != "group"
                    ):
                        record = {
                            "task_id": task_id,
                            "episode": episode,
                            "kind": "group",
                            "path": f"/{episode_key}",
                            "category": (
                                "missing_episode"
                                if not reference_exists or not generated_exists
                                else "episode_kind"
                            ),
                            "reference_exists": reference_exists,
                            "generated_exists": generated_exists,
                            "reference_kind": reference_kind,
                            "generated_kind": generated_kind,
                            "strict_equal": False,
                        }
                        leaves.write(record)
                        differences.write(record)
                        reference_digest = _missing_scope_digest()
                        generated_digest = _missing_scope_digest()
                        if reference_obj is not None:
                            reference_digest = _episode_root_digest(reference_obj)
                        if generated_obj is not None:
                            generated_digest = _episode_root_digest(generated_obj)
                        episode_report["reference_canonical_sha256"] = reference_digest
                        episode_report["generated_canonical_sha256"] = generated_digest
                    else:
                        compared = _compare_scope(
                            reference_obj,
                            generated_obj,
                            task_id=task_id,
                            episode=episode,
                            scope=episode_key,
                            exclude_episode_roots=False,
                            emit_leaf=leaves.write,
                            emit_difference=differences.write,
                            rtol=rtol,
                            atol=atol,
                        )
                        episode_report.update(compared)
                        reference_digest = compared["reference_canonical_sha256"]
                        generated_digest = compared["generated_canonical_sha256"]
                    reference_episode_digests.append((episode, reference_digest))
                    generated_episode_digests.append((episode, generated_digest))
                    selected_equal &= bool(episode_report["strict_equal"])
                    task_report["episodes"].append(episode_report)

                reference_task_sha = _task_digest(
                    shared["reference_canonical_sha256"], reference_episode_digests
                )
                generated_task_sha = _task_digest(
                    shared["generated_canonical_sha256"], generated_episode_digests
                )
                task_report["reference_canonical_sha256"] = reference_task_sha
                task_report["generated_canonical_sha256"] = generated_task_sha
                task_report["canonical_sha256_equal"] = (
                    reference_task_sha == generated_task_sha
                )
                task_report["strict_equal"] = bool(
                    selected_equal and reference_task_sha == generated_task_sha
                )
            task_reports.append(task_report)

        inventory_equal = not any(
            inventory[key]
            for key in (
                "reference_missing",
                "reference_unexpected",
                "generated_missing",
                "generated_unexpected",
            )
        )
        passed = bool(inventory_equal and all(task["strict_equal"] for task in task_reports))
        summary = {
            "schema_version": 1,
            "comparison_scope": "selected_episodes_only",
            "workspace_root": str(workspace),
            "reference_dir": str(reference),
            "generated_dir": str(generated),
            "reference_open_mode": "read_only",
            "report_dir": str(report),
            "selected_episodes": list(selected_episodes),
            "task_ids": list(normalized_task_ids),
            "tolerances": {
                "rtol": rtol,
                "atol": atol,
                "joint_action_max_abs_diff": joint_action_max_abs_diff,
            },
            "file_inventory": inventory,
            "tasks": task_reports,
            "counts": {
                "expected_tasks": len(normalized_task_ids),
                "tasks_strict_equal": sum(bool(task["strict_equal"]) for task in task_reports),
                "selected_task_episodes": len(normalized_task_ids) * len(selected_episodes),
                "selected_episodes_strict_equal": sum(
                    bool(episode["strict_equal"])
                    for task in task_reports
                    for episode in task["episodes"]
                ),
                "leaf_records": leaves.count,
                "differences": differences.count,
            },
            "leaf_comparisons_path": str(leaves_path),
            "differences_path": str(differences_path),
            "strict_equal": passed,
            "passed": passed,
        }

    allowed_difference_count = 0
    rejected_difference_count = 0
    with differences_path.open("r", encoding="utf-8") as difference_file:
        for line in difference_file:
            record = json.loads(line)
            metrics = record.get("metrics", {})
            max_abs_diff = metrics.get("max_abs_diff")
            within_absolute_cap = bool(
                isinstance(max_abs_diff, (int, float))
                and math.isfinite(float(max_abs_diff))
                and float(max_abs_diff) <= joint_action_max_abs_diff
            )
            allowed = bool(
                allow_joint_action_allclose
                and record.get("category") == "dataset_content"
                and str(record.get("path", "")).endswith("/action/joint_action")
                and record.get("metadata_equal") is True
                and record.get("attributes_equal") is True
                and metrics.get("allclose") is True
                and within_absolute_cap
            )
            if allowed:
                allowed_difference_count += 1
            else:
                rejected_difference_count += 1

    accepted = bool(
        passed
        or (
            allow_joint_action_allclose
            and allowed_difference_count > 0
            and rejected_difference_count == 0
        )
    )
    summary["acceptance_policy"] = {
        "name": (
            "joint_action_allclose"
            if allow_joint_action_allclose
            else "strict_only"
        ),
        "rtol": rtol,
        "atol": atol,
        "joint_action_max_abs_diff": joint_action_max_abs_diff,
        "allowed_difference_count": allowed_difference_count,
        "rejected_difference_count": rejected_difference_count,
    }
    summary["accepted"] = accepted

    comparison_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "严格比较官方与重新生成的 RoboMME HDF5；默认扫描 16 个任务的 episode 0..8。"
        )
    )
    parser.add_argument("--reference-dir", required=True, help="官方参考 HDF5 目录（只读打开）")
    parser.add_argument("--generated-dir", required=True, help="重新生成的 HDF5 目录")
    parser.add_argument("--report-dir", required=True, help="仓库内审查报告输出目录")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPISODES),
        help="要比较的 episode 编号，默认：0 1 2 3 4 5 6 7 8",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=EXPECTED_TASK_IDS,
        default=list(EXPECTED_TASK_IDS),
        help="要比较的任务，默认全部 16 个任务",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="允许输入和输出所在的仓库根目录，默认当前目录",
    )
    parser.add_argument("--rtol", type=float, default=1e-7, help="数值 allclose 相对容差")
    parser.add_argument("--atol", type=float, default=0.0, help="数值 allclose 绝对容差")
    parser.add_argument(
        "--allow-joint-action-allclose",
        action="store_true",
        help="允许仅 action/joint_action 的浮点差异在 allclose 范围内通过；仍保留 strict_equal=false",
    )
    parser.add_argument(
        "--joint-action-max-abs-diff",
        type=float,
        default=DEFAULT_JOINT_ACTION_MAX_ABS_DIFF,
        help="joint_action 容差策略允许的最大绝对误差，默认 1e-12",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        summary = compare_generated_dataset(
            reference_dir=args.reference_dir,
            generated_dir=args.generated_dir,
            report_dir=args.report_dir,
            episodes=args.episodes,
            workspace_root=args.workspace_root,
            task_ids=args.tasks,
            rtol=args.rtol,
            atol=args.atol,
            allow_joint_action_allclose=args.allow_joint_action_allclose,
            joint_action_max_abs_diff=args.joint_action_max_abs_diff,
        )
    except (UnsafePathError, UnsafeHDF5LinkError, OSError, ValueError) as exc:
        raise SystemExit(f"比较输入无效：{exc}") from exc
    comparison_path = Path(summary["report_dir"]) / "comparison.json"
    if not summary["accepted"]:
        raise SystemExit(f"比较未通过所选验收策略；报告：{comparison_path}")
    if summary["strict_equal"]:
        print(f"严格比较通过；报告：{comparison_path}")
    else:
        print(f"严格比较未通过，但 joint_action 容差策略通过；报告：{comparison_path}")


if __name__ == "__main__":
    main()

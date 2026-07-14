#!/usr/bin/env python3
"""独立 No-Patch RoboMME 数据生成、验证与 joint_action 对比入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
METADATA_ROOT = SRC_ROOT / "robomme" / "env_metadata" / "train"
REFERENCE_ROOT = REPO_ROOT / "data" / "robomme_data_h5"
ALL_TASKS = (
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap",
    "VideoUnmask", "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick",
    "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight", "InsertPeg",
    "MoveCube", "PatternLock", "RouteStick",
)
STICK_TASKS = frozenset(("PatternLock", "RouteStick"))
MAX_EPISODES = 9
MAX_ABS_DIFF = 1e-8
MAX_ERRORS = 200
EPISODE_RE = re.compile(r"^episode_(\d+)$")
TIMESTEP_RE = re.compile(r"^timestep_(\d+)$")


class DatasetGenerationError(RuntimeError):
    pass


class PlannerExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodeJob:
    task: str
    episode: int
    seed: int
    difficulty: str
    worker_dir: str
    gpu: str
    repo_root: str

    @property
    def recovery_mode(self) -> str | None:
        if self.episode <= 2:
            return "z"
        if self.episode <= 5:
            return "xy"
        return None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_layout() -> None:
    required = (REPO_ROOT / "uv.lock", SRC_ROOT, METADATA_ROOT, REFERENCE_ROOT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise DatasetGenerationError("缺少必需路径: " + ", ".join(missing))


def _prepare_output(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise DatasetGenerationError(f"输出目录不能是符号链接: {requested}")
    parent = requested.parent
    while parent != parent.parent:
        if parent.exists() and parent.is_symlink():
            raise DatasetGenerationError(f"输出路径包含符号链接父目录: {parent}")
        if parent == REPO_ROOT:
            break
        parent = parent.parent
    output = requested.resolve()
    repo = REPO_ROOT.resolve()
    reference = REFERENCE_ROOT.resolve()
    if output == repo or not _inside(output, repo):
        raise DatasetGenerationError(f"输出目录必须位于仓库内: {output}")
    if _inside(output, reference):
        raise DatasetGenerationError(f"输出目录不能位于参考数据内: {output}")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise DatasetGenerationError(f"输出目录必须不存在或为空: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _integer(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DatasetGenerationError(f"{path}: {field} 必须是整数，得到 {value!r}")
    return int(value)


def _read_train_metadata() -> dict[str, dict[int, dict[str, Any]]]:
    expected_files = {f"record_dataset_{task}_metadata.json" for task in ALL_TASKS}
    actual_files = {path.name for path in METADATA_ROOT.glob("record_dataset_*_metadata.json")}
    if actual_files != expected_files:
        raise DatasetGenerationError(
            "train metadata 文件集合不匹配固定 16 环境: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    all_records: dict[str, dict[int, dict[str, Any]]] = {}
    for task in ALL_TASKS:
        path = METADATA_ROOT / f"record_dataset_{task}_metadata.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetGenerationError(f"无法读取 {path}: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("env_id") != task:
            raise DatasetGenerationError(f"{path}: env_id 不匹配")
        records = payload.get("records")
        if not isinstance(records, list):
            raise DatasetGenerationError(f"{path}: records 必须是 list")
        if _integer(payload.get("record_count"), "record_count", path) != len(records):
            raise DatasetGenerationError(f"{path}: record_count 不等于 records 长度")
        if len(records) != 100:
            raise DatasetGenerationError(f"{path}: records 必须严格为 100 条")
        indexed: dict[int, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise DatasetGenerationError(f"{path}: record 必须是 object")
            if any(key not in record for key in ("task", "episode", "seed", "difficulty")):
                raise DatasetGenerationError(f"{path}: record 缺少 task/episode/seed/difficulty")
            if record["task"] != task:
                raise DatasetGenerationError(f"{path}: record task 不匹配")
            episode = _integer(record["episode"], "episode", path)
            seed = _integer(record["seed"], "seed", path)
            difficulty = record["difficulty"]
            if not isinstance(difficulty, str) or not difficulty:
                raise DatasetGenerationError(f"{path}: difficulty 必须为非空字符串")
            if episode in indexed:
                raise DatasetGenerationError(f"{path}: episode {episode} 重复")
            indexed[episode] = {
                "task": task,
                "episode": episode,
                "seed": seed,
                "difficulty": difficulty,
            }
        if set(indexed) != set(range(100)):
            raise DatasetGenerationError(f"{path}: episode 集合必须严格为 0..99")
        all_records[task] = indexed
    return all_records


def _parse_tasks(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_TASKS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise DatasetGenerationError("--env 不能为空且不能重复")
    unknown = sorted(set(names) - set(ALL_TASKS))
    if unknown:
        raise DatasetGenerationError("未知环境: " + ", ".join(unknown))
    return [task for task in ALL_TASKS if task in names]


def _parse_gpus(value: str | Sequence[str | int]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    gpus = tuple(str(item).strip() for item in raw if str(item).strip())
    if not gpus or any(re.fullmatch(r"\d+", item) is None for item in gpus):
        raise DatasetGenerationError("GPU 必须为逗号分隔的非负整数")
    return gpus


def _add_error(section: dict[str, Any], message: str) -> None:
    section["error_count"] = int(section.get("error_count", 0)) + 1
    errors = section.setdefault("errors", [])
    if len(errors) < MAX_ERRORS:
        errors.append(message)


def _timestep_indices(group: h5py.Group, source: str) -> tuple[list[int], list[str]]:
    errors: list[str] = []
    indices: list[int] = []
    for name in group.keys():
        if name == "setup":
            continue
        match = TIMESTEP_RE.fullmatch(name)
        if match is None or not isinstance(group[name], h5py.Group):
            errors.append(f"{source}: 非法 timestep {name!r}")
        else:
            indices.append(int(match.group(1)))
    indices.sort()
    if not indices:
        errors.append(f"{source}: 没有 timestep")
    elif indices != list(range(len(indices))):
        errors.append(f"{source}: timestep 必须从 0 连续，实际 {indices[:12]}")
    return indices, errors


def _completed(group: h5py.Group, source: str) -> tuple[bool | None, list[str]]:
    indices, errors = _timestep_indices(group, source)
    if errors:
        return None, errors
    try:
        dataset = group[f"timestep_{indices[-1]}"]["info"]["is_completed"]
    except KeyError:
        return None, [f"{source}: 最终 timestep 缺少 info/is_completed"]
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != () or np.dtype(dataset.dtype) != np.dtype(bool):
        return None, [f"{source}: info/is_completed 必须为 bool 标量"]
    value = dataset[()]
    if not isinstance(value, (bool, np.bool_)):
        return None, [f"{source}: info/is_completed 不是 bool"]
    return bool(value), []

def _runtime_bool(value: Any, torch_module: Any) -> bool:
    if isinstance(value, torch_module.Tensor):
        if value.numel() != 1:
            raise DatasetGenerationError("evaluate 返回了非标量 Tensor")
        return bool(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise DatasetGenerationError("evaluate 返回了非标量 ndarray")
        return bool(value.item())
    return bool(value)


def _is_failure(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) == -1


def _planner_classes(arm_base: type, stick_base: type, screw_error: type[BaseException]) -> tuple[type, type]:
    class ScrewThenRRT:
        def move_to_pose_with_screw(self, *args: Any, **kwargs: Any) -> Any:
            last_error: BaseException | None = None
            for _ in range(3):
                try:
                    result = super().move_to_pose_with_screw(*args, **kwargs)
                except screw_error as exc:
                    last_error = exc
                    continue
                if not _is_failure(result):
                    return result
            for _ in range(3):
                try:
                    result = super().move_to_pose_with_RRTStar(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    continue
                if not _is_failure(result):
                    return result
            suffix = f": {last_error}" if last_error is not None else ""
            raise PlannerExhausted("screw 3 次与 RRTStar 3 次均失败" + suffix)

    class NoPatchArm(ScrewThenRRT, arm_base):
        pass

    class NoPatchStick(ScrewThenRRT, stick_base):
        pass

    return NoPatchArm, NoPatchStick


def _execute_tasks(record_env: Any, planner: Any, torch_module: Any, job: EpisodeJob) -> None:
    task_list = list(getattr(record_env.unwrapped, "task_list", []) or [])
    if not task_list:
        raise DatasetGenerationError(f"{job.task}/episode_{job.episode}: task_list 为空")
    for entry in task_list:
        solve = entry.get("solve") if isinstance(entry, Mapping) else None
        if not callable(solve):
            raise DatasetGenerationError(f"{job.task}/episode_{job.episode}: task 没有 solve")
        record_env.unwrapped.evaluate(solve_complete_eval=True)
        result = solve(record_env, planner)
        if _is_failure(result):
            raise PlannerExhausted(f"{job.task}/episode_{job.episode}: solve 返回 -1")
        evaluation = record_env.unwrapped.evaluate(solve_complete_eval=True)
        if _runtime_bool(evaluation.get("fail", False), torch_module):
            raise DatasetGenerationError(f"{job.task}/episode_{job.episode}: 环境报告 fail")
        if _runtime_bool(evaluation.get("success", False), torch_module):
            return
    evaluation = record_env.unwrapped.evaluate(solve_complete_eval=True)
    if not _runtime_bool(evaluation.get("success", False), torch_module):
        raise DatasetGenerationError(f"{job.task}/episode_{job.episode}: 完整 task_list 后未成功")


def _raw_summary(path: Path, job: EpisodeJob) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetGenerationError(f"缺少 raw HDF5: {path}")
    name = f"episode_{job.episode}"
    with h5py.File(path, "r") as handle:
        if name not in handle or not isinstance(handle[name], h5py.Group):
            raise DatasetGenerationError(f"{path}: 缺少 {name}")
        done, errors = _completed(handle[name], f"{path}/{name}")
        if errors:
            raise DatasetGenerationError("; ".join(errors))
        if done is not True:
            raise DatasetGenerationError(f"{path}/{name}: 最终 is_completed 不是 true")
        timesteps, _ = _timestep_indices(handle[name], f"{path}/{name}")
    return {"raw_h5_path": str(path), "timestep_count": len(timesteps)}


def _worker(job: EpisodeJob) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = job.gpu
    worker_dir = Path(job.worker_dir)
    raw_path = worker_dir / "hdf5_files" / f"{job.task}_ep{job.episode}_seed{job.seed}.h5"
    record_env: Any | None = None
    caught: BaseException | None = None
    error_traceback: str | None = None
    try:
        source_root = Path(job.repo_root) / "src"
        if not source_root.is_dir():
            raise DatasetGenerationError(f"src 不存在: {source_root}")
        sys.path.insert(0, str(source_root))
        import gymnasium as gym
        import torch
        import robomme.robomme_env
        from robomme.env_record_wrapper import FailsafeTimeout, RobommeRecordWrapper
        from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
        from robomme.robomme_env.utils.planner_fail_safe import (
            FailAwarePandaArmMotionPlanningSolver,
            FailAwarePandaStickMotionPlanningSolver,
            ScrewPlanFailure,
        )

        arm_cls, stick_cls = _planner_classes(
            FailAwarePandaArmMotionPlanningSolver,
            FailAwarePandaStickMotionPlanningSolver,
            ScrewPlanFailure,
        )
        kwargs: dict[str, Any] = {
            "obs_mode": "rgb+depth+segmentation",
            "control_mode": "pd_joint_pos",
            "render_mode": "rgb_array",
            "reward_mode": "dense",
            "seed": job.seed,
            "difficulty": job.difficulty,
        }
        if job.recovery_mode is not None:
            kwargs["robomme_failure_recovery"] = True
            kwargs["robomme_failure_recovery_mode"] = job.recovery_mode
        worker_dir.mkdir(parents=True, exist_ok=False)
        base_env = gym.make(job.task, **kwargs)
        record_env = RobommeRecordWrapper(
            base_env,
            dataset=str(worker_dir),
            env_id=job.task,
            episode=job.episode,
            seed=job.seed,
            save_video=True,
        )
        record_env.reset()
        planner_kwargs: dict[str, Any] = {
            "debug": False,
            "vis": False,
            "base_pose": record_env.unwrapped.agent.robot.pose,
            "visualize_target_grasp_pose": False,
            "print_env_info": False,
        }
        if job.task in STICK_TASKS:
            planner_kwargs["joint_vel_limits"] = 0.3
            planner = stick_cls(record_env, **planner_kwargs)
        else:
            planner = arm_cls(record_env, **planner_kwargs)
        _execute_tasks(record_env, planner, torch, job)
    except (SceneGenerationError, FailsafeTimeout, PlannerExhausted,
            ScrewPlanFailure, DatasetGenerationError) as exc:
        caught = exc
        error_traceback = traceback.format_exc()
    except Exception as exc:
        caught = exc
        error_traceback = traceback.format_exc()
    finally:
        if record_env is not None:
            try:
                record_env.close()
            except Exception as close_exc:
                if caught is None:
                    caught = close_exc
                    error_traceback = traceback.format_exc()

    base = {
        "task": job.task,
        "episode": job.episode,
        "seed": job.seed,
        "difficulty": job.difficulty,
        "gpu": job.gpu,
        "recovery_mode": job.recovery_mode,
        "attempt_count": 1,
    }
    if caught is not None:
        return {
            **base,
            "ok": False,
            "error_type": type(caught).__name__,
            "error": str(caught),
            "traceback": error_traceback,
        }
    try:
        return {**base, "ok": True, **_raw_summary(raw_path, job)}
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _episode_groups(handle: h5py.File, label: str, section: dict[str, Any]) -> dict[int, h5py.Group]:
    groups: dict[int, h5py.Group] = {}
    for name in handle.keys():
        match = EPISODE_RE.fullmatch(name)
        if match is None or not isinstance(handle[name], h5py.Group):
            _add_error(section, f"{label}: 非法根对象 {name!r}")
            continue
        groups[int(match.group(1))] = handle[name]
    return groups


def _text(dataset: h5py.Dataset, source: str) -> str:
    if dataset.shape != ():
        raise DatasetGenerationError(f"{source}: 字符串必须是标量")
    value = dataset.asstr()[()]
    if not isinstance(value, str):
        raise DatasetGenerationError(f"{source}: 不是字符串")
    return value


def _audit_episode(
    group: h5py.Group,
    task: str,
    episode: int,
    record: Mapping[str, Any],
    label: str,
    section: dict[str, Any],
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "episode": episode,
        "timestep_count": 0,
        "final_is_completed": None,
        "joint_shape": None,
        "joint_dtype": None,
    }
    setup = group.get("setup")
    if not isinstance(setup, h5py.Group):
        _add_error(section, f"{label}: 缺少 setup")
    else:
        seed = setup.get("seed")
        difficulty = setup.get("difficulty")
        if not isinstance(seed, h5py.Dataset) or seed.shape != () or int(seed[()]) != int(record["seed"]):
            _add_error(section, f"{label}: setup/seed 不匹配")
        if not isinstance(difficulty, h5py.Dataset):
            _add_error(section, f"{label}: 缺少 setup/difficulty")
        else:
            try:
                if _text(difficulty, f"{label}: setup/difficulty") != record["difficulty"]:
                    _add_error(section, f"{label}: setup/difficulty 不匹配")
            except DatasetGenerationError as exc:
                _add_error(section, str(exc))
    steps, errors = _timestep_indices(group, label)
    for error in errors:
        _add_error(section, error)
    if errors:
        return detail
    detail["timestep_count"] = len(steps)
    done, done_errors = _completed(group, label)
    for error in done_errors:
        _add_error(section, error)
    detail["final_is_completed"] = done
    signatures: set[tuple[tuple[int, ...], str]] = set()
    for timestep in steps:
        try:
            joint = group[f"timestep_{timestep}"]["action"]["joint_action"]
        except KeyError:
            _add_error(section, f"{label}: timestep_{timestep} 缺少 action/joint_action")
            continue
        if not isinstance(joint, h5py.Dataset):
            _add_error(section, f"{label}: joint_action 必须是 dataset")
            continue
        signature = (tuple(joint.shape), str(joint.dtype))
        signatures.add(signature)
        if tuple(joint.shape) != (8,) or np.dtype(joint.dtype) != np.dtype(np.float64):
            _add_error(section, f"{label}: joint_action 必须为 (8,) float64")
            continue
        values = np.asarray(joint[()])
        if not np.all(np.isfinite(values)):
            _add_error(section, f"{label}: joint_action 包含非有限值")
            continue
        section["joint_vector_count"] += 1
        section["joint_element_count"] += int(values.size)
    if len(signatures) == 1:
        shape, dtype = next(iter(signatures))
        detail["joint_shape"], detail["joint_dtype"] = list(shape), dtype
    elif signatures:
        _add_error(section, f"{label}: joint_action shape/dtype 不一致")
    return detail

def _audit_file(
    path: Path,
    task: str,
    records: Mapping[int, Mapping[str, Any]],
    episodes: Sequence[int],
    exact_episodes: bool,
    label: str,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "label": label,
        "task": task,
        "path": str(path),
        "episodes": [],
        "completed_count": 0,
        "joint_vector_count": 0,
        "joint_element_count": 0,
        "error_count": 0,
        "errors": [],
    }
    if not path.is_file():
        _add_error(section, f"{label}/{task}: HDF5 不存在")
        return section
    try:
        with h5py.File(path, "r") as handle:
            groups = _episode_groups(handle, f"{label}/{task}", section)
            expected = set(episodes)
            actual = set(groups)
            if not expected.issubset(actual):
                _add_error(section, f"{label}/{task}: 缺少 episode {sorted(expected - actual)}")
            if exact_episodes and actual != expected:
                _add_error(section, f"{label}/{task}: episode 集合不严格匹配")
            for episode in episodes:
                if episode not in groups:
                    continue
                detail = _audit_episode(
                    groups[episode],
                    task,
                    episode,
                    records[episode],
                    f"{label}/{task}/episode_{episode}",
                    section,
                )
                section["episodes"].append(detail)
                if detail["final_is_completed"] is True:
                    section["completed_count"] += 1
    except OSError as exc:
        _add_error(section, f"{label}/{task}: 无法读取 HDF5: {exc}")
    return section


def _audit_metadata(path: Path, task: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    section: dict[str, Any] = {"task": task, "path": str(path), "error_count": 0, "errors": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _add_error(section, f"{task}: metadata 无法读取: {exc}")
        return section
    if not isinstance(payload, Mapping) or payload.get("env_id") != task:
        _add_error(section, f"{task}: metadata env_id 不匹配")
    if payload.get("record_count") != len(records):
        _add_error(section, f"{task}: metadata record_count 不匹配")
    if payload.get("records") != list(records):
        _add_error(section, f"{task}: metadata records 与 train metadata 不一致")
    return section


def _compare(
    output: Path,
    tasks: Sequence[str],
    episodes: Sequence[int],
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "joint_vector_count": 0,
        "joint_element_count": 0,
        "different_element_count": 0,
        "max_abs_diff": None,
        "max_abs_diff_location": None,
        "error_count": 0,
        "errors": [],
    }
    for task in tasks:
        reference_path = REFERENCE_ROOT / f"record_dataset_{task}.h5"
        generated_path = output / f"record_dataset_{task}.h5"
        if not reference_path.is_file() or not generated_path.is_file():
            _add_error(section, f"{task}: 比较文件不存在")
            continue
        try:
            with h5py.File(reference_path, "r") as reference, h5py.File(generated_path, "r") as generated:
                reference_groups = _episode_groups(reference, f"reference/{task}", section)
                generated_groups = _episode_groups(generated, f"generated/{task}", section)
                for episode in episodes:
                    left = reference_groups.get(episode)
                    right = generated_groups.get(episode)
                    if left is None or right is None:
                        _add_error(section, f"{task}/episode_{episode}: 缺少 reference 或 generated")
                        continue
                    left_steps, left_errors = _timestep_indices(left, f"reference/{task}/episode_{episode}")
                    right_steps, right_errors = _timestep_indices(right, f"generated/{task}/episode_{episode}")
                    for error in left_errors + right_errors:
                        _add_error(section, error)
                    if left_errors or right_errors or left_steps != right_steps:
                        if not left_errors and not right_errors:
                            _add_error(section, f"{task}/episode_{episode}: timestep 集合不一致")
                        continue
                    for timestep in left_steps:
                        location = f"{task}/episode_{episode}/timestep_{timestep}"
                        try:
                            a = left[f"timestep_{timestep}"]["action"]["joint_action"]
                            b = right[f"timestep_{timestep}"]["action"]["joint_action"]
                        except KeyError:
                            _add_error(section, f"{location}: 缺少 action/joint_action")
                            continue
                        if not isinstance(a, h5py.Dataset) or not isinstance(b, h5py.Dataset):
                            _add_error(section, f"{location}: joint_action 不是 dataset")
                            continue
                        if tuple(a.shape) != tuple(b.shape) or np.dtype(a.dtype) != np.dtype(b.dtype):
                            _add_error(section, f"{location}: joint_action shape/dtype 不一致")
                            continue
                        if tuple(a.shape) != (8,) or np.dtype(a.dtype) != np.dtype(np.float64):
                            _add_error(section, f"{location}: joint_action 不是 (8,) float64")
                            continue
                        x, y = np.asarray(a[()]), np.asarray(b[()])
                        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                            _add_error(section, f"{location}: joint_action 包含非有限值")
                            continue
                        delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                        section["joint_vector_count"] += 1
                        section["joint_element_count"] += int(delta.size)
                        section["different_element_count"] += int(np.count_nonzero(delta != 0.0))
                        maximum = float(np.max(delta))
                        if section["max_abs_diff"] is None or maximum > section["max_abs_diff"]:
                            index = int(np.argmax(delta))
                            section["max_abs_diff"] = maximum
                            section["max_abs_diff_location"] = {
                                "task": task,
                                "episode": episode,
                                "timestep": timestep,
                                "element_index": index,
                                "reference_value": float(x.reshape(-1)[index]),
                                "generated_value": float(y.reshape(-1)[index]),
                            }
        except OSError as exc:
            _add_error(section, f"{task}: 比较 HDF5 读取失败: {exc}")
    if section["joint_element_count"] == 0:
        _add_error(section, "没有比较任何 joint_action 元素")
    section["within_max_abs_diff"] = bool(
        section["max_abs_diff"] is not None and section["max_abs_diff"] <= MAX_ABS_DIFF
    )
    return section


def _merge(output: Path, task: str, results: Sequence[Mapping[str, Any]]) -> Path:
    target = output / f"record_dataset_{task}.h5"
    temporary = output / f".record_dataset_{task}.h5.tmp"
    try:
        with h5py.File(temporary, "w") as merged:
            for result in sorted(results, key=lambda item: int(item["episode"])):
                episode = int(result["episode"])
                name = f"episode_{episode}"
                with h5py.File(str(result["raw_h5_path"]), "r") as raw:
                    if name not in raw:
                        raise DatasetGenerationError(f"{raw.filename}: 缺少 {name}")
                    raw.copy(raw[name], merged, name=name)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_metadata(output: Path, task: str, records: Sequence[Mapping[str, Any]]) -> None:
    payload = {"env_id": task, "record_count": len(records), "records": list(records)}
    _write_text(
        output / f"record_dataset_{task}_metadata.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _summary(audits: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("completed_count", "joint_vector_count", "joint_element_count", "error_count")
    return {
        "file_count": len(audits),
        "episode_count": sum(len(audit["episodes"]) for audit in audits),
        **{key: sum(int(audit.get(key, 0)) for audit in audits) for key in keys},
    }


def _validate(
    output: Path,
    tasks: Sequence[str],
    records_by_task: Mapping[str, Mapping[int, Mapping[str, Any]]],
    episodes: Sequence[int],
) -> dict[str, Any]:
    generated, official, metadata = [], [], []
    for task in tasks:
        records = [records_by_task[task][episode] for episode in episodes]
        metadata.append(_audit_metadata(output / f"record_dataset_{task}_metadata.json", task, records))
        generated.append(_audit_file(output / f"record_dataset_{task}.h5", task, records_by_task[task], episodes, True, "generated"))
        official.append(_audit_file(REFERENCE_ROOT / f"record_dataset_{task}.h5", task, records_by_task[task], episodes, False, "official"))
    generated_summary, official_summary = _summary(generated), _summary(official)
    comparison = _compare(output, tasks, episodes)
    metadata_errors = sum(int(item["error_count"]) for item in metadata)
    expected = len(tasks) * len(episodes)
    scope = {
        "tasks": list(tasks),
        "episode_indices": list(episodes),
        "expected_episode_count": expected,
        "full_16x9": list(tasks) == list(ALL_TASKS) and list(episodes) == list(range(MAX_EPISODES)),
    }
    passed = (
        metadata_errors == 0
        and generated_summary["error_count"] == 0
        and official_summary["error_count"] == 0
        and comparison["error_count"] == 0
        and generated_summary["completed_count"] == expected
        and official_summary["completed_count"] == expected
        and comparison["within_max_abs_diff"]
    )
    return {
        "passed": passed,
        "scope": scope,
        "metadata": {"error_count": metadata_errors, "audits": metadata},
        "generated": {**generated_summary, "audits": generated},
        "official": {**official_summary, "audits": official},
        "joint_action_comparison": comparison,
        "acceptance": {
            "expected_final_completed": expected,
            "generated_final_completed": generated_summary["completed_count"],
            "official_final_completed": official_summary["completed_count"],
            "max_allowed_abs_diff": MAX_ABS_DIFF,
            "max_abs_diff": comparison["max_abs_diff"],
        },
    }

def _markdown(report: Mapping[str, Any]) -> str:
    validation = report.get("validation", {})
    scope = validation.get("scope", {}) if isinstance(validation, Mapping) else {}
    acceptance = validation.get("acceptance", {}) if isinstance(validation, Mapping) else {}
    comparison = validation.get("joint_action_comparison", {}) if isinstance(validation, Mapping) else {}
    generation = report.get("generation", {})
    lines = [
        "# No-Patch 数据生成报告",
        "",
        f"- 状态：{report.get('status')}",
        f"- 当前 HEAD：{report.get('current_head')}",
        f"- uv.lock SHA-256：{report.get('uv_lock_sha256')}",
        f"- 生成时间（UTC）：{report.get('generated_at_utc')}",
        "",
        "参数 JSON：",
        json.dumps(report.get("parameters", {}), ensure_ascii=False, indent=2),
        "",
        "## 结果",
        "",
        f"- worker 成功数：{generation.get('success_count', 0)}",
        f"- worker 失败数：{generation.get('failure_count', 0)}",
        f"- 验证范围完整 16×9：{scope.get('full_16x9', False)}",
        f"- 官方最终完成：{acceptance.get('official_final_completed', 0)}/{acceptance.get('expected_final_completed', 0)}",
        f"- 生成最终完成：{acceptance.get('generated_final_completed', 0)}/{acceptance.get('expected_final_completed', 0)}",
        f"- joint_action 比较元素数：{comparison.get('joint_element_count', 0)}",
        f"- 最大绝对差：{comparison.get('max_abs_diff')}",
        f"- 最大允许绝对差：{acceptance.get('max_allowed_abs_diff')}",
        f"- 验证通过：{validation.get('passed', False)}",
    ]
    if report.get("error"):
        lines.extend(("", "## 错误", "", json.dumps(report["error"], ensure_ascii=False, indent=2)))
    return "\n".join(lines) + "\n"


def _write_reports(output: Path, report: dict[str, Any]) -> None:
    json_path = output / "no_patch_generation_report.json"
    markdown_path = output / "no_patch_generation_report.md"
    report["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
    _write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_text(markdown_path, _markdown(report))


def generate_dataset(
    output_dir: str | Path,
    env: str = "all",
    episodes: int = MAX_EPISODES,
    workers: int = 1,
    gpus: str | Sequence[str | int] = "0",
) -> dict[str, Any]:
    _ensure_layout()
    if not 1 <= episodes <= MAX_EPISODES:
        raise DatasetGenerationError(f"episodes 必须在 1..{MAX_EPISODES}")
    if workers < 1:
        raise DatasetGenerationError("workers 必须大于 0")
    output = _prepare_output(output_dir)
    tasks, gpu_ids, episode_indices = _parse_tasks(env), _parse_gpus(gpus), list(range(episodes))
    records_by_task = _read_train_metadata()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_head": _head(),
        "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
        "parameters": {
            "output_dir": str(output),
            "env": env,
            "tasks": tasks,
            "episodes": episodes,
            "workers": workers,
            "gpus": list(gpu_ids),
            "metadata_root": str(METADATA_ROOT),
            "reference_root": str(REFERENCE_ROOT),
            "seed_attempts_per_episode": 1,
            "save_video_for_recording": True,
        },
    }
    temporary = output / ".workers"
    results: list[dict[str, Any]] = []
    try:
        temporary.mkdir()
        jobs: list[EpisodeJob] = []
        for task in tasks:
            for episode in episode_indices:
                record = records_by_task[task][episode]
                jobs.append(EpisodeJob(
                    task=task,
                    episode=episode,
                    seed=int(record["seed"]),
                    difficulty=str(record["difficulty"]),
                    worker_dir=str(temporary / f"{task}_episode_{episode}"),
                    gpu=gpu_ids[len(jobs) % len(gpu_ids)],
                    repo_root=str(REPO_ROOT),
                ))
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {executor.submit(_worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results.append(future.result())
                except BaseException as exc:
                    results.append({
                        "task": job.task, "episode": job.episode, "seed": job.seed,
                        "difficulty": job.difficulty, "gpu": job.gpu,
                        "recovery_mode": job.recovery_mode, "attempt_count": 1,
                        "ok": False, "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc(),
                    })
        results.sort(key=lambda item: (str(item["task"]), int(item["episode"])))
        failures = [item for item in results if not item.get("ok")]
        report["generation"] = {
            "requested_count": len(jobs),
            "success_count": len(results) - len(failures),
            "failure_count": len(failures),
            "results": results,
        }
        if failures:
            details = "; ".join(
                f"{item['task']}/episode_{item['episode']}: {item.get('error_type')} {item.get('error')}"
                for item in failures
            )
            raise DatasetGenerationError(f"单次原 seed 生成失败: {details}")
        for task in tasks:
            task_results = [item for item in results if item["task"] == task]
            _merge(output, task, task_results)
            _write_metadata(
                output,
                task,
                [records_by_task[task][episode] for episode in episode_indices],
            )
        report["validation"] = _validate(output, tasks, records_by_task, episode_indices)
        report["status"] = "passed" if report["validation"]["passed"] else "failed"
        if not report["validation"]["passed"]:
            raise DatasetGenerationError("生成后验证或 joint_action 比较未达到验收条件")
        _write_reports(output, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if "generation" not in report:
            failures = [item for item in results if not item.get("ok")]
            report["generation"] = {
                "requested_count": len(results),
                "success_count": len(results) - len(failures),
                "failure_count": len(failures),
                "results": results,
            }
        try:
            _write_reports(output, report)
        except Exception as report_error:
            raise DatasetGenerationError(f"{exc}; 同时无法写报告: {report_error}") from exc
        if isinstance(exc, DatasetGenerationError):
            raise
        raise DatasetGenerationError(str(exc)) from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立 No-Patch RoboMME 数据生成与验证")
    parser.add_argument("--output-dir", required=True, help="仓库内不存在或为空的输出目录")
    parser.add_argument("--env", "--environment", default="all", help="all 或逗号分隔环境名")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES, help="每环境从 episode 0 开始的数量")
    parser.add_argument("--workers", "--max-workers", dest="workers", type=int, default=1)
    parser.add_argument("--gpus", "--gpu", dest="gpus", default="0", help="例如 0 或 0,1")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        report = generate_dataset(
            output_dir=args.output_dir,
            env=args.env,
            episodes=args.episodes,
            workers=args.workers,
            gpus=args.gpus,
        )
    except DatasetGenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report["status"],
        "report_paths": report["report_paths"],
        "max_abs_diff": report["validation"]["joint_action_comparison"]["max_abs_diff"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

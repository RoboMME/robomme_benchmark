#!/usr/bin/env python3
"""独立 No-Patch RoboMME 数据生成入口。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_generated_dataset_contract import (
    MAX_EPISODES,
    METADATA_ROOT,
    REFERENCE_ROOT,
    DatasetContractError,
    inspect_episode_terminal,
    parse_tasks,
    read_train_metadata,
)
from write_generation_report import (
    build_validation_report,
    new_generation_report,
    write_generation_report,
    write_text_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
STICK_TASKS = frozenset(("PatternLock", "RouteStick"))


class DatasetGenerationError(RuntimeError):
    """生成、合并或生成后验收未满足 No-Patch 契约。"""


class PlannerExhausted(RuntimeError):
    """本地 planner 的 screw 与 RRTStar 重试均已耗尽。"""


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


def _parse_gpus(value: str | Sequence[str | int]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    gpus = tuple(str(item).strip() for item in raw if str(item).strip())
    if not gpus or any(re.fullmatch(r"\d+", item) is None for item in gpus):
        raise DatasetGenerationError("GPU 必须为逗号分隔的非负整数")
    return gpus


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


def _planner_classes(
    arm_base: type,
    stick_base: type,
    screw_error: type[BaseException],
) -> tuple[type, type]:
    """以本地子类实现 screw 三次后 RRTStar 三次的无 monkeypatch 回退。"""

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
            raise PlannerExhausted(
                f"{job.task}/episode_{job.episode}: solve 返回 -1"
            )
        evaluation = record_env.unwrapped.evaluate(solve_complete_eval=True)
        if _runtime_bool(evaluation.get("fail", False), torch_module):
            raise DatasetGenerationError(
                f"{job.task}/episode_{job.episode}: 环境报告 fail"
            )
        if _runtime_bool(evaluation.get("success", False), torch_module):
            return
    evaluation = record_env.unwrapped.evaluate(solve_complete_eval=True)
    if not _runtime_bool(evaluation.get("success", False), torch_module):
        raise DatasetGenerationError(
            f"{job.task}/episode_{job.episode}: 完整 task_list 后未成功"
        )


def _raw_summary(path: Path, job: EpisodeJob) -> dict[str, Any]:
    """在 worker 成功后立即用共享 contract helper 检查 raw 轨迹 terminal。"""
    if not path.is_file():
        raise DatasetGenerationError(f"缺少 raw HDF5: {path}")
    name = f"episode_{job.episode}"
    with h5py.File(path, "r") as handle:
        if name not in handle or not isinstance(handle[name], h5py.Group):
            raise DatasetGenerationError(f"{path}: 缺少 {name}")
        timesteps, done, errors = inspect_episode_terminal(
            handle[name],
            f"{path}/{name}",
        )
        if errors:
            raise DatasetGenerationError("; ".join(errors))
        if done is not True:
            raise DatasetGenerationError(f"{path}/{name}: 最终 is_completed 不是 true")
    return {"raw_h5_path": str(path), "timestep_count": len(timesteps)}


def _worker(job: EpisodeJob) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = job.gpu
    worker_dir = Path(job.worker_dir)
    raw_path = worker_dir / "hdf5_files" / (
        f"{job.task}_ep{job.episode}_seed{job.seed}.h5"
    )
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
    except (
        SceneGenerationError,
        FailsafeTimeout,
        PlannerExhausted,
        ScrewPlanFailure,
        DatasetGenerationError,
    ) as exc:
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


def _write_metadata(
    output: Path,
    task: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    payload = {"env_id": task, "record_count": len(records), "records": list(records)}
    write_text_atomic(
        output / f"record_dataset_{task}_metadata.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def generate_dataset(
    output_dir: str | Path,
    env: str = "all",
    episodes: int = MAX_EPISODES,
    workers: int = 1,
    gpus: str | Sequence[str | int] = "0",
) -> dict[str, Any]:
    """生成、合并并在同一进程执行拆分后的合约验证与数值比较。"""
    _ensure_layout()
    output = _prepare_output(output_dir)
    temporary = output / ".workers"
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "parameters": {
            "output_dir": str(output),
            "env": env,
            "episodes": episodes,
            "workers": workers,
            "requested_gpus": str(gpus),
            "metadata_root": str(METADATA_ROOT),
            "reference_root": str(REFERENCE_ROOT),
            "max_abs_diff": 1e-8,
            "seed_attempts_per_episode": 1,
            "save_video_for_recording": True,
        },
    }
    try:
        if not 1 <= episodes <= MAX_EPISODES:
            raise DatasetGenerationError(f"episodes 必须在 1..{MAX_EPISODES}")
        if workers < 1:
            raise DatasetGenerationError("workers 必须大于 0")
        report = new_generation_report(report["parameters"])
        tasks = parse_tasks(env)
        gpu_ids = _parse_gpus(gpus)
        records_by_task = read_train_metadata()
        episode_indices = list(range(episodes))
        report["parameters"].update(
            {
                "tasks": tasks,
                "gpus": list(gpu_ids),
            }
        )
        temporary.mkdir()
        jobs: list[EpisodeJob] = []
        for task in tasks:
            for episode in episode_indices:
                record = records_by_task[task][episode]
                jobs.append(
                    EpisodeJob(
                        task=task,
                        episode=episode,
                        seed=int(record["seed"]),
                        difficulty=str(record["difficulty"]),
                        worker_dir=str(temporary / f"{task}_episode_{episode}"),
                        gpu=gpu_ids[len(jobs) % len(gpu_ids)],
                        repo_root=str(REPO_ROOT),
                    )
                )
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
                    results.append(
                        {
                            "task": job.task,
                            "episode": job.episode,
                            "seed": job.seed,
                            "difficulty": job.difficulty,
                            "gpu": job.gpu,
                            "recovery_mode": job.recovery_mode,
                            "attempt_count": 1,
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
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
                f"{item['task']}/episode_{item['episode']}: "
                f"{item.get('error_type')} {item.get('error')}"
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
        report["validation"] = build_validation_report(
            output,
            tasks,
            episode_indices,
            records_by_task=records_by_task,
            reference_root=REFERENCE_ROOT,
            metadata_root=METADATA_ROOT,
            max_abs_diff=1e-8,
        )
        report["status"] = "passed" if report["validation"]["passed"] else "failed"
        if not report["validation"]["passed"]:
            raise DatasetGenerationError("生成后验证或 joint_action 比较未达到验收条件")
        write_generation_report(output, report)
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
            write_generation_report(output, report)
        except Exception as report_error:
            raise DatasetGenerationError(
                f"{exc}; 同时无法写报告: {report_error}"
            ) from exc
        if isinstance(exc, DatasetGenerationError):
            raise
        raise DatasetGenerationError(str(exc)) from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立 No-Patch RoboMME 数据生成与验证；最新完整报告固定写入 scripts/data-generation-v2-noPatch/reports/")
    parser.add_argument("--output-dir", required=True, help="仓库内不存在或为空的输出目录")
    parser.add_argument("--env", "--environment", default="all", help="all 或逗号分隔环境名")
    parser.add_argument(
        "--episodes",
        type=int,
        default=MAX_EPISODES,
        help="每环境从 episode 0 开始的数量",
    )
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
    comparison = report["validation"]["joint_action_comparison"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_paths": report["report_paths"],
                "max_abs_diff": comparison["max_abs_diff"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

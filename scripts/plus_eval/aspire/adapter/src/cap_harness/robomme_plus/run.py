"""Run an upstream ASPIRE program on one budgeted RoboMME-Plus training episode.

This is a trusted operator entry point. Use an isolated worker before accepting
model-written programs: upstream ProgramExecutor is not an OS security sandbox.
"""
import argparse
import json
from pathlib import Path

from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, TracedProvider
from cap_harness.providers.sam3.client import Sam3Provider
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.pyroki.client import PyRokiProvider
from cap_harness.providers.curobo.client import CuRoboProvider
from cap_harness.registry import ROBOMME_PLUS_PUBLIC_TOOL_NAMES, ToolRegistry
from cap_harness.runtime import ProgramExecutor
from .adapter import RoboMMEPlusAdapter


def run(*, study_root, output_root, task, episode, program, purpose="development"):
    from benchmark_io.protocol import Study
    study = Study.open(study_root)
    if task not in study.config["records"] or not 0 <= episode < len(study.config["records"][task]):
        raise ValueError("Only provided training episodes may be used")
    study_path = Path(output_root)
    limit = study.config["records"][task][episode]["max_steps"]
    recorder = RunRecorder(output_root=study_path, benchmark="robomme_plus", suite="robomme_plus",
        task_id=sorted(study.config["records"]).index(task), task_name=task, seed=episode,
        max_steps=limit, camera_width=256, camera_height=256,
        providers={"segmentation":"sam3", "grasping":"contact-graspnet", "ik":["pyroki","curobo"]})
    adapter = RoboMMEPlusAdapter(study_root=study_root, output_root=recorder.root / "native",
                                 purpose=purpose, run_observer=recorder)
    execution = None
    error = None
    success = False
    try:
        recorder.save_program(Path(program))
        observation = adapter.reset(task, episode)
        recorder.on_reset(observation, adapter.get_task_metadata())
        segmentation = TracedProvider(Sam3Provider(), recorder, "sam3")
        grasp = TracedProvider(ContactGraspNetProvider(), recorder, "contact_graspnet")
        pyroki = TracedProvider(PyRokiProvider(), recorder, "pyroki")
        curobo = TracedProvider(CuRoboProvider(control_period_s=adapter.control_period_s), recorder, "curobo")
        api = CapApi(adapter, segmentation_provider=segmentation,
            grasp_providers={"contact-graspnet":grasp}, ik_providers={"pyroki":pyroki,"curobo":curobo},
            trajectory_planning_providers={"curobo":curobo}, integrated_pose_planning_providers={"curobo-integrated":curobo},
            default_camera="front", recorder=recorder)
        registry = api.register_tools(ToolRegistry(public_extension_allowlist=ROBOMME_PLUS_PUBLIC_TOOL_NAMES))
        execution = ProgramExecutor(registry, recorder=recorder).execute_program(Path(program).read_text())
        success = adapter.check_success()
        return dict(program_ok=execution.ok, task_success=success, run_dir=str(recorder.root))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        adapter.close()
        recorder.finalize(program_ok=bool(execution and execution.ok), task_success=success,
            termination_reason="harness_error" if error else "program_completed", error=error,
            program_result=execution.result if execution else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--program", required=True)
    args = parser.parse_args()
    print(json.dumps(run(study_root=args.study, output_root=args.output, task=args.task,
                         episode=args.episode, program=args.program)))


if __name__ == "__main__":
    main()

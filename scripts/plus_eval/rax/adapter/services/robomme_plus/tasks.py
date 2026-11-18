"""Host-only task catalog. Episode identifiers are released indices, not new seeds."""
import json
import os
from pathlib import Path

CAMERAS = ("front", "wrist")
CONTROL_HZ, IMAGE_SIZE, ACTION_DIM, STATE_DIM, FRAME_STRIDE = 20, 256, 7, 8, 5
POSITION_SCALE_M, ROTATION_SCALE_RAD = 0.05, 0.5
EXPLORATION_COLLECTIONS, COLLECTION_SUFFIX = (0,), {1: "test"}
COLLECTION_NAMES = {0: "train", 1: "test"}


def source_root():
    if os.environ.get("ROBOMME_PLUS_ROOT"):
        return Path(os.environ["ROBOMME_PLUS_ROOT"]).resolve()
    from importlib.util import find_spec
    spec = find_spec("robomme_plus")
    if spec is None or spec.origin is None:
        raise RuntimeError("Set ROBOMME_PLUS_ROOT to the pinned benchmark checkout")
    return Path(spec.origin).resolve().parents[2]


def runtime_root():
    return Path(os.environ.get("RAX_ARTIFACT_ROOT", "runs/agentic/rax"))


def task_names():
    return tuple(p.name.removeprefix("record_dataset_").removesuffix("_metadata.json")
                 for p in sorted((source_root() / "src/robomme_plus/env_metadata/train").glob("*_metadata.json")))


def task_info(task):
    if task not in task_names():
        raise ValueError(f"Unknown RoboMME-Plus task: {task}")
    return dict(official_name=task)


def instruction(task):
    task_info(task)
    return f"Follow the {task} episode goal and conditioning video returned at reset."


def task_timeouts(task):
    task_info(task)
    if os.environ.get('ROBOMME_AGENTIC_RUNTIME_POLICY'):
        from benchmark_io.runtime_policy import configured
        limits = configured()
        return dict(startup_seconds=limits.startup_seconds,
                    episode_seconds=limits.startup_seconds+limits.controller_seconds)
    return dict(startup_seconds=600, episode_seconds=3600)


def layout(task, collection, index):
    task_info(task)
    if collection not in COLLECTION_NAMES:
        raise ValueError("Only the released training and test collections exist")
    path = source_root() / "src/robomme_plus/env_metadata" / COLLECTION_NAMES[collection] / f"record_dataset_{task}_metadata.json"
    records = json.loads(path.read_text())["records"]
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError("Episode index outside the released split")
    return records[index]

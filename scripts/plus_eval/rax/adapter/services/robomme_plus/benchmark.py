"""Registry hooks for the upstream RAX harness; no learning policy is defined here."""
from pathlib import Path
import os
import shutil

from .tasks import task_names, task_timeouts, layout


def tool_definitions():
    from .tools import tool_definitions as definitions
    return definitions()


def backend(config):
    from .backend import RoboMMEPlusBackend
    return RoboMMEPlusBackend(config)


def study():
    from benchmark_io.protocol import Study
    return Study.open(os.environ["ROBOMME_PLUS_STUDY"])


def validate_config(config):
    s = study()
    if s.config["method"] != "rax" or config.task not in s.config["records"]:
        raise ValueError("RAX configuration does not match its operator study")
    if config.eval_seed != 0 or config.formal_collection != 1:
        raise ValueError("Development uses train; formal evaluation uses test")
    if config.agent_cli != "claude" or config.agent_effort not in ("medium", "high"):
        raise ValueError("This study uses upstream LIBERO's Claude model with its pinned reasoning effort")
    demos = s.config["setting"] != "AfI"
    if config.demonstration_context != ("completion_sequence" if demos else "none"):
        raise ValueError("Demonstration visibility differs from the adaptation setting")
    if len(config.exploration_seeds) > s.config["budget"]:
        raise ValueError("Exploration exceeds the task attempt budget")
    if config.task_rules != "public" or config.agent_formal or config.adaptation_source:
        raise ValueError("Use public task information and frozen compiled-controller evaluation")
    validate_layouts(config)


def validate_settings(values):
    if values["formal_seed"] + values["formal_episodes"] > 50:
        raise ValueError("Only 50 held-out episodes per task")


def validate_layouts(config):
    for i in config.exploration_seeds:
        layout(config.task, 0, i)
    for i in range(config.formal_seed, config.formal_seed + config.formal_episodes):
        layout(config.task, 1, i)


def exploration_seeds(task, collection, start, episodes, **_):
    if collection != 0 or not 0 <= episodes <= 10:
        raise ValueError("Use at most ten starts on the released training split")
    count = len(study().config["records"][task])
    return tuple((start + i) % count for i in range(episodes))


def task_context(config, demonstration, *, live_formal):
    setting = study().config["setting"]
    return (f"# Task: {config.task}\n\n"
            "Use the episode's instruction and the complete conditioning-video sequence returned at reset. "
            "The sequence ends with the first execution observation. Front and wrist cameras are available. "
            "Native task deadlines, step budgets and outcomes are authoritative.\n\n"
            f"Setting: {setting}. Released training recordings are "
            + ("unavailable. " if setting == "AfI" else "available as ordered visual evidence. ")
            + "Every simulator start, preview, rehearsal and verifier attempt consumes the same per-task budget. "
            "Use only the provided training episode indices; no task source or hidden object state. "
            "Freeze the controller before held-out evaluation; do not adapt from test results.\n"
            + ("AfD has zero simulator attempts. Register and submit your offline-developed bundle without rehearsal.\n"
               if setting == "AfD" else ""))


def refresh_workspace(config, workspace):
    pass


def reference_mounts(config):
    return []


def fetch_demonstration(demos_root, task, kind):
    if kind != "none":
        study().require_demos()


def provision_demonstration(config, target, demos_root):
    s = study()
    target = Path(target)
    if s.config["setting"] == "AfI":
        target.mkdir(parents=True, exist_ok=True)
        if any(target.iterdir()):
            raise ValueError("AfI demonstration directory must be empty")
        return dict(kind="none", image_count=0, root="runtime/demonstrations", manifest_path=None)
    from benchmark_io.demonstrations import export_task
    manifest = export_task(s, config.task, os.environ.get("ROBOMME_PLUS_DEMOS", "data/robomme_plus_train/h5"), target)
    return dict(kind="completion_sequence", image_count=manifest["image_count"],
                demonstration_count=len(manifest["demonstrations"]), root="runtime/demonstrations",
                manifest_path="runtime/demonstrations/manifest.json")


def episode_reviewer(root):
    return None


def review_analysis():
    return None


def unscoped_call(name, arguments):
    raise ValueError(f"Unsupported unscoped tool: {name}")

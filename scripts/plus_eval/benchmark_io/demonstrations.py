"""Offline conversion of released demos, with explicit phase and source provenance."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def export_task(study, task, source_root, target):
    study.require_demos()  # Gate BEFORE reading any demonstration file.
    if task not in study.config["records"]:
        raise ValueError("Unknown study task")
    target = Path(target)
    source = Path(source_root) / f"record_dataset_{task}.h5"
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # Validate the current study even when the visual export is cached.
    with h5py.File(source, "r") as h5:
        for episode, row in enumerate(study.config["records"][task]):
            src = h5[f"episode_{episode}"]
            if int(src["setup/seed"][()]) != row["seed"]:
                raise ValueError("Training demonstration and seed metadata differ")
            saved = json.loads(src["setup/episode_metadata"][()].decode())
            for key in ("difficulty", "variant", "perceptual_variation", "mechanism_variation"):
                if saved.get(key) != row.get(key):
                    raise ValueError(f"Training demonstration has stale {key}")
    existing = target / "manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        if manifest["source_sha256"] != digest:
            raise ValueError("Demonstrations changed since export")
        return manifest
    target.mkdir(parents=True, exist_ok=True)
    episodes = []
    with h5py.File(source, "r") as h5:
        for episode, row in enumerate(study.config["records"][task]):
            src = h5[f"episode_{episode}"]
            dest = target / f"episode_{episode}"
            dest.mkdir(exist_ok=True)
            goal = src["setup/task_goal"][()]
            if isinstance(goal, np.ndarray):
                goal = goal.reshape(-1)[0]
            if isinstance(goal, bytes):
                goal = goal.decode()
            frames = []
            keys = sorted((k for k in src if k.startswith("timestep_")), key=lambda k: int(k.split("_")[-1]))
            for i, key in enumerate(keys):
                step = src[key]
                image = np.concatenate([step[f"obs/{c}_rgb"][()] for c in ("front", "wrist")], axis=1)
                relative = f"episode_{episode}/frame_{i:06d}.jpg"
                Image.fromarray(image).save(target / relative, quality=95)
                frames.append(dict(file=relative, frame=i, timestamp_s=i / 20,
                                   phase="video_demo" if bool(step["info/is_video_demo"][()]) else "expert_execution"))
            episodes.append(dict(episode=episode, goal=str(goal), control_hz=20, frames=frames,
                                 ordered_images=[f["file"] for f in frames]))
    manifest = dict(schema_version=1, source="released_robomme_plus_training", source_sha256=digest,
                    task=task, context_kind="completion_sequence", camera="front_left_wrist_right",
                    contains=["rgb_images", "goal", "phase", "recording_time"], demonstrations=episodes,
                    ordered_images=[f for e in episodes for f in e["ordered_images"]])
    manifest["image_count"] = len(manifest["ordered_images"])
    temporary = target / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(existing)
    return manifest

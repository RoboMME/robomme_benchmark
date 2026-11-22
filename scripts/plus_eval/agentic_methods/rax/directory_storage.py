"""Local directory workspaces for RoboMME-Plus; no loop-device mounts.

The native workspace size is checked at launch/resume, not a block-device quota.
Container mounts and isolation remain controlled by the original RAX launcher.
"""
import json
import os
from pathlib import Path


def provision(root, image, megabytes):
    from services.storage_root import artifact_root
    root = Path(root).resolve()
    if not root.is_relative_to(artifact_root()) or type(megabytes) is not int or megabytes < 64:
        raise ValueError('Invalid private workspace root or size')
    target = root/'agent-storage'
    target.mkdir(mode=0o700)
    (root/'agent-storage.json').write_text(json.dumps(dict(
        backend='directory', workspace_mb=megabytes, image=image, hard_quota=False)))
    return target


def inspect(root, metadata):
    target = Path(root).resolve()/'agent-storage'
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError('Missing private directory workspace')
    used = 0
    for directory, dirs, files in os.walk(target, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            path = Path(directory)/name
            if not path.is_symlink():
                try: used += path.stat().st_size
                except FileNotFoundError: pass
    return dict(metadata, mounted=False, used_bytes=used,
                capacity_bytes=metadata['workspace_mb']*1024**2,
                available_bytes=max(0, metadata['workspace_mb']*1024**2-used))


def validate(root, megabytes, workspace, home, status):
    target = Path(root).resolve()/'agent-storage'
    if status['workspace_mb'] != megabytes or status['used_bytes'] > megabytes*1024**2:
        raise RuntimeError('Workspace exceeds the configured startup size limit')
    for path in (workspace, home):
        if not Path(path).resolve().is_relative_to(target) or not Path(path).is_dir():
            raise RuntimeError('Writable path escaped the private workspace directory')
    return status

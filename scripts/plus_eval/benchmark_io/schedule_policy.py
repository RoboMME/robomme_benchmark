"""Operator scheduling controls; never change a policy or simulator allowance."""
import json
from pathlib import Path
import sqlite3

from .study_configurations import configurations


def development_slots(root):
    """Reload operator concurrency independently of experiment/model settings."""
    path = Path(root)/'schedule_policy.json'
    policy = json.loads(path.read_text()) if path.exists() else {}
    total = policy.get('development_workers_total', 6)
    per_gpu = policy.get('development_workers_per_gpu', 3)
    if (type(total) is not int or type(per_gpu) is not int
            or not 1 <= total <= 18 or not 1 <= per_gpu <= 18 or total > 2*per_gpu):
        raise ValueError('Invalid development concurrency for two GPUs')
    return total, per_gpu


def evaluation_slots(root):
    """Legacy total/maximum interface for callers with a scalar GPU limit."""
    total, limits = evaluation_placement(root)
    return total, max(limits.values())


def evaluation_placement(root):
    """Reload shared evaluation capacity and explicit limits for each GPU."""
    path = Path(root)/'schedule_policy.json'
    policy = json.loads(path.read_text()) if path.exists() else {}
    total = policy.get('evaluation_workers_total', 16)
    limits = policy.get('evaluation_workers_by_gpu')
    if limits is None:
        per_gpu = policy.get('evaluation_workers_per_gpu', 8)
        limits = {'0': per_gpu, '1': per_gpu}
    if (type(total) is not int or not 1 <= total <= 18
            or not isinstance(limits, dict) or set(limits) != {'0', '1'}
            or any(type(n) is not int or not 0 <= n <= 18 for n in limits.values())
            or total > sum(limits.values())):
        raise ValueError('Invalid evaluation concurrency for two GPUs')
    return total, limits


def evaluation_gate(root):
    root = Path(root)
    path = root/'schedule_policy.json'
    mode = json.loads(path.read_text()).get('evaluation_mode') if path.exists() else 'asynchronous'
    if mode not in ('asynchronous', 'after_all_development', 'held'):
        raise ValueError('Unknown evaluation scheduling mode')
    completed = 0
    expected = {key for key, _, _, _ in configurations(root)}
    total = len(expected)
    if (root/'queue.sqlite3').is_file():
        with sqlite3.connect(root/'queue.sqlite3') as db:
            # The marker alone does not acknowledge a finished worker. Require
            # both the frozen handoff and successful development job completion.
            finished = db.execute("SELECT c.key FROM cases c JOIN jobs j ON j.case_key=c.key "
                "WHERE j.kind='develop' AND j.state='done' AND c.ready=1").fetchall()
            completed = len(expected.intersection(row[0] for row in finished))
    enabled = mode == 'asynchronous' or (mode == 'after_all_development' and completed == total)
    return dict(mode=mode, enabled=enabled, development_completed=completed,
                development_total=total)


def evaluation_excluded_cases(root):
    """Allow completed cases to run while another awaits an operator review."""
    path = Path(root)/'schedule_policy.json'
    policy = json.loads(path.read_text()) if path.exists() else {}
    excluded = policy.get('evaluation_excluded_cases', [])
    if not isinstance(excluded, list) or any(not isinstance(key, str) or not key for key in excluded):
        raise ValueError('evaluation_excluded_cases must be a list of case keys')
    return sorted(set(excluded))


def evaluation_episode_indices(root):
    """Optional dispatch subset; frozen episode permissions are never expanded."""
    path = Path(root)/'schedule_policy.json'
    policy = json.loads(path.read_text()) if path.exists() else {}
    indices = policy.get('evaluation_episode_indices')
    if indices is None:
        return None
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < 50 for i in indices):
        raise ValueError('evaluation_episode_indices must contain released test indices 0–49')
    return sorted(set(indices))

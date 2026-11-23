"""Operator-owned per-task development clocks (not simulator step limits).

The clock lives beside the training ledger, outside the agent's mounts. Resuming
a conversation never resets it. Reporting after expiry cannot buy coding time.
"""
import fcntl
import json
import os
from pathlib import Path
import time

TASK_SECONDS = 150 * 60
WRAPUP_SECONDS = 120
WRAPUP = (
    "The 150-minute per-task development deadline has been reached. Wrap up the "
    "policy now; no more modifications, debugging, verification or simulator "
    "interactions. Preserve the existing code. Report its current status and "
    "limitations. You may submit an existing eligible policy using the native "
    "submission rules, but may not run another rehearsal or change its code."
)


class DevelopmentExpired(RuntimeError):
    pass


class TaskClock:
    def __init__(self, study, task):
        if Path(task).name != task or task in ('.', '..'):
            raise ValueError('Invalid task name')
        self.directory = Path(study) / 'development_clocks'
        self.path = self.directory / (task + '.json')
        self.task = task

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def update(self, **changes):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / (self.task + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            row = self.read() or {'task': self.task}
            row.update(changes)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(row, indent=2))
            os.replace(temporary, self.path)
        return row

    def start(self, started=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / (self.task + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            row = self.read()
            if row is None:
                started = time.time() if started is None else started
                row = dict(task=self.task, started=started, seconds=TASK_SECONDS,
                           deadline=started + TASK_SECONDS)
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(row, indent=2))
                os.replace(temporary, self.path)
        return row

    def remaining(self, now=None):
        row = self.read()
        if row is None:
            return TASK_SECONDS
        return max(0, row['deadline'] - (time.time() if now is None else now))

    def require_open(self):
        if self.read() is not None and self.remaining() <= 0:
            raise DevelopmentExpired(f'{self.task}: {WRAPUP}')

    def expire(self):
        if self.remaining() > 0:
            raise ValueError('Cannot expire a task before its deadline')
        row = self.read()
        return row if row.get('expired_at') else self.update(expired_at=time.time(), wrapup_prompt=WRAPUP)


def rax_clock(config):
    study = os.environ.get('ROBOMME_PLUS_STUDY')
    if not study or config.benchmark != 'robomme_plus':
        return None
    clock = TaskClock(study, config.task)
    # The process record predates provisioning and survives operator restarts.
    record = Path(study).parent / 'logs' / f'develop_{config.task}_process.json'
    started = json.loads(record.read_text()).get('started') if record.exists() else None
    clock.start(started)
    return clock

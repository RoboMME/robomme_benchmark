"""Cross-process GPU slots shared by all frozen evaluation queues."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import threading
import time


class EvaluationSlots:
    def __init__(self, root, gpus, workers_per_gpu):
        if not gpus or len(set(gpus)) != len(gpus) or workers_per_gpu < 1:
            raise ValueError('Use distinct GPUs and a positive evaluation worker count')
        self.root = Path(root) / 'runtime/evaluation_slots'
        self.root.mkdir(parents=True, exist_ok=True)
        # Interleave GPUs so a partial wave uses both devices.
        self.slots = [(str(gpu), index) for index in range(workers_per_gpu) for gpu in gpus]
        self.capacity = len(self.slots)

    @contextmanager
    def acquire(self, owner, *, gpu=None):
        candidates = [slot for slot in self.slots if gpu is None or slot[0] == str(gpu)]
        if not candidates:
            raise ValueError('Requested GPU is outside the evaluation pool')
        while True:
            for gpu, index in candidates:
                path = self.root / f'gpu{gpu}_{index}.lock'
                stream = path.open('a+')
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    stream.seek(0)
                    stream.truncate()
                    json.dump(dict(owner=owner, gpu=gpu, slot=index, pid=os.getpid(),
                        thread=threading.get_ident(), started=time.time()), stream)
                    stream.flush()
                    yield gpu, index
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
            time.sleep(.2)


def active_slots(root):
    """Read only currently locked slots, ignoring stale metadata after release."""
    rows = []
    for path in sorted((Path(root)/'runtime/evaluation_slots').glob('*.lock')):
        with path.open('r') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                try:
                    rows.append(json.load(stream))
                except json.JSONDecodeError:
                    pass  # An owner is updating its slot record.
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
    return rows

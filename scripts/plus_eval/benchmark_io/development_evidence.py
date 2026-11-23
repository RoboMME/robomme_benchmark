"""Operator-owned tool events and content-addressed development snapshots.

Only the hook client and a Unix socket enter the agent container. Evidence and
source objects stay on the host. These records supplement native/API transcripts.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socketserver
import threading
import tempfile
import time
import uuid


MEDIA = {'.jpg', '.jpeg', '.png', '.mp4', '.webm', '.h5', '.hdf5', '.parquet', '.npy', '.npz'}
EXCLUDED = {'demonstrations', 'published', 'runtime', '__pycache__', '.git', '.venv', 'node_modules'}


class EvidenceRecorder:
    def __init__(self, root, workspace):
        self.root, self.workspace = Path(root).resolve(), Path(workspace).resolve()
        (self.root / 'objects').mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def record(self, event):
        with self.lock, (self.root/'record.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            files = {}
            for directory, dirs, names in os.walk(self.workspace, followlinks=False):
                dirs[:] = [d for d in dirs if d not in EXCLUDED and not (Path(directory)/d).is_symlink()]
                for name in names:
                    path = Path(directory)/name
                    if path.is_symlink() or not path.is_file() or path.suffix.lower() in MEDIA or path.suffix == '.pyc':
                        continue
                    try:
                        content = path.read_bytes()
                    except FileNotFoundError:
                        continue
                    digest = hashlib.sha256(content).hexdigest()
                    target = self.root/'objects'/digest
                    if not target.exists():
                        # Content-addressed: a partial write must never occupy the digest name.
                        partial = target.with_name(f'{digest}.{uuid.uuid4().hex}.tmp')
                        partial.write_bytes(content)
                        partial.replace(target)
                    files[str(path.relative_to(self.workspace))] = digest
            revision = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            record = dict(id=uuid.uuid4().hex, time=time.time(), revision=revision, files=files, event=event)
            with (self.root/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary = self.root/'latest.json.tmp'
            temporary.write_text(json.dumps({k:record[k] for k in ('id','time','revision')}))
            temporary.replace(self.root/'latest.json')
            return revision


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline(64*1024**2 + 1)
            if len(raw) > 64*1024**2 or not raw.endswith(b'\n'):
                raise ValueError('Oversized or incomplete audit event')
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise ValueError('Audit event must be an object')
            revision = self.server.recorder.record(event)
            reply = dict(ok=True, revision=revision)
        except Exception as exc:
            reply = dict(ok=False, error=type(exc).__name__)
        self.wfile.write(json.dumps(reply).encode()+b'\n')


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@contextmanager
def capture(root, workspace, settings_file):
    """Install synchronous hooks, then serve them for one native agent launch."""
    root = Path(root).resolve()
    recorder = EvidenceRecorder(root/'history', workspace)
    # Unix sockets have a short path limit; experiment/task names can be long.
    socket_context = tempfile.TemporaryDirectory(prefix='robomme-audit-')
    socket_dir = Path(socket_context.name)
    path = socket_dir/'audit.sock'
    if path.exists():
        path.unlink()  # Caller owns this task/campaign's single worker lock.
    settings_file = Path(settings_file)
    settings = json.loads(settings_file.read_text())
    hooks = settings.setdefault('hooks', {})
    command = '/usr/local/bin/python /opt/robomme-audit/hook.py'
    for name in ('Setup','SessionStart','PreToolUse','PostToolUse','PostToolUseFailure','SubagentStart','SubagentStop','Stop','SessionEnd'):
        entries = hooks.setdefault(name, [])
        if not any(any(h.get('command') == command for h in e.get('hooks', [])) for e in entries):
            entries.append(dict(matcher='', hooks=[dict(type='command', command=command, timeout=120)]))
    settings_file.write_text(json.dumps(settings, indent=2))
    mounts = [(socket_dir, '/run/robomme-audit', True),
              (Path(__file__).with_name('evidence_hook.py'), '/opt/robomme-audit/hook.py', True)]
    recorder.record(dict(hook_event_name='OperatorStart'))
    try:
        with _Server(str(path), _Handler) as server:
            server.recorder = recorder
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield mounts
            finally:
                server.shutdown()
                thread.join(5)
                recorder.record(dict(hook_event_name='OperatorStop'))
    finally:
        socket_context.cleanup()


def latest_revision(root):
    path = Path(root)/'history/latest.json'
    if not path.is_file():
        raise RuntimeError('Development code must be archived before a simulator start')
    return json.loads(path.read_text())['revision']

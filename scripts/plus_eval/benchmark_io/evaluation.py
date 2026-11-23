"""Operator-only frozen evaluation manifest, separate from training attempts.

The developing agent never receives this capability or these outcomes. Each
authorized held-out episode can be reserved once against immutable policy files.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

from .protocol import Study


def execution_hashes(root):
    root = Path(root)
    paths = [p for directory in ('runtime_source', 'protocol') for p in (root/directory).rglob('*')
             if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts]
    paths += [root/name for name in ('runtime_policy.json', 'study_plan.json')]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Frozen execution files must not be symlinks')
    # RAX creates this empty synchronization file when its storage helper first
    # runs. It contains no code or configuration. Keep all other runtime files
    # pinned, and reject a symlink or unexpected contents instead of ignoring it.
    storage_lock = root/'runtime_source/rax/runtime/workspace-storage.lock'
    if storage_lock in paths:
        if storage_lock.read_bytes() != b'':
            raise ValueError('RAX workspace storage lock must be empty')
        paths.remove(storage_lock)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


class EvaluationStudy(Study):
    dataset = 'test'

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'frozen_evaluation.json').read_text())
        self.config = self.manifest['config']
        self.path = self.root / 'evaluation_attempts.sqlite3'
        self.verify_frozen()
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, task TEXT NOT NULL, '
                'episode INTEGER NOT NULL, seed INTEGER NOT NULL, purpose TEXT NOT NULL, '
                'started REAL NOT NULL, ended REAL, outcome TEXT, UNIQUE(task,episode))')

    @classmethod
    def freeze(cls, root, *, development_study, policy_directory, episodes, benchmark_root, artifact_profile="full"):
        if artifact_profile not in ('full', 'logs'):
            raise ValueError('Unknown evaluation artifact profile')
        root = Path(root).resolve()
        if root.exists():
            raise FileExistsError('Never overwrite a frozen evaluation')
        root.mkdir(parents=True)
        source = Path(policy_directory).resolve()
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Frozen policy must not contain symlinks')
        shutil.copytree(source, root / 'policy')
        hashes = {str(p.relative_to(root / 'policy')): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((root / 'policy').rglob('*')) if p.is_file()}
        if not hashes:
            raise ValueError('Cannot freeze an empty policy')
        records = {}
        from robomme_plus import BenchmarkEnvBuilder
        for task, indices in episodes.items():
            builder = BenchmarkEnvBuilder(task, dataset='test', action_space='joint_angle')
            if not indices or any(type(i) is not int or not 0 <= i < 50 for i in indices):
                raise ValueError('Only released test episode indices are authorized')
            records[task] = [builder.get_episode_metadata(i) for i in range(50)]
        study = Study.open(development_study)
        manifest = dict(kind='frozen_operator_only_evaluation', created=time.time(),
            development_study=str(study.root), policy_sha256=hashes, episodes=episodes,
            config=dict(method=study.config['method'], setting=study.config['setting'], records=records,
                        task_order=sorted(study.config['records']), artifact_profile=artifact_profile),
            development_attempt_count=len(study.attempts()), test_feedback_to_agent=False)
        # Fresh studies pin the whole harness/protocol, not only the selected code.
        for parent in study.root.parents:
            if (parent/'study_plan.json').is_file() and (parent/'runtime_source').is_dir():
                manifest['execution_root'] = str(parent)
                manifest['execution_sha256'] = execution_hashes(parent)
                break
        (root / 'frozen_evaluation.json').write_text(json.dumps(manifest, indent=2))
        from .protocol import source_fingerprint
        (root / 'benchmark_source.json').write_text(json.dumps(source_fingerprint(benchmark_root), indent=2))
        return cls(root)

    def verify_frozen(self):
        files = {str(p.relative_to(self.root / 'policy')): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((self.root / 'policy').rglob('*')) if p.is_file()}
        if files != self.manifest['policy_sha256']:
            raise ValueError('Frozen policy files changed')
        if 'execution_sha256' in self.manifest:
            root = Path(self.manifest['execution_root'])
            if execution_hashes(root) != self.manifest['execution_sha256']:
                raise ValueError('Frozen harness/protocol changed')
        study = Study.open(self.manifest['development_study'])
        if len(study.attempts()) != self.manifest['development_attempt_count']:
            raise ValueError('Development continued after this evaluation was frozen')

    def reserve(self, task, episode=None, *, purpose):
        self.verify_frozen()
        if episode not in self.manifest['episodes'].get(task, []):
            raise PermissionError('Held-out episode not authorized by the frozen manifest')
        row = self.config['records'][task][episode]
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM attempts WHERE task=? AND episode=?', (task, episode)).fetchone():
                raise PermissionError('Held-out episode already reserved; do not overwrite its result')
            cursor = db.execute('INSERT INTO attempts(task,episode,seed,purpose,started) VALUES(?,?,?,?,?)',
                                (task, episode, row['seed'], purpose, time.time()))
            return dict(id=cursor.lastrowid, task=task, episode=episode, seed=row['seed'],
                        purpose=purpose, ordinal=1, dataset='test')

    def require_demos(self):
        raise PermissionError('Frozen evaluation does not expose training recordings')

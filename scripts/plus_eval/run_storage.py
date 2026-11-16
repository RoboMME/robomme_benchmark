"""Episode selection, recoverable replacement and summaries for RoboMME-Plus runs.

No simulator, model or API imports: saved runs can be repaired offline.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import uuid

from demonstration_sampling import CAPPED_METHODS, MAX_DEMONSTRATION_FRAMES
from astra_image_sampling import MAX_DEMONSTRATION_IMAGES, MAX_ONE_SHOT_CONTEXT_IMAGES, image_count

OPERATIONAL = {'tasks', 'episodes', 'exclude_tasks', 'output_dir', 'output_root',
               'resume', 'retry_errors', 'rerun', 'fail_fast', 'api_pricing_json',
               'host', 'port', 'gpu', 'workers', 'vla_python', 'simplememvla_python'}


def read(path):
    return json.loads(Path(path).read_text())


def mode_from_args(args):
    return next((name for name in ('resume', 'retry_errors', 'rerun') if getattr(args, name, False)), 'new')


def add_run_modes(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true', help='Run missing, interrupted, errored, incomplete or outdated-demo episodes; keep compatible outcomes.')
    group.add_argument('--retry-errors', action='store_true', help='Retry selected saved errors only (use --resume for unfinished episodes too).')
    group.add_argument('--rerun', action='store_true', help='Replace every selected episode, archiving previous rollouts.')


@contextmanager
def run_lock(directory, name='.evaluation.lock'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another worker owns this output directory') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def key(case):
    return case['task'], case['episode']


def episode_name(case):
    return f"{case['task']}_ep{case['episode']}"


def saved_result(directory, case):
    try:
        result = read(Path(directory) / episode_name(case) / 'result.json')
        if key(result) != key(case) or result.get('status') not in ('success', 'fail', 'timeout', 'error'):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def valid_outcome(directory, case, result=None):
    result = saved_result(directory, case) if result is None else result
    if not result or result.get('interrupted') or result['status'] == 'error':
        return False
    if result.get('artifacts_complete') is False or result.get('artifact_errors'):
        return False
    episode = Path(directory) / episode_name(case)
    steps = episode / 'steps.jsonl'
    if not steps.is_file() or (result.get('env_step_calls', result.get('steps', 0)) and not steps.stat().st_size):
        return False
    return all((episode / name).is_file() and (episode / name).stat().st_size > 0
               for name in ('episode.mp4', 'actions.npy', 'frames.jsonl'))


def requires_demonstration_resampling(directory, case, result):
    """Reject outcomes whose demo preprocessing differs from this method's policy."""
    if not result:
        return False
    sampling = result.get('demonstration_preprocessing')
    method = result.get('method')
    if method is None:
        config = Path(directory) / 'config.json'
        method = read(config).get('method') if config.exists() else None
    if method not in CAPPED_METHODS:
        # Repair outcomes produced when the FrameSamp-only cap was mistakenly
        # applied in the shared runner. Unchanged short demos remain reusable.
        if sampling and (sampling.get('applied') or
                         sampling.get('input_demo_frames') != sampling.get('source_demo_frames')):
            return True
        if method in ('embodied-astra', 'embodied-astra-fs'):
            limit = (MAX_ONE_SHOT_CONTEXT_IMAGES if method == 'embodied-astra-fs'
                     else MAX_DEMONSTRATION_IMAGES)
            if method == 'embodied-astra-fs':
                report = result.get('one_shot_adaptation')
                if report is not None and report.get('first_turn_images') is not None:
                    return report['first_turn_images'] > limit
            report = result.get('astra_demonstration_sampling')
            if report is not None and method == 'embodied-astra':
                return report['input_images'] > limit
            # Legacy results have the exact initial request saved. Inspect it
            # rather than guessing image counts from simulator frame rates.
            calls = Path(directory) / episode_name(case) / 'calls'
            initial = next(iter(sorted(calls.glob('*_astra_api/input.json'))), None)
            if initial is not None:
                try:
                    return image_count(read(initial)) > limit
                except (OSError, ValueError):
                    return True
        return False
    if sampling is not None:
        return sampling['input_demo_frames'] > MAX_DEMONSTRATION_FRAMES
    try:
        with (Path(directory) / episode_name(case) / 'frames.jsonl').open() as file:
            first = json.loads(next(file))
        return first.get('phase') == 'demonstration' and first.get('batch_size', 0) - 1 > MAX_DEMONSTRATION_FRAMES
    except (OSError, ValueError, StopIteration):
        return False  # Missing/incomplete artifacts are handled by valid_outcome.


def compatible_metadata(directory, case, expected):
    if expected is None:
        return True
    try:
        from robomme_plus.evaluation.metadata import replay_identity
        saved = read(Path(directory) / episode_name(case) / 'metadata.json')
        return replay_identity(case['task'], saved) == replay_identity(case['task'], expected)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def benchmark_sources():
    root = Path(__file__).resolve().parents[2]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / 'src/robomme_plus').rglob('*.py'))}


def check_benchmark_sources(old, new, mode):
    old_sources, new_sources = old.get('benchmark_source_sha256'), new.get('benchmark_source_sha256')
    if old_sources != new_sources and (old_sources or new_sources) and mode != 'rerun':
        raise ValueError('Benchmark source changed. Use --rerun for the selected episodes or a new '
                         'output directory; saved outcomes cannot establish compatibility with new task code.')


def select_pending(directory, cases, mode, episode_metadata=None, source_hashes=None):
    selected = []
    state_path = Path(directory) / 'run_state.json'
    state = read(state_path) if mode == 'resume' and state_path.exists() else {}
    forced = {key(c) for c in state.get('pending', [])}
    config_path = Path(directory) / 'config.json'
    config = read(config_path) if source_hashes is not None and config_path.exists() else {}
    snapshots = {}
    for case in cases:
        result = saved_result(directory, case)
        expected = (episode_metadata or {}).get(episode_name(case))
        compatible_source = True
        if source_hashes is not None and result is not None:
            relative = config.get('case_provenance', {}).get(episode_name(case))
            if relative not in snapshots:
                try:
                    snapshot = read(_inside(Path(directory), relative)) if relative else config
                    snapshots[relative] = snapshot.get('benchmark_source_sha256') == source_hashes
                except (OSError, ValueError, TypeError):
                    snapshots[relative] = False
            compatible_source = snapshots[relative]
        already_published = result and result.get('attempt_id') == state.get('attempt_id') and valid_outcome(directory, case, result)
        if (mode in ('new', 'rerun') or
                (mode == 'resume' and ((key(case) in forced and not already_published)
                                      or not valid_outcome(directory, case, result)
                                      or not compatible_metadata(directory, case, expected)
                                      or not compatible_source
                                      or requires_demonstration_resampling(directory, case, result))) or
                (mode == 'retry_errors' and result and result['status'] == 'error' and not result.get('interrupted'))):
            selected.append(case)
    return selected


def argument_identity(arguments):
    # Paths are compared without persisting personal directories or credentials.
    identity = {}
    for name, value in arguments.items():
        if name in OPERATIONAL:
            continue
        if isinstance(value, Path):
            identity[name] = {'path_sha256': hashlib.sha256(str(value.expanduser().absolute()).encode()).hexdigest()}
        else:
            identity[name] = value
    return identity


def check_configuration(old, new, mode='resume'):
    if old['method'] != new['method']:
        raise ValueError('Existing run belongs to a different method')
    check_benchmark_sources(old, new, mode)
    if new['method'] == 'embodied-astra-fs':
        current_context = new['arguments'].get('training_observation_context_version')
        previous_context = old.get('arguments', {}).get('training_observation_context_version', 1)
        if current_context is not None and current_context != previous_context:
            raise ValueError('Training observation context changed. Use a new output directory '
                             'to keep one-shot runs with different prompts separate.')
    previous = old.get('argument_identity')
    current = argument_identity(new['arguments'])
    if previous is None:
        # Legacy configs have redacted paths. Compare all remaining known settings.
        previous = {k: v for k, v in old.get('arguments', {}).items()
                    if k not in OPERATIONAL and v != '[LOCAL_PATH]'}
    for name, value in previous.items():
        if name in current and current[name] != value:
            raise ValueError(f'Inference setting changed: {name}. Use a new output directory for a different model configuration.')


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Invalid replacement journal path')
    return path


def recover_replacement(directory):
    """Recover a process killed between the old/new directory renames."""
    directory = Path(directory)
    journal = directory / '.replacement.json'
    if not journal.exists():
        return
    record = read(journal)
    target, source, archive = [_inside(directory, record[k]) for k in ('target', 'source', 'archive')]
    if not target.exists() and archive.exists():
        archive.rename(target)  # Preserve the old result; resume will retry the attempt.
    elif not target.exists() and source.exists():
        source.rename(target)
    elif not target.exists():
        raise RuntimeError('Replacement journal has no recoverable episode')
    journal.unlink()


def replace_episode(directory, source, case, attempt):
    from evaluation import write_json
    directory, source = Path(directory).resolve(), Path(source).resolve()
    target = directory / episode_name(case)
    if source == target:
        return
    archive = directory / '.history' / attempt / target.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    journal = directory / '.replacement.json'
    write_json(journal, {k: str(p.relative_to(directory)) for k, p in
                        [('source', source), ('target', target), ('archive', archive)]})
    try:
        if target.exists():
            target.rename(archive)
        source.rename(target)
    except BaseException:
        if not target.exists() and archive.exists():
            archive.rename(target)
        raise
    finally:
        if target.exists():
            journal.unlink(missing_ok=True)


def rebuild_task(directory):
    from evaluation import summarize, write_json, utc_now
    directory = Path(directory)
    config = read(directory / 'config.json')
    results = [r for case in config['cases'] if (r := saved_result(directory, case)) is not None]
    report = summarize(config['cases'], results, config['method'])
    state = directory / 'run_state.json'
    report['pending_replacements'] = read(state).get('pending', []) if state.exists() else []
    report['updated_utc'] = utc_now()
    write_json(directory / 'summary.json', report)
    return report


def rebuild_batch(directory, batch=None, *, refresh_tasks=True):
    """Aggregate once per episode. Live coordinators only read worker summaries.

    Rebuilding task files is safe only after their workers have stopped; each
    worker owns its own summary's atomic publication while evaluating.
    """
    from evaluation import summarize, write_json, utc_now
    directory = Path(directory)
    batch = read(directory / 'batch.json') if batch is None else batch
    cases, results = [], []
    for task, row in batch['tasks'].items():
        task_dir = directory / task
        expected = [{'task': task, 'episode': ep} for ep in row['episodes']]
        cases.extend(expected)
        if not (task_dir / 'config.json').exists():
            continue
        if refresh_tasks:
            report = rebuild_task(task_dir)
        elif (task_dir / 'summary.json').exists():
            report = read(task_dir / 'summary.json')
        else:
            continue
        allowed = {key(c) for c in expected}
        results.extend(r for r in report['results'] if key(r) in allowed)
        row.update(completed=report['completed'], counts=report['counts'],
                   success_rate=report['success_rate'], metrics=report['metrics'])
    report = summarize(cases, results, batch['method'])
    report['updated_utc'] = utc_now()
    if 'workers' in batch:
        report['execution'] = {name: batch[name] for name in
            ('workers', 'peak_concurrent_tasks', 'invocation_started_utc', 'invocation_elapsed_seconds')
            if name in batch}
    write_json(directory / 'summary.json', report)
    for name in ('requested', 'completed', 'pending', 'counts', 'success_rate', 'metrics'):
        batch[name] = report[name]
    batch['updated_utc'] = report['updated_utc']
    write_json(directory / 'batch.json', batch)
    return report


class RunSession:
    """Caller holds the run lock for this session's entire lifetime."""
    def __init__(self, directory, config, mode):
        from evaluation import write_json, utc_now
        self.directory = Path(directory)
        recover_replacement(self.directory)
        path = self.directory / 'config.json'
        old = read(path) if path.exists() else None
        if mode == 'new' and any(p.name != '.evaluation.lock' for p in self.directory.iterdir()):
            raise ValueError('Output exists. Use --resume, --retry-errors or --rerun explicitly.')
        if old:
            check_configuration(old, config, mode)
        elif any(not p.name.startswith('.') for p in self.directory.iterdir()):
            raise ValueError('Cannot reuse a nonempty directory without config.json')
        self.id = utc_now().replace(':', '').replace('+', '-') + '-' + uuid.uuid4().hex[:8]
        self.attempt = self.directory / '.attempts' / self.id
        self.attempt.mkdir(parents=True)
        self.selected = config['cases']
        self.todo = select_pending(self.directory, self.selected, mode, config.get('episode_metadata'),
                                   config.get('benchmark_source_sha256'))
        old_state = read(self.directory / 'run_state.json') if (self.directory / 'run_state.json').exists() else {}
        selected_keys = {key(c) for c in self.selected}
        self.deferred = [c for c in old_state.get('pending', []) if key(c) not in selected_keys]
        self.remaining = list(self.todo)
        if old_state:
            write_json(self.attempt / 'previous_run_state.json', old_state)
        config['argument_identity'] = argument_identity(config['arguments'])
        config['attempt_id'] = self.id
        config['attempt_started_utc'] = utc_now()
        config['operation'] = mode
        if old:
            write_json(self.attempt / 'previous_config.json', old)
            config['started_utc'] = old['started_utc']
        write_json(self.attempt / 'config.json', config)
        self.config = dict(config)
        for name in ('policy_configuration', 'model_setup_seconds'):
            if old and name in old:
                self.config[name] = old[name]
        merged = {key(c): c for c in (old or {}).get('cases', [])}
        merged.update({key(c): c for c in self.selected})
        self.config['cases'] = list(merged.values())
        # Retained episodes keep the configuration that actually produced them,
        # including when a subset is rerun with a newer benchmark revision.
        self.config['case_provenance'] = dict((old or {}).get('case_provenance', {}))
        if old:
            previous = str((self.attempt / 'previous_config.json').relative_to(self.directory))
            for case in old.get('cases', []):
                self.config['case_provenance'].setdefault(episode_name(case), previous)
        self.config['updated_utc'] = utc_now()
        write_json(path, self.config)
        rebuild_task(self.directory)

    def episode_directory(self, case):
        target = self.directory / episode_name(case)
        return self.attempt / target.name if target.exists() else target

    @property
    def pending(self):
        return self.deferred + self.remaining

    def record(self, case, result, path):
        from evaluation import write_json
        result['attempt_id'] = self.id
        write_json(path / 'result.json', result)
        # An interrupted replacement must not displace a finished old rollout.
        if not result.get('interrupted'):
            replace_episode(self.directory, path, case, self.id)
            self.remaining = [c for c in self.remaining if key(c) != key(case)]
            self.config['case_provenance'][episode_name(case)] = str(
                (self.attempt / 'config.json').relative_to(self.directory))
            write_json(self.directory / 'config.json', self.config)
        return rebuild_task(self.directory)

    def setup_complete(self, seconds, policy_config):
        from evaluation import write_json
        self.config.update(model_setup_seconds=seconds, policy_configuration=policy_config)
        write_json(self.directory / 'config.json', self.config)
        snapshot = read(self.attempt / 'config.json')
        snapshot.update(model_setup_seconds=seconds, policy_configuration=policy_config)
        write_json(self.attempt / 'config.json', snapshot)

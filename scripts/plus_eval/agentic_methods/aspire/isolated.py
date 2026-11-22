"""Original ASPIRE runtime in an isolated worker, with host-owned public tools.

Only sensor/robot contract data traverses JSON RPC. The worker cannot import the
simulator, reset it, access host files, or reach a network/provider credential.
"""
import json
import hashlib
import os
from pathlib import Path
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
import uuid

from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, TracedProvider
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.pyroki.client import PyRokiProvider
from cap_harness.providers.curobo.client import CuRoboProvider
from cap_harness.registry import ROBOMME_PLUS_PUBLIC_TOOL_NAMES, ToolRegistry
from cap_harness.runtime import ProgramExecutor
from cap_harness.robomme_plus.adapter import RoboMMEPlusAdapter
from benchmark_io.protocol import Study
from benchmark_io.runtime_policy import configured, ProgressWatchdog, observation_memory_gib
from .wire import dumps, loads, write_reply
from .sam3_reliability import ReliableSam3Provider

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'ASPIRE/src/cap_harness'


def communicate_until_episode_end(process, episode, stop_worker, *, timeout=3600, terminal_grace=1, watchdog=None):
    """End execution at the simulator's terminal boundary, even if the policy loops."""
    deadline = time.monotonic() + timeout
    terminal_since = None
    while True:
        now = time.monotonic()
        development_clock = getattr(episode, 'development_clock', None)
        if development_clock and development_clock.remaining() <= 0:
            episode.abort_requested = 'development_deadline'
            stop_worker()
            stdout, stderr = process.communicate(timeout=30)
            return stdout, stderr, 'development_deadline'
        if watchdog is not None and not (episode.terminated or episode.truncated):
            reason = watchdog.check(episode.steps, now=now)
            if reason:
                episode.abort_requested = reason
                stop_worker()
                stdout, stderr = process.communicate(timeout=30)
                return stdout, stderr, reason
        if episode.terminated or episode.truncated:
            if terminal_since is None:
                terminal_since = now
            if now - terminal_since >= terminal_grace:
                reason = 'episode_terminated' if episode.terminated else 'episode_truncated'
                stop_worker()
                stdout, stderr = process.communicate(timeout=30)
                return stdout, stderr, reason
        if now >= deadline:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            stdout, stderr = process.communicate(timeout=min(.25, deadline - now))
            return stdout, stderr, None
        except subprocess.TimeoutExpired:
            pass


def prepare_runtime(destination):
    """Copy only upstream public contracts and runtime, never the simulator/API host."""
    destination = Path(destination)
    package = destination / 'cap_harness'
    package.mkdir(parents=True, exist_ok=True)
    (package / '__init__.py').write_text('')
    for name in ('contracts.py', 'errors.py', 'runtime.py', 'registry.py'):
        shutil.copy2(UPSTREAM / name, package / name)
    for name in ('wire.py', 'program_worker.py'):
        shutil.copy2(HERE / name, destination / name)


class ToolHandler(socketserver.StreamRequestHandler):
    def handle(self):
        name = None
        try:
            request = loads(self.rfile.readline(512 * 1024**2))
            name = request['name']
            if name not in self.server.public_tools:
                raise PermissionError('Tool is not in the public ASPIRE registry')
            if getattr(self.server.episode, 'abort_requested', None):
                raise RuntimeError('Operator watchdog ended this episode')
            if self.server.episode.terminated or self.server.episode.truncated:
                raise RuntimeError('Episode has ended; execution is closing')
            result = self.server.public_tools[name](*request.get('args', []), **request.get('kwargs', {}))
            response = {'value': result}
        except Exception as exc:
            response = {'error': f'{type(exc).__name__}: {exc}'}
        try:
            delivery = write_reply(self.wfile, response,
                                   video=name == 'robomme_plus.get_conditioning_video')
            if name == 'robomme_plus.get_conditioning_video' and getattr(self.server, 'delivery_log', None):
                with self.server.delivery_log.open('a') as log:
                    log.write(json.dumps(dict(time=time.time(), tool=name, **delivery)) + '\n')
        except (BrokenPipeError, ConnectionResetError):
            # The operator may close the worker immediately at episode termination.
            pass


class ToolServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


def run_program(*, study_root, output_root, task, episode, code, image,
                purpose='task_solving_sanity', provider_port_base=18110,
                study_override=None, dataset='train'):
    study = study_override or Study.open(study_root)
    if dataset != getattr(study, 'dataset', 'train'):
        raise ValueError('Requested dataset must match the operator-owned study capability')
    if task not in study.config['records']:
        raise ValueError('Task is not in this study')
    if episode is None:
        attempts = [a for a in study.attempts() if a['task'] == task]
        episode = min(range(len(study.config['records'][task])),
                      key=lambda i: (sum(a['episode'] == i for a in attempts), i))
    if type(episode) is not int or not 0 <= episode < len(study.config['records'][task]):
        raise ValueError('Only released episode indices are allowed')
    limit = study.config['records'][task][episode]['max_steps']
    recorder_type = RunRecorder
    if study.config.get('artifact_profile') == 'logs':
        from .lean_recording import LogRecorder
        recorder_type = LogRecorder
    recorder = recorder_type(output_root=Path(output_root), benchmark='robomme_plus', suite='robomme_plus',
        task_id=study.config.get('task_order', sorted(study.config['records'])).index(task), task_name=task, seed=episode,
        max_steps=limit, camera_width=256, camera_height=256,
        providers={'segmentation': 'sam3', 'grasping': 'contact-graspnet', 'ik': ['pyroki', 'curobo']})
    source = recorder.root / 'submitted_program.py'
    source.write_text(code)
    recorder.save_program(source)
    adapter = RoboMMEPlusAdapter(study_root=study_root, output_root=recorder.root / 'native',
                                 purpose=purpose, run_observer=recorder,
                                 policy_revision=hashlib.sha256(code.encode()).hexdigest() if dataset == 'train' else None)
    if study_override is not None:
        adapter.study = study_override
    execution = None
    error = None
    success = False
    process = None
    terminal_stop = None
    sam3 = None
    container = 'aspire-program-' + uuid.uuid4().hex
    limits = configured()
    try:
        observation = adapter.reset(task, episode)
        if dataset == 'train':
            from benchmark_io.development_deadline import TaskClock
            adapter.episode.development_clock = TaskClock(study_root, task)
        recorder.on_reset(observation, adapter.get_task_metadata())
        url = lambda offset: f'http://127.0.0.1:{provider_port_base + offset}'
        bounded = dict(timeout_s=limits.provider_seconds,max_retries=0) if limits else {}
        sam3 = ReliableSam3Provider(url(4), log_path=recorder.root/'sam3_requests.jsonl',
                                   timeout_s=limits.provider_seconds if limits else 60)
        segmentation = TracedProvider(sam3, recorder, 'sam3')
        grasp = TracedProvider(ContactGraspNetProvider(url(5), **bounded), recorder, 'contact_graspnet')
        pyroki = TracedProvider(PyRokiProvider(url(6), **(bounded or dict(timeout_s=180))), recorder, 'pyroki')
        curobo = TracedProvider(CuRoboProvider(url(8), **(bounded or dict(timeout_s=240)),
            control_period_s=adapter.control_period_s), recorder, 'curobo')
        api = CapApi(adapter, segmentation_provider=segmentation,
            grasp_providers={'contact-graspnet': grasp}, ik_providers={'pyroki': pyroki, 'curobo': curobo},
            trajectory_planning_providers={'curobo': curobo},
            integrated_pose_planning_providers={'curobo-integrated': curobo},
            default_camera='front', recorder=recorder)
        registry = api.register_tools(ToolRegistry(public_extension_allowlist=ROBOMME_PLUS_PUBLIC_TOOL_NAMES))
        host_runtime = ProgramExecutor(registry, recorder=recorder)
        with tempfile.TemporaryDirectory(prefix='aspire-program-') as temporary:
            temp = Path(temporary)
            prepare_runtime(temp / 'runtime')
            (temp / 'rpc').mkdir()
            with ToolServer(str(temp / 'rpc/tool.sock'), ToolHandler) as server:
                server.episode = adapter.episode
                server.delivery_log = recorder.root/'input_delivery.jsonl'
                server.public_tools = {spec.name: host_runtime._wrapped_tool(spec.name, spec.function)
                                       for spec in registry.all_specs(public_only=True)}
                threading.Thread(target=server.serve_forever, daemon=True).start()
                memory_gib = observation_memory_gib(adapter._demonstration,limits) if limits else 4
                if limits:
                    (recorder.root/'resource_limits.json').write_text(json.dumps(dict(
                        memory_gib=memory_gib,controller_seconds=limits.controller_seconds,
                        initial_no_progress_seconds=limits.initial_no_progress_seconds,
                        no_progress_seconds=limits.no_progress_seconds,provider_seconds=limits.provider_seconds)))
                command = ['docker', 'run', '--pull=never', '--name', container,
                    '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
                    '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=128',
                    f'--memory={memory_gib}g', '--cpus=2', '--ipc=private', '--log-driver=none',
                    '--tmpfs', '/tmp:rw,nosuid,nodev,size=128m,mode=1777',
                    '--mount', f'type=bind,src={temp / "runtime"},dst=/runtime,readonly',
                    '--mount', f'type=bind,src={temp / "rpc"},dst=/run/aspire,readonly',
                    '--mount', f'type=bind,src={source},dst=/program/program.py,readonly',
                    '--env', 'PYTHONPATH=/runtime', '--env', 'PYTHONDONTWRITEBYTECODE=1',
                    '--env', 'OPENBLAS_NUM_THREADS=1', '--entrypoint', '/usr/local/bin/python',
                    image, '/runtime/program_worker.py']
                (recorder.root / 'worker_command.json').write_text(json.dumps(command, indent=2))
                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr, terminal_stop = communicate_until_episode_end(
                        process, adapter.episode,
                        lambda: subprocess.run(['docker', 'kill', container], capture_output=True, timeout=30),
                        timeout=limits.controller_seconds+5 if limits else 3600,
                        watchdog=ProgressWatchdog(limits) if limits else None)
                    inspection = subprocess.run(['docker','inspect','--format','{{json .State}}',container],
                                                capture_output=True,timeout=15)
                    if inspection.returncode == 0:
                        state = json.loads(inspection.stdout)
                        (recorder.root/'worker_state.json').write_text(json.dumps(
                            {k:state.get(k) for k in ('Status','ExitCode','OOMKilled','Error','StartedAt','FinishedAt')}))
                    (recorder.root / 'worker_stderr.log').write_bytes(stderr)
                    (recorder.root / 'worker_stdout.json').write_bytes(stdout)
                    if process.returncode and not terminal_stop:
                        raise RuntimeError(f'Isolated program worker exited {process.returncode}: {stderr[-1500:].decode(errors="replace")}')
                    if not terminal_stop:
                        execution = loads(stdout)
                    success = adapter.check_success()
                finally:
                    adapter.episode.abort_requested = terminal_stop or 'worker_closed'
                    subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=30)
                    if process is not None and process.poll() is None:
                        process.kill(); process.wait(timeout=10)
                    server.shutdown()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if adapter.episode is not None:
            success = adapter.check_success()
        adapter.close()
        recorder.finalize(program_ok=bool(execution and execution.ok), task_success=success,
            termination_reason=terminal_stop or ('harness_error' if error else ('program_error' if execution and not execution.ok else 'program_completed')),
            error=error, program_result=execution.result if execution else None)
    provider_failures = sam3.infrastructure_failures if sam3 else 0
    return dict(program_ok=bool(execution and execution.ok), task_success=success,
                run_dir=str(recorder.root), error=error or (str(execution.error) if execution and execution.error else None),
                termination_reason=terminal_stop or ('harness_error' if error else 'program_completed'),
                control_steps=adapter.episode.steps if adapter.episode else 0, episode=episode,
                provider_infrastructure_failures=provider_failures,
                failure_kind='infrastructure' if provider_failures else
                    ('policy_timeout' if terminal_stop in ('controller_wall_timeout','controller_no_progress') else None))

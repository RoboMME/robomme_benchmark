"""Full multi-task ASPIRE development with its upstream coordinator and library.

Only experiment plumbing lives here: task routing, isolated simulator processes,
budget enforcement, archival and the native coding-agent launch. The coordinator
chooses the curriculum, programs and skill proposals using upstream prompts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid

from benchmark_io.protocol import Study
from benchmark_io.demonstrations import export_task
from benchmark_io.prompt_bundle import install as install_protocol
from benchmark_io.development_deadline import TaskClock, TASK_SECONDS, WRAPUP
from .development_deadline import CampaignClocks
from .development_runtime import AgentHandler, AgentServer, provision, HERE, UPSTREAM


def resolve_task(request, tasks):
    name, index = request.get('task'), request.get('task_id')
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            raise ValueError('task-id must be an integer') from None
        if not 0 <= index < len(tasks):
            raise ValueError('Unknown task-id')
        if name is not None and name != tasks[index]:
            raise ValueError('task and task-id disagree')
        name = tasks[index]
    if name not in tasks:
        raise ValueError('Every full-campaign run requires --task NAME or --task-id INDEX')
    return name


class Campaign:
    def __init__(self, args):
        self.args, self.study = args, Study.open(args.study)
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=args.resume)
        self.tasks = sorted(self.study.config['records'])
        self.workspace, self.published = self.root / 'workspace', self.root / 'published'
        self.workspace.mkdir(exist_ok=args.resume)
        self.published.mkdir(exist_ok=args.resume)
        self.token = args.token_file.read_text().strip()
        self.locks = {task: threading.Lock() for task in self.tasks}
        self.archive_lock = threading.Lock()
        self.slots = queue.Queue()
        for index, gpu in enumerate(args.gpus):
            self.slots.put((gpu, args.provider_port_base + 100 * index))
        self.replies = []
        self.clocks = CampaignClocks(self.study.root, self.root/'claude-home', self.tasks)
        if (self.root / 'run_replies.json').is_file():
            self.replies = json.loads((self.root / 'run_replies.json').read_text())

    def status(self):
        attempts = self.study.attempts()
        rows = []
        for i, task in enumerate(self.tasks):
            used = sum(a['task'] == task for a in attempts)
            rows.append(dict(task=task, task_id=i, starts_used=used,
                starts_remaining=self.study.config['budget'] - used,
                development_clock=TaskClock(self.study.root, task).read(),
                training_episode_indices=list(range(len(self.study.config['records'][task])))))
        return dict(setting=self.study.config['setting'], budget_per_task=self.study.config['budget'],
            tasks=rows, demonstrations_available=self.study.config['setting'] != 'interaction-only',
            test_access=False, simulation_workers=len(self.args.gpus))

    def snapshot(self):
        """Content-address code and notes at every API turn and simulator request."""
        with self.archive_lock:
            archive = self.root / 'history'
            (archive / 'objects').mkdir(parents=True, exist_ok=True)
            files = {}
            for directory, dirs, names in os.walk(self.workspace):
                dirs[:] = [d for d in dirs if d not in ('demonstrations', 'published', '__pycache__', '.git')]
                for name in names:
                    path = Path(directory) / name
                    if path.is_symlink() or path.suffix not in ('.py', '.md', '.json', '.sh', '.txt'):
                        continue
                    try:
                        content = path.read_bytes()
                    except FileNotFoundError:
                        continue
                    digest = hashlib.sha256(content).hexdigest()
                    target = archive / 'objects' / digest
                    if not target.exists():
                        target.write_bytes(content)
                    files[str(path.relative_to(self.workspace))] = digest
            with (archive / 'snapshots.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), files=files)) + '\n')

    def run(self, request):
        task = resolve_task(request, self.tasks)
        with self.locks[task]:
            clock = TaskClock(self.study.root, task)
            clock.start()
            clock.require_open()
            status = next(r for r in self.status()['tasks'] if r['task'] == task)
            if not status['starts_remaining']:
                return dict(ok=False, rejected=True, error='Per-task simulator budget exhausted', task=task)
            code = request.get('code')
            if not isinstance(code, str) or not 0 < len(code) <= 1024 * 1024:
                raise ValueError('Provide a Python program of at most 1 MiB')
            episode = request.get('episode')
            if episode is not None and (type(episode) is not int or episode not in status['training_episode_indices']):
                raise ValueError('Only released training episode indices are permitted')
            self.snapshot()
            ident = f'{task}/attempt_{status["starts_used"]+1:03d}_{uuid.uuid4().hex[:8]}'
            run_root = self.root / 'runs' / ident
            run_root.mkdir(parents=True)
            request = {**request, 'task': task}
            request_path = run_root / 'request.json'
            request_path.write_text(json.dumps(request, indent=2))
            result_path = run_root / 'result.json'
            slot = self.slots.get()
            gpu, port = slot
            command = [sys.executable, '-m', 'aspire.run_episode', '--study', str(self.study.root),
                '--output', str(run_root / 'artifacts'), '--request', str(request_path),
                '--result', str(result_path), '--image', self.args.image, '--provider-port-base', str(port)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2'}
            (run_root / 'operator_command.json').write_text(json.dumps(dict(argv=command, gpu=gpu), indent=2))
            try:
                with (run_root / 'operator.log').open('w') as log:
                    from benchmark_io.runtime_policy import configured
                    limits=configured()
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=limits.worker_seconds if limits else 3900)
                if not result_path.is_file():
                    raise RuntimeError(f'Simulator worker exited {result.returncode}; see operator log')
                result = json.loads(result_path.read_text())
                source = Path(result['run_dir'])
                target = self.published / ident
                shutil.copytree(source, target, ignore=shutil.ignore_patterns('native', 'worker_command.json'))
                native = next((source / 'native').glob('*'), None)
                if native:
                    for name in ('rollout.mp4', 'actions.jsonl', 'observations.jsonl', 'outcome.json'):
                        if (native / name).is_file():
                            shutil.copy2(native / name, target / ('native_' + name))
                reply = {**result, 'run_dir': '/workspace/published/' + ident,
                         'ok': result['error'] is None, 'budget': self.status()}
            finally:
                self.slots.put(slot)
            with self.archive_lock:
                self.replies.append(reply)
                (self.root / 'run_replies.json').write_text(json.dumps(self.replies, indent=2))
            return reply


class CampaignHandler(AgentHandler):
    def do_POST(self):
        if self.path.split('?')[0] == '/v1/messages':
            self.server.session.snapshot()
        super().do_POST()


def prepare(session):
    args = session.args
    model, home, bridge = provision(session)
    original = UPSTREAM / '.claude/skills/iterative-debugging'
    shutil.copytree(original / 'scripts', session.workspace / '.claude/skills/iterative-debugging/scripts')
    if session.study.config['setting'] != 'interaction-only':
        for task in session.tasks:
            export_task(session.study, task, args.demos, session.root / 'demonstrations' / task)
    tasks = [dict(benchmark='robomme_plus', suite='robomme_plus', task_id=i, task_name=task,
                  training_episode_indices=list(range(len(session.study.config['records'][task]))))
             for i, task in enumerate(session.tasks)]
    (session.workspace / 'campaign/tasks.json').write_text(json.dumps(tasks, indent=2))
    (session.workspace / 'campaign/campaign.json').write_text(json.dumps(dict(dev_seeds=[0,1,2],
        unseen_seeds=[], heldout_count=0, setting=session.study.config['setting'],
        budget_per_task=session.study.config['budget'], task_episode_indices={r['task_name']:r['training_episode_indices'] for r in tasks}), indent=2))
    text = f'''# ASPIRE full RoboMME-Plus campaign

Use ASPIRE's original multi-task coordinator, task workers, trace-guided repair,
independent skill verifier and shared library. Read the original prompts under
.claude/skills/iterative-debugging/ and docs/robomme-plus.md. The following changes
only the benchmark access, budget and evaluation protocol, not your learning strategy.

- Setting: {session.study.config['setting']}. Model: Opus 5.5, high reasoning.
- RUN_ROOT=/workspace/campaign; BENCHMARK=robomme_plus; SUITE=robomme_plus.
  All 16 task names and stable IDs are in campaign/tasks.json. Choose your curriculum
  as in ASPIRE. Maintain the upstream task/skill handoff and library provenance.
- Task directories follow the original convention:
  /workspace/campaign/robomme_plus/robomme_plus/task_<TASK_ID>/.
  Finish each task's selected program there as fix_code.py, with task_analysis.md,
  skill_report.json, versioned code, and evidence. Keep a campaign_report.md.
- At most {session.study.config['budget']} simulator starts PER TASK across all workers,
  scene probes, repairs, batches, and independent verifiers. No free resets.
  `cap-harness status` is authoritative. Failed starts remain charged.
- `cap-harness run --task TASK_NAME --program FILE [--seed INDEX] --output-root DIR`
  uses the original ASPIRE API and providers in an isolated simulator worker.
  `--task-id INDEX` also works. A task identifier is REQUIRED. Omitting --seed
  chooses the least-used released training episode. run-batch charges every seed.
- Training episode indices are 0,1,2 (MoveCube 0..4), NOT arbitrary procedural
  seeds. No extra unseen training split exists. Never substitute test seeds.
  ASPIRE's original verifier criterion remains unchanged: do not claim a five-seed
  verification from three distinct seeds. Report insufficient evidence as untested;
  do not promote unverified proposals. demo-only has no simulator verification.
- interaction-only has no expert recordings. demo-only and demo&interaction have all 50 released recordings under
  /workspace/demonstrations/<TASK>/manifest.json (front/wrist RGB, goal and phase).
  Recordings are fixed evidence, not access to oracle actions or simulator code.
- The host provides {len(args.gpus)} simulator slots, one on each assigned GPU, and
  routes jobs automatically. Keep at most {len(args.gpus)} task/verifier jobs in flight.
  Provider startup, installations and GPU management are operator-only.
- Use public APIs in docs/api-reference.md and docs/robomme-plus.md. The available
  extensions are robomme_plus.get_task_metadata(), get_controller_metadata(), and
  get_conditioning_video(). An episode's goal comes from get_task_context().
- Read original per-call traces, keyframes and full rollout videos in the returned
  /workspace/published directories. Preserve every program version and report.
- The host already initialized this campaign. Do not run LIBERO init_run,
  run_pipeline, validation or simulator source-scanning commands. The original
  bookkeeping helpers are available; pass --run-root /workspace/campaign.
- All held-out evaluation is deferred until the ENTIRE campaign is frozen. No
  Stage 2 results, held-out-driven revisits or test feedback are available during
  development. For scheduling, release a task slot after its Stage 1 handoff.
  At a hard budget limit, finalize the best policy supported by training evidence
  with honest remaining failures; do not invent success or skip difficult tasks.

The container is offline, without simulator source, host paths or credentials.
`cap_harness` is ASPIRE's own upstream runtime, unrelated to the separate CaP baseline.
'''
    text += install_protocol(session.workspace, 'aspire')
    (session.workspace / 'CLAUDE.md').write_text(text)
    return model, home, bridge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--demos', default=os.environ.get('ROBOMME_PLUS_DEMOS','data/robomme_plus_train/h5'))
    p.add_argument('--gpus', nargs='+', default=['0','1'])
    p.add_argument('--provider-port-base', type=int, default=18110)
    p.add_argument('--proxy-url', default='http://127.0.0.1:18177')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    session = Campaign(args)
    (session.root/'deadline_enforcement.json').write_text(json.dumps(
        dict(pid=os.getpid(), task_seconds=TASK_SECONDS, activated=time.time()), indent=2))
    if args.resume:
        model, home, bridge = 'us.anthropic.claude-opus-5-5', session.root / 'claude-home', session.root / 'bridge'
    else:
        model, home, bridge = prepare(session)
    rpc = session.root / 'rpc'
    rpc.mkdir(exist_ok=True)
    (rpc / 'agent.sock').unlink(missing_ok=True)
    container = 'aspire-campaign-' + uuid.uuid4().hex
    prompt = ('Act as the ASPIRE multi-task coordinator. Read CLAUDE.md and the original coordinator '
        'prompt. Complete Stage 1 for all 16 tasks with the original task-worker and independent '
        'skill-verification workflow and shared library. Choose your own curriculum. Follow the '
        'fixed setting and per-task rollout ceiling. Save every version, command and report, '
        'and each selected fix_code.py in its task directory. The operator evaluates only after '
        'all development ends; no test feedback is available. '
        f'The host enforces {TASK_SECONDS // 60} minutes per task from its first worker assignment, '
        'including offline tests, API calls and simulator work. Save fix_code.py early. '
        'At expiry, wrap up with the existing code; no further modifications or simulator starts. '
        'Expired task directories are frozen. Report their limitations and continue other tasks; '
        'never dispatch a new worker or verifier for an expired task.')
    command = ['docker','run','--rm','--pull=never','--name',container,'--network=none',
        '--read-only','--user',f'{os.getuid()}:{os.getgid()}','--cap-drop=ALL',
        '--security-opt=no-new-privileges','--pids-limit=512','--memory=16g','--cpus=4',
        '--ipc=private','--init','--log-driver=none','--tmpfs','/tmp:rw,nosuid,nodev,size=1g,mode=1777',
        '--workdir','/workspace','--env','HOME=/claude-home','--env','CLAUDE_CONFIG_DIR=/claude-home',
        '--env','PATH=/opt/bridge:/usr/local/bin:/usr/bin:/bin','--env','PYTHONDONTWRITEBYTECODE=1',
        '--env','OPENBLAS_NUM_THREADS=1','--env','CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0',
        '--env','CLAUDE_CODE_AUTO_COMPACT_WINDOW=250000','--env','ENABLE_PROMPT_CACHING_1H=1']
    mounts = [(session.workspace,'/workspace',False),(home,'/claude-home',False),
        (bridge,'/opt/bridge',True),(rpc,'/run/aspire',True),(session.published,'/workspace/published',True),
        (Path(os.environ.get('ROBOMME_CLAUDE_CLI') or shutil.which('claude') or 'claude').resolve(),'/opt/claude',True),
        (home/'settings.json','/claude-home/settings.json',True),(home/'mcp.json','/claude-home/mcp.json',True)]
    if session.study.config['setting'] != 'interaction-only':
        mounts.append((session.root/'demonstrations','/workspace/demonstrations',True))
    for source,target,readonly in mounts:
        command += ['--mount',f'type=bind,src={source},dst={target}'+(',readonly' if readonly else '')]
    command += ['--entrypoint','/usr/local/bin/python',args.image,'/opt/bridge/agent_bridge.py',
        '/opt/claude','-p','--model',model,'--effort','high','--strict-mcp-config',
        '--mcp-config','/claude-home/mcp.json','--permission-mode','bypassPermissions',
        '--disallowedTools','WebFetch','WebSearch','--output-format','stream-json','--verbose']
    if args.resume:
        transcripts = sorted((home/'projects/-workspace').glob('*.jsonl'),key=lambda f:f.stat().st_mtime)
        if not transcripts:
            raise RuntimeError('No native conversation to resume')
        command += ['--resume',transcripts[-1].stem]
        prompt = 'Continue the existing campaign. Read cap-harness status; prior starts stay charged. ' + prompt
    base_command = command[:]
    command += ['--',prompt]
    (session.root/'launch_command.json').write_text(json.dumps(command,indent=2))
    (session.root/'development_prompt.txt').write_text(prompt)
    with AgentServer(str(rpc/'agent.sock'),CampaignHandler) as server:
        server.session = session
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with (session.root/'agent_stdout.jsonl').open('a') as out, (session.root/'agent_stderr.log').open('a') as err:
                while True:
                    # Frozen mounts apply both after a timeout and an operator
                    # restart. The final selector reads these immutable copies.
                    expired = [t for t in session.tasks if (TaskClock(session.study.root, t).read() or {}).get('expired_at')]
                    command = base_command[:]
                    for task in expired:
                        index = session.tasks.index(task)
                        frozen = session.root/'deadline_frozen'/task
                        target = f'/workspace/campaign/robomme_plus/robomme_plus/task_{index}'
                        if frozen.is_dir():
                            command[command.index('--entrypoint'):command.index('--entrypoint')] = [
                                '--mount', f'type=bind,src={frozen},dst={target},readonly']
                    command += ['--', prompt]
                    with (session.root/'launch_commands.jsonl').open('a') as launches:
                        launches.write(json.dumps(dict(time=time.time(), argv=command))+'\n')
                    result = subprocess.Popen(command, stdout=out, stderr=err)
                    while result.poll() is None:
                        overdue = session.clocks.overdue()
                        if overdue:
                            for task in overdue:
                                TaskClock(session.study.root, task).expire()
                            subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=30)
                            result.wait(timeout=30)
                            for task in overdue:
                                index = session.tasks.index(task)
                                source = session.workspace/'campaign/robomme_plus/robomme_plus'/f'task_{index}'
                                frozen = session.root/'deadline_frozen'/task
                                if source.is_dir():
                                    shutil.copytree(source, frozen, symlinks=False,
                                        ignore=shutil.ignore_patterns('__pycache__', '*.mp4', '*.h5'))
                                else:
                                    frozen.mkdir(parents=True)
                            session.snapshot()
                            transcripts = sorted((home/'projects/-workspace').glob('*.jsonl'), key=lambda f:f.stat().st_mtime)
                            if '--resume' in base_command:
                                position = base_command.index('--resume')
                                del base_command[position:position+2]
                            base_command += ['--resume', transcripts[-1].stem]
                            prompt = (WRAPUP + ' This applies to: ' + ', '.join(overdue) + '. '
                                      'All their sandbox commands have been stopped and their task directories are read-only. '
                                      'Report those tasks using the saved code; do not dispatch them again. '
                                      'The deadline status overrides any pending/incomplete entry in gen_progress or missing skill report: '
                                      'record these as deadline-limited, with unverified skills, and skip further work on them. '
                                      'Continue the other unfinished tasks; their original clocks and rollout counts remain in force.')
                            with (session.root/'deadline_events.jsonl').open('a') as events:
                                events.write(json.dumps(dict(time=time.time(), tasks=overdue, prompt=prompt))+'\n')
                            break
                        time.sleep(1)
                    else:
                        break
                    # A deadline interrupted the container; resume its native
                    # coordinator conversation, not a fresh learning session.
            session.snapshot()
            selected = {}
            for i, task in enumerate(session.tasks):
                path = session.workspace/'campaign/robomme_plus/robomme_plus'/f'task_{i}'/'fix_code.py'
                if (TaskClock(session.study.root, task).read() or {}).get('expired_at'):
                    path = session.root/'deadline_frozen'/task/'fix_code.py'
                selected[task] = os.path.relpath(path, session.workspace) if path.is_file() else None
            report = dict(returncode=result.returncode,selected_programs=selected,status=session.status(),
                          deadline_tasks=[t for t in session.tasks if (TaskClock(session.study.root, t).read() or {}).get('expired_at')],
                          held_out_feedback_provided=False,effort='high',finished=time.time())
            (session.root/'development_result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
        finally:
            subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=30)
            server.shutdown()


if __name__ == '__main__':
    main()

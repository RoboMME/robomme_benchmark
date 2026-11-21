"""Full-study launch using RAX's native exploration/register/rehearse/submit flow."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from benchmark_io.protocol import Study
from benchmark_io.evaluation import EvaluationStudy
from benchmark_io.prompt_bundle import install as install_protocol
from harness.claude_cli.auto_research import deploy, redeploy, launch
from services.controller.config import Configuration, Limits
from services.controller.supervisor import Supervisor
from services.robomme_plus.backend import RoboMMEPlusBackend






def prune_media(root):
    """Drop completed evaluation media, preserving logs and code, never active inputs."""
    root = Path(root)
    removed = 0
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() in ('.png','.jpg','.jpeg','.mp4','.webm','.npy','.npz','.h5') or path.name in ('actions.jsonl','observations.jsonl'):
            removed += path.stat().st_size
            path.unlink()
        elif path.name == 'response.json':
            data = json.loads(path.read_text())
            for block in data.get('content', []):
                if block.get('type') == 'image':
                    block.pop('data', None)
                    block['omitted_by_artifact_profile'] = 'logs'
            path.write_text(json.dumps(data))
    return removed


class EvaluationSupervisor(Supervisor):
    def _run_episode(self, bundle_id, *, formal, formal_index=0):
        before = len(self.state['results'])
        try:
            return super()._run_episode(bundle_id, formal=formal, formal_index=formal_index)
        except Exception:
            # Native cleanup can raise after durably saving a finished episode.
            # Keep that outcome and its disclosed shutdown error; never replay it.
            rows = self.state['results'][before:]
            if (not formal or len(rows) != 1 or self.state.get('active')
                    or rows[0].get('bundle') != bundle_id
                    or rows[0].get('formal_episode_index') != formal_index + 1
                    or not rows[0].get('timing')
                    or not any(e.get('stage') == 'environment_shutdown'
                               for e in rows[0].get('infrastructure_errors', []))):
                raise
            root = os.environ.get('ROBOMME_PLUS_FROZEN_EVALUATION')
            if not root:
                raise
            attempts = EvaluationStudy(root).attempts()
            attempt = next((a for a in attempts if a['task'] == self.config.task
                            and a['episode'] == rows[0].get('layout_id')), None)
            if not attempt or attempt['ended'] is None:
                raise
            return rows[0]

    def resume_unstarted(self, frozen):
        """Operator-only continuation: every previously reserved episode stays closed."""
        from services.controller.storage import verify
        batch = self.state.get('formal_batch')
        submission = self.state.get('submission') or {}
        if (not batch or batch.get('status') != 'interrupted'
                or self.config.formal_workers != 1 or self.state.get('active')
                or batch.get('bundle') != submission.get('bundle')
                or batch.get('sha256') != submission.get('sha256')):
            raise RuntimeError('Only an idle, interrupted frozen bundle batch may continue')
        bundle_id = submission['bundle']
        manifest = self.state['bundles'][bundle_id]
        if manifest['files'] != frozen.manifest['policy_sha256']:
            raise ValueError('Continuation policy differs from the frozen policy')
        verify(self.config.root/'bundles'/manifest['directory'], manifest)
        rows = [r for r in self.state['results'] if r.get('mode') == 'formal']
        attempts = frozen.attempts()
        done = {r.get('layout_id') for r in rows}
        expected = set(range(self.config.formal_seed, self.config.formal_seed + self.config.formal_episodes))
        if (len(done) != len(rows) or not done <= expected
                or {a['episode'] for a in attempts} != done
                or len(attempts) != len(done)
                or any(a['task'] != self.config.task or a['ended'] is None for a in attempts)
                or any(r.get('bundle') != bundle_id or r.get('sha256') != manifest['sha256']
                       or r.get('formal_episode_index') != r['layout_id'] - self.config.formal_seed + 1
                       or r.get('driver', 'bundle') != 'bundle' or r.get('session_stopped')
                       or not r.get('timing') for r in rows)):
            raise RuntimeError('Continuation requires one finalized result for every reserved episode')
        self.audit(dict(event='operator_continue_unstarted_formal', preserved_episodes=sorted(done),
                        remaining_episodes=sorted(expected-done), policy_modified=False,
                        repeated_attempts=0, api_calls=0))
        # A shutdown exception can save the result before _record_formal counts it.
        # Recount from the saved evidence using exactly the native scoring function.
        batch = dict(batch, status='running', completed_episodes=0, success_count=0,
                     unsuccessful_count=0, error_count=0, infrastructure_error_count=0)
        for row in rows:
            self._record_formal(batch, row)
        return self._formal_batch(bundle_id, manifest, driver='bundle', batch=batch,
            done={episode-self.config.formal_seed+1 for episode in done})

    def _end_recording(self, previous, bundle, output, result):
        recording = self._recording
        native = self.backend.sim.output_dir if self.backend else None
        super()._end_recording(previous, bundle, output, result)
        if self.phase == 'formal' and os.environ.get('ROBOMME_PLUS_EVAL_ARTIFACTS') == 'logs':
            roots = [recording.root, output]
            if native:
                roots.append(native)
            if self.backend:
                roots.extend(directory / self.backend.sim.run_id for directory in self.backend.frame_root.iterdir()
                             if directory.is_dir() and self.backend.sim.run_id)
            removed = sum(prune_media(path) for path in roots if Path(path).is_dir())
            (recording.root/'retention.json').write_text(json.dumps(dict(profile='logs',discarded_media_bytes=removed)))


def develop(args):
    study = Study.open(args.study)
    budget = study.config['budget']
    if args.output.exists() and not args.resume:
        raise FileExistsError('Full development requires a fresh independent session')
    gpu = subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    limit = Limits(wall_seconds=3600,memory_mb=8192,cpus=2,pids=256,scratch_mb=512,artifact_bytes=2*1024**3)
    config = Configuration(root=args.output.resolve(),image=args.image,task=args.task,
        exploration_seeds=tuple(i % len(study.config['records'][args.task]) for i in range(budget)),
        formal_seed=0,eval_seed=0,formal_eval_seed=1,formal_episodes=50,formal_workers=1,
        sim_gpu=gpu,development=limit,formal=limit,training=limit,workspace_mb=4096,
        demonstration_context='none' if study.config['setting']=='interaction-only' else 'completion_sequence',
        benchmark='robomme_plus',agent_cli='claude',agent_provider='custom',agent_effort='high',
        external_perception=False,observation_profile='rgbd')
    if args.resume:
        config = Configuration.from_dict(json.loads((args.output/'configuration.json').read_text()))
        deployed = redeploy(config, claude=Path(os.environ.get('ROBOMME_CLAUDE_CLI') or shutil.which('claude') or 'claude'))
    else:
        deployed = deploy(config,provider='custom',key_file=args.token_file,
            model='us.anthropic.claude-opus-5-5',claude=Path(os.environ.get('ROBOMME_CLAUDE_CLI') or shutil.which('claude') or 'claude'),base_url=args.proxy_url)
    from benchmark_io.development_deadline import rax_clock, TASK_SECONDS
    clock = rax_clock(config)
    prompt = (
        'Develop a controller for the assigned RoboMME-Plus task using the installed original RAX '
        'instructions and exploration/compile/rehearse/submit workflow. Read AGENTS.md, TASK.md and '
        'the native API/controller contract. This is a fresh independent experiment, not a continuation '
        f'of another budget or smoke test. The TOTAL per-task simulator-start ceiling is R{budget}. '
        'Every preview, interaction, reset and rehearsal counts. Status is authoritative. '
        'Choose your own strategy under that ceiling. demo-only is offline with the released task recordings; '
        'interaction-only has no expert recordings; demo+interaction has recordings and the same bounded simulator tools. '
        'Keep all program versions, commands, reasoning notes and evidence. Register your selected '
        'bundle and use the native submit tool to freeze it. For interaction settings, the original '
        'clean rehearsal of that exact bundle is required and consumes one of the same allowed starts. '
        'There is no extra rehearsal allowance. demo-only has the documented zero-interaction exception. '
        'Write memory/development_report.md with the submitted bundle ID and unresolved limitations. '
        'Submitting closes exploration but does not show any test outcomes. Do not run formal evaluation; '
        'the operator evaluates all 50 held-out episodes only after every task finishes development. '
        'No test feedback will be returned. Do not access simulator source, other methods or hidden state. '
        'The optional Gemini service is disabled, as configured by the operator. Native coding-model '
        'vision and the original RGB-D geometry tools remain available. Use Opus 5.5 high reasoning.'
    )
    prompt += (f' The host enforces a {TASK_SECONDS // 60}-minute wall-clock development limit for this task, '
               'including API calls, offline work and simulator attempts. Save a usable policy early. '
               'At expiry, all development stops and only final reporting/submission of existing code is allowed.')
    supplement = '' if args.resume else install_protocol(deployed['workspace'], 'rax')
    prompt += supplement
    if supplement:
        with (deployed['workspace']/'CLAUDE.md').open('a') as stream:
            stream.write(supplement)
    if args.resume:
        prompt = ('Resume this same task and conversation. All previous simulator starts and elapsed '
                  'development time remain charged; no budget resets. ' + prompt)
    (args.output/('resume_prompt.txt' if args.resume else 'development_prompt.txt')).write_text(prompt)
    (args.output/'development_config.json').write_text(json.dumps(asdict(config),default=str,indent=2))
    result = launch(config,deployed,prompt=prompt,capture=True,calls_file=args.output/'model-call-count.json')
    with (args.output/'agent_stdout.jsonl').open('a') as stream:
        stream.write(result.stdout or '')
    with (args.output/'agent_stderr.log').open('a') as stream:
        stream.write(result.stderr or '')
    (args.output/'launch_command.json').write_text(json.dumps(result.args,indent=2))
    state = json.loads((args.output/'state.json').read_text())
    report = dict(method='rax',setting=study.config['setting'],budget=budget,task=args.task,
        returncode=result.returncode,submission=state.get('submission'),bundles=list(state.get('bundles',{})),
        attempts=[a for a in study.attempts() if a['task']==args.task],effort='high',
        external_gemini=False,held_out_feedback_provided=False,finished=time.time(),
        development_clock=clock.read() if clock else None)
    (args.output/'development_result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def evaluate(args):
    state = json.loads((args.development/'state.json').read_text())
    config = Configuration.from_dict(state['configuration'])
    submission = state.get('submission')
    if not submission:
        raise ValueError('No native submitted bundle; no operator selection or free rehearsal is allowed')
    manifest = state['bundles'][submission['bundle']]
    import robomme_plus
    if not (args.output/'frozen_evaluation.json').is_file():
        frozen = EvaluationStudy.freeze(args.output,development_study=args.study,
            policy_directory=args.development/'bundles'/manifest['directory'],
            episodes={config.task:list(range(50))},benchmark_root=Path(robomme_plus.__file__).parent,
            artifact_profile=args.artifacts)
    else:
        frozen = EvaluationStudy(args.output)
    os.environ['ROBOMME_PLUS_FROZEN_EVALUATION'] = str(frozen.root)
    os.environ['ROBOMME_PLUS_EVAL_ARTIFACTS'] = args.artifacts
    os.environ['ROBOMME_PLUS_PURPOSE'] = 'frozen_full_evaluation'
    gpu = subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    # Placement is operator-owned; retain the native session's immutable learning config.
    backend_config = replace(config, sim_gpu=gpu)
    (frozen.root/'execution_placement.json').write_text(json.dumps(dict(gpu=gpu),indent=2))
    supervisor = EvaluationSupervisor(config,lambda:RoboMMEPlusBackend(backend_config))
    try:
        evaluation = (supervisor.resume_unstarted(frozen) if args.resume_unstarted
                      else supervisor.run_submission())
        if evaluation is None:
            evaluation = supervisor.state.get('formal_batch')
        (frozen.root/'result.json').write_text(json.dumps(evaluation,indent=2))
        if not evaluation or evaluation.get('completed_episodes') != 50:
            raise RuntimeError('Native formal batch is incomplete; preserve evidence for operator recovery')
        print(json.dumps(evaluation),flush=True)
    finally:
        supervisor.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('develop','evaluate'))
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--development',type=Path)
    p.add_argument('--task')
    p.add_argument('--gpu',default='0')
    p.add_argument('--image',default=os.environ.get('ROBOMME_AGENTIC_IMAGE','robomme-agentic:latest'))
    p.add_argument('--token-file',type=Path)
    p.add_argument('--proxy-url',default='http://127.0.0.1:18177')
    p.add_argument('--artifacts',choices=('full','logs'),default='full')
    p.add_argument('--resume', action='store_true', help='Continue the same unfinished development and clock')
    p.add_argument('--resume-unstarted', action='store_true',
                   help='Operator continuation of an interrupted batch; never repeat a reserved episode')
    args=p.parse_args()
    os.environ['ROBOMME_PLUS_STUDY']=str(args.study.resolve())
    os.environ['ROBOMME_PLUS_PURPOSE']='full_development'
    os.environ['RAX_ARTIFACT_ROOT']=str((args.development or args.output).resolve().parent)
    if args.phase=='develop':
        develop(args)
    else:
        evaluate(args)


if __name__=='__main__':
    main()

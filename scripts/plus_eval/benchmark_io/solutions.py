"""Evaluate published frozen solutions; never run model development or selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .evaluation import EvaluationStudy
from .full_study import Pipeline, REPO, write
from .runtime_source import assemble


def verified_files(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    for relative, expected in manifest['policy_files'].items():
        path = directory/relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Policy paths must stay inside their solution directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Published policy changed: {relative}')
    return manifest


def resolve_image(image):
    if re.fullmatch(r'(?:[A-Za-z0-9_./:-]+@)?sha256:[a-f0-9]{64}', image):
        return image
    return subprocess.check_output(['docker','image','inspect','--format={{.Id}}',image], text=True).strip()


def main(method, evaluate, method_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=['AfD','AfI','AfD+I'], required=True)
    parser.add_argument('--budget', type=int, choices=[0,1,3,5,10], required=True)
    parser.add_argument('--solutions', type=Path, default=Path(method_dir)/'solutions')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--episodes', nargs='+', type=int, default=list(range(50)))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent episode processes on this GPU')
    parser.add_argument('--image', default=os.environ.get('ROBOMME_AGENTIC_IMAGE','robomme-agentic:latest'))
    parser.add_argument('--artifacts', choices=['full','logs'], default='full')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.setting=='AfD') != (args.budget==0):
        parser.error('AfD uses R0; interaction settings use R1/R3/R5/R10')
    if args.workers<1 or not args.episodes or any(i<0 or i>=50 for i in args.episodes):
        parser.error('Use positive worker counts and test episode indices 0–49')
    args.output_dir=args.output_dir.resolve();args.solutions=args.solutions.resolve()
    if args._worker:
        task,index=args.tasks[0],args.episodes[0]
        output=args.output_dir/'episodes'/task/f'episode_{index:03d}'
        frozen=EvaluationStudy(args.output_dir/'evaluation'/task)
        try:
            if any(a['task']==task and a['episode']==index for a in frozen.attempts()):
                raise RuntimeError('This frozen episode was already reserved; preserve its result')
            row=evaluate(args,frozen,output)
        except Exception as exc:
            row=dict(task_success=False,failure_kind='infrastructure',error=f'{type(exc).__name__}: {exc}')
        row.update(task=task,episode=index)
        write(output/'result.json',row)
        return
    import robomme_plus
    from robomme_plus import BenchmarkEnvBuilder
    tasks=args.tasks or sorted(BenchmarkEnvBuilder.get_task_list())
    if len(set(tasks))!=len(tasks) or not set(tasks)<=set(BenchmarkEnvBuilder.get_task_list()):
        parser.error('Choose distinct released task names')
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    image=resolve_image(args.image)
    config=dict(method=method,setting=args.setting,budget=args.budget,solutions=str(args.solutions),
                image=image,artifact_profile=args.artifacts)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text())!=config:
        raise ValueError('Use a new output directory for different solutions or settings')
    write(root/'config.json',config)
    if not (root/'runtime_source').exists():assemble(root/'runtime_source')
    for package in ('robomme','robomme_plus'):
        destination=root/'benchmark/src'/package
        if not destination.exists():
            shutil.copytree(REPO/'src'/package,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    policy=args.solutions/'runtime_policy.json'
    if not (root/'runtime_policy.json').exists():shutil.copy2(policy,root/'runtime_policy.json')
    runtime=root/'runtime_source'
    pipeline=Pipeline(root,[args.gpu])
    pipeline.env.update(ROBOMME_AGENTIC_CODE=str(runtime),ROBOMME_AGENTIC_REPO=str(REPO),
        PYTHONPATH=os.pathsep.join(map(str,[root/'benchmark/src',runtime,runtime/'rax',runtime/'aspire/ASPIRE/src'])),
        ROBOMME_AGENTIC_RUNTIME_POLICY=str(root/'runtime_policy.json'),ROBOMME_AGENTIC_IMAGE=image)
    study,_=pipeline.prepare(root,method,args.setting,args.budget)
    for task in tasks:
        source=args.solutions/args.setting/f'R{args.budget}'/task
        provenance=verified_files(source)
        frozen_root=root/'evaluation'/task
        if not frozen_root.exists():
            with tempfile.TemporaryDirectory() as temporary:
                for name in provenance['policy_files']:
                    target=Path(temporary)/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source/name,target)
                EvaluationStudy.freeze(frozen_root,development_study=study.root,policy_directory=temporary,
                    episodes={task:list(range(50))},benchmark_root=root/'benchmark/src/robomme_plus',
                    artifact_profile=args.artifacts)
            write(frozen_root/'published_solution.json',provenance)
        elif json.loads((frozen_root/'published_solution.json').read_text())!=provenance:
            raise ValueError('Published solution selection changed')
        EvaluationStudy(frozen_root).verify_frozen()
    if method=='aspire':pipeline.providers()
    def run(pair):
        task,index=pair;output=root/'episodes'/task/f'episode_{index:03d}'
        if (output/'result.json').exists():return
        output.mkdir(parents=True,exist_ok=True)
        env=dict(pipeline.env,CUDA_VISIBLE_DEVICES=args.gpu,
                 ROBOMME_AGENTIC_JOB=f'{root}::{task}/{index}')
        command=[sys.executable,str(runtime/method/'eval_plus.py'),'--_worker','--setting',args.setting,
                 '--budget',str(args.budget),'--solutions',str(args.solutions),'--output-dir',str(root),
                 '--tasks',task,'--episodes',str(index),'--gpu',args.gpu,'--image',image,'--artifacts',args.artifacts]
        write(output/'command.json',dict(argv=command))
        try:
            with (output/'eval.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                               timeout=2100,check=True,start_new_session=True)
        except (subprocess.SubprocessError,OSError) as exc:
            from .job_cleanup import cleanup
            cleanup(env['ROBOMME_AGENTIC_JOB'],output)
            if not (output/'result.json').exists():
                write(output/'result.json',dict(task=task,episode=index,task_success=False,
                      failure_kind='infrastructure',error=f'{type(exc).__name__}: {exc}'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run,[(task,i) for task in tasks for i in sorted(set(args.episodes))]))
    rows=[json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*/*/result.json'))]
    write(root/'results.json',rows)
    summary=dict(method=method,setting=args.setting,budget=args.budget,total=len(rows),
        successes=sum(bool(r.get('task_success')) for r in rows),evaluation_api_cost_usd=0,
        imported_frozen_solutions=True,new_development_attempts=0,
        infrastructure_errors=sum(r.get('failure_kind')=='infrastructure' for r in rows))
    write(root/'summary.json',summary);print(json.dumps(summary))

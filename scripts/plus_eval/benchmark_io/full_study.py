"""Durable development and frozen-evaluation queues, independently resumable."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import time
import threading
import uuid
import urllib.request

from .protocol import Study, source_fingerprint
from .evaluation import EvaluationStudy
from .costs import summarize
from .publish_progress import publish
from .evaluation_slots import EvaluationSlots
from .development_selection import rax_source, training_accounting

REPO = Path(os.environ.get('ROBOMME_AGENTIC_REPO',Path(__file__).resolve().parents[3]))
IMAGE = os.environ.get('ROBOMME_AGENTIC_IMAGE', 'robomme-agentic:latest')
TOKEN = Path(os.environ.get('ROBOMME_AGENTIC_TOKEN_FILE', Path.home()/'.config/robomme/relay-token'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def stop_owned(record):
    if not Path(record).is_file():
        return
    row = json.loads(Path(record).read_text())
    pid = row['pid']
    if not alive(pid):
        return
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    # Do not act on a recycled PID or another experiment's process.
    if argv[:len(row['argv'])] != row['argv']:
        raise RuntimeError(f'Owned process identity changed: {record}')
    os.kill(pid, signal.SIGTERM)
    deadline=time.time()+30
    while alive(pid) and time.time()<deadline:
        time.sleep(.2)
    if alive(pid):
        raise RuntimeError(f'Owned process did not stop: {record}')


class Pipeline:
    def __init__(self, root, gpus, *, method=None, proxy_port=18177,
                 eval_gpus=None, eval_workers_per_gpu=1, lane='all'):
        self.root, self.gpus = Path(root).resolve(), gpus
        self.methods = (method,) if method else ("aspire", "rax")
        self.proxy_port = proxy_port
        self.lane = lane
        self.eval_gpus = list(eval_gpus or gpus)
        self.eval_workers_per_gpu = eval_workers_per_gpu
        self.eval_slots = EvaluationSlots(self.root, self.eval_gpus, eval_workers_per_gpu)
        self.state_root = self.root / method if method else self.root
        if lane != 'all':
            self.state_root = self.state_root / 'lanes' / lane
        self.state_root.mkdir(parents=True, exist_ok=True)
        code = Path(os.environ.get('ROBOMME_AGENTIC_CODE',REPO/'scripts/plus_eval'))
        self.env = {**os.environ, 'PYTHONPATH': ':'.join(map(str,[self.root/'benchmark/src',
            code,code/'rax',code/'aspire/ASPIRE/src'])),
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
            'ROBOMME_PLUS_ROOT':str(self.root/'benchmark'),'ROBOMME_PLUS_PYTHON':os.environ.get('ROBOMME_PLUS_PYTHON',sys.executable),
            'ROBOMME_PLUS_DEMOS':str(self.root/'benchmark/h5')}
        self.current = {}
        self.slots=queue.Queue()
        for gpu in gpus:
            self.slots.put((gpu,self.provider_port(gpu)))

    @staticmethod
    def provider_port(gpu):
        return int(os.environ.get('ROBOMME_AGENTIC_PROVIDER_BASE','18110')) + 100 * int(gpu)

    def status(self, **fields):
        self.current.update(fields)
        data = dict(updated=time.time(),pid=os.getpid(),lane=self.lane,**self.current)
        write(self.state_root/'status.json',data)
        if self.lane != 'all' and self.current.get('case'):
            write(Path(self.current['case'])/f'{self.lane}_status.json',data)

    def heartbeat(self, stop):
        while not stop.wait(20):
            try:
                current = dict(self.current)
                case = Path(current['case']) if current.get('case') else None
                counts = {}
                if case:
                    if self.lane != 'evaluation':
                        write(case/'api_cost_summary.json',summarize(case/'development_api'))
                    manifests = [case/'evaluation/frozen_evaluation.json']
                    manifests += list((case/'evaluation').glob('*/frozen_evaluation.json'))
                    for manifest in manifests:
                        if not manifest.is_file():
                            continue
                        study = EvaluationStudy(manifest.parent)
                        attempts = study.attempts()
                        counts[str(manifest.parent.relative_to(case))] = dict(
                            started=len(attempts), finished=sum(a['ended'] is not None for a in attempts))
                write(self.state_root/'heartbeat.json',dict(updated=time.time(),pid=os.getpid(),
                    current=current,evaluation_attempts=counts,free_bytes=shutil.disk_usage(self.root).free))
                publish(self.root,self.root/'progress.json')
            except Exception as exc:
                write(self.state_root/'heartbeat_error.json',dict(updated=time.time(),error=str(exc)))

    def launch(self, command, log, record, env=None):
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        with Path(log).open('a') as stream:
            process=subprocess.Popen(command,cwd=REPO,env=env or self.env,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        write(record,dict(pid=process.pid,argv=command,started=time.time()))
        return process

    def run(self, command, log, env=None, record=None):
        if record is not None:
            process = self.launch(command, log, record, env)
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f'Worker exited {returncode}; inspect {log}')
            return
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        with Path(log).open('a') as stream:
            result=subprocess.run(command,cwd=REPO,env=env or self.env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Worker exited {result.returncode}; inspect {log}')

    def prepare(self, case, method, setting, budget):
        import robomme_plus
        from robomme_plus import BenchmarkEnvBuilder
        tasks=sorted(BenchmarkEnvBuilder.get_task_list())
        records={}
        for task in tasks:
            builder=BenchmarkEnvBuilder(task,dataset='train')
            records[task]=[builder.get_episode_metadata(i) for i in range(builder.get_episode_num())]
        study=Study(case/'study',method=method,setting=setting,budget=budget,records=records)
        pin=source_fingerprint(Path(robomme_plus.__file__).parent)
        path=study.root/'benchmark_source.json'
        if path.exists() and json.loads(path.read_text())!=pin:
            raise RuntimeError('Prepared benchmark changed')
        write(path,pin)
        write(study.root/'study.json',study.config)
        return study,tasks

    def relay(self, case, method, setting, budget):
        record=case/'proxy_process.json'
        if record.is_file() and alive(json.loads(record.read_text())['pid']):
            return
        write(case/'api_context.json',dict(method=method,setting=setting,budget=budget,phase='development'))
        command=[sys.executable,'-m','benchmark_io.bedrock_proxy','--output',str(case/'development_api'),
            '--token-file',str(TOKEN),'--port',str(self.proxy_port),'--max-calls','100000','--effort','high',
            '--context',str(case/'api_context.json')]
        process=self.launch(command,case/'proxy.log',record)
        time.sleep(1)
        if process.poll() is not None:
            raise RuntimeError(f'Model relay failed; inspect {case}/proxy.log')

    def providers(self, gpus=None):
        directory=self.root/'runtime/providers'
        directory.mkdir(parents=True,exist_ok=True)
        for gpu in (gpus or self.gpus):
            # Both lanes may need the same native provider stack at once.
            with (directory/f'gpu{gpu}_startup.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                self.start_providers([gpu])

    def start_providers(self, gpus):
        root=self.root/'runtime/providers'
        base=Path(os.environ.get('ROBOMME_ASPIRE_PROVIDER_ROOT',REPO/'scripts/plus_eval/aspire/ASPIRE'))
        for gpu in (gpus or self.gpus):
            for name,offset in [('sam3',4),('graspnet',5),('pyroki',6),('curobo',8)]:
                port=self.provider_port(gpu)+offset
                record=root/f'gpu{gpu}_{name}.json'
                url=f'http://127.0.0.1:{port}/healthz'
                def healthy():
                    try:
                        with urllib.request.urlopen(url,timeout=2) as response:
                            return response.status==200
                    except (OSError,ValueError):
                        return False
                if healthy():
                    if not record.is_file():
                        raise RuntimeError(f'Provider port {port} belongs to another run')
                    continue
                if record.is_file():
                    stop_owned(record)
                command=[str(base/f'.venv-{name}/bin/python'),'-m',f'cap_harness.providers.{name}.service',
                         '--port',str(port)]
                if name=='sam3':
                    command+=['--checkpoint',str(Path(os.environ.get('ROBOMME_SAM3_CHECKPOINT',base/'checkpoints/sam3.pt')))]
                env={**self.env,'CUDA_VISIBLE_DEVICES':gpu,'CAP_HARNESS_VENDOR_ROOT':str(base/'third_party'),
                     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu'}
                env.pop('ROBOMME_AGENTIC_JOB',None)  # Shared services outlive individual jobs.
                process=self.launch(command,root/f'gpu{gpu}_{name}.log',record,env)
                deadline=time.time()+240
                while not healthy():
                    if process.poll() is not None or time.time()>deadline:
                        raise RuntimeError(f'{name} provider startup failed on GPU {gpu}')
                    self.status(phase='starting_providers',provider=name,gpu=gpu)
                    time.sleep(5)

    def wait_development(self, case, study):
        result=case/'development/development_result.json'
        record=case/'development_process.json'
        while not result.is_file():
            if not record.is_file() or not alive(json.loads(record.read_text())['pid']):
                raise RuntimeError(f'Development exited without a final report: {case}')
            self.status(phase='development',training_starts=len(study.attempts()),
                        api_requests=len(list((case/'development_api').glob('*/request.json'))))
            write(case/'api_cost_summary.json',summarize(case/'development_api'))
            time.sleep(20)
        data=json.loads(result.read_text())
        if data['returncode'] or any(not path and task not in data.get('deadline_tasks', [])
                                    for task, path in data['selected_programs'].items()):
            raise RuntimeError(f'ASPIRE campaign incomplete; preserve and resume its native conversation: {case}')

    def aspire_development(self, case, study):
        result=case/'development/development_result.json'
        record=case/'development_process.json'
        if not result.is_file() and not (record.is_file() and alive(json.loads(record.read_text())['pid'])):
            command=[sys.executable,'-m','aspire.campaign','--study',str(study.root),
                '--output',str(case/'development'),'--token-file',str(TOKEN),'--image',IMAGE,
                '--demos',str(self.root/'benchmark/h5'),'--proxy-url',f'http://127.0.0.1:{self.proxy_port}',
                '--provider-port-base',str(self.provider_port(self.gpus[0])),'--gpus',*self.gpus]
            if (case/'development').exists():
                command.append('--resume')
            self.launch(command,case/'development.log',record)
        self.wait_development(case,study)

    def freeze_policies(self, case, study, tasks, budget):
        """Publish immutable policies before allowing the next development session."""
        import robomme_plus
        root=case/'evaluation'
        profile = 'full' if budget in (0,10) else 'logs'
        if study.config['method'] == 'rax':
            for task in tasks:
                development, development_study = rax_source(case, task)
                report = json.loads((development/'development_result.json').read_text())
                if report['returncode']:
                    raise RuntimeError(f'RAX development failed: {task}')
                if not report.get('submission'):
                    continue
                state = json.loads((development/'state.json').read_text())
                if state.get('submission') != report['submission']:
                    raise RuntimeError(f'Native submitted selection changed: {task}')
                output = root/task
                if (output/'frozen_evaluation.json').is_file():
                    EvaluationStudy(output)
                    continue
                manifest = state['bundles'][state['submission']['bundle']]
                EvaluationStudy.freeze(output,development_study=development_study,
                    policy_directory=development/'bundles'/manifest['directory'],
                    episodes={task:list(range(50))},benchmark_root=Path(robomme_plus.__file__).parent,
                    artifact_profile=profile)
            return
        if not (root/'frozen_evaluation.json').is_file():
            selected=json.loads((case/'development/development_result.json').read_text())['selected_programs']
            staging=case/'selected_policies'
            staging.mkdir(exist_ok=True)
            for task,path in selected.items():
                if path:
                    shutil.copy2(case/'development/workspace'/path,staging/f'{task}.py')
            write(staging/'selection.json',selected)
            EvaluationStudy.freeze(root,development_study=study.root,policy_directory=staging,
                episodes={task:list(range(50)) for task in tasks},benchmark_root=Path(robomme_plus.__file__).parent,
                artifact_profile=profile)
        else:
            EvaluationStudy(root)

    def aspire_evaluation(self, case, study, tasks, budget):
        root=case/'evaluation'
        if not (root/'frozen_evaluation.json').is_file():
            self.freeze_policies(case, study, tasks, budget)
        frozen=EvaluationStudy(root)
        def episode(task,index):
            output=root/'episodes'/task/f'episode_{index:03d}'
            path=output/'result.json'
            if path.is_file():
                return json.loads(path.read_text())
            record = output/'worker_process.json'
            saved = json.loads(record.read_text()) if record.is_file() else {}
            adopt = bool(saved and alive(saved['pid']))
            if adopt:
                actual = Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().split('\0')
                if actual[:len(saved['argv'])] != saved['argv']:
                    raise RuntimeError('Saved evaluation PID belongs to another process')
            placement = json.loads((output/'execution_placement.json').read_text()) if adopt else {}
            with self.eval_slots.acquire(f'aspire/{study.config["setting"]}/R{budget}/{task}/{index}',
                                         gpu=placement.get('gpu')) as (gpu, slot):
                port = self.provider_port(gpu)
                if adopt:
                    while not path.is_file():
                        if not alive(saved['pid']):
                            raise RuntimeError('Adopted frozen episode exited without its result')
                        time.sleep(1)
                    return json.loads(path.read_text())
                # A crashed reserved test is retained as infrastructure failure, never silently rerun.
                prior=next((a for a in frozen.attempts() if a['task']==task and a['episode']==index),None)
                if prior:
                    row=dict(task=task,episode=index,task_success=False,error='Interrupted reserved episode',
                             failure_kind='infrastructure',prior_attempt=prior)
                    write(path,row)
                    return row
                request=output/'request.json'
                write(request,dict(task=task,episode=index,code=(root/'policy'/f'{task}.py').read_text(),
                                   purpose='frozen_full_evaluation'))
                command=[sys.executable,'-m','aspire.run_episode','--study',str(study.root),
                    '--output',str(output/'artifacts'),'--request',str(request),'--result',str(path),
                    '--image',IMAGE,'--provider-port-base',str(port),'--evaluation',str(root)]
                env={**self.env,'CUDA_VISIBLE_DEVICES':gpu}
                write(output/'execution_placement.json',dict(gpu=gpu,slot=slot,provider_port=port))
                self.run(command,output/'operator.log',env,record=record)
                return json.loads(path.read_text())
        rows=[]
        self.status(phase='evaluation',evaluation_workers=self.eval_slots.capacity,
                    evaluation_gpus=self.eval_gpus,workers_per_gpu=self.eval_workers_per_gpu)
        with ThreadPoolExecutor(max_workers=self.eval_slots.capacity) as pool:
            jobs=[pool.submit(episode,task,index) for task in tasks for index in range(50)]
            for future in as_completed(jobs):
                rows.append(future.result())
                self.status(phase='evaluation',completed=len(rows),total=800,
                            successes=sum(bool(r.get('task_success')) for r in rows))
                write(root/'progress.json',dict(completed=len(rows),total=800,
                    successes=sum(bool(r.get('task_success')) for r in rows)))
        write(root/'results.json',sorted(rows,key=lambda r:(r['task'],r['episode'])))
        return rows

    def rax_development(self, case, study, tasks):
        def task_job(task):
            slot=self.slots.get();gpu,_=slot
            try:
                output, study_root = rax_source(case, task)
                if (output/'development_result.json').is_file():
                    return
                command=[sys.executable,'-m','robomme_full','develop','--study',str(study_root),
                    '--output',str(output),'--task',task,'--gpu',gpu,'--image',IMAGE,'--token-file',str(TOKEN),
                    '--proxy-url',f'http://127.0.0.1:{self.proxy_port}']
                record=case/'logs'/f'develop_{task}_process.json'
                if record.is_file():
                    saved=json.loads(record.read_text())
                    pid=saved['pid']
                    if alive(pid):
                        actual=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
                        if actual[:len(saved['argv'])]!=saved['argv']:
                            raise RuntimeError('Development PID no longer belongs to this task')
                    else:
                        if not (output/'configuration.json').is_file():
                            raise RuntimeError(f'Unfinished RAX development has no configuration: {task}')
                        # Retain the original per-task start before replacing the
                        # process record; this is not another experiment.
                        from .development_deadline import TaskClock
                        TaskClock(study_root, task).start(saved['started'])
                        command.append('--resume')
                        process=self.launch(command,case/'logs'/f'develop_{task}.log',record)
                        pid=process.pid
                else:
                    process=self.launch(command,case/'logs'/f'develop_{task}.log',record)
                    pid=process.pid
                while not (output/'development_result.json').is_file():
                    if not alive(pid):
                        raise RuntimeError(f'RAX development exited without a report: {task}')
                    time.sleep(2)
                report=json.loads((output/'development_result.json').read_text())
                if report['returncode']:
                    raise RuntimeError(f'RAX development failed: {task}')
            finally:
                self.slots.put(slot)
        with ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            jobs=[pool.submit(task_job,task) for task in tasks]
            for i,future in enumerate(as_completed(jobs),1):
                future.result()
                self.status(phase='development',tasks_finished=i,tasks_total=len(tasks),training_starts=len(study.attempts()))
                write(case/'api_cost_summary.json',summarize(case/'development_api'))
    def rax_evaluation(self, case, study, tasks, budget):
        def task_job(task):
            with self.eval_slots.acquire(f'rax/{study.config["setting"]}/R{budget}/{task}') as (gpu, slot):
                development, development_study=rax_source(case,task)
                result=json.loads((development/'development_result.json').read_text())
                if result['returncode']:
                    raise RuntimeError(f'RAX development process failed: {task}')
                output=case/'evaluation'/task
                if not result.get('submission'):
                    rows=[dict(task=task,episode=i,task_success=False,
                               failure_kind='no_submitted_policy',executed=False) for i in range(50)]
                    write(output/'results.json',rows)
                    return rows
                if not (output/'results.json').is_file():
                    command=[sys.executable,'-m','robomme_full','evaluate','--study',str(development_study),
                        '--development',str(development),'--output',str(output),'--gpu',gpu,
                        '--artifacts','full' if budget in (0,10) else 'logs']
                    record = case/'logs'/f'eval_{task}_process.json'
                    saved = json.loads(record.read_text()) if record.is_file() else {}
                    if saved and alive(saved['pid']):
                        actual = Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().split('\0')
                        if actual[:len(saved['argv'])] != saved['argv']:
                            raise RuntimeError('Saved RAX evaluation PID has a different owner')
                        if saved['argv'][saved['argv'].index('--gpu')+1] != gpu:
                            raise RuntimeError('Cannot move a running RAX batch to another GPU')
                        while alive(saved['pid']):
                            time.sleep(2)
                        if not (output/'result.json').is_file():
                            raise RuntimeError(f'Adopted RAX batch exited without final result: {task}')
                    elif not (output/'result.json').is_file():
                        if saved:
                            command.append('--resume-unstarted')
                        write(output/'worker_placement.json',dict(gpu=gpu,slot=slot))
                        self.run(command,case/'logs'/f'eval_{task}.log',record=record)
                    state=json.loads((development/'state.json').read_text())
                    rows=[dict(task=task,episode=r['formal_episode_index']-1,
                               task_success=r.get('formal_outcome')=='success',native_result=r,executed=True)
                          for r in state['results'] if r.get('mode')=='formal']
                    if len(rows)!=50 or len({r['episode'] for r in rows})!=50:
                        raise RuntimeError(f'RAX incomplete or duplicate formal results: {task}')
                    write(output/'results.json',rows)
                return json.loads((output/'results.json').read_text())
        rows=[]
        self.status(phase='evaluation',evaluation_workers=self.eval_slots.capacity,
                    evaluation_gpus=self.eval_gpus,workers_per_gpu=self.eval_workers_per_gpu)
        with ThreadPoolExecutor(max_workers=self.eval_slots.capacity) as pool:
            for future in as_completed([pool.submit(task_job,task) for task in tasks]):
                rows.extend(future.result())
                self.status(phase='evaluation',completed=len(rows),total=800,
                            successes=sum(r['task_success'] for r in rows))
                write(case/'evaluation/progress.json',dict(completed=len(rows),total=800,
                    successes=sum(r['task_success'] for r in rows)))
        write(case/'evaluation/results.json',sorted(rows,key=lambda r:(r['task'],r['episode'])))
        return rows

    def configurations(self):
        for method in self.methods:
            for setting,budgets in [('AfD',[0]),('AfI',[1,3,5,10]),('AfD+I',[1,3,5,10])]:
                for budget in budgets:
                    yield method,setting,budget,self.root/method/setting/f'R{budget}'

    def develop_case(self, case, method, setting, budget):
        study,tasks=self.prepare(case,method,setting,budget)
        development_done = ((case/'development/development_result.json').is_file() if method=='aspire'
            else all((case/'development'/task/'development_result.json').is_file() for task in tasks))
        if not development_done:
            self.relay(case,method,setting,budget)
            self.status(phase='development')
        if method=='aspire':
            if budget and not development_done:
                self.providers()
            self.aspire_development(case,study)
        else:
            self.rax_development(case,study,tasks)
        stop_owned(case/'proxy_process.json')
        write(case/'api_cost_summary.json',summarize(case/'development_api'))
        self.status(phase='freezing')
        self.freeze_policies(case,study,tasks,budget)
        write(case/'development_complete.json',dict(method=method,setting=setting,budget=budget,
            completed=time.time(),**training_accounting(case),
            policies_frozen=True,test_feedback_to_development=False))
        self.status(phase='awaiting_evaluation')

    def evaluate_case(self, case, method, setting, budget):
        study=Study.open(case/'study')
        tasks=sorted(study.config['records'])
        if method=='aspire':
            self.providers(self.eval_gpus)
            rows=self.aspire_evaluation(case,study,tasks,budget)
        else:
            rows=self.rax_evaluation(case,study,tasks,budget)
        summary=dict(method=method,setting=setting,budget=budget,total=800,
            successes=sum(bool(r['task_success']) for r in rows),
            tasks={task:dict(successes=sum(bool(r['task_success']) for r in rows if r['task']==task),total=50) for task in tasks},
            evaluation_api_cost_usd=0.,evaluation_api_requests=0,
            evaluation_api_reason='Frozen isolated programs; RAX optional Gemini disabled',
            **training_accounting(case),completed=time.time())
        summary['success_rate']=summary['successes']/800
        write(case/'evaluation/api_usage.json',dict(phase='evaluation',requests=0,cost_usd=0.,
              reason=summary['evaluation_api_reason']))
        write(case/'completed.json',summary)
        self.status(phase='completed',completed=800,total=800,successes=summary['successes'])

    def execute(self):
        for method,setting,budget,case in self.configurations():
            if (case/'completed.json').is_file():
                continue
            ready = case/'development_complete.json'
            if self.lane == 'development' and ready.is_file():
                continue
            if shutil.disk_usage(self.root).free < 150*1024**3:
                raise RuntimeError('Less than 150 GiB free; preserve data and stop new work')
            self.current=dict(method=method,setting=setting,budget=budget,case=str(case))
            if self.lane == 'evaluation':
                self.status(phase='waiting_for_development')
                while not ready.is_file():
                    time.sleep(5)
                if not json.loads(ready.read_text()).get('policies_frozen'):
                    raise RuntimeError('Development ready marker does not certify frozen policies')
            elif not ready.is_file():
                self.status(phase='preparing')
                self.develop_case(case,method,setting,budget)
            if self.lane != 'development':
                self.evaluate_case(case,method,setting,budget)
        # Provider services are shared with the other lane; only the monitor owns cleanup.
        self.current={}
        self.status(phase='complete',configurations_completed=9*len(self.methods))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',default=['0','1'])
    p.add_argument('--method',choices=('aspire','rax'))
    p.add_argument('--proxy-port',type=int,default=18177)
    p.add_argument('--eval-gpus',nargs='+')
    p.add_argument('--eval-workers-per-gpu',type=int,default=1)
    p.add_argument('--lane',choices=('all','development','evaluation'),default='all')
    args=p.parse_args()
    pipeline=Pipeline(args.root,args.gpus,method=args.method,proxy_port=args.proxy_port,
        eval_gpus=args.eval_gpus,eval_workers_per_gpu=args.eval_workers_per_gpu,lane=args.lane)
    with (pipeline.state_root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop = threading.Event()
        monitor = threading.Thread(target=pipeline.heartbeat,args=(stop,),daemon=True)
        monitor.start()
        try:
            pipeline.execute()
        except BaseException as exc:
            pipeline.status(phase='needs_attention',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            stop.set()
            monitor.join(timeout=5)


if __name__=='__main__':
    main()

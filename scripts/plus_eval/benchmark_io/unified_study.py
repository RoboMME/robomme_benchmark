"""Configurable shared development and isolated evaluation slots."""
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from .full_study import Pipeline, REPO, IMAGE, alive, write
from .protocol import Study
from .evaluation import EvaluationStudy
from .runtime_policy import RuntimePolicy, configured
from .costs import summarize
from .schedule_policy import development_slots, evaluation_slots, evaluation_placement, evaluation_gate, evaluation_excluded_cases, evaluation_episode_indices


def configurations():
    for setting,budget in [('AfD',0),('AfI',1),('AfD+I',1),('AfI',3),('AfD+I',3),
                           ('AfI',5),('AfD+I',5),('AfI',10),('AfD+I',10)]:
        for method in ('aspire','rax'):
            yield f'{method}/{setting}/R{budget}',method,setting,budget


class Queue:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'queue.sqlite3'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS cases (
                  key TEXT PRIMARY KEY, method TEXT, setting TEXT, budget INTEGER, ordinal INTEGER,
                  ready INTEGER DEFAULT 0, dispatched INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, kind TEXT, case_key TEXT, task TEXT, episode INTEGER,
                  state TEXT DEFAULT 'queued', gpu TEXT, pid INTEGER, argv TEXT,
                  launched REAL, last_activity REAL, finished REAL, error TEXT);
            ''')

    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        return db

    def seed(self):
        with self.connect() as db:
            for ordinal,(key,method,setting,budget) in enumerate(configurations()):
                db.execute('INSERT OR IGNORE INTO cases(key,method,setting,budget,ordinal) VALUES(?,?,?,?,?)',
                           (key,method,setting,budget,ordinal))
                db.execute('INSERT OR IGNORE INTO jobs(id,kind,case_key) VALUES(?,?,?)',('develop/'+key,'develop',key))

    def jobs(self, state=None, kind=None):
        clauses,values=[],[]
        for key,value in [('state',state),('kind',kind)]:
            if value is not None:clauses.append('j.'+key+'=?');values.append(value)
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT j.*,c.method,c.setting,c.budget,c.ordinal FROM jobs j '
                'JOIN cases c ON c.key=j.case_key'+(' WHERE '+' AND '.join(clauses) if clauses else '')+
                ' ORDER BY c.ordinal,j.episode,j.task',values)]

    def update(self, ident, **fields):
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(k+'=?' for k in fields)+' WHERE id=?',[*fields.values(),ident])

    def ready(self, key, tasks):
        marker=self.root/key/'development_complete.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('policies_frozen'):
            raise ValueError('Evaluation requires a frozen development handoff')
        with self.connect() as db:
            for episode in range(50):
                for task in tasks:
                    db.execute('INSERT OR IGNORE INTO jobs(id,kind,case_key,task,episode) VALUES(?,?,?,?,?)',
                               (f'evaluate/{key}/{task}/{episode:03d}','evaluate',key,task,episode))
            db.execute('UPDATE cases SET ready=1 WHERE key=?',(key,))

    def next_job(self, kind):
        if kind=='evaluate' and not evaluation_gate(self.root)['enabled']:
            return None
        excluded = evaluation_excluded_cases(self.root) if kind=='evaluate' else []
        exclusion = ('AND c.key NOT IN ('+','.join('?' for _ in excluded)+') ') if excluded else ''
        indices = evaluation_episode_indices(self.root) if kind=='evaluate' else None
        if indices == []:
            return None
        subset = ('AND j.episode IN ('+','.join('?' for _ in indices)+') ') if indices is not None else ''
        # Give each ready configuration a worker before assigning extra slots.
        # A newly frozen case must not monopolize the queue while catching up
        # with cumulative dispatch counts from the completed sanity pass.
        active = ("LEFT JOIN (SELECT case_key,COUNT(*) AS count FROM jobs "
                  "WHERE kind='evaluate' AND state IN ('running','starting') GROUP BY case_key) "
                  "active ON active.case_key=c.key ") if kind=='evaluate' else ''
        with self.connect() as db:
            row=db.execute('SELECT j.*,c.method,c.setting,c.budget,c.ordinal FROM jobs j JOIN cases c ON c.key=j.case_key '
                +active+"WHERE j.state='queued' AND j.kind=? "+('AND c.ready=1 ' if kind=='evaluate' else '')+exclusion+subset+
                ('ORDER BY COALESCE(active.count,0),c.dispatched,c.ordinal,j.episode,j.task ' if kind=='evaluate' else 'ORDER BY c.ordinal ')+
                'LIMIT 1',(kind,*excluded,*(indices or []))).fetchone()
            return dict(row) if row else None

    def mark_dispatched(self,key):
        with self.connect() as db:db.execute('UPDATE cases SET dispatched=dispatched+1 WHERE key=?',(key,))


def job_output(root, job):
    case=Path(root)/job['case_key']
    return case/'development_job' if job['kind']=='develop' else case/'evaluation/episodes'/job['task']/f'episode_{job["episode"]:03d}'


def job_tag(root, ident):
    return str(Path(root).resolve())+'::'+ident


def find_worker(root, job):
    """Adopt an existing child across a scheduler crash, without restarting physics."""
    candidates=[]
    expected=json.loads(job['argv']) if job.get('argv') else None
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if argv!=expected or not alive(int(proc.name)):continue
            env=(proc/'environ').read_bytes().split(b'\0')
            if ('ROBOMME_AGENTIC_JOB='+job_tag(root,job['id'])).encode() in env:
                candidates.append(int(proc.name))
        except (OSError,UnicodeError):continue
    if len(candidates)>1:raise RuntimeError('Duplicate live worker identities')
    return candidates[0] if candidates else None


def environment(root):
    root=Path(root).resolve()
    env=Pipeline(root,['0','1']).env
    code = root/'runtime_source'
    if code.is_dir():
        env.update(ROBOMME_AGENTIC_CODE=str(code),ROBOMME_AGENTIC_REPO=str(REPO),
                   PYTHONPATH=':'.join(map(str,[root/'benchmark/src',code,code/'rax',code/'aspire/ASPIRE/src'])))
    plan = root/'study_plan.json'
    if plan.exists():
        env['ROBOMME_AGENTIC_IMAGE'] = json.loads(plan.read_text())['image']
    token = root/'runtime/private/relay-token'
    if token.exists():
        env['ROBOMME_AGENTIC_TOKEN_FILE'] = str(token)
    protocol = root/'protocol'
    selection = root/'future_development_protocol.json'
    if selection.exists():
        protocol = (root/json.loads(selection.read_text())['relative_directory']).resolve()
        if not protocol.is_relative_to(root) or not (protocol/'bundle.json').is_file():
            raise ValueError('Invalid future development protocol selection')
    env.update(ROBOMME_AGENTIC_RUNTIME_POLICY=str(root/'runtime_policy.json'),
               ROBOMME_AGENTIC_PROMPTS=str(protocol),ROBOMME_AGENTIC_PROVIDER_BASE='23110')
    return env


class CasePipeline(Pipeline):
    def __init__(self,root,job):
        super().__init__(root,[job['gpu']],method=job['method'],proxy_port=24100+job['ordinal'],
                         eval_gpus=['0','1'],eval_workers_per_gpu=evaluation_slots(root)[1],lane='development')
        self.state_root=Path(root)/job['case_key']/'development_job'
        self.state_root.mkdir(parents=True,exist_ok=True)
        self.current=dict(method=job['method'],setting=job['setting'],budget=job['budget'],case=str(Path(root)/job['case_key']))


def evaluate_one(root,job):
    case=Path(root)/job['case_key']; output=job_output(root,job)
    task,index=job['task'],job['episode']
    if (output/'result.json').exists():return json.loads((output/'result.json').read_text())
    if job['method']=='aspire':
        from aspire.isolated import run_program
        frozen=EvaluationStudy(case/'evaluation')
        if not (frozen.root/'policy'/f'{task}.py').exists():
            return dict(task=task,episode=index,task_success=False,executed=False,
                        failure_kind='no_policy_at_development_deadline')
        row=run_program(study_root=case/'study',output_root=output/'artifacts',task=task,episode=index,
            code=(frozen.root/'policy'/f'{task}.py').read_text(),image=IMAGE,purpose='frozen_full_evaluation_v2',
            provider_port_base=Pipeline.provider_port(job['gpu']),study_override=frozen,dataset='test')
        row.update(task=task)
        return row
    from services.controller.config import Configuration
    from services.controller.storage import verify
    from services.controller.supervisor import formal_outcome
    from services.robomme_plus.backend import RoboMMEPlusBackend
    from robomme_full import EvaluationSupervisor
    development=case/'development'/task
    report=json.loads((development/'development_result.json').read_text())
    if not report.get('submission'):
        return dict(task=task,episode=index,task_success=False,executed=False,failure_kind='no_submitted_policy')
    frozen=EvaluationStudy(case/'evaluation'/task)
    state=json.loads((development/'state.json').read_text());original=state['bundles'][state['submission']['bundle']]
    selected = original
    verify(frozen.root/'policy',selected)
    os.environ.update(ROBOMME_PLUS_STUDY=str(case/'study'),ROBOMME_PLUS_FROZEN_EVALUATION=str(frozen.root),
                      ROBOMME_PLUS_EVAL_ARTIFACTS=frozen.config['artifact_profile'],
                      ROBOMME_PLUS_PURPOSE='frozen_full_evaluation_v2',RAX_ARTIFACT_ROOT=str(case))
    gpu_uuid=subprocess.check_output(['nvidia-smi','-i',job['gpu'],'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    config=replace(Configuration.from_dict(state['configuration']),root=output,formal_episodes=1,
                   formal_seed=index,formal_workers=1,sim_gpu=gpu_uuid)
    supervisor=EvaluationSupervisor(config,lambda:RoboMMEPlusBackend(config))
    try:
        imported,manifest=supervisor.register(frozen.root/'policy',workspace_path=selected['workspace_path'])
        if manifest['sha256']!=selected['sha256']:raise ValueError('Frozen bundle identity changed')
        supervisor.state['submission']=dict(bundle=imported,sha256=manifest['sha256'])
        supervisor.state['formal_reserved']=True;supervisor._save()
        native=supervisor.run_submission();native['formal_outcome']=formal_outcome(native)
        row = dict(task=task,episode=index,task_success=native['formal_outcome']=='success',native_result=native,
                   executed=native.get('start_step_id') is not None)
        return row
    finally:
        supervisor.close()


def worker(root,ident):
    queue=Queue(root);job=next(j for j in queue.jobs() if j['id']==ident)
    output=job_output(root,job);output.mkdir(parents=True,exist_ok=True)
    case=Path(root)/job['case_key']
    worker_lock=(output/'worker.lock').open('a')
    fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(environment(root),CUDA_VISIBLE_DEVICES=job['gpu'],ROBOMME_AGENTIC_JOB=job_tag(root,ident))
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):write(output/'heartbeat.json',dict(pid=os.getpid(),time=time.time()))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if job['kind']=='develop':
            pipeline=CasePipeline(root,job)
            monitor=threading.Thread(target=pipeline.heartbeat,args=(stop,),daemon=True);monitor.start()
            pipeline.develop_case(case,job['method'],job['setting'],job['budget'])
            write(output/'result.json',dict(ok=True,policies_frozen=True))
        else:
            write(output/'result.json',evaluate_one(root,job))
    except BaseException as exc:
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
        row=dict(ok=False,error=f'{type(exc).__name__}: {exc}',failure_kind='infrastructure')
        if job['kind']=='evaluate':row.update(task=job['task'],episode=job['episode'],task_success=False)
        write(output/'result.json',row)
        raise
    finally:
        stop.set();thread.join(2)
        worker_lock.close()


def aggregate(root,key):
    case=Path(root)/key
    rows=[json.loads(p.read_text()) for p in (case/'evaluation/episodes').glob('*/*/result.json')]
    rows.sort(key=lambda r:(r['task'],r['episode']))
    successes=sum(bool(r.get('task_success')) for r in rows)
    write(case/'evaluation/progress.json',dict(completed=len(rows),total=800,successes=successes))
    if len(rows)==800 and len({(r['task'],r['episode']) for r in rows})==800:
        method,setting,budget=key.split('/')
        write(case/'evaluation/results.json',rows)
        write(case/'completed.json',dict(method=method,setting=setting,budget=int(budget[1:]),total=800,
            successes=successes,success_rate=successes/800,completed=time.time(),
            tasks={task:dict(total=50,successes=sum(bool(r.get('task_success')) for r in rows if r['task']==task)) for task in sorted({r['task'] for r in rows})},
            infrastructure_errors=sum(r.get('failure_kind')=='infrastructure' for r in rows),
            evaluation_api_requests=0,evaluation_api_cost_usd=0.0))
        write(case/'evaluation_status.json',dict(phase='completed',updated=time.time()))
        write(case/'evaluation/api_usage.json',dict(requests=0,cost_usd=0.0,reason='Frozen programs; optional Gemini disabled'))


def memory_available_gib():
    rows={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return rows['MemAvailable']/1024**2,rows['MemTotal']/1024**2


class Scheduler:
    def __init__(self,root):
        self.root=Path(root).resolve();self.queue=Queue(root);self.queue.seed()
        self.children={}
        self.policy=RuntimePolicy(**json.loads((self.root/'runtime_policy.json').read_text()))
        self.env=environment(root)
        self.last_snapshot=0

    def admit(self,job,running):
        available,total=memory_available_gib()
        def estimate(j):return (24 if j['method']=='aspire' else 12) if j['kind']=='develop' else (16 if j['method']=='aspire' else 8)
        return available>=32 and sum(estimate(j) for j in running)+estimate(job)<=total-64

    def start(self,job,gpu):
        job={**job,'gpu':gpu};output=job_output(self.root,job);output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','benchmark_io.unified_study','worker','--root',str(self.root),'--job',job['id']]
        self.queue.update(job['id'],state='starting',gpu=gpu,launched=time.time(),last_activity=time.time(),argv=json.dumps(command))
        env={**self.env,'CUDA_VISIBLE_DEVICES':gpu,'ROBOMME_AGENTIC_JOB':job_tag(self.root,job['id'])}
        with (output/'operator.log').open('a') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(output/'process.json',dict(pid=process.pid,argv=command,started=time.time(),gpu=gpu,job=job['id']))
        self.queue.update(job['id'],state='running',pid=process.pid)
        self.children[job['id']]=process
        if job['kind']=='evaluate':
            self.queue.mark_dispatched(job['case_key'])
            write(self.root/job['case_key']/'evaluation_status.json',dict(phase='evaluation',updated=time.time(),shared_queue=True))

    def poll(self):
        from .job_cleanup import cleanup
        for job in self.queue.jobs('running')+self.queue.jobs('starting'):
            output=job_output(self.root,job);result=output/'result.json'
            if job['id'] in self.children:self.children[job['id']].poll()
            pid=find_worker(self.root,job)
            running=pid is not None
            if running and (pid!=job['pid'] or job['state']=='starting'):
                self.queue.update(job['id'],pid=pid,state='running');job['pid']=pid
            elapsed=time.time()-(job['launched'] or time.time())
            limit=self.policy.worker_seconds if job['kind']=='evaluate' else self.policy.development_seconds
            stale=False
            if job['kind']=='develop':
                case=self.root/job['case_key']
                activity=max([job['last_activity'] or job['launched']]+[p.stat().st_mtime for p in (case/'development_api').glob('*/request.json')])
                # Active simulator attempts have their own shorter watchdogs.
                ledgers=case/'study/attempts.sqlite3'
                if ledgers.exists():
                    with sqlite3.connect(f'file:{ledgers}?mode=ro',uri=True) as db:
                        last=db.execute('SELECT MAX(COALESCE(ended,started)) FROM attempts').fetchone()[0]
                    activity=max(activity,last or 0)
                stale=time.time()-activity>self.policy.development_idle_seconds
            if running and not result.exists() and elapsed<limit and not stale:continue
            # A finished worker gets a short cleanup grace; orphan descendants
            # are also collected when their parent already exited successfully.
            if running and result.exists() and time.time()-result.stat().st_mtime<10:continue
            artifact=self.root/job['case_key']/'development' if job['kind']=='develop' else output
            write(output/'cleanup.json',cleanup(job_tag(self.root,job['id']),artifact))
            if not result.exists():
                row=dict(ok=False,failure_kind='infrastructure',error='Worker watchdog expired' if elapsed>=limit or stale else 'Worker exited without result')
                if job['kind']=='evaluate':row.update(task=job['task'],episode=job['episode'],task_success=False)
                write(result,row)
            data=json.loads(result.read_text())
            failed=data.get('failure_kind')=='infrastructure' or (job['kind']=='develop' and not data.get('ok'))
            self.queue.update(job['id'],state='failed' if failed else 'done',finished=time.time(),error=data.get('error'))
            child=self.children.pop(job['id'],None)
            if child:child.poll()
            if job['kind']=='develop' and not failed:
                study=Study.open(self.root/job['case_key']/'study');self.queue.ready(job['case_key'],sorted(study.config['records']))
            elif job['kind']=='evaluate':aggregate(self.root,job['case_key'])

        # Reconcile the crash window between final worker acknowledgement and
        # enqueueing the immutable test handoff. INSERT OR IGNORE is idempotent.
        with self.queue.connect() as db:
            pending=[r['key'] for r in db.execute('SELECT key FROM cases WHERE ready=0')]
        for key in pending:
            marker=self.root/key/'development_complete.json'
            if marker.is_file() and json.loads(marker.read_text()).get('policies_frozen'):
                study=Study.open(self.root/key/'study')
                self.queue.ready(key,sorted(study.config['records']))

    def dispatch(self):
        running=self.queue.jobs('running')+self.queue.jobs('starting')
        development_total, development_per_gpu = development_slots(self.root)
        evaluation_total, evaluation_by_gpu = evaluation_placement(self.root)
        for kind,capacity in [('develop',development_total),('evaluate',evaluation_total)]:
            while sum(j['kind']==kind for j in running)<capacity:
                job=self.queue.next_job(kind)
                if not job or not self.admit(job,running):break
                counts={gpu:sum(j['kind']==kind and j['gpu']==gpu for j in running) for gpu in ('0','1')}
                limits = {g: development_per_gpu for g in counts} if kind=='develop' else evaluation_by_gpu
                # A lowered limit drains existing episodes without interrupting
                # physics; only new dispatches are subject to the new placement.
                choices=[gpu for gpu in counts if counts[gpu]<limits[gpu]]
                if not choices:break
                available=[]
                for candidate in choices:
                    try:
                        free=float(subprocess.check_output(['nvidia-smi','-i',candidate,'--query-gpu=memory.free',
                            '--format=csv,noheader,nounits'],text=True,timeout=10).strip())
                        if free>=2048:available.append(candidate)
                    except (OSError,ValueError,subprocess.SubprocessError):continue
                if not available:break
                gpu=min(available,key=lambda g:(counts[g],
                    sum(j['kind']==kind and j['gpu']==g and j['method']==job['method'] for j in running),g))
                self.start(job,gpu);running.append({**job,'gpu':gpu})

    def snapshot(self):
        jobs=self.queue.jobs();running=[j for j in jobs if j['state']=='running']
        failed=[j['id'] for j in jobs if j['state']=='failed']
        gate=evaluation_gate(self.root)
        episode_indices=evaluation_episode_indices(self.root)
        development_total, development_per_gpu = development_slots(self.root)
        evaluation_total, evaluation_by_gpu = evaluation_placement(self.root)
        evaluation_per_gpu = evaluation_by_gpu['0'] if len(set(evaluation_by_gpu.values())) == 1 else None
        write(self.root/'runtime/evaluation_config.json',dict(gpus=['0','1'],workers_per_gpu=evaluation_per_gpu,total_workers=evaluation_total,
            workers_by_gpu=evaluation_by_gpu,
            scope='shared_methods',method_gpus={'aspire':['0','1'],'rax':['0','1']},
            asynchronous_development_evaluation=gate['mode']=='asynchronous',development_workers_total=development_total,
            development_workers_per_gpu=development_per_gpu,
            evaluation_gate=gate,evaluation_episode_indices=episode_indices))
        write(self.root/'status.json',dict(updated=time.time(),
            phase='shared_queue' if gate['enabled'] else 'development_first',pid=os.getpid(),
            evaluation_gate=gate,evaluation_episode_indices=episode_indices,
            development_running=sum(j['kind']=='develop' for j in running),development_capacity=development_total,
            active_development_by_gpu={g:sum(j['kind']=='develop' and j['gpu']==g for j in running) for g in ('0','1')},
            evaluation_running=sum(j['kind']=='evaluate' for j in running),evaluation_capacity=evaluation_total,
            evaluation_capacity_by_gpu=evaluation_by_gpu,
            active_evaluation_by_gpu={g:sum(j['kind']=='evaluate' and j['gpu']==g for j in running) for g in ('0','1')},
            jobs_failed=failed,queued=sum(j['state']=='queued' for j in jobs),
            completed=sum(j['state']=='done' for j in jobs)))
        write(self.root/'heartbeat.json',dict(updated=time.time(),pid=os.getpid(),needs_attention=failed))
        for job in jobs:
            if job['state']=='failed':
                name='development_status.json' if job['kind']=='develop' else 'evaluation_status.json'
                write(self.root/job['case_key']/name,dict(updated=time.time(),phase='needs_attention',error=job['error']))
        for key,method,setting,budget in configurations():
            case=self.root/key
            if (case/'development_api').exists():write(case/'api_cost_summary.json',summarize(case/'development_api'))
        from .publish_progress import publish
        publish(self.root,self.root/'progress.json')

    def run(self):
        with (self.root/'scheduler.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            os.environ.update(self.env)
            stop=threading.Event()
            def maintain_providers():
                pipeline=Pipeline(self.root,['0','1']);pipeline.env.update(self.env)
                pipeline.state_root=self.root/'runtime/providers/monitor'
                pipeline.state_root.mkdir(parents=True,exist_ok=True)
                while not stop.is_set():
                    try:
                        pipeline.providers(['0','1'])
                        write(self.root/'runtime/provider_monitor.json',dict(updated=time.time(),ok=True))
                    except Exception as exc:
                        write(self.root/'runtime/provider_monitor.json',dict(updated=time.time(),ok=False,error=str(exc)))
                    stop.wait(15)
            threading.Thread(target=maintain_providers,daemon=True).start()
            while True:
                self.poll();self.dispatch()
                if time.time()-self.last_snapshot>20:
                    self.snapshot();self.last_snapshot=time.time()
                if not any(j['state'] in ('queued','running','starting') for j in self.queue.jobs()):
                    self.snapshot();stop.set();break
                time.sleep(5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['seed','run','worker'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--job')
    args=parser.parse_args()
    if args.command=='seed':Queue(args.root).seed()
    elif args.command=='worker':worker(args.root,args.job)
    else:Scheduler(args.root).run()


if __name__=='__main__':main()

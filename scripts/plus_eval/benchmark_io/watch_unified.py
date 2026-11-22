"""Monitor the durable shared queue; restart its scheduler, never replay a trial."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .full_study import REPO, write
from .unified_study import Queue, environment


def scheduler_environment(root, scheduler_code=None):
    """Select operator scheduling code without replacing the frozen worker runtime."""
    env = environment(root)
    if scheduler_code is not None:
        code = Path(scheduler_code).resolve()
        if not (code/'benchmark_io/unified_study.py').is_file():
            raise ValueError('Scheduler code must contain benchmark_io/unified_study.py')
        env['PYTHONPATH'] = str(code) + os.pathsep + env['PYTHONPATH']
    return env


def run(root, scheduler_code=None):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failures=0
        while not (root/'STOP').exists():
            pending=[j for j in Queue(root).jobs() if j['state'] in ('queued','running','starting')]
            if not pending:break
            command=[sys.executable,'-m','benchmark_io.unified_study','run','--root',str(root)]
            with (root/'scheduler.log').open('a') as log:
                process=subprocess.Popen(command,cwd=REPO,env=scheduler_environment(root, scheduler_code),stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
            write(root/'scheduler_process.json',dict(pid=process.pid,argv=command,started=time.time()))
            while process.poll() is None:
                write(root/'monitor_status.json',dict(pid=os.getpid(),scheduler_pid=process.pid,
                    updated=time.time(),scheduler_restarts=failures,phase='monitoring'))
                time.sleep(10)
            if process.returncode==0:break
            failures+=1
            write(root/'monitor_status.json',dict(updated=time.time(),phase='scheduler_restart',
                returncode=process.returncode,scheduler_restarts=failures))
            if failures>=3:
                write(root/'monitor_status.json',dict(updated=time.time(),phase='needs_attention',
                    reason='Scheduler failed three times; existing workers remain accounted for'))
                return
            time.sleep(10)
        failed=[j['id'] for j in Queue(root).jobs('failed')]
        write(root/'monitor_status.json',dict(updated=time.time(),phase='finished_with_errors' if failed else 'finished',
              failed_jobs=failed,scheduler_restarts=failures))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scheduler-code', type=Path,
                   help='Operator-only scheduler snapshot; episode workers still use the study runtime')
    args = p.parse_args()
    run(args.root, args.scheduler_code)

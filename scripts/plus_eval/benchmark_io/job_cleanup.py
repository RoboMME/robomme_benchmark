"""Bounded cleanup restricted to a scheduler job's inherited identity and mounts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def cleanup(job_tag, artifact_root):
    owned = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():
            continue
        try:
            env=dict(item.split('=',1) for item in (directory/'environ').read_bytes().decode().split('\0') if '=' in item)
        except (OSError,UnicodeError):
            continue
        if env.get('ROBOMME_AGENTIC_JOB')==job_tag:
            owned.append(int(directory.name))
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid in owned:
            try:
                # Check identity again after the TERM grace period (PID reuse).
                entries=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                if ('ROBOMME_AGENTIC_JOB='+job_tag).encode() in entries:os.kill(pid,sig)
            except (OSError,ProcessLookupError): pass
        if sig==signal.SIGTERM: time.sleep(2)
    ids=subprocess.run(['docker','ps','-q'],capture_output=True,text=True,timeout=15).stdout.split()
    removed=[]
    if ids:
        result=subprocess.run(['docker','inspect',*ids],capture_output=True,text=True,timeout=20)
        if result.returncode==0:
            for row in json.loads(result.stdout):
                if any(Path(m.get('Source','/')).is_relative_to(Path(artifact_root).resolve()) for m in row.get('Mounts',[])):
                    subprocess.run(['docker','rm','-f',row['Id']],capture_output=True,timeout=20)
                    removed.append(row['Id'])
    return dict(processes=owned,containers=removed)

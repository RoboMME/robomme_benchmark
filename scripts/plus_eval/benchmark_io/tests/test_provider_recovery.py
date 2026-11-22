import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_io.full_study import Pipeline, provider_launch_config, write
from benchmark_io.unified_study import Queue, Scheduler


def installation(tmp_path):
    binary=tmp_path/'installed/.venv-sam3/bin/python'
    binary.parent.mkdir(parents=True);binary.write_text('');binary.chmod(0o700)
    checkpoint=tmp_path/'weights/sam3.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'checkpoint')
    record=tmp_path/'gpu0_sam3.json'
    write(record,dict(pid=123,argv=[str(binary),'-m','cap_harness.providers.sam3.service',
        '--port','26114','--checkpoint',str(checkpoint)]))
    return binary,checkpoint,record


def test_restart_uses_recorded_installation_and_checkpoint(tmp_path,monkeypatch):
    monkeypatch.delenv('ROBOMME_ASPIRE_PROVIDER_ROOT',raising=False)
    monkeypatch.delenv('ROBOMME_SAM3_CHECKPOINT',raising=False)
    binary,checkpoint,record=installation(tmp_path)
    command,base=provider_launch_config(record,'sam3',26114)
    assert command==json.loads(record.read_text())['argv']
    assert base==binary.parents[2] and command[-1]==str(checkpoint)


def test_missing_launch_files_fail_before_stopping_existing_service(tmp_path,monkeypatch):
    binary,_,record=installation(tmp_path);binary.unlink()
    monkeypatch.setenv('ROBOMME_ASPIRE_PROVIDER_ROOT',str(binary.parents[2]))
    pipeline=Pipeline(tmp_path,['0'])
    target=tmp_path/'runtime/providers/gpu0_sam3.json';target.parent.mkdir(parents=True)
    target.write_text(record.read_text().replace('26114','18114'))
    monkeypatch.setattr('benchmark_io.full_study.urllib.request.urlopen',lambda *a,**k:(_ for _ in ()).throw(OSError('down')))
    monkeypatch.setattr('benchmark_io.full_study.alive',lambda pid:False)
    monkeypatch.setattr('benchmark_io.full_study.stop_owned',lambda *a:pytest.fail('Must validate replacement before stopping'))
    with pytest.raises(FileNotFoundError,match='interpreter'):pipeline.start_providers(['0'])


def test_provider_port_mismatch_is_rejected(tmp_path):
    _,_,record=installation(tmp_path)
    with pytest.raises(ValueError,match='does not match'):provider_launch_config(record,'sam3',26214)


def test_unhealthy_aspire_does_not_block_rax_dispatch(tmp_path,monkeypatch):
    q=Queue(tmp_path)
    with q.connect() as db:
        for ordinal,method in enumerate(('aspire','rax')):
            key=f'{method}/demo-only/R0'
            db.execute('INSERT INTO cases(key,method,setting,budget,ordinal,ready) VALUES(?,?,?,?,?,1)',
                (key,method,'demo-only',0,ordinal))
            db.execute('INSERT INTO jobs(id,kind,case_key,task,episode) VALUES(?,?,?,?,0)',
                ('evaluate/'+key,'evaluate',key,'BinFill'))
    scheduler=Scheduler.__new__(Scheduler)
    scheduler.root=tmp_path;scheduler.queue=q;scheduler.env={'ROBOMME_AGENTIC_PROVIDER_BASE':'26110'}
    scheduler.admit=lambda *args:True
    monkeypatch.setattr('benchmark_io.unified_study.provider_stack_health',lambda base,gpu:dict(sam3=False))
    monkeypatch.setattr('benchmark_io.unified_study.subprocess.check_output',lambda *a,**k:'20000')
    launched=[]
    def start(job,gpu):
        launched.append(job['method']);q.update(job['id'],state='running',gpu=gpu)
    scheduler.start=start;scheduler.dispatch()
    assert launched==['rax']
    assert q.next_job('evaluate')['method']=='aspire'
    assert q.next_job('evaluate',excluded_methods=['aspire']) is None


def test_single_health_timeout_does_not_restart_busy_provider(tmp_path,monkeypatch):
    root=tmp_path/'runtime/providers';root.mkdir(parents=True)
    for name in ('sam3','graspnet','pyroki','curobo'):write(root/f'gpu0_{name}.json',dict(pid=123))
    calls=0
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def request(*a,**k):
        nonlocal calls
        calls+=1
        if calls==1:raise TimeoutError('busy')
        return Response()
    monkeypatch.setattr('benchmark_io.full_study.urllib.request.urlopen',request)
    monkeypatch.setattr('benchmark_io.full_study.alive',lambda pid:True)
    monkeypatch.setattr('benchmark_io.full_study.time.sleep',lambda _:None)
    monkeypatch.setattr('benchmark_io.full_study.stop_owned',lambda *a:pytest.fail('Transient timeout must not stop service'))
    Pipeline(tmp_path,['0']).start_providers(['0'])
    assert calls==5

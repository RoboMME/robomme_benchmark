from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

import robomme_full
from robomme_full import EvaluationSupervisor
from services.controller.supervisor import Supervisor


def supervisor_fixture(tmp_path, monkeypatch):
    supervisor = EvaluationSupervisor.__new__(EvaluationSupervisor)
    supervisor.config = SimpleNamespace(root=tmp_path, task='PickXtimes', formal_workers=1,
                                       formal_seed=0, formal_episodes=4)
    supervisor.published_root = tmp_path/'published'
    supervisor._state_lock = threading.RLock()
    supervisor._save = lambda: None
    supervisor.audit = lambda event: None
    supervisor.budget = lambda: {}
    manifest = dict(files={'controller.py':'hash'}, sha256='bundle-hash', directory='bundle')
    first = dict(mode='formal', bundle='bundle', sha256='bundle-hash', layout_id=0,
                 formal_episode_index=1, reason='exit', returncode=0, task_complete=True,
                 formal_outcome='success', timing={'total_seconds':1})
    # Native record saved on cleanup, but not yet included in batch counters.
    second = dict(first, layout_id=1, formal_episode_index=2, task_complete=None,
                  reason='timeout', error_type='TimeoutError',
                  infrastructure_errors=[{'stage':'environment_shutdown'}])
    second.pop('formal_outcome')
    supervisor.state = dict(active=None, bundles={'bundle':manifest}, results=[first,second],
        submission=dict(bundle='bundle',sha256='bundle-hash'),
        formal_batch=dict(bundle='bundle',sha256='bundle-hash',status='interrupted',
                          episode_count=4,completed_episodes=1,success_count=1))
    attempts = [dict(task='PickXtimes',episode=i,ended=10) for i in range(2)]
    frozen = SimpleNamespace(manifest={'policy_sha256':manifest['files']}, attempts=lambda:attempts)
    monkeypatch.setattr('services.controller.storage.verify',lambda *args:None)
    return supervisor, frozen, attempts


def test_continue_only_unstarted_preserves_success_and_timeout(tmp_path, monkeypatch):
    supervisor,frozen,_ = supervisor_fixture(tmp_path,monkeypatch)
    original=deepcopy(supervisor.state['results'])
    called=[]
    def run(bundle, *, formal, formal_index):
        called.append(formal_index)
        return dict(mode='formal',bundle=bundle,formal_episode_index=formal_index+1,
                    reason='exit',returncode=0,task_complete=False)
    supervisor._run_episode=run
    result=supervisor.resume_unstarted(frozen)
    assert called==[2,3]
    assert result['completed_episodes']==4 and result['success_count']==1
    assert result['error_count']==1 and result['unsuccessful_count']==2
    assert supervisor.state['results'][0]==original[0]
    assert supervisor.state['results'][1]==dict(original[1],formal_outcome='error')


@pytest.mark.parametrize('fault',['unfinalized','missing_result','duplicate','wrong_policy'])
def test_continuation_refuses_uncertain_or_changed_attempts(tmp_path,monkeypatch,fault):
    supervisor,frozen,attempts=supervisor_fixture(tmp_path,monkeypatch)
    if fault=='unfinalized': attempts[1]['ended']=None
    elif fault=='missing_result': supervisor.state['results'].pop()
    elif fault=='duplicate': supervisor.state['results'].append(deepcopy(supervisor.state['results'][0]))
    else: frozen.manifest={'policy_sha256':{'controller.py':'changed'}}
    supervisor._run_episode=lambda *a,**kw:pytest.fail('No simulator may start')
    with pytest.raises((RuntimeError,ValueError)):
        supervisor.resume_unstarted(frozen)


@pytest.mark.parametrize('finalized',[True,False])
def test_cleanup_failure_keeps_saved_result_only_if_ledger_finalized(tmp_path,monkeypatch,finalized):
    supervisor,frozen,attempts=supervisor_fixture(tmp_path,monkeypatch)
    result=deepcopy(supervisor.state['results'][1])
    supervisor.state['results']=[]
    attempts[1]['ended']=10 if finalized else None
    monkeypatch.setenv('ROBOMME_PLUS_FROZEN_EVALUATION',str(tmp_path))
    monkeypatch.setattr(robomme_full,'EvaluationStudy',lambda _:frozen)
    def failed_cleanup(self,*args,**kwargs):
        self.state['results'].append(result)
        raise ValueError('RPC response identity mismatch')
    monkeypatch.setattr(Supervisor,'_run_episode',failed_cleanup)
    if finalized:
        assert supervisor._run_episode('bundle',formal=True,formal_index=1)==result
    else:
        with pytest.raises(ValueError,match='identity mismatch'):
            supervisor._run_episode('bundle',formal=True,formal_index=1)

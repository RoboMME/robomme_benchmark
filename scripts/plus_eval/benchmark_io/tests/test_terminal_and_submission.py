import json
import subprocess
from types import SimpleNamespace

import pytest

from aspire.isolated import communicate_until_episode_end
from benchmark_io.protocol import Study
from services.controller.config import Configuration, Limits
from services.controller.frontend import ResearchFrontend
from services.controller.supervisor import Supervisor


@pytest.mark.parametrize('terminal', ['terminated', 'truncated'])
def test_terminal_simulator_stops_looping_worker_without_waiting_wall_limit(terminal):
    episode = SimpleNamespace(terminated=False, truncated=False)
    stopped = []
    class Process:
        args = ['fake-worker']
        def communicate(self, timeout):
            if stopped:
                return b'', b'preserved stderr'
            setattr(episode, terminal, True)
            raise subprocess.TimeoutExpired(self.args, timeout)
    out, err, reason = communicate_until_episode_end(Process(), episode,
        lambda: stopped.append(True), terminal_grace=0)
    assert stopped == [True]
    assert err == b'preserved stderr'
    assert reason == 'episode_' + terminal


def test_nonterminal_worker_completion_is_preserved():
    episode = SimpleNamespace(terminated=False, truncated=False)
    process = SimpleNamespace(communicate=lambda **kw: (b'normal result', b''))
    assert communicate_until_episode_end(process, episode,
        lambda: pytest.fail('A running episode must not be stopped')) == (b'normal result', b'', None)


@pytest.mark.parametrize('setting,budget', [('demo-only', 0), ('interaction-only', 1), ('demo+interaction', 1)])
@pytest.mark.parametrize('expired', [False, True])
def test_native_submit_accepts_offline_demo_only_but_still_requires_interaction_rehearsal(tmp_path, monkeypatch, setting, budget, expired):
    study = Study(tmp_path/'study', method='rax', setting=setting, budget=budget,
        records={'BinFill': [dict(seed=i, max_steps=100, control_freq=20) for i in range(3)]})
    monkeypatch.setenv('ROBOMME_PLUS_STUDY', str(study.root))
    monkeypatch.setenv('RAX_ARTIFACT_ROOT', str(tmp_path))
    if expired:
        from benchmark_io.development_deadline import TaskClock
        TaskClock(study.root, 'BinFill').start(0)
    limits = Limits(600, 1024, 1., 128, 128, 1000000)
    config = Configuration(root=tmp_path/'session', image='sha256:'+'0'*64, task='BinFill',
        exploration_seeds=(0,)*budget, formal_seed=0, eval_seed=0, formal_eval_seed=1,
        sim_gpu='GPU-00000000-0000-0000-0000-000000000000', development=limits, formal=limits,
        training=limits, benchmark='robomme_plus', agent_cli='claude', agent_effort='high',
        demonstration_context='none' if setting=='interaction-only' else 'completion_sequence')
    supervisor = Supervisor(config, lambda: pytest.fail('No simulator may start during submission'),
        runner_factory=lambda *a: SimpleNamespace(preflight=lambda: None))
    frontend = ResearchFrontend(supervisor, tmp_path/'workspace')
    try:
        source = frontend.workspace/'code/controller'
        source.mkdir(parents=True)
        (source/'controller.py').write_text('def main(ctx): pass\n')
        bundle, _ = supervisor.register(source, workspace_path='code/controller')
        if setting == 'demo-only':
            response = frontend.call('submit', {'bundle': bundle})
            assert json.loads(response['content'][0]['text'])['status'] == 'submitted'
            assert supervisor.state['submission']['bundle'] == bundle
            assert supervisor.state['formal_reserved']
        else:
            with pytest.raises(RuntimeError, match='rehearsal'):
                frontend.call('submit', {'bundle': bundle})
            assert supervisor.state.get('submission') is None
        assert not study.attempts()
    finally:
        frontend.close()

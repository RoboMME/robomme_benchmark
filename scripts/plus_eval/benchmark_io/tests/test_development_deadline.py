from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from benchmark_io.development_deadline import TaskClock, DevelopmentExpired, TASK_SECONDS
from benchmark_io.protocol import Study
from aspire.development_deadline import CampaignClocks
from aspire.isolated import communicate_until_episode_end
from harness.claude_cli.auto_research import claude_arguments


def test_clock_is_150_minutes_and_resume_cannot_reset_it(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmark_io.development_deadline.time.time', lambda: 10000)
    clock = TaskClock(tmp_path, 'BinFill')
    assert TASK_SECONDS == 9000
    assert clock.start(1000)['deadline'] == 10000
    assert TaskClock(tmp_path, 'BinFill').start(9999)['started'] == 1000
    with pytest.raises(DevelopmentExpired):
        clock.require_open()
    assert clock.expire()['expired_at'] == 10000
    assert TaskClock(tmp_path, 'OtherTask').remaining() == 9000


def test_expired_task_cannot_reserve_even_with_rollouts_remaining(tmp_path):
    study = Study(tmp_path, method='rax', setting='AfI', budget=10,
                  records={'BinFill': [dict(seed=i) for i in range(3)]})
    TaskClock(study.root, 'BinFill').start(0)
    with pytest.raises(DevelopmentExpired):
        study.reserve('BinFill', purpose='deadline_rehearsal')
    assert study.attempts() == []


def assignment(ident, index, timestamp):
    return dict(type='assistant', timestamp=timestamp, message=dict(content=[dict(
        type='tool_use', name='Agent', id=ident,
        input=dict(prompt=f'Read /workspace/campaign/prompts/worker_robomme_plus_{index}.txt'))]))


def test_native_aspires_assignments_completions_and_revisits_do_not_reset_clock(tmp_path):
    clocks = CampaignClocks(tmp_path, tmp_path/'home', ['BinFill', 'StopCube'])
    row = assignment('first', 0, '2026-01-01T00:00:00Z')
    clocks.consume(row)
    first = TaskClock(tmp_path, 'BinFill').read()['started']
    clocks.consume(dict(type='user', timestamp='2026-01-01T00:10:00Z', message=dict(content=
        '<task-notification><tool-use-id>first</tool-use-id><status>completed</status></task-notification>')))
    clocks.consume(row)  # duplicate stream delivery must not reopen the worker
    assert clocks.overdue() == []
    clocks.consume(assignment('second', 0, '2026-01-01T03:00:00Z'))
    assert clocks.overdue() == ['BinFill']
    assert TaskClock(tmp_path, 'BinFill').read()['started'] == first
    assert TaskClock(tmp_path, 'StopCube').read() is None


def test_incomplete_transcript_line_is_consumed_after_completion(tmp_path):
    home = tmp_path/'home'
    directory = home/'projects/-workspace'
    directory.mkdir(parents=True)
    transcript = directory/'session.jsonl'
    row = json.dumps(assignment('one', 0, datetime.now(timezone.utc).isoformat()))
    transcript.write_text(row[:50])
    clocks = CampaignClocks(tmp_path/'study', home, ['BinFill'])
    clocks.refresh()
    assert TaskClock(tmp_path/'study', 'BinFill').read() is None
    with transcript.open('a') as stream:
        stream.write(row[50:]+'\n')
    clocks.refresh()
    assert TaskClock(tmp_path/'study', 'BinFill').read() is not None


def test_deadline_stops_inflight_aspire_program_and_records_reason(tmp_path):
    clock = TaskClock(tmp_path, 'BinFill')
    clock.start(0)
    stopped = []
    episode = SimpleNamespace(terminated=False, truncated=False, development_clock=clock)
    process = SimpleNamespace(communicate=lambda **kw: (b'partial output', b''))
    out, _, reason = communicate_until_episode_end(process, episode, lambda: stopped.append(True))
    assert stopped == [True]
    assert out == b'partial output'
    assert reason == episode.abort_requested == 'development_deadline'


def test_wrapup_has_no_builtin_execution_or_editing_tools():
    args = claude_arguments(dict(model='test', deadline_wrapup=True, subagents=0),
                            prompt='Wrap up', json_output=True)
    assert args[args.index('--tools')+1] == 'Read,Glob,Grep'
    assert 'Agent' in args and 'Task' in args  # disallowed, never spawned


@pytest.mark.parametrize('name', ['start_episode', 'rehearse', 'evaluate', 'gemini_generate'])
def test_rax_frontend_rejects_late_actions_before_dispatch(tmp_path, monkeypatch, name):
    from services.controller.frontend import ResearchFrontend
    monkeypatch.setenv('ROBOMME_PLUS_STUDY', str(tmp_path))
    TaskClock(tmp_path, 'BinFill').start(0)
    frontend = object.__new__(ResearchFrontend)
    frontend.s = SimpleNamespace(config=SimpleNamespace(benchmark='robomme_plus', task='BinFill'))
    with pytest.raises(DevelopmentExpired):
        frontend.call(name, {})


def test_rax_timeout_stops_background_work_resumes_readonly_and_finishes_queue(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from harness.claude_cli import auto_research as launcher
    monkeypatch.setenv('ROBOMME_PLUS_STUDY', str(tmp_path/'study'))
    root = tmp_path/'session'
    home = root/'claude-home'
    transcripts = home/'projects/-workspace'
    transcripts.mkdir(parents=True)
    (transcripts/'session-id.jsonl').write_text('{}\n')
    (root/'state.json').write_text('{"bundles": {}, "submission": null}')
    config = SimpleNamespace(root=root, benchmark='robomme_plus', task='BinFill',
                             training=None, image='fake', workspace_mb=4096)
    deployed = dict(configuration=root/'config.json', workspace=root/'workspace', claude_home=home)
    monkeypatch.setattr(launcher, 'DockerSandbox', lambda *a: SimpleNamespace(preflight=lambda: None))
    monkeypatch.setattr(launcher.workspace_storage, 'validate', lambda *a: None)
    monkeypatch.setattr(launcher, 'relay_credential', lambda *a: None)
    monkeypatch.setattr(launcher, 'model_relay', lambda *a, **k: object())
    stopped, launched = [], []
    def frontend(args, log):
        Path(args[args.index('--socket')+1]).touch()
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(launcher, 'start_frontend', frontend)
    monkeypatch.setattr(launcher, 'stop_services', lambda name, agent, server, model: stopped.append(name))
    def command(config, deployed, relay, name, **kwargs):
        launched.append(dict(deployed=deployed.copy(), prompt=kwargs['prompt']))
        return ['fake', name]
    monkeypatch.setattr(launcher, 'container_command', command)
    processes = []
    class Process:
        returncode = 0
        def __init__(self, *args, **kwargs):
            self.first = not processes
            processes.append(self)
            self.calls = 0
        def communicate(self, timeout=None):
            self.calls += 1
            if self.first and self.calls == 1:
                TaskClock(tmp_path/'study', 'BinFill').update(deadline=0)
                raise subprocess.TimeoutExpired('fake', timeout)
            return ('original transcript\n' if self.first else 'wrapup transcript\n', '')
    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    result = launcher.launch(config, deployed, prompt='Original task', capture=True)
    assert result.returncode == 0
    assert len(stopped) == 2  # before reporting and final cleanup
    assert launched[1]['deployed']['deadline_wrapup'] is True
    assert launched[1]['deployed']['resume_session'] == 'session-id'
    assert 'no more modifications' in launched[1]['prompt']
    assert result.stdout == 'original transcript\nwrapup transcript\n'
    assert json.loads((root/'deadline_frozen_state.json').read_text())['submission'] is None
    assert TaskClock(tmp_path/'study', 'BinFill').read()['expired_at']

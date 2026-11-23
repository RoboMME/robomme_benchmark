import json
from pathlib import Path
import socket

import pytest

from benchmark_io.development_evidence import EvidenceRecorder, capture
from benchmark_io.evaluation import execution_hashes


def test_failed_revision_retained_and_external_symlink_not_read(tmp_path):
    workspace = tmp_path/'workspace'; workspace.mkdir()
    code = workspace/'policy.py'; code.write_text('raise ValueError("first")\n')
    hidden = tmp_path/'private.py'; hidden.write_text('not agent evidence')
    (workspace/'link.py').symlink_to(hidden)
    recorder = EvidenceRecorder(tmp_path/'audit', workspace)
    first = recorder.record({'hook_event_name':'PreToolUse','tool_name':'Bash','tool_input':{'command':'python policy.py'}})
    code.write_text('print("second")\n')
    second = recorder.record({'hook_event_name':'PostToolUseFailure','error':'failure'})
    events = [json.loads(s) for s in (tmp_path/'audit/events.jsonl').read_text().splitlines()]
    assert first != second
    assert (tmp_path/'audit/objects'/events[0]['files']['policy.py']).read_text() == 'raise ValueError("first")\n'
    assert 'link.py' not in events[0]['files']
    assert events[1]['event']['error'] == 'failure'


def test_hook_socket_records_tools_and_preserves_existing_settings(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    (workspace/'policy.py').write_text('pass\n')
    settings=tmp_path/'settings.json';settings.write_text(json.dumps({'env':{'example':'retained'}}))
    with capture(tmp_path/'audit',workspace,settings) as mounts:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.connect(str(mounts[0][0]/'audit.sock'))
            connection.sendall(json.dumps({'hook_event_name':'PostToolUse','tool_use_id':'one','tool_response':'saved'}).encode()+b'\n')
            response=json.loads(connection.makefile('rb').readline())
        assert response['ok']
        assert all(readonly for _,_,readonly in mounts)
    saved=json.loads(settings.read_text())
    assert saved['env']=={'example':'retained'}
    assert 'PreToolUse' in saved['hooks'] and 'PostToolUseFailure' in saved['hooks']
    events=[json.loads(s) for s in (tmp_path/'audit/history/events.jsonl').read_text().splitlines()]
    assert [e['event']['hook_event_name'] for e in events]==['OperatorStart','PostToolUse','OperatorStop']
    assert events[1]['event']['tool_response']=='saved'


def test_execution_pin_detects_added_modified_deleted_files(tmp_path):
    for directory in ('runtime_source','protocol'):(tmp_path/directory).mkdir()
    for name in ('runtime_policy.json','study_plan.json'):(tmp_path/name).write_text('{}')
    code=tmp_path/'runtime_source/runner.py';code.write_text('pass\n')
    pinned=execution_hashes(tmp_path)
    code.write_text('print(1)\n');assert execution_hashes(tmp_path)!=pinned
    code.write_text('pass\n')
    extra=tmp_path/'protocol/extra.md';extra.write_text('new input');assert execution_hashes(tmp_path)!=pinned
    extra.unlink();assert execution_hashes(tmp_path)==pinned
    code.unlink();assert execution_hashes(tmp_path)!=pinned
    code.symlink_to(tmp_path/'study_plan.json')
    with pytest.raises(ValueError,match='symlink'):execution_hashes(tmp_path)

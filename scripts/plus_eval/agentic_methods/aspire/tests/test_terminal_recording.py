"""Terminal simulator outcomes must survive artifact finalization."""
import json

import pytest

from aspire.isolated import recorder_termination_reason
from aspire.lean_recording import LogRecorder
from cap_harness.artifacts import RunRecorder


@pytest.mark.parametrize('recorder_type', [RunRecorder, LogRecorder])
@pytest.mark.parametrize('stop,success,expected', [
    ('episode_terminated', False, 'environment_terminated'),
    ('episode_terminated', True, 'task_succeeded'),
    ('episode_truncated', False, 'environment_truncated'),
    ('episode_truncated', True, 'environment_truncated'),
    ('program_error', False, 'program_error'),
])
def test_finalize_preserves_terminal_outcome(tmp_path, recorder_type, stop, success, expected):
    recorder = recorder_type(output_root=tmp_path, benchmark='robomme_plus',
        suite='test', task_id=0, task_name='ButtonUnmaskSwap', seed=16,
        max_steps=100, camera_width=4, camera_height=4, providers={})
    recorder.terminated = stop == 'episode_terminated'
    recorder.truncated = stop == 'episode_truncated'
    recorder.step_count = 42
    recorder.finalize(program_ok=False, task_success=success,
        termination_reason=recorder_termination_reason(stop, task_success=success),
        program_result=None)
    outcome = json.loads((recorder.root / 'outcome.json').read_text())
    assert outcome['task_success'] is success
    assert outcome['steps_executed'] == 42
    assert outcome['termination_reason'] == expected
    assert outcome['environment_terminated'] == recorder.terminated
    assert outcome['environment_truncated'] == recorder.truncated
    assert outcome['error'] is None

import json
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from cap_harness.contracts import CameraObservation, Pose
from aspire.sam3_reliability import ReliableSam3Provider


class Session:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(status_code=reply, text='test', json=lambda: {'results': []})


@pytest.fixture
def observation():
    return CameraObservation(rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                             depth_m=np.ones((4, 4)), intrinsics=np.eye(3),
                             camera_pose=Pose(np.zeros(3), np.array([1., 0, 0, 0]), frame='world'), frame='front')


@pytest.mark.parametrize('first', [500, requests.ConnectionError('unavailable'), requests.Timeout('late')])
def test_transient_failure_recovers_on_third_attempt(tmp_path, observation, first):
    session = Session([first, 503, 200])
    provider = ReliableSam3Provider('http://test', log_path=tmp_path/'calls.jsonl', session=session)
    delays = []
    provider._http._sleep = delays.append
    result = provider.segment_text(observation, 'cube')
    assert result.error.code.value == 'no_segmentation'
    assert len(session.calls) == 3 and delays == [1, 2]
    assert session.calls[0][2]['json'] == session.calls[2][2]['json']
    assert provider.infrastructure_failures == 0
    assert len((tmp_path/'calls.jsonl').read_text().splitlines()) == 3


@pytest.mark.parametrize('operation', ['segment_text', 'segment_points'])
def test_exhaustion_is_bounded_and_recorded(tmp_path, observation, operation):
    session = Session([500, 500, 500])
    provider = ReliableSam3Provider('http://test', log_path=tmp_path/'calls.jsonl', session=session)
    provider._http._sleep = lambda _: None
    prompt = 'cube' if operation == 'segment_text' else np.array([[1, 1]])
    result = getattr(provider, operation)(observation, prompt)
    assert result.error.code.value == 'provider_unavailable'
    assert result.error.details['attempts'] == 3
    assert len(session.calls) == 3 and provider.infrastructure_failures == 1
    events = [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'request_failed'


@pytest.mark.parametrize('status', [200, 400])
def test_detection_miss_and_invalid_requests_are_not_retried(tmp_path, observation, status):
    session = Session([status])
    provider = ReliableSam3Provider('http://test', log_path=tmp_path/'calls.jsonl', session=session)
    provider.segment_text(observation, 'cube')
    assert len(session.calls) == 1 and provider.infrastructure_failures == 0

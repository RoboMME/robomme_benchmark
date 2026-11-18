from io import BytesIO

import numpy as np
import pytest

from aspire import wire
from cap_harness.contracts import CameraObservation, Observation, RobotState, Pose


def frame(index):
    pose = Pose([1., 2., 3.], [1., 0., 0., 0.], 'robot_base')
    camera = CameraObservation(np.full((8, 8, 3), index, np.uint8),
        np.full((8, 8), .125 + index, np.float32), np.eye(3), 'front', pose)
    state = RobotState(joint_positions={'primary': np.arange(7.)},
        joint_velocities={'primary': np.zeros(7)}, end_effector_poses={'primary': pose},
        gripper_positions={'primary': .5}, base_frame='robot_base', embodiment='robomme_plus')
    return Observation(cameras={'front': camera, 'wrist': camera}, robot_state=state, timestamp_s=index / 20.)


def test_video_larger_than_reply_limit_arrives_whole_without_schema_change(monkeypatch):
    monkeypatch.setattr(wire, 'REPLY_LIMIT', 4096)
    monkeypatch.setattr(wire, 'VIDEO_FRAME_LIMIT', 4096)
    frames = tuple(frame(i) for i in range(30))
    with pytest.raises(ValueError, match='exceeds'):
        wire.write_reply(BytesIO(), {'value': frames})
    stream = BytesIO()
    wire.write_reply(stream, {'value': frames}, video=True)
    assert stream.tell() > wire.REPLY_LIMIT
    assert max(map(len, stream.getvalue().splitlines(keepends=True))) <= wire.VIDEO_FRAME_LIMIT
    stream.seek(0)
    result = wire.read_reply(stream, video=True)['value']
    assert len(result) == len(frames)
    for actual, expected in zip(result, frames):
        assert type(actual) is Observation
        assert actual.timestamp_s == expected.timestamp_s
        for name in ('front', 'wrist'):
            a, b = actual.cameras[name], expected.cameras[name]
            np.testing.assert_array_equal(a.rgb, b.rgb)
            np.testing.assert_array_equal(a.depth_m, b.depth_m)
            np.testing.assert_array_equal(a.intrinsics, b.intrinsics)
            np.testing.assert_array_equal(a.camera_pose.position, b.camera_pose.position)
            assert a.rgb.dtype == b.rgb.dtype and not a.rgb.flags.writeable
        np.testing.assert_array_equal(actual.robot_state.joint_positions['primary'], np.arange(7.))


@pytest.mark.parametrize('rows', [
    [{'index': 1, 'value': frame(0)}, {'end': 1}],
    [{'index': 0, 'value': frame(0)}, {'end': 2}],
    [{'index': 0, 'value': 'not a sensor observation'}, {'end': 1}],
    [{'index': 0, 'value': frame(0)}],
])
def test_partial_or_reordered_video_is_never_returned(rows):
    data = [{'__stream__': wire.VIDEO_STREAM, 'count': 1}, *rows]
    stream = BytesIO(b''.join(wire.dumps(row) + b'\n' for row in data))
    with pytest.raises(ValueError):
        wire.read_reply(stream, video=True)


def test_nonvideo_replies_and_errors_keep_existing_contract():
    for reply in ({'value': {'ok': True}}, {'error': 'known error'}):
        stream = BytesIO()
        wire.write_reply(stream, reply, video=True)
        stream.seek(0)
        assert wire.read_reply(stream, video=True) == reply
    stream = BytesIO()
    wire.write_reply(stream, {'value': ()}, video=True)
    stream.seek(0)
    assert wire.read_reply(stream, video=True) == {'value': []}


def test_truncated_ordinary_reply_is_explicitly_rejected(monkeypatch):
    monkeypatch.setattr(wire, 'REPLY_LIMIT', 16)
    with pytest.raises(ValueError, match='oversized'):
        wire.read_reply(BytesIO(b'{"value":"' + b'a' * 100 + b'"}\n'))

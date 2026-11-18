import numpy as np
import pytest

from aspire.wire import dumps, loads
from cap_harness.contracts import CameraObservation, Pose, MotionStrategy


def test_calibrated_camera_survives_transport_without_becoming_writable():
    camera = CameraObservation(rgb=np.zeros((4, 5, 3), np.uint8), depth_m=np.ones((4, 5)),
        intrinsics=np.eye(3), frame='front',
        camera_pose=Pose(position=[1, 2, 3], quaternion_wxyz=[1, 0, 0, 0], frame='robot_base'))
    restored = loads(dumps(camera))
    np.testing.assert_array_equal(restored.rgb, camera.rgb)
    np.testing.assert_array_equal(restored.depth_m, camera.depth_m)
    np.testing.assert_array_equal(restored.camera_pose.position, [1, 2, 3])
    assert not restored.rgb.flags.writeable
    assert loads(dumps(MotionStrategy())).ik_solver == MotionStrategy().ik_solver


def test_unknown_classes_and_object_arrays_cannot_cross_boundary():
    with pytest.raises(ValueError, match='Unrecognized'):
        loads(b'{"__type__":"Popen","fields":{"args":["echo","bad"]}}')
    with pytest.raises(ValueError, match='Object arrays'):
        dumps(np.array([object()], dtype=object))


def test_array_declared_shape_must_match_actual_payload():
    import json
    value = json.loads(dumps(np.ones((2, 2))))
    value['shape'] = [100, 100]
    with pytest.raises(ValueError, match='payload size differs'):
        loads(json.dumps(value).encode())

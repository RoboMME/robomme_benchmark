"""Offline synthetic public Observation; no simulator or task imports."""
import numpy as np
from cap_harness.contracts import CameraObservation, Observation, Pose, RobotState


def observation(timestamp_s=0.0):
    pose=Pose(position=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.],frame='robot_base')
    cameras={name:CameraObservation(rgb=np.zeros((8,8,3),dtype=np.uint8),
        depth_m=np.ones((8,8),dtype=np.float32),intrinsics=np.eye(3),
        frame=name,camera_pose=pose,timestamp_s=timestamp_s) for name in ('front','wrist')}
    state=RobotState(joint_positions={'primary':np.zeros(7)},
        joint_velocities={'primary':np.zeros(7)},end_effector_poses={'primary':pose},
        gripper_positions={'primary':1.0},base_frame='robot_base',
        embodiment='robomme_plus',timestamp_s=timestamp_s)
    return Observation(cameras=cameras,robot_state=state,timestamp_s=timestamp_s)


if __name__=='__main__':
    video=(observation(0.0),observation(5.0))
    assert video[1].timestamp_s-video[0].timestamp_s==5.0
    assert video[0].cameras['wrist'].rgb.shape==(8,8,3)
    print('Synthetic ASPIRE observation schema verified; no simulator started.')

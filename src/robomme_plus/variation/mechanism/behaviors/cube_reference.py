"""Native-speed physical cube pickup with an alternate square-face grasp."""
import numpy as np
import sapien
from transforms3d.quaternions import quat2mat
from robomme_plus.common.sim import position,scalar


def pickup(env,planner,cube):
    native=env.unwrapped
    def move(point,q):
        pose=sapien.Pose(point,q)
        path=planner.move_to_pose_with_screw(pose,dry_run=True)
        if not isinstance(path,dict) or path.get('status')!='Success':
            path=planner.move_to_pose_with_RRTConnect(pose,dry_run=True)
        if not isinstance(path,dict) or path.get('status')!='Success':return False
        if len(path['position']):planner.follow_path(path)
        return True
    for axis in (0,1):
        planner.open_gripper(t=12)
        p=position(native.agent.tcp,False).copy();p[2]=.22
        q=native.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1,4)[0]
        if not move(p,q):continue
        rotation=quat2mat(cube.pose.sp.q);closing=rotation[:,axis].copy();closing[2]=0;closing/=np.linalg.norm(closing)
        p=position(cube,False).copy()
        q=native.agent.build_grasp_pose(np.array([0,0,-1]),closing,p).q
        hover=p.copy();hover[2]=.18
        if not move(hover,q):continue
        p=position(cube,False).copy()
        if not move(p,q):continue
        planner.close_gripper(t=16)
        if not scalar(native.agent.is_grasping(cube)):continue
        if not move(hover,q):continue
        if scalar(native.agent.is_grasping(cube)):return
    raise RuntimeError('Physical cube pickup failed with both square-face grasps')

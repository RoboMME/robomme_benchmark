"""Native-speed physical placement for moving-ring demonstrations."""
import numpy as np
import sapien


def place(wrapped,planner,cube,target):
    from robomme_plus.common.sim import position,scalar
    native=wrapped.unwrapped;routes=native._limit_routes;index=routes.actors.index(target)
    assert scalar(native.agent.is_grasping(cube)),'Placement starts without a grasped cube'
    q=native.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1,4)[0]
    planner.gripper_state=planner.CLOSED
    offset=position(cube,False)-position(native.agent.tcp,False)
    hover=np.r_[position(target,False)[:2]-offset[:2],.15]
    if planner.move_to_pose_with_screw(sapien.Pose(hover,q))==-1:return -1
    used=0
    while used<360:
        if not scalar(native.agent.is_grasping(cube)):
            raise RuntimeError('Demonstration grasp slipped before release')
        offset=position(cube,False)-position(native.agent.tcp,False)
        target_xy=position(target,False)[:2].copy()
        goal=np.r_[target_xy,position(cube,True)[2]+.003]-offset
        path=planner.move_to_pose_with_screw(sapien.Pose(goal,q),dry_run=True)
        if not isinstance(path,dict) or path.get('status')!='Success':return -1
        changed=False
        for start in range(0,len(path['position']),8):
            end=min(start+8,len(path['position']))
            planner.follow_path({**path,'position':path['position'][start:end],
                                 'velocity':path['velocity'][start:end]})
            used+=end-start
            if np.linalg.norm(position(target,False)[:2]-target_xy)>.006:
                changed=True;break
        if changed:continue
        if any(index in pair for pair in routes.pairs):
            planner.close_gripper(t=1);used+=1;continue
        if np.linalg.norm(position(cube,False)[:2]-position(target,False)[:2])>.018:
            planner.close_gripper(t=1);used+=1;continue
        planner.open_gripper(t=20)
        point=position(native.agent.tcp,False).copy();point[2]=.20
        if planner.move_to_pose_with_screw(sapien.Pose(point,q))==-1:return -1
        return planner.open_gripper(t=12)
    raise RuntimeError('Native-speed moving-target placement did not converge')

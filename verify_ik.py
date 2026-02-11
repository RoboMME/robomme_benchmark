import gymnasium as gym
import numpy as np
import torch
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

# Create a dummy env
env = gym.make("PickCube-v1", obs_mode="none", control_mode="pd_joint_pos")
env.reset()
planner = PandaArmMotionPlanningSolver(
    env,
    debug=False,
    vis=False,
    base_pose=env.unwrapped.agent.robot.pose,
)

# Get current state
# Ensure we get numpy arrays
tcp_pose = env.unwrapped.agent.tcp.pose
p = tcp_pose.p
if isinstance(p, torch.Tensor):
    p = p.cpu().numpy()
p = p.flatten()

q = tcp_pose.q
if isinstance(q, torch.Tensor):
    q = q.cpu().numpy()
q = q.flatten()

qpos = env.unwrapped.agent.robot.get_qpos()
if isinstance(qpos, torch.Tensor):
    qpos = qpos.cpu().numpy()
qpos = qpos.flatten() # This might include all joints of articulation

print(f"Current TCP Position: {p}")
print(f"Current TCP Quat (wxyz): {q}")

# CASE 1: Assume IK expects WXYZ (matches tcp.pose.q)
goal_wxyz = np.concatenate([p, q]) # [x,y,z, w,x,y,z]
# Transform to base
try:
    goal_base_wxyz = planner.planner.transform_goal_to_wrt_base(goal_wxyz)
    print("\n--- CASE 1: Pass WXYZ (Identity from TCP) ---")
    status_wxyz, sol_wxyz = planner.planner.IK(goal_base_wxyz, qpos)
    print(f"IK Status: {status_wxyz}")
    if len(sol_wxyz) > 0:
        # Assuming solution is 7DOF
        sol_q = sol_wxyz[0][:7]
        # Compare with current qpos (Panda usually first 7 joints)
        current_q = qpos[:7]
        diff = np.linalg.norm(sol_q - current_q)
        print(f"Joint Diff from nominal: {diff:.4f}")
        if diff < 0.1:
            print(">> Confirmed: IK expects WXYZ (or rotation is symmetric)")
        else:
            print(f">> Suspicious: IK returned varying solution ({diff:.4f})")
except Exception as e:
    print(f"CASE 1 Error: {e}")

# CASE 2: Assume IK expects XYZW
# Construct XYZW from valid WXYZ
# q is [w, x, y, z]
q_xyzw_val = np.array([q[1], q[2], q[3], q[0]])
goal_xyzw = np.concatenate([p, q_xyzw_val])

try:
    goal_base_xyzw = planner.planner.transform_goal_to_wrt_base(goal_xyzw)
    print("\n--- CASE 2: Pass XYZW ---")
    status_xyzw, sol_xyzw = planner.planner.IK(goal_base_xyzw, qpos)
    print(f"IK Status: {status_xyzw}")
    if len(sol_xyzw) > 0:
        sol_q = sol_xyzw[0][:7]
        current_q = qpos[:7]
        diff = np.linalg.norm(sol_q - current_q)
        print(f"Joint Diff from nominal: {diff:.4f}")
        if diff < 0.1:
            print(">> Confirmed: IK expects XYZW")
        else:
            print(">> Confirmed: IK likely DOES NOT expect XYZW (interpreted as wrong rotation)")
except Exception as e:
    print(f"CASE 2 Error: {e}")

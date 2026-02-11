import gymnasium as gym
import numpy as np
import sapien.physx as physx
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

env = gym.make("PickCube-v1", obs_mode="none", control_mode="pd_joint_pos")
env.reset()

agent = env.unwrapped.agent
tcp_pose = agent.tcp.pose
# Find the EE link. Usually last link or 'panda_hand'
ee_link = agent.robot.get_links()[-1] # Usually right finger or something.
# We need the link that IK solves for.
# MPLIB usually uses 'panda_hand'.
for link in agent.robot.get_links():
    if link.name == "panda_hand":
        ee_link = link
        break
        
link_pose = ee_link.pose

print(f"TCP Pose: p={tcp_pose.p}, q={tcp_pose.q}")
print(f"Link Pose: p={link_pose.p}, q={link_pose.q}")

# Calculate Offset T_link_tcp = inv(T_link) * T_tcp
# Sapien Pose inverse/mult
T_link = link_pose.to_transformation_matrix()
T_tcp = tcp_pose.to_transformation_matrix()
T_offset = np.linalg.inv(T_link) @ T_tcp

print("Offset Matrix:")
print(T_offset)

from scipy.spatial.transform import Rotation
rot = Rotation.from_matrix(T_offset[:3, :3])
rpy = rot.as_euler('xyz', degrees=True)
print(f"Rotation Offset (RPY deg): {rpy}")
print(f"Translation Offset: {T_offset[:3, 3]}")

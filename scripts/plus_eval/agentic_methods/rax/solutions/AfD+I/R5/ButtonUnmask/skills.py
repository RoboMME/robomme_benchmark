import numpy as np
from percep import find_container

def place_back(R, xy, z=0.04, log=print):
    R.move_to([xy[0], xy[1], z + 0.005], 1, max_steps=30, chunk=5, gain=0.6)
    R.hold(6, -1)
    R.move_to([xy[0], xy[1], 0.16], -1, max_steps=25, chunk=5, gain=0.7)

def lift_container(R, xy, yaw, log=print, grasp_z=0.035, lift_z=0.20):
    R.move_to([xy[0], xy[1], 0.15], -1, max_steps=40, chunk=5, gain=0.7, yaw_target=yaw)
    R.move_to([xy[0], xy[1], grasp_z], -1, max_steps=30, chunk=4, gain=0.6, yaw_target=yaw)
    R.hold(8, 1)
    w = R.grip_open()
    log('grip width after close', w)
    R.move_to([xy[0], xy[1], lift_z], 1, max_steps=30, chunk=5, gain=0.7)
    w2 = R.grip_open()
    log('grip width after lift', w2, 'pos', R.pos())
    return w2 > 0.03

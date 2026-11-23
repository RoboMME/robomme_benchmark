import numpy as np
from PIL import Image

def quat_to_R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    h,w=depth_m.shape
    fx,fy,cx,cy=K[0][0],K[1][1],K[0][2],K[1][2]
    u,v=np.meshgrid(np.arange(w),np.arange(h))
    d=depth_m
    pc=np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d],-1)
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    return pc@R.T+t

def load_depth(path):
    return np.array(Image.open(path)).astype(np.float64)*0.001

def load_rgb(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.int32)

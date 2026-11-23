import numpy as np
from PIL import Image

def quat_to_R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW meters (0 invalid). returns HxWx3 world points, valid mask"""
    H,W=depth_m.shape
    K=K["matrix"] if isinstance(K,dict) else K
    fx,fy,cx,cy=K[0][0],K[1][1],K[0][2],K[1][2]
    u,v=np.meshgrid(np.arange(W),np.arange(H))
    d=depth_m
    pc=np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d],-1)
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    pw=pc@R.T+t
    return pw, d>0

def load_depth(path, mpu=0.001):
    return np.asarray(Image.open(path)).astype(np.float64)*mpu

def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.int16)

def project(pts, K, pose):
    K=K["matrix"] if isinstance(K,dict) else K
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    pc=(np.asarray(pts)-t)@R
    d=-pc[:,2]
    u=K[0][0]*pc[:,0]/d+K[0][2]; v=-K[1][1]*pc[:,1]/d+K[1][2]
    return np.stack([u,v],1)

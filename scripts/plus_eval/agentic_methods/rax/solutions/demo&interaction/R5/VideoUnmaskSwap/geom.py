import numpy as np
from PIL import Image

def quat_to_R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def depth_to_world(depth_m, K, pose):
    """depth_m HxW meters (0 invalid). returns HxWx3 world points, valid mask."""
    H,W=depth_m.shape
    fx,fy,cx,cy=K[0][0],K[1][1],K[0][2],K[1][2]
    u,v=np.meshgrid(np.arange(W)+0.0,np.arange(H)+0.0)
    d=depth_m
    cam=np.stack([(u-cx)*d/fx,-(v-cy)*d/fy,-d],-1)
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    pts=cam@R.T+t
    return pts, d>0

def world_to_pixel(p, K, pose):
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    c=(np.asarray(p)-t)@R  # camera coords
    d=-c[...,2]
    u=K[0][0]*c[...,0]/d+K[0][2]; v=-K[1][1]*c[...,1]/d+K[1][2]
    return u,v,d

def load_frame(root, frame, cam):
    files={(f['camera'],f['kind']):f['path'] for f in frame['files']}
    rgb=np.asarray(Image.open(f"{root}/{files[(cam,'rgb')]}").convert('RGB'))
    dep=np.asarray(Image.open(f"{root}/{files[(cam,'depth')]}")).astype(np.float32)*frame['depth'][cam].get('meters_per_unit',0.001)
    return rgb,dep

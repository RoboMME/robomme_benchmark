import numpy as np, json, colorsys
from PIL import Image

def quat_to_R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def deproject(depth_m, K, pose):
    H,W=depth_m.shape
    fx,fy,cx,cy=K[0][0],K[1][1],K[0][2],K[1][2]
    u,v=np.meshgrid(np.arange(W)+0.0,np.arange(H)+0.0)
    d=depth_m
    cam=np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d],-1)
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    return cam@R.T+t

def hsv_img(rgb):
    a=rgb.astype(np.float32)/255.
    mx=a.max(-1); mn=a.min(-1); d=mx-mn+1e-9
    r,g,b=a[...,0],a[...,1],a[...,2]
    h=np.where(mx==r,((g-b)/d)%6,np.where(mx==g,(b-r)/d+2,(r-g)/d+4))*60
    s=np.where(mx>0,(mx-mn)/(mx+1e-9),0)
    return h,s,mx

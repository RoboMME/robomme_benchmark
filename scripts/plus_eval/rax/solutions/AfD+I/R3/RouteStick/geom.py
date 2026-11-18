import numpy as np
def quat_to_R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def project(pts, K, pose):
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    pc=(np.asarray(pts)-t)@R  # world->cam (OpenGL)
    d=-pc[:,2]; fx,cx,fy,cy=K[0][0],K[0][2],K[1][1],K[1][2]
    u=pc[:,0]/d*fx+cx; v=-pc[:,1]/d*fy+cy
    return np.stack([u,v],1)
def backproject(u,v,d,K,pose):
    R=quat_to_R(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    fx,cx,fy,cy=K[0][0],K[0][2],K[1][1],K[1][2]
    pc=np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d],-1)
    return pc@R.T+t

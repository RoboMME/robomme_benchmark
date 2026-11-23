import numpy as np
from PIL import Image

def quat_to_R(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres (0 invalid) -> world points HxWx3 (nan invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose['quaternion_wxyz'])
    P = cam @ R.T + np.array(pose['position_m'])
    P[d <= 0] = np.nan
    return P

def load_frame(root, frame):
    """Return dict cam -> (rgb uint8 HxWx3, world pts HxWx3)."""
    files = {(f['camera'], f['kind']): f['path'] for f in frame['files']}
    out = {}
    for cam in ('front', 'wrist'):
        if (cam, 'rgb') not in files or (cam, 'depth') not in files:
            continue
        rgb = np.asarray(Image.open(f"{root}/{files[(cam,'rgb')]}").convert('RGB'))
        dep = np.asarray(Image.open(f"{root}/{files[(cam,'depth')]}")).astype(np.float64)
        mpu = frame.get('depth', {}).get(cam, {}).get('meters_per_unit', 0.001)
        dep = dep * mpu
        K = frame['camera_intrinsics'][cam]['matrix']
        pose = frame['camera_poses'][cam]
        out[cam] = (rgb, backproject(dep, K, pose))
    return out

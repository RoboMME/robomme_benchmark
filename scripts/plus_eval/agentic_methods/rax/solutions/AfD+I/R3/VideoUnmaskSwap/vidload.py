import json, numpy as np
from PIL import Image
from mcpio import depth_m, backproject
ROOT = '/workspace/'
def load_manifest(path):
    m = json.load(open(path if path.startswith('/') else ROOT + path))
    return m
def load_frame(fr, root=ROOT):
    files = {(f['camera'], f['kind']): f['path'] for f in fr['files']}
    out = {}
    for k, p in files.items():
        out[k] = np.array(Image.open(root + p))
    return out
def world_points(fr, imgs, cam):
    d = depth_m(imgs[(cam, 'depth')], fr.get('depth', {}).get(cam, {}).get('meters_per_unit', 0.001))
    return backproject(d, fr['camera_intrinsics'][cam]['matrix'], fr['camera_poses'][cam])

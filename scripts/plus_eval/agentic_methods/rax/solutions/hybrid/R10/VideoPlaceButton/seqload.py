import json
import numpy as np
from pathlib import Path
from PIL import Image

WS = Path('/workspace')

def load_sequence(manifest_path, want_depth=False):
    m = json.loads((WS / manifest_path).read_text())
    frames = sorted(m['frames'], key=lambda f: f['frame_index'])
    out = []
    for f in frames:
        files = {(x['camera'], x['kind']): x['path'] for x in f['files']}
        rec = dict(phase=f['phase'], t=float(f.get('timestamp_s', 0.0)), idx=f['frame_index'],
                   src=f.get('source_frame_index'), state=f.get('state'), cal=dict(K=f.get('camera_intrinsics'), P=f.get('camera_poses')))
        with Image.open(WS / files[('front', 'rgb')]) as im:
            rec['front'] = np.array(im.convert('RGB'))
        with Image.open(WS / files[('wrist', 'rgb')]) as im:
            rec['wrist'] = np.array(im.convert('RGB'))
        if want_depth and ('front', 'depth') in files:
            with Image.open(WS / files[('front', 'depth')]) as im:
                rec['front_depth'] = np.array(im).astype(np.float32) * 0.001
        out.append(rec)
    return m, out

import json
from pathlib import Path
import numpy as np
from PIL import Image

def load_sequence(manifest_path, root='/workspace'):
    root = Path(root)
    m = json.loads((root / manifest_path).read_text())
    out = []
    fr = sorted(m['frames'], key=lambda f: f['frame_index'])
    for f in fr:
        files = {(x['camera'], x['kind']): x['path'] for x in f['files']}
        if ('front', 'rgb') not in files or ('wrist', 'rgb') not in files:
            continue
        a = np.array(Image.open(root / files[('front', 'rgb')]).convert('RGB'))
        w = np.array(Image.open(root / files[('wrist', 'rgb')]).convert('RGB'))
        t = float(f.get('timestamp_s', 0.0))
        if f['phase'] != 'video_demo':
            t = out[-1][0] + 0.1 if out else 0.0
        out.append((t, a, w))
    return out, m

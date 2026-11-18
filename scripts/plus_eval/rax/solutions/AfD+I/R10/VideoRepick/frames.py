import json
import numpy as np
from pathlib import Path
from PIL import Image
ROOT = Path('/workspace')

def load_manifest(manifest_path):
    p = Path(manifest_path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text())

def frame_files(frame):
    return {(f['camera'], f['kind']): f['path'] for f in frame['files']}

def load_rgb(path):
    with Image.open(ROOT / path) as im:
        return np.asarray(im.convert('RGB'))

def load_depth(path, scale=0.001):
    with Image.open(ROOT / path) as im:
        return np.asarray(im).astype(np.float32) * scale

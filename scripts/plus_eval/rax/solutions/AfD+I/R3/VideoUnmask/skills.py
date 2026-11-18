import math, os, json
import numpy as np
from geom import backproject, load_rgb, load_depth
from perception import white_points, locate_container
import io, base64

def live_points(robot, cam):
    """Load current observation images of a camera from the reply's saved frame files or attachments."""
    d = robot.last
    return None

def obs_cloud(ctx, robot, cam):
    """Fetch an observation and decode RGB + depth of cam into world points."""
    from PIL import Image
    r = ctx.call('robomme_observe')
    content = r.get('content', r) if isinstance(r, dict) else r
    meta = json.loads(content[0]['text']); robot.last = meta
    rgb = dep = None
    for a in meta['attachments']:
        if a['camera'] != cam:
            continue
        blk = content[a['content_index']]
        raw = base64.b64decode(blk['data'])
        im = Image.open(io.BytesIO(raw))
        if a['kind'] == 'rgb':
            rgb = np.asarray(im.convert('RGB'))
        else:
            dep = np.asarray(im).astype(np.float64) * meta['depth'][cam].get('meters_per_unit', 0.001)
    pts = backproject(dep, meta['camera_intrinsics'][cam]['matrix'], meta['camera_poses'][cam])
    return rgb, pts, meta

def refine_container(ctx, robot, guess_xy, log=print):
    best = None
    for cam in ('wrist', 'front'):
        try:
            rgb, pts, meta = obs_cloud(ctx, robot, cam)
            W = white_points(rgb, pts, zmin=0.02, zmax=0.12)
            c = locate_container(W, guess_xy, radius=0.04)
            log(f'refine {cam}: {c}')
            if c and c['n_top'] > 30 and abs(c['ext'][0] - 0.038) < 0.012 and abs(c['ext'][1] - 0.038) < 0.012:
                return c, cam
            if c and best is None:
                best = c
        except Exception as e:
            log(f'refine {cam} error {e!r}')
    return best, 'fallback'

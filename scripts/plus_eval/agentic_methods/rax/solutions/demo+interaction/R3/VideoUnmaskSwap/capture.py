import json, os, numpy as np
from PIL import Image
from mcpio import parse

def wait_until_still(ctx, outdir=None, max_steps=150, still_needed=8, log=print):
    """Step zero actions one at a time until the front image stops changing. Returns list of (meta, imgs)."""
    seq = []; prev = None; still = 0
    for k in range(max_steps):
        r = ctx.call('robomme_step', actions=[[0, 0, 0, 0, 0, 0, -1]])
        meta, imgs = parse(r)
        seq.append((meta, imgs))
        if outdir:
            for (cam, kind), a in imgs.items():
                Image.fromarray(a).save(f'{outdir}/x{k:03d}_{cam}_{kind}.png')
        f = imgs[('front', 'rgb')].astype(int)
        if prev is not None:
            ch = int((np.abs(f - prev).max(2) > 30).sum())
            still = still + 1 if ch < 5 else 0
        prev = f
        if meta.get('episode_ended'): break
        if still >= still_needed: break
    log(f'wait_until_still: {len(seq)} steps')
    return seq

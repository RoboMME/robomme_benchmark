"""Measure a held/lying tool's geometry from front-camera depth."""
import math
import numpy as np
import perception as P

def scene_points(meta, imgs, cam='front'):
    D = imgs[(cam, 'depth')]; c = P.Cam(meta, cam)
    vs, us = np.nonzero(D > 0); Pimg = np.zeros(D.shape + (3,)); Pimg[vs, us] = c.backproject(us, vs, D[vs, us])
    return Pimg, D > 0

def held_tool_points(meta, imgs, gpos, radius=0.42, zmin=0.004, colors=('salmon', 'gray', 'teal', 'tan', 'slate')):
    a = imgs[('front', 'rgb')]
    Pimg, valid = scene_points(meta, imgs)
    m = np.zeros(a.shape[:2], bool)
    fn = dict(salmon=P.m_salmon, gray=P.m_gray, teal=P.m_teal, tan=P.m_tan, slate=P.m_slate)
    for c in colors: m |= fn[c](a)
    rel = Pimg - gpos
    sel = valid & m & (np.hypot(rel[..., 0], rel[..., 1]) < radius) & (Pimg[..., 2] > zmin)
    return Pimg[sel], a[sel]

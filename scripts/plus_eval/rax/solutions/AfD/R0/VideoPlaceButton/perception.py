"""Pure image perception for VideoPlaceButton (no MCP calls, numpy/scipy only).

Front camera (256x256): bullseye targets (ring colour has blue > green, unlike the
orange wood), cube colour masks.  Wrist camera: held cube (large colour area at the
bottom centre), button press (large solid white square right under the camera).
Thresholds were measured on the three released demonstrations (pink and purple
targets; red, green, blue cubes).
"""
import numpy as np
from scipy import ndimage

CUBE_COLORS = ("red", "green", "blue")
HELD_WRIST_PIXELS = 3000      # held cube in wrist bottom-centre box: 7000-9900 px in demos; released ~900
PRESS_WHITE_PIXELS = 4000     # solid white button square in wrist view: 5100-6700 px at press in demos
PRESS_SOLIDITY = 0.9          # target rings are hollow (area/filled ~0.4)
SLOT_FREQ = 0.4               # a slot pixel is target-coloured in >40% of video frames
SLOT_MIN_PIXELS = 40
BLOB_MIN_PIXELS = 15
EIGHT = np.ones((3, 3), bool)


def _split(a):
    a = np.asarray(a).astype(np.int16)
    return a[..., 0], a[..., 1], a[..., 2]


def color_masks(a):
    r, g, b = _split(a)
    return {
        "red": (r > 150) & (g < 70) & (b < 70),
        "green": (g > 140) & (r < 110) & (b < 110),
        "blue": (b > 110) & (r < 70) & (g < 80),
    }


def cube_info(a, box=None):
    """{color: (pixels, x, y)}; optional box=(x0,y0,x1,y1) restricts the count."""
    out = {}
    for k, m in color_masks(a).items():
        if box is not None:
            x0, y0, x1, y1 = box
            mm = np.zeros_like(m)
            mm[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            m = mm
        ys, xs = np.nonzero(m)
        out[k] = (int(len(xs)), float(xs.mean()) if len(xs) else -1.0, float(ys.mean()) if len(xs) else -1.0)
    return out


def cube_blob(front, color, min_pixels=12):
    """Largest connected blob of a cube colour in an image: (x, y, pixels, mask) or None."""
    m = color_masks(front)[color]
    lab, n = ndimage.label(m, structure=EIGHT)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, np.arange(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    if sizes[i - 1] < min_pixels:
        return None
    reg = lab == i
    ys, xs = np.nonzero(reg)
    return float(xs.mean()), float(ys.mean()), int(len(xs)), reg


def target_mask(a):
    r, g, b = _split(a)
    return (b > g + 12) & (r > 120) & (r > b - 60)


def tw_wood(a):
    """(target-or-white mask, wood mask)."""
    r, g, b = _split(a)
    ring = (b > g + 12) & (r > 120) & (r > b - 60)
    white = (r > 200) & (g > 200) & (b > 200)
    wood = (r > g + 15) & (g > b + 5) & (r > 90)
    return ring | white, wood


def components(mask, min_pixels):
    """Hole-filled 8-connected components: list of (x, y, pixels, region)."""
    lab, n = ndimage.label(mask, structure=EIGHT)
    out = []
    for i in range(1, n + 1):
        reg = ndimage.binary_fill_holes(lab == i)
        cnt = int(reg.sum())
        if cnt >= min_pixels:
            ys, xs = np.nonzero(reg)
            out.append((float(xs.mean()), float(ys.mean()), cnt, reg))
    return out


def target_blobs(front):
    """Target-coloured blobs in one frame: list of (x, y, pixels)."""
    return [(x, y, c) for x, y, c, _ in components(target_mask(front), BLOB_MIN_PIXELS)]


def press_feature(wrist):
    """(largest white component pixels, solidity, cx, cy) in the wrist image."""
    r, g, b = _split(wrist)
    white = (r > 215) & (g > 215) & (b > 215)
    lab, n = ndimage.label(white)
    if n == 0:
        return 0, 0.0, -1.0, -1.0
    sizes = ndimage.sum(white, lab, np.arange(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    comp = lab == i
    area = int(comp.sum())
    filled = int(ndimage.binary_fill_holes(comp).sum())
    ys, xs = np.nonzero(comp)
    return area, area / max(filled, 1), float(xs.mean()), float(ys.mean())


def is_press(wrist):
    """Button pressed: big solid white square right under the empty gripper (centroid ~(150,175))."""
    area, sol, cx, cy = press_feature(wrist)
    return (area >= PRESS_WHITE_PIXELS and sol >= PRESS_SOLIDITY and 95 <= cx <= 205 and cy >= 135
            and held_color(wrist) is None)


HELD_BOX = (100, 150, 256, 256)   # wrist region where a grasped cube sits (centroid ~(180,215) in demos)


def held_color(wrist):
    info = cube_info(wrist, HELD_BOX)
    best = max(info.items(), key=lambda kv: kv[1][0])
    return best[0] if best[1][0] >= HELD_WRIST_PIXELS else None


def looks_like_wrist(img):
    """Wrist images show the dark gripper finger at the right bottom and a grey one at left bottom."""
    a = np.asarray(img).astype(np.int16)
    reg = a[185:250, 225:256]
    dark = ((reg.max(axis=2) < 70) & (np.abs(reg[..., 0] - reg[..., 2]) < 20)).mean()
    reg2 = a[200:256, 0:60]
    grey = ((np.abs(reg2[..., 0] - reg2[..., 2]) < 18) & (np.abs(reg2[..., 1] - reg2[..., 2]) < 18)).mean()
    return dark > 0.25 or grey > 0.5

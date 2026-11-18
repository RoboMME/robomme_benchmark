"""Conditioning-video analysis: where each coloured cube instance was before it got covered.

Uses both the front and the wrist camera of every video frame (depth + calibration), so cubes
hidden from one camera (e.g. behind occluders) are still found.
"""
import glob
import re
from pathlib import Path
import numpy as np
import perception as P

CHROMA = {"red": 0, "orange": 29, "yellow": 42, "green": 85, "cyan": 128, "blue": 170,
          "purple": 200, "magenta": 218, "pink": 233}
ALIASES = {"violet": "purple", "lime": "green", "teal": "cyan", "gold": "yellow", "grey": "gray"}
TABLE_HUE = 15
COLOR_WORDS = list(CHROMA.keys()) + ["black"]


def parse_targets(instruction):
    """Ordered [(colour, all_flag)] from e.g. 'the container hiding the green cube, followed by all
    containers hiding blue cubes'."""
    text = (instruction or "").lower()
    words = re.findall(r"[a-z]+", text)
    words = [ALIASES.get(w, w) for w in words]
    out = []
    for i, w in enumerate(words):
        if w not in COLOR_WORDS:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        prev = " ".join(words[max(0, i - 6):i])
        plural = nxt in ("cubes", "blocks", "ones") or bool(re.search(r"\b(all|both|every|each)\b", prev))
        if any(c == w for c, _ in out):
            continue
        out.append((w, plural))
    return out


def parse_colors(instruction):
    return [c for c, _ in parse_targets(instruction)]


def find_manifest(meta):
    seq = (meta or {}).get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if mp and (P.WS / mp).exists():
        return mp
    # fallback: an initial sequence of *this* episode only
    ep = (meta or {}).get("episode_id")
    cands = sorted(glob.glob(str(P.WS / "runtime/autonomous_controller/runs/*/frames/sequence_*/manifest.json")),
                   key=lambda p: Path(p).stat().st_mtime)
    for c in reversed(cands):
        try:
            m = P.load_manifest(str(Path(c).relative_to(P.WS)))
            if ep and (m.get("transition") or {}).get("episode_id") != ep:
                continue
            if any(f.get("phase") == "video_demo" for f in m.get("frames", [])):
                return str(Path(c).relative_to(P.WS))
        except Exception:
            continue
    return None


def label_image(hsv, tol=12):
    """Per-pixel colour index into COLOR_WORDS, -1 for none (table, white, grey...)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = (v >= 50) & ((s >= 200) | ((s >= 110) & (P.hue_dist(h, TABLE_HUE) >= 10)))
    lab = np.full(h.shape, -1)
    best = np.full(h.shape, 999)
    for i, c in enumerate(CHROMA):
        d = P.hue_dist(h, CHROMA[c])
        upd = chroma & (d < best) & (d <= tol)
        lab[upd] = i
        best = np.where(upd, d, best)
    black = (v <= 75) & (s <= 70)
    lab[black] = COLOR_WORDS.index("black")
    return lab


def detections(manifest_path, log=print, want=None):
    """List of dict(colour, frame, cam, xy, npx) for cube blobs in the video frames."""
    man = P.load_manifest(manifest_path)
    frames = [f for f in man["frames"] if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: f.get("frame_index", 0))
    dets = []
    for fr in frames:
        files = P.frame_files(fr)
        for cam, min_px in (("front", 6), ("wrist", 12)):
            if (cam, "rgb") not in files or (cam, "depth") not in files:
                continue
            try:
                rgb, hsv = P.load_rgb_hsv(files[(cam, "rgb")][0])
                path, mpu = files[(cam, "depth")]
                d = P.load_depth(path, mpu)
                K = fr["camera_intrinsics"][cam]
                pose = fr["camera_poses"][cam]
            except Exception as e:  # missing file / calibration
                log(f"video frame {fr.get('frame_index')} {cam}: {e}")
                continue
            lab = label_image(hsv)
            for i, c in enumerate(COLOR_WORDS):
                if want is not None and c not in want:
                    continue
                mk = lab == i
                if mk.sum() < min_px:
                    continue
                for n, vs, us in P.components(mk, min_px):
                    m2 = np.zeros(mk.shape, bool)
                    m2[vs, us] = True
                    pts = P.deproject_mask(m2, d, K, pose)
                    pts = pts[(pts[:, 2] > -0.01) & (pts[:, 2] < 0.07)]
                    if len(pts) < max(4, min_px // 2):
                        continue
                    xy = np.median(pts[:, :2], 0)
                    dets.append({"colour": c, "frame": fr.get("frame_index", 0), "cam": cam,
                                 "xy": xy, "npx": int(n)})
    return dets, len(frames)


def instances(dets, radius=0.03, min_dets=3):
    """Cluster detections per colour into cube instances: colour -> list of dict(xy, n, last_frame)."""
    out = {}
    by_c = {}
    for d in dets:
        by_c.setdefault(d["colour"], []).append(d)
    for c, ds in by_c.items():
        xy = np.array([d["xy"] for d in ds])
        left = np.ones(len(ds), bool)
        inst = []
        while left.any():
            # seed at the densest remaining detection
            idx = np.nonzero(left)[0]
            dens = [(np.linalg.norm(xy[idx] - xy[i], axis=1) < radius).sum() for i in idx]
            centre = xy[idx[int(np.argmax(dens))]]
            for _ in range(5):
                mem = left & (np.linalg.norm(xy - centre, axis=1) < radius)
                if not mem.any():
                    break
                centre = np.median(xy[mem], 0)
            mem = left & (np.linalg.norm(xy - centre, axis=1) < radius)
            if not mem.any():
                left[idx[int(np.argmax(dens))]] = False
                continue
            left &= ~mem
            members = [ds[k] for k in np.nonzero(mem)[0]]
            frames = sorted(set(m["frame"] for m in members))
            if len(members) >= min_dets:
                # prefer the larger (less occluded) detections for the position
                npx = np.array([m["npx"] for m in members])
                big = npx >= 0.5 * npx.max()
                pos = np.median(np.array([m["xy"] for m, b in zip(members, big) if b]), 0)
                inst.append({"xy": pos, "n": len(members), "frames": len(frames),
                             "first_frame": frames[0], "last_frame": frames[-1],
                             "cams": sorted(set(m["cam"] for m in members))})
        # merge instances that ended up very close
        merged = []
        for it in sorted(inst, key=lambda t: -t["n"]):
            if all(np.linalg.norm(it["xy"] - m["xy"]) > 0.04 for m in merged):
                merged.append(it)
        if merged:
            out[c] = merged
    return out

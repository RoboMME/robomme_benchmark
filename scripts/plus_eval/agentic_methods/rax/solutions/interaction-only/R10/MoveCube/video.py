"""Classify the manner shown in the conditioning video from robot states + RGB-D."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from geom import deproject_all

WS = Path("/workspace")

def masks(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return {
        "red": (r > 120) & (g < 50) & (b < 50),
        "purple": (r > 130) & (b > 170) & (g < 140) & (b - g > 70),
        "teal": (g - r > 45) & (b - r > 45) & (np.abs(g - b) < 25) & (r < 110),
    }

def load_frame(fr):
    files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
    out = {}
    for cam in ("front", "wrist"):
        if (cam, "rgb") in files:
            out[(cam, "rgb")] = np.array(Image.open(WS / files[(cam, "rgb")]).convert("RGB"))
        if (cam, "depth") in files:
            out[(cam, "depth")] = np.array(Image.open(WS / files[(cam, "depth")])).astype(np.float32) * 0.001
    return out

def blob_world(meta, imgs, cam, color, zmax=0.1):
    rgb = imgs.get((cam, "rgb")); dep = imgs.get((cam, "depth"))
    if rgb is None or dep is None:
        return None, 0
    m = masks(rgb)[color]
    if m.sum() < 3:
        return None, int(m.sum())
    P = deproject_all(meta, cam, dep)
    pts = P[m]
    pts = pts[np.isfinite(pts).all(1)]
    pts = pts[pts[:, 2] < zmax]
    if len(pts) < 3:
        return None, int(m.sum())
    return np.median(pts, axis=0), len(pts)

def _ball_grasp(vid, imgs0, res, S, width):
    """Video grasps the yellow ball: record release point relative to the inclined board."""
    try:
        from perception import locate_ball, board_pose, board_z
        bxyz, cam = locate_ball(vid[0], imgs0)
        gp = np.array(res["grasp_pos"])
        if bxyz is None or np.linalg.norm(bxyz[:2] - gp[:2]) > 0.04:
            return False
        gi = res["grasp_index"]
        rel = np.where(width[gi:] > 0.07)[0]
        if not len(rel):
            return False
        ri = gi + int(rel[0])
        rp = S[max(gi, ri - 1), :3]
        res["release_index"] = ri; res["release_pos"] = rp.round(4).tolist()
        bd = board_pose(vid[0], imgs0)
        if bd is None:
            return True
        s = float((rp[:2] - bd["c"]) @ bd["g"]); l = float((rp[:2] - bd["c"]) @ bd["w"])
        res["ramp"] = {"slope": bd["slope"], "s_from_low": bd["s_low"] - s, "surf_from_low": (bd["s_low"] - s) / np.cos(bd["slope"]),
                       "lat": l, "dz_surface": float(rp[2] - board_z(bd, rp[:2])),
                       "h_rel": float(board_z(bd, rp[:2]) - board_z(bd, bd["c"] + bd["g"] * bd["s_low"])),
                       "d_ct": None if (res.get("cube0") is None or res.get("target0") is None) else
                           float(np.linalg.norm(np.array(res["cube0"][:2]) - np.array(res["target0"][:2]))),
                       "cube_lat": None if res.get("cube0") is None else float((np.array(res["cube0"][:2]) - bd["c"]) @ bd["w"])}
        return True
    except Exception as e:
        res["ramp_err"] = repr(e)
        return False

def analyze(manifest_path):
    man = json.loads((WS / manifest_path).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    res = {"n_video": len(vid)}
    if not vid:
        res["manner"] = "unknown"
        return res
    S = np.array([f["state"] for f in vid], dtype=float)
    width = S[:, 6] + S[:, 7]
    res["width_min"] = float(width.min())
    imgs0 = load_frame(vid[0])
    cube0, n = blob_world(vid[0], imgs0, "front", "red")
    tgt0, nt = blob_world(vid[0], imgs0, "front", "purple")
    res["cube0"] = None if cube0 is None else cube0.round(4).tolist()
    res["target0"] = None if tgt0 is None else tgt0.round(4).tolist()
    closed = np.where(width < 0.072)[0]
    if len(closed):
        gi = int(closed[0])
        # settled closed width: median of the next frames while closed
        seg = width[gi:gi + 30]
        wc = float(np.median(seg[-10:])) if len(seg) else float(width[gi])
        gp = S[gi, :3]
        res["grasp_index"] = gi
        res["grasp_pos"] = gp.round(4).tolist()
        res["closed_width"] = wc
        if cube0 is not None:
            res["grasp_to_cube_xy"] = float(np.linalg.norm(gp[:2] - cube0[:2]))
    # lowest TCP height and where
    zi = int(np.argmin(S[:, 2]))
    res["zmin"] = float(S[zi, 2]); res["zmin_pos"] = S[zi, :3].round(4).tolist()
    # manner decision
    if "grasp_index" not in res or res["closed_width"] < 0.006:
        manner = "push"
    elif res.get("grasp_to_cube_xy", 1.0) < 0.04:
        manner = "pick"
    elif _ball_grasp(vid, imgs0, res, S, width):
        manner = "ramp"
    else:
        manner = "tool"
        try:
            from perception import find_tools
            tools = [t for t in find_tools(vid[0], imgs0) if t["kind"] != "unknown"]
            gp = np.array(res["grasp_pos"])
            if tools:
                best = min(tools, key=lambda t: np.linalg.norm(t["H"][:2] - gp[:2]))
                res["tool_dist"] = float(np.linalg.norm(best["H"][:2] - gp[:2]))
                if res["tool_dist"] < 0.06:
                    res["tool_kind"] = best["kind"]
                    res["tool_L"] = best["L_end"]
                res["tools"] = [(t["kind"], t["cam"], np.round(t["H"], 3).tolist()) for t in tools]
        except Exception as e:
            res["tool_err"] = repr(e)
    res["manner"] = manner
    return res

if __name__ == "__main__":
    import sys
    print(json.dumps(analyze(sys.argv[1]), indent=1))

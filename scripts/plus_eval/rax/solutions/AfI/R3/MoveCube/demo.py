"""Analyse the conditioning video: which object was grasped and the tool/cube geometry during the push."""
import json, os, math
import numpy as np
from perception import obs_from_manifest_frame, cube_estimate, target_estimate, components, masks


def _ang(v):
    return math.atan2(v[1], v[0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def tool_list(o, cams=("front", "wrist")):
    """All tools: teal handle + pink head paired by shaft coverage. Returns list of dicts."""
    out = []
    for cam in cams:
        if cam not in o.rgb:
            continue
        pinks = [c for c in components(o, cam, "pink", 5) if c["diag"] < 0.2]
        teals = [c for c in components(o, cam, "teal", 5) if c["diag"] < 0.12]
        sm = masks(o.rgb[cam])["shaft"]
        SP = o.cloud(cam)[sm]
        SP = SP[np.isfinite(SP).all(1)]
        SP = SP[(SP[:, 2] > -0.02) & (SP[:, 2] < 0.3)][:, :2]
        pairs = []
        for i, pk in enumerate(pinks):
            for j, tl in enumerate(teals):
                a, b = tl["c"][:2], pk["c"][:2]
                d = np.linalg.norm(b - a)
                if d < 0.06 or d > 0.5:
                    continue
                cov = 0.0
                if len(SP):
                    ts = np.linspace(0.2, 0.8, 13)
                    cov = float(np.mean([np.min(np.linalg.norm(SP - (a + t * (b - a)), axis=1)) < 0.02 for t in ts]))
                if cov < 0.5:
                    continue
                pairs.append((cov * 100 - d * 10, i, j, cov))
        pairs.sort(reverse=True)
        used_p, used_t = set(), set()
        for sc, i, j, cov in pairs:
            if i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j)
            pk, tl = pinks[i], teals[j]
            head = pk["P"].mean(0)
            dvec = head[:2] - tl["c"][:2]
            L = float(np.linalg.norm(dvec))
            out.append(dict(cam=cam, handle=tl["c"], head=head, d=dvec / L, length=L, head_diag=pk["diag"],
                            n_pink=pk["n"], cov=cov, pink_P=pk["P"]))
        if out:
            break  # first camera with tools
    return out


def find_manifest(meta, root="/workspace"):
    seq = meta.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if mp:
        return mp if os.path.isabs(mp) else os.path.join(root, mp)
    import glob
    eid = meta.get("episode_id")
    cands = sorted(glob.glob(os.path.join(root, "runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json")),
                   key=os.path.getmtime, reverse=True)
    for p in cands:
        try:
            m = json.load(open(p))
            if eid is None or (m.get("transition") or {}).get("episode_id") == eid:
                return p
        except Exception:
            pass
    return None


def load_frames(meta, root="/workspace"):
    path = find_manifest(meta, root)
    if not path:
        return None
    m = json.load(open(path))
    frames = [f for f in m.get("frames", []) if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: f.get("frame_index", 0))
    return frames


def analyse(frames, log=print):
    S = np.array([f["state"] for f in frames], float)
    w = S[:, 6] + S[:, 7]
    w_open = float(np.median(w[: max(3, len(w) // 20)]))
    closed = w < w_open - 0.012
    res = dict(n=len(frames), w_open=w_open)
    o0 = obs_from_manifest_frame(frames[0])
    cube0 = cube_estimate(o0)
    tgt0 = target_estimate(o0)
    res["cube0"] = cube0; res["target0"] = tgt0
    tools0 = tool_list(o0)
    res["tools0"] = [dict(handle=t["handle"], head=t["head"], length=t["length"], head_diag=t["head_diag"]) for t in tools0]
    if closed.sum() < 3:
        res["kind"] = "push_open"
        return res
    i0 = int(np.argmax(closed))
    g = S[i0, :3]
    res["grasp_xyz"] = g; res["w_closed"] = float(np.median(w[closed]))
    dc = np.linalg.norm(g[:2] - cube0[:2]) if cube0 is not None else 9
    dt = [np.linalg.norm(g[:2] - t["handle"][:2]) for t in tools0]
    res["d_cube"] = float(dc); res["d_tools"] = [float(x) for x in dt]
    if dt and min(dt) < 0.05 and min(dt) < dc:
        k = int(np.argmin(dt))
        t = tools0[k]
        res["kind"] = "tool"
        res["tool"] = dict(handle=t["handle"], head=t["head"], length=t["length"], head_diag=t["head_diag"],
                           grasp_offset=float(np.dot(g[:2] - t["handle"][:2], t["d"])))
        # geometry while the cube moves
        recs = []
        prev = None
        for k2 in range(i0, len(frames), 6):
            if not closed[k2] or S[k2, 2] > 0.06:
                continue
            o = obs_from_manifest_frame(frames[k2])
            c = cube_estimate(o, cams=("front",))
            if c is None:
                continue
            pk = [p for p in components(o, "front", "pink", 4) if abs(p["diag"] - t["head_diag"]) < 0.04]
            if not pk:
                continue
            # head nearest the previous head / EE
            ee = S[k2, :2]
            pk.sort(key=lambda p: np.linalg.norm(p["c"][:2] - ee) if prev is None else np.linalg.norm(p["c"][:2] - prev))
            hd = pk[0]["P"].mean(0)
            prev = hd[:2]
            recs.append((k2, c[:2], hd[:2], ee.copy()))
        if len(recs) >= 3:
            cs = np.array([r[1] for r in recs])
            mv = np.linalg.norm(cs - cs[0], axis=1)
            moving = [r for r, m in zip(recs, mv) if m > 0.01]
            res["cube_travel"] = float(mv.max())
            if len(moving) >= 2 and mv.max() > 0.03:
                cend = cs[np.argmax(mv)]
                u = cend - cs[0]; u /= np.linalg.norm(u)
                rel = []; phis = []
                for k2, c, hd, ee in moving:
                    r = hd - c
                    a = -_ang(u)
                    rel.append([math.cos(a) * r[0] - math.sin(a) * r[1], math.sin(a) * r[0] + math.cos(a) * r[1]])
                    phis.append(_wrap(_ang(hd - ee) - _ang(u)))
                rel = np.array(rel)
                res["u_demo"] = u
                res["head_rel_u"] = np.median(rel, 0)
                res["phi_rel"] = float(np.median(phis))
                res["n_moving"] = len(moving)
        return res
    if res["w_closed"] < 0.008:
        res["kind"] = "push_closed"
    elif dc < 0.04:
        res["kind"] = "pick"
    else:
        res["kind"] = "unknown_grasp"
    return res

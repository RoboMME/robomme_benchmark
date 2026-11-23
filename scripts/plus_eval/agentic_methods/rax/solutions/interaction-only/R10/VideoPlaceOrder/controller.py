"""VideoPlaceOrder controller v4 (target identity tracking through pair-swap shuffles).
Reads the conditioning video's robot-state trajectory, extracts cube release locations in order,
chooses the k-th (from instruction ordinal), matches it to a live target, picks the cube and
places it with a low (contact) release, as the video does."""
import json, math, time, traceback
from pathlib import Path
import numpy as np
import vlib as V

OUT = Path("/workspace/output")
PLACE_Z = 0.018      # EE z at release (video releases at 0.016-0.023)
PICK_Z = 0.020
CARRY_Z = 0.15

def releases_from_video(man):
    ev, frames = V.video_events(man)
    rel = []
    for i, e in enumerate(ev):
        if e["kind"] == "hold->open" and e["xyz"][2] < 0.06:
            rel.append(e)
    return rel, ev

def choose_target(meta, imgs, xy):
    tg = V.find_targets(meta, imgs, "front")
    if not tg:
        return np.array(xy), None
    d = [np.linalg.norm(t["xyz"][:2] - np.asarray(xy)) for t in tg]
    j = int(np.argmin(d))
    if d[j] < 0.05:
        return tg[j]["xyz"][:2].copy(), d[j]
    return np.array(xy), d[j]

def pick(ctx, meta, imgs, color):
    est = V.find_cube_color(meta, imgs, "front", color)
    if est is None:
        est = V.find_cube_color(meta, imgs, "wrist", color)
    if est is None:
        raise RuntimeError("cube not found")
    c = est[0]
    V.log("cube front est", np.round(c, 3))
    for attempt in range(3):
        meta, imgs = V.move_to(ctx, [c[0] - 0.01, c[1], CARRY_Z], -1)
        e2 = V.find_cube_color(meta, imgs, "wrist", color, near=c, radius=0.08)
        if e2: c = e2[0]
        meta, imgs = V.move_to(ctx, [c[0], c[1], 0.08], -1, tol=0.004)
        e3 = V.find_cube_color(meta, imgs, "wrist", color, near=c, radius=0.05)
        if e3: c = e3[0]
        V.log("cube wrist est", np.round(c, 3))
        meta, imgs = V.move_to(ctx, [c[0], c[1], PICK_Z + 0.002], -1, tol=0.004)
        meta, imgs = V.gripper(ctx, 1, 10)
        w = meta["state"][6] + meta["state"][7]
        V.log("closed width", round(w, 4), "step", meta["step_id"])
        meta, imgs = V.move_to(ctx, [c[0], c[1], CARRY_Z], 1)
        w = meta["state"][6] + meta["state"][7]
        if w > 0.02:
            return meta, imgs, True
        V.log("grasp missed, retry")
        meta, imgs = V.gripper(ctx, -1, 6)
        est = V.find_cube_color(meta, imgs, "front", color)
        if est: c = est[0]
    return meta, imgs, False

def main(ctx):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {}
    try:
        meta, imgs = V.observe(ctx)
        instr = meta["instruction"]; V.log("instruction:", instr)
        color = V.parse_color(instr); k = V.parse_ordinal(instr) or 1
        man = V.load_video(meta)
        rel, ev = releases_from_video(man) if man else ([], [])
        for e in ev: V.log("event", e["kind"], e["frame"], np.round(e["xyz"], 3))
        cube0 = V.find_cube_color(meta, imgs, "front", color)
        chain = V.cube_chain(ev, cube0[0][:2]) if cube0 is not None else None
        if not chain:
            V.log("chain not found; using all releases"); chain = rel
        V.log("chain", [np.round(e["xyz"][:2], 3).tolist() for e in chain])
        if not chain:
            raise RuntimeError("no releases in video")
        # RULE v5: targets shuffle (pair swaps) during the video; follow the identity of the
        # target that received the k-th placement of this cube to its final spot.
        # Swaps are detected by point-symmetry of the two moving rings about the pair midpoint,
        # with both spots vacated >= 3 frames (identity_answer_sym). v4 pairs tracker is fallback.
        goal_xy = None
        try:
            t0 = time.time()
            dets = V.video_detections(man)
            ans, diag = V.identity_answer_sym(man, chain, k, dets, min_sym=3.0)
            V.log("sym ids", diag.get("ids"), "perm", diag.get("perm"), "t=%.1fs" % (time.time() - t0))
            for sh in diag.get("shuffles", []):
                V.log("  shuffle", sh)
            if ans is not None:
                goal_xy = np.array(ans)
            else:
                ans, diag = V.identity_answer_pairs(man, chain, k, dets)
                V.log("pairs fallback ids", diag.get("ids"))
                if ans is not None:
                    goal_xy = np.array(ans)
        except Exception as ex:
            V.log("identity tracking error", repr(ex))
        if goal_xy is None:
            idx = max(0, min(k - 1, len(chain) - 1))
            goal_xy = np.array(chain[idx]["xyz"][:2]); V.log("fallback to k-th release location")
        tgt, dist = choose_target(meta, imgs, goal_xy)
        V.log("k", k, "video xy", np.round(goal_xy, 3), "-> live target", np.round(tgt, 3), "dist", dist)
        rec.update(instr=instr, color=color, k=k, goal_xy=goal_xy.tolist(), target=tgt.tolist())
        meta, imgs, ok = pick(ctx, meta, imgs, color)
        if not ok:
            V.log("pick failed; trying anyway")
        meta, imgs = V.move_to(ctx, [tgt[0], tgt[1], CARRY_Z], 1)
        meta, imgs = V.move_to(ctx, [tgt[0], tgt[1], 0.05], 1, tol=0.004)
        meta, imgs = V.move_to(ctx, [tgt[0], tgt[1], PLACE_Z], 1, tol=0.003, max_iter=6)
        V.log("pre-release state", np.round(meta["state"], 3), "step", meta["step_id"])
        meta, imgs = V.gripper(ctx, -1, 6)
        V.log("released; ended", meta.get("episode_ended"), "step", meta["step_id"])
        if not meta.get("episode_ended"):
            meta, imgs = V.move_to(ctx, [tgt[0], tgt[1], CARRY_Z], -1)
            V.log("lifted; ended", meta.get("episode_ended"), "step", meta["step_id"])
        rec["ended"] = meta.get("episode_ended")
    except Exception as ex:
        V.log("ERROR", repr(ex)); V.log(traceback.format_exc())
        rec["error"] = repr(ex)
    finally:
        try:
            (OUT / "vpo_log.txt").write_text("\n".join(V.LOG))
            (OUT / "vpo_rec.json").write_text(json.dumps(rec, default=str))
        except Exception:
            pass

"""Defensive I/O over the RAX MCP robot tools.

The exact JSON layout of replies was never observed (demo-only: zero simulator starts), so
every field is located by key search and every image is labelled by the best
available evidence, in this order:
  1. the documented frame-sequence manifest (files with camera/kind/path) when its
     last frame is the current step,
  2. labels attached to the image blocks or captions/metadata lists next to them,
  3. the block order learnt by matching blocks against labelled manifest files,
  4. default order front, wrist.
"""
import base64
import io
import json
import os
import time
import traceback

import numpy as np
from PIL import Image

WS = "/workspace"


# ----------------------------------------------------------------------------- helpers
def find_key(obj, key, maxdepth=8):
    """Breadth-first search for `key`; returns the shallowest value or None."""
    q = [(obj, 0)]
    while q:
        o, d = q.pop(0)
        if isinstance(o, dict):
            if key in o:
                return o[key]
            if d < maxdepth:
                q.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list) and d < maxdepth:
            q.extend((v, d + 1) for v in o)
    return None


def find_all_keys(obj, key, maxdepth=8):
    out, q = [], [(obj, 0)]
    while q:
        o, d = q.pop(0)
        if isinstance(o, dict):
            if key in o:
                out.append(o[key])
            if d < maxdepth:
                q.extend((v, d + 1) for v in o.values())
        elif isinstance(o, list) and d < maxdepth:
            q.extend((v, d + 1) for v in o)
    return out


def is_num_list(v, n=None):
    if not isinstance(v, (list, tuple)):
        return False
    if n is not None and len(v) != n:
        return False
    return len(v) > 0 and all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in v)


def any_true(obj, keys):
    for k in keys:
        for v in find_all_keys(obj, k):
            if v is True:
                return True
    return False


def strip_data(block):
    return {k: v for k, v in block.items() if k not in ("data", "blob")}


def decode_image_bytes(raw):
    im = Image.open(io.BytesIO(raw))
    im.load()
    return im


def pil_to_arrays(im):
    """Return ('rgb', HxWx3 uint8) or ('depth', HxW uint16/float)."""
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.array(im)
        return "depth", arr
    if mode == "F":
        return "depth", np.array(im)
    if mode in ("L",):
        return "gray", np.array(im)
    return "rgb", np.array(im.convert("RGB"))


def load_disk_image(path):
    p = path if os.path.isabs(path) else os.path.join(WS, path)
    with Image.open(p) as im:
        im.load()
        return pil_to_arrays(im)


# ----------------------------------------------------------------------------- calibration
def _cam_entry(v, cam):
    if isinstance(v, dict):
        if cam in v:
            return v[cam]
        for k, x in v.items():
            if cam in str(k).lower():
                return x
    if isinstance(v, list):
        for x in v:
            if isinstance(x, dict) and str(x.get("camera", x.get("name", ""))).lower() == cam:
                return x
    return None


def parse_K(v):
    """-> (fx, fy, cx, cy) or None."""
    if v is None:
        return None
    if isinstance(v, dict):
        if all(k in v for k in ("fx", "fy", "cx", "cy")):
            return tuple(float(v[k]) for k in ("fx", "fy", "cx", "cy"))
        for k in ("K", "matrix", "intrinsic", "intrinsics", "camera_matrix", "values"):
            if k in v:
                r = parse_K(v[k])
                if r:
                    return r
        return None
    if isinstance(v, list):
        a = np.array(v, dtype=float) if all(isinstance(t, (list, int, float)) for t in v) else None
        if a is None:
            return None
        if a.shape == (3, 3):
            return float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])
        if a.size == 9:
            a = a.reshape(3, 3)
            return float(a[0, 0]), float(a[1, 1]), float(a[0, 2]), float(a[1, 2])
        if a.size == 4:
            return tuple(float(t) for t in a)
    return None


def parse_pose(v):
    """-> (R 3x3 camera->world, t, convention) or None."""
    from geom import quat_wxyz_to_mat
    if v is None:
        return None
    conv = "opengl"
    if isinstance(v, dict):
        c = str(v.get("convention", v.get("frame_convention", "opengl"))).lower()
        conv = "opencv" if "cv" in c else "opengl"
        pos = None
        for k in ('position_m', "position", "p", "translation", "t", "xyz"):
            if is_num_list(v.get(k), 3):
                pos = np.array(v[k], float)
                break
        q = None
        for k in ("quaternion_wxyz", "quaternion", "q", "wxyz", "orientation"):
            if is_num_list(v.get(k), 4):
                q = v[k]
                break
        if pos is not None and q is not None:
            return quat_wxyz_to_mat(q), pos, conv
        for k in ("matrix", "pose", "T", "cam2world", "transform", "extrinsics"):
            if k in v:
                r = parse_pose(v[k])
                if r:
                    return r[0], r[1], conv
        return None
    if isinstance(v, list):
        try:
            a = np.array(v, float)
        except Exception:
            return None
        if a.shape == (4, 4):
            return a[:3, :3], a[:3, 3], conv
    return None


def own_backproject(obs, cam, u, v, radius=1):
    dep = obs.depth.get(cam)
    K = parse_K(_cam_entry(obs.intrinsics, cam))
    pose = parse_pose(_cam_entry(obs.poses, cam))
    if dep is None or K is None or pose is None:
        return None
    H, W = dep.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return None
    win = dep[max(0, vi - radius):vi + radius + 1, max(0, ui - radius):ui + radius + 1]
    win = win[win > 0]
    if len(win) == 0:
        return None
    d = float(np.median(win))
    fx, fy, cx, cy = K
    R, t, conv = pose
    if conv == "opengl":
        pc = np.array([(u - cx) / fx * d, -(v - cy) / fy * d, -d])
    else:
        pc = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d])
    return R @ pc + t


# ----------------------------------------------------------------------------- data
class Obs:
    def __init__(self):
        self.step_id = None
        self.state = None
        self.instruction = None
        self.rgb = {}      # camera -> HxWx3
        self.depth = {}    # camera -> HxW float metres
        self.intrinsics = {}
        self.poses = {}
        self.meta = {}
        self.ended = False
        self.complete = False
        self.label_source = None

    @property
    def pos(self):
        return np.array(self.state[0:3], float)

    @property
    def rotvec(self):
        return np.array(self.state[3:6], float)

    @property
    def fingers(self):
        return float(self.state[6]) + float(self.state[7])


class StepResult:
    def __init__(self, meta, ended, complete, ok=True, error=None):
        self.meta, self.ended, self.complete, self.ok, self.error = meta, ended, complete, ok, error


# ----------------------------------------------------------------------------- IO
class RobotIO:
    END_KEYS = ("episode_ended",)
    COMPLETE_KEYS = ("task_complete",)

    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.block_order = None   # learnt mapping: index among rgb blocks -> camera
        self.ended = False
        self.complete = False
        self.last_obs = None
        self.n_calls = 0
        self.steps_sent = 0
        self.remaining = None
        self.max_steps = None
        self.raw_dump_budget = 6
        self.px_parse_fail = 0
        self.own_used = False
        self.timing = {}
        self.n_observe = 0
        self.own_check = None

    # ------------------------------------------------------------- raw calls
    def call(self, tool, **args):
        self.n_calls += 1
        t = time.time()
        try:
            res = self.ctx.call(tool, **args)
        finally:
            tt = self.timing.setdefault(tool, [0, 0.0])
            tt[0] += 1
            tt[1] += time.time() - t
        meta, blocks = self._split(res)
        if self.raw_dump_budget > 0:
            self.raw_dump_budget -= 1
            try:
                self._dump_raw(tool, args, res)
            except Exception:
                pass
        return meta, blocks

    def _dump_raw(self, tool, args, res):
        os.makedirs(os.path.join(WS, "output"), exist_ok=True)
        slim = dict(res)
        cont = []
        for b in res.get("content", []) or []:
            if isinstance(b, dict):
                bb = strip_data(b)
                if bb.get("type") == "text" and len(bb.get("text", "")) > 20000:
                    bb["text"] = bb["text"][:20000] + "...<cut>"
                cont.append(bb)
        slim["content"] = cont
        with open(os.path.join(WS, "output", "raw_replies.jsonl"), "a") as f:
            f.write(json.dumps({"tool": tool, "args": {k: v for k, v in args.items() if k != "actions"},
                                "n_actions": len(args.get("actions", []) or []),
                                "reply": slim}, default=str)[:60000] + "\n")

    def _split(self, res):
        meta = {}
        blocks = res.get("content", []) if isinstance(res, dict) else []
        sc = res.get("structuredContent") if isinstance(res, dict) else None
        if isinstance(sc, dict):
            meta.update(sc)
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text", "")
                try:
                    j = json.loads(t)
                except Exception:
                    continue
                if isinstance(j, dict):
                    for k, v in j.items():
                        meta.setdefault(k, v)
        return meta, blocks

    # ------------------------------------------------------------- status
    def status(self):
        try:
            meta, _ = self.call("robomme_status")
        except Exception as e:
            self.log(f"robomme_status failed: {e}")
            return {}
        self._update_flags(meta)
        return meta

    def _update_flags(self, meta):
        if any_true(meta, self.END_KEYS):
            self.ended = True
        if any_true(meta, self.COMPLETE_KEYS):
            self.complete = True
        lim = find_key(meta, "episode_limits")
        if isinstance(lim, dict):
            for k, v in lim.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    kl = k.lower()
                    if "remain" in kl:
                        self.remaining = int(v)
                    elif "max" in kl or "limit" in kl or kl in ("steps", "horizon"):
                        self.max_steps = int(v)

    # ------------------------------------------------------------- observe
    def observe(self):
        self.n_observe += 1
        meta, blocks = self.call("robomme_observe")
        self._update_flags(meta)
        obs = self._parse_obs(meta, blocks)
        self.last_obs = obs
        return obs

    def _parse_obs(self, meta, blocks):
        o = Obs()
        o.meta = meta
        for st in find_all_keys(meta, "state"):
            if isinstance(st, dict):
                for k in ("values", "vector", "data"):
                    if is_num_list(st.get(k)) and len(st[k]) >= 8:
                        st = st[k]
                        break
            if is_num_list(st) and len(st) >= 8:
                o.state = [float(t) for t in st[:8]]
                break
        if o.state is None:
            for k in ("proprio", "proprioception", "robot_state", "proprioceptive_state"):
                v = find_key(meta, k)
                if is_num_list(v) and len(v) >= 8:
                    o.state = [float(t) for t in v[:8]]
                    break
        for sid in find_all_keys(meta, "step_id"):
            if isinstance(sid, (int, float)) and not isinstance(sid, bool):
                o.step_id = int(sid)
                break
        ins = find_key(meta, "instruction")
        if isinstance(ins, str):
            o.instruction = ins
        o.intrinsics = find_key(meta, "camera_intrinsics") or {}
        o.poses = find_key(meta, "camera_poses") or {}
        o.ended = any_true(meta, self.END_KEYS)
        o.complete = any_true(meta, self.COMPLETE_KEYS)
        self._parse_images(o, meta, blocks)
        return o

    # ------------------------------------------------------------- images
    def _manifest_images(self, meta, step_id):
        """Labelled images from a frame-sequence manifest if its last frame is current."""
        mp = None
        fs = find_key(meta, "frame_sequence")
        if isinstance(fs, dict):
            mp = fs.get("manifest_path")
        if mp is None:
            mp = find_key(meta, "manifest_path")
        if not isinstance(mp, str):
            return None, None
        p = mp if os.path.isabs(mp) else os.path.join(WS, mp)
        if not os.path.exists(p):
            return None, None
        with open(p) as f:
            man = json.load(f)
        frames = man.get("frames") or []
        if not frames:
            return None, None
        fr = max(frames, key=lambda r: (r.get("frame_index", 0) if isinstance(r.get("frame_index"), int) else 0))
        ex = [r for r in frames if r.get("phase") == "execution"]
        if ex:
            fr = max(ex, key=lambda r: (r.get("step_id") or 0, r.get("frame_index") or 0))
        fsid = fr.get("step_id")
        if isinstance(fsid, (int, float)) and step_id is not None:
            if int(fsid) != int(step_id):
                return None, fr
        elif self.n_observe > 1:
            # cannot prove the manifest frame is current: only trusted on the first observation
            return None, fr
        out = {}
        for fi in fr.get("files", []) or []:
            cam, kind, path = fi.get("camera"), fi.get("kind"), fi.get("path")
            if not (cam and kind and path):
                continue
            try:
                k, arr = load_disk_image(path)
            except Exception as e:
                self.log(f"manifest image load failed {path}: {e}")
                continue
            if kind == "depth":
                mpu = fi.get("meters_per_unit", 0.001) or 0.001
                out[(cam, "depth")] = arr.astype(np.float64) * float(mpu)
            else:
                if k == "rgb":
                    out[(cam, "rgb")] = arr
        return out, fr

    def _block_label(self, blocks, i):
        b = blocks[i]
        txt = json.dumps(strip_data(b), default=str).lower()
        cam = "front" if "front" in txt else ("wrist" if ("wrist" in txt or "hand" in txt) else None)
        kind = "depth" if "depth" in txt else None
        if cam is None and i > 0:
            # caption text block right before
            pb = blocks[i - 1]
            if isinstance(pb, dict) and pb.get("type") == "text":
                t = pb.get("text", "")
                if len(t) < 400:
                    tl = t.lower()
                    if "front" in tl and "wrist" not in tl:
                        cam = "front"
                    elif "wrist" in tl and "front" not in tl:
                        cam = "wrist"
                    if "depth" in tl:
                        kind = "depth"
        return cam, kind

    def _descriptor_labels(self, meta, img_blocks, o):
        """Metadata list of {camera, kind} entries in the same order as the image blocks."""
        q = [meta]
        while q:
            x = q.pop(0)
            if isinstance(x, dict):
                q.extend(x.values())
            elif isinstance(x, list):
                if len(x) == len(img_blocks) and len(x) >= 2 and all(isinstance(t, dict) and "camera" in t for t in x):
                    got = {}
                    for d, b in zip(x, img_blocks):
                        cam = str(d.get("camera")).lower()
                        kind = str(d.get("kind", d.get("type", "rgb"))).lower()
                        if "depth" in kind or b["kind"] == "depth":
                            if b["kind"] == "depth":
                                o.depth[cam] = b["arr"].astype(np.float64) * float(d.get("meters_per_unit", 0.001) or 0.001)
                        elif b["kind"] == "rgb":
                            got[cam] = b["arr"]
                    if "front" in got and "wrist" in got:
                        o.rgb.update(got)
                        return True
                q.extend(x)
        return False

    def _parse_images(self, o, meta, blocks):
        img_blocks = []
        for i, b in enumerate(blocks):
            if isinstance(b, dict) and b.get("type") == "image" and b.get("data"):
                try:
                    im = decode_image_bytes(base64.b64decode(b["data"]))
                    kind, arr = pil_to_arrays(im)
                except Exception as e:
                    self.log(f"image decode failed: {e}")
                    continue
                cam, lk = self._block_label(blocks, i)
                img_blocks.append(dict(idx=i, kind=kind, arr=arr, cam=cam, lkind=lk))
        man_imgs, man_frame = None, None
        try:
            man_imgs, man_frame = self._manifest_images(meta, o.step_id)
        except Exception as e:
            self.log(f"manifest parse failed: {e}")
        if o.state is None and isinstance(man_frame, dict) and is_num_list(man_frame.get("state")):
            o.state = [float(t) for t in man_frame["state"][:8]]

        rgb_blocks = [b for b in img_blocks if b["kind"] == "rgb" and b["lkind"] != "depth"]
        depth_blocks = [b for b in img_blocks if b["kind"] == "depth" or b["lkind"] == "depth"]

        # learn block order from manifest images (pixel match)
        if man_imgs and rgb_blocks:
            order = []
            for b in rgb_blocks:
                best, bd = None, 1e9
                for cam in ("front", "wrist"):
                    ref = man_imgs.get((cam, "rgb"))
                    if ref is None or ref.shape != b["arr"].shape:
                        continue
                    d = float(np.mean(np.abs(ref.astype(np.int16) - b["arr"].astype(np.int16))))
                    if d < bd:
                        best, bd = cam, d
                order.append(best if bd < 12 else None)
            if order.count("front") == 1 and order.count("wrist") == 1:
                if self.block_order != order:
                    self.log(f"learnt rgb block order from manifest: {order}")
                self.block_order = order

        # rgb assignment
        src = None
        if man_imgs and ("front", "rgb") in man_imgs and ("wrist", "rgb") in man_imgs:
            o.rgb["front"], o.rgb["wrist"] = man_imgs[("front", "rgb")], man_imgs[("wrist", "rgb")]
            src = "manifest"
        else:
            labelled = {b["cam"]: b["arr"] for b in rgb_blocks if b["cam"]}
            if "front" in labelled and "wrist" in labelled:
                o.rgb.update(labelled)
                src = "block_labels"
            elif self.block_order and len(rgb_blocks) == len(self.block_order):
                for b, cam in zip(rgb_blocks, self.block_order):
                    if cam is not None:
                        o.rgb[cam] = b["arr"]
                src = "learnt_order"
            elif self._descriptor_labels(meta, img_blocks, o):
                src = "meta_descriptors"
            elif len(rgb_blocks) >= 2:
                o.rgb["front"], o.rgb["wrist"] = rgb_blocks[0]["arr"], rgb_blocks[1]["arr"]
                src = "default_order"
            elif len(rgb_blocks) == 1:
                o.rgb[rgb_blocks[0]["cam"] or "front"] = rgb_blocks[0]["arr"]
                src = "single"
        o.label_source = src

        # depth assignment
        if man_imgs:
            for cam in ("front", "wrist"):
                if (cam, "depth") in man_imgs:
                    o.depth[cam] = man_imgs[(cam, "depth")]
        if not o.depth and depth_blocks:
            labelled = {b["cam"]: b["arr"] for b in depth_blocks if b["cam"]}
            if labelled:
                for cam, arr in labelled.items():
                    o.depth[cam] = arr.astype(np.float64) * (0.001 if arr.dtype != np.float32 else 1.0)
            elif len(depth_blocks) >= 2:
                for cam, b in zip(("front", "wrist"), depth_blocks):
                    o.depth[cam] = b["arr"].astype(np.float64) * 0.001

    # ------------------------------------------------------------- step
    def step(self, rows):
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        try:
            meta, _ = self.call("robomme_step", actions=rows)
        except Exception as e:
            self.log(f"robomme_step error: {e}")
            # do not retry blindly; refresh status
            if "ended" in str(e).lower() or "terminal" in str(e).lower():
                self.ended = True
            self.status()
            return StepResult({}, self.ended, self.complete, ok=False, error=str(e))
        self.steps_sent += len(rows)
        if self.remaining is not None:
            self.remaining -= len(rows)
        self._update_flags(meta)
        return StepResult(meta, self.ended, self.complete)

    # ------------------------------------------------------------- 3-D points
    def px2world(self, step_id, sels, radius=1):
        """sels: list of (camera, u, v). Returns list of np.array(3) or None."""
        out = []
        for i in range(0, len(sels), 32):
            chunk = sels[i:i + 32]
            args = dict(observation_step_id=int(step_id if step_id is not None else 0),
                        selections=[dict(camera=c, u=float(u), v=float(v)) for c, u, v in chunk],
                        depth_window_radius=int(radius))
            pts = [None] * len(chunk)
            try:
                meta, _ = self.call("robomme_pixel_to_position", **args)
                pts = self._parse_points(meta, len(chunk))
            except Exception as e:
                self.log(f"pixel_to_position failed: {e}")
                self.px_parse_fail += 1
            if all(p is None for p in pts) and self.last_obs is not None and self.last_obs.step_id == step_id:
                own = [own_backproject(self.last_obs, c, u, v, max(1, radius)) for c, u, v in chunk]
                if any(p is not None for p in own):
                    if not self.own_used:
                        self.log("pixel_to_position gave nothing; using own depth back-projection")
                    self.own_used = True
                    pts = own
            elif self.own_check is None and self.last_obs is not None and self.last_obs.step_id == step_id:
                try:
                    own = [own_backproject(self.last_obs, c, u, v, max(1, radius)) for c, u, v in chunk]
                    diffs = [float(np.linalg.norm(a - b)) for a, b in zip(pts, own) if a is not None and b is not None]
                    if diffs:
                        self.own_check = float(np.median(diffs))
                        self.log(f"own back-projection vs tool: median diff {self.own_check:.4f} m over {len(diffs)}")
                except Exception as e:
                    self.own_check = -1.0
                    self.log(f"own back-projection check failed: {e}")
            out.extend(pts)
        return out

    def _parse_points(self, meta, n):
        cands = []
        for k in ("points", "results", "positions", "selections", "world_points", "items"):
            v = find_key(meta, k)
            if isinstance(v, list) and len(v) == n:
                cands.append(v)
        if not cands:
            # any list of length n of dicts
            q = [meta]
            while q:
                o = q.pop(0)
                if isinstance(o, dict):
                    q.extend(o.values())
                elif isinstance(o, list):
                    if len(o) == n and all(isinstance(t, (dict, list)) for t in o):
                        cands.append(o)
                    q.extend(o)
        res = [None] * n
        if not cands:
            self.px_parse_fail += 1
            return res
        lst = cands[0]
        for i, it in enumerate(lst):
            res[i] = self._point_from_item(it)
        return res

    @staticmethod
    def _point_from_item(it):
        if is_num_list(it, 3):
            return np.array(it, float)
        if not isinstance(it, dict):
            return None
        if it.get("valid") is False:
            return None
        for k in ("position", "world_position", "world", "point", "xyz", "position_m",
                  "world_position_m", "world_point", "position_world", "p_world"):
            v = it.get(k)
            if is_num_list(v, 3):
                return np.array(v, float)
            if isinstance(v, dict) and all(a in v for a in "xyz"):
                return np.array([v["x"], v["y"], v["z"]], float)
        if all(a in it for a in "xyz") and all(isinstance(it[a], (int, float)) for a in "xyz"):
            return np.array([it["x"], it["y"], it["z"]], float)
        for k, v in it.items():
            kl = k.lower()
            if is_num_list(v, 3) and ("pos" in kl or "world" in kl or "point" in kl) and "cam" not in kl:
                return np.array(v, float)
        return None

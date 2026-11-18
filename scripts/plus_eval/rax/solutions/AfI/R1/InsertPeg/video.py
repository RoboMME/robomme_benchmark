"""Conditioning-video parsing: frames, robot-state trajectory, gripper events."""
import json
import os
import numpy as np

from geom import rotvec_to_mat, mat_to_rotvec, slerp_mat
from mcpio import log, _is_num_list

WS = "/workspace"


def resolve(path, manifest_path=None):
    if path is None:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    cands = [os.path.join(WS, path)]
    if manifest_path:
        cands.append(os.path.join(os.path.dirname(manifest_path), path))
    cands.append(path)
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def load_manifest(mpath):
    full = resolve(mpath)
    with open(full) as fh:
        m = json.load(fh)
    return m, full


def frame_files(fr, manifest_full):
    out = {}
    for f in fr.get("files", []) or []:
        try:
            cam, kind = f.get("camera"), f.get("kind")
            out[(cam, kind)] = (resolve(f.get("path"), manifest_full), f.get("meters_per_unit", 0.001))
        except Exception:
            pass
    return out


def split_frames(m):
    frames = list(m.get("frames", []) or [])
    frames.sort(key=lambda f: (f.get("frame_index", 0)))
    exe = [f for f in frames if f.get("phase") == "execution"]
    if exe:
        vid = [f for f in frames if f.get("phase") != "execution"]
    elif frames:
        exe, vid = frames[-1:], frames[:-1]
    else:
        vid = []
    return vid, exe


class Demo:
    """Robot-state trajectory recorded in the conditioning video."""

    def __init__(self, vid_frames):
        T, S = [], []
        for f in vid_frames:
            st = f.get("state")
            ts = f.get("timestamp_s")
            if _is_num_list(st, 8) and isinstance(ts, (int, float)):
                T.append(float(ts))
                S.append([float(x) for x in st])
        self.ok = False
        self.reason = ""
        if len(T) < 5:
            self.reason = "too few states (%d)" % len(T)
            return
        T = np.array(T)
        S = np.array(S)
        # sort/unique by time
        o = np.argsort(T, kind="stable")
        T, S = T[o], S[o]
        keep = np.concatenate([[True], np.diff(T) > 1e-6])
        T, S = T[keep], S[keep]
        self.T_raw, self.S_raw = T, S
        pos = S[:, :3]
        motion = float(np.max(np.linalg.norm(pos - pos[0], axis=1)))
        self.width_raw = S[:, 6] + S[:, 7]
        if motion < 0.02:
            self.reason = "robot hardly moves in video (%.3f m)" % motion
            return
        self.ok = True
        self._resample()

    def _resample(self, dt=0.05):
        T, S = self.T_raw, self.S_raw
        n = int(np.floor((T[-1] - T[0]) / dt + 1e-6)) + 1
        t = T[0] + dt * np.arange(n)
        self.t = t
        P = np.stack([np.interp(t, T, S[:, i]) for i in range(3)], axis=1)
        W = np.interp(t, T, self.width_raw)
        Rs_raw = [rotvec_to_mat(s[3:6]) for s in S]
        Rs = []
        j = 0
        for tk in t:
            while j < len(T) - 2 and T[j + 1] < tk:
                j += 1
            t0, t1 = T[j], T[min(j + 1, len(T) - 1)]
            a = 0.0 if t1 <= t0 else min(1.0, max(0.0, (tk - t0) / (t1 - t0)))
            Rs.append(slerp_mat(Rs_raw[j], Rs_raw[min(j + 1, len(T) - 1)], a))
        self.P, self.R, self.W = P, Rs, W
        self.n = n
        self._events()

    def _events(self):
        """Gripper command per resampled step from finger-width changes."""
        W = self.W
        n = self.n
        wmax = float(np.max(W))
        self.wmax = wmax
        # look-ahead width changes at two scales (fast and slow grippers)
        dW2 = np.zeros(n)
        dW2[:-2] = W[2:] - W[:-2]
        dW6 = np.zeros(n)
        if n > 6:
            dW6[:-6] = W[6:] - W[:-6]
        dW = np.where(np.abs(dW6) * 0.75 > np.abs(dW2), dW6 * 0.75, dW2)
        cmd = np.zeros(n)
        state = -1.0 if W[0] > 0.6 * wmax and wmax > 0.03 else 1.0
        if wmax < 0.01:
            state = 1.0
        events = []
        def onset(i, sign):
            # last frame before the width actually starts moving in direction sign
            j = i
            while j < min(n - 1, i + 8) and sign * (W[j + 1] - W[j]) < 0.0001:
                j += 1
            return j

        i = 0
        cmd[:] = state
        while i < n:
            if dW[i] < -0.0015 and state < 0:
                j = onset(i, -1.0)
                state = 1.0
                events.append(("close", j))
                cmd[j:] = state
                i = j + 1
                continue
            if dW[i] > 0.0015 and state > 0:
                j = onset(i, 1.0)
                state = -1.0
                events.append(("open", j))
                cmd[j:] = state
                i = j + 1
                continue
            i += 1
        self.cmd = cmd
        self.events = events
        self.init_cmd = cmd[0] if n else -1.0
        # holding intervals: [close_end, next open)
        self.holds = []
        for idx, (e, i) in enumerate(events):
            if e != "close":
                continue
            # end of the closing motion: width stops changing
            j = i
            # wait for the closing to start, then for it to stop
            while j < min(n - 1, i + 15) and W[j] > W[i] - 0.001:
                j += 1
            while j < n - 1 and W[j + 1] < W[j] - 0.0002:
                j += 1
            jend = n - 1
            for e2, i2 in events[idx + 1:]:
                if e2 == "open":
                    jend = i2
                    break
            wgrasp = float(W[min(j + 1, n - 1)])
            self.holds.append({"close": i, "closed": j, "release": jend, "width": wgrasp,
                               "object": wgrasp > 0.004})
        # first moving step
        d = np.linalg.norm(self.P - self.P[0], axis=1)
        mv = np.nonzero(d > 0.002)[0]
        self.first_move = int(max(0, mv[0] - 2)) if len(mv) else 0

    def grasp_hold(self):
        """First hold interval that holds an object, or None."""
        for h in self.holds:
            if h["object"]:
                return h
        return None

    def summary(self):
        if not self.ok:
            return "Demo(not ok: %s)" % self.reason
        return "Demo(n=%d, dur=%.2fs, events=%s, holds=%s, first_move=%d, wmax=%.4f)" % (
            self.n, self.t[-1] - self.t[0], self.events, self.holds, self.first_move, self.wmax)

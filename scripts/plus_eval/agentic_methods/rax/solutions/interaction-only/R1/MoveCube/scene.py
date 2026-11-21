"""Scene perception built on Perceiver: cube, tool, target in world frame."""
import math

import numpy as np

import perception as P

GRIP_EXCL_R = 0.045       # m, horizontal radius around TCP treated as gripper
GRIP_EXCL_DZ = 0.012      # points higher than TCP z - this are gripper


COLOR_WORDS = {"red": (200, 30, 30), "green": (30, 170, 50), "blue": (30, 60, 200), "yellow": (220, 200, 30),
               "purple": (130, 40, 170), "orange": (240, 130, 20), "pink": (240, 120, 180), "cyan": (30, 200, 210),
               "white": (235, 235, 235), "black": (25, 25, 25), "brown": (120, 70, 30), "gray": (128, 128, 128),
               "grey": (128, 128, 128)}


def instruction_color(text):
    t = (text or "").lower()
    import re
    hits = [w for w in COLOR_WORDS if re.search(r"\b" + w + r"\b(?=\s+(cube|block|box))", t)]
    return COLOR_WORDS[hits[0]] if len(hits) == 1 else None


class Scene:
    def __init__(self, robot, perceiver, log=print):
        self.robot, self.per, self.log = robot, perceiver, log
        self.table_z = None
        self.cube_color = None
        self.cube_size = 0.04
        self.tool_color = None
        self.target = None
        self.cube = None
        self.tool = None
        self._cache = {}
        self.want_color = None

    def _obs(self):
        obs = self.robot.obs()
        if obs.depth("front") is None or obs.rgb("front") is None:
            self.robot.observe()
            obs = self.robot.obs()
        return obs

    def cloud(self, cam="front", stride=1):
        rid = id(self.robot.reply)
        if self._cache.get("rid") != rid:
            self._cache = {"rid": rid}
        key = (cam, stride)
        if key in self._cache:
            return self._cache[key]
        obs = self._obs()
        if id(self.robot.reply) != rid:      # observe() refreshed the reply
            self._cache = {"rid": id(self.robot.reply)}
        cl = self.per.cloud(obs, cam, stride)
        self._cache[key] = cl
        return cl

    def grip_mask(self, cl, r=GRIP_EXCL_R, dz=GRIP_EXCL_DZ):
        s = self.robot.state
        if s is None:
            return np.zeros(cl.shape, bool)
        d = np.linalg.norm(cl.xyz[..., :2] - s[:2], axis=-1)
        return (d < r) & (cl.xyz[..., 2] > s[2] - dz)

    def survey(self):
        """Full perception from the front camera. Returns dict summary."""
        cl = self.cloud("front", 2)
        if cl is None:
            self.log("survey: no front cloud")
            return None
        if self.table_z is None:
            self.table_z = self.per.fit_table(cl)
            self.log(f"table z = {self.table_z}")
        tz = self.table_z
        comps, _ = self.per.components(cl, tz, exclude=self.grip_mask(cl, 0.08, 0.03))
        cubes, tools, others = P.classify_objects(comps)
        self.log("survey comps:", [(round(o["L"], 3), round(o["W"], 3), round(o["hmax"], 3), o["n"],
                                    np.round(o["center"], 3).tolist(), np.round(o["color"]).tolist()) for o in comps
                                   if o["hmax"] < 0.15][:12])
        out = {"cubes": cubes, "tools": tools, "others": others}
        if cubes:
            c = cubes[0]
            if self.cube_color is not None and len(cubes) > 1:
                c = min(cubes, key=lambda o: P.color_dist(o["color"], self.cube_color))
            elif self.want_color is not None and len(cubes) > 1:
                c = min(cubes, key=lambda o: P.color_dist(o["color"], self.want_color))
                self.log("cube chosen by instruction colour", np.round(c["color"]).tolist())
            self.cube_color = c["color"] if self.cube_color is None else self.cube_color
            self.cube_size = float(np.clip(c["hmax"], 0.02, 0.08))
            self.cube = self.refine_cube(c["center"], cams=("front",)) or self._cube_from(c)
        if tools:
            t = tools[0]
            self.tool_color = t["color"]
            full = self.cloud("front", 1)
            tt = P.color_track(full, tz, t["color"], near=t["center"], radius=0.5 * t["L"] + 0.03, thr=0.12,
                               hmin=0.004, hmax=0.12) if full is not None else None
            self.tool = t if tt is None or tt["n"] < t["n"] else self._merge_tool(t, tt)
        avoid = []
        tgt = P.find_target(cl, tz, [o for o in comps if o["hmax"] < 0.15], log=self.log)
        if tgt is None:
            # fallback: a low, wide non-cube non-tool object (e.g. a plate/pad target)
            flat = [o for o in others if o["hmax"] < 0.03 and 0.04 < o["L"] < 0.3 and o["n"] >= 10]
            if flat:
                o = max(flat, key=lambda o: o["n"])
                tgt = {"center": o["center"], "radius": 0.5 * o["L"], "n": o["n"], "score": 0, "src": "raised",
                       "color": o["color"]}
        if tgt is not None and self.target is None:
            self.target = tgt      # the target is static: keep the first (least occluded) estimate
        out["target"] = tgt
        self.log("survey: cube", None if self.cube is None else {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in self.cube.items() if k in ("center", "yaw", "size", "n")},
                 "tool", None if self.tool is None else (np.round(self.tool["center"], 3).tolist(), round(self.tool["L"], 3)),
                 "target", None if tgt is None else (np.round(tgt["center"], 3).tolist(), round(tgt["radius"], 3), np.round(tgt["color"]).tolist()))
        return out

    def _merge_tool(self, t, tt):
        return P.describe(tt["pts"], tt["cols"], self.table_z) if tt["L"] >= 0.8 * t["L"] else t

    def _cube_from(self, o):
        top = o["top"] if len(o["top"]) >= 4 else o["pts"]
        yaw, ctr, ext = P.min_area_yaw(top[:, :2])
        return {"center": ctr, "yaw": float(yaw), "size": self.cube_size, "n": o["n"], "zmax": o["zmax"]}

    def refine_cube(self, near, cams=("front", "wrist"), radius=0.07, excl_zmin=None):
        """Colour-based cube localisation near `near` with gripper exclusion."""
        if self.cube_color is None:
            return None
        best = None
        zmin = self.robot.state[2] - GRIP_EXCL_DZ if excl_zmin is None else excl_zmin
        for cam in cams:
            if best is not None and best["n"] >= 25:
                break
            cl = self.cloud(cam, 1)
            if cl is None:
                continue
            m = P.color_track(cl, self.table_z, self.cube_color, near=near, radius=radius, thr=0.13,
                              hmin=0.004, hmax=0.2,
                              exclude_xy=self.robot.state[:2], exclude_r=GRIP_EXCL_R,
                              exclude_zmin=zmin)
            if m is None or m["n"] < 5:
                continue
            top = m["top"] if len(m["top"]) >= 4 else m["pts"]
            occ = self.robot.state[:2] if np.linalg.norm(self.robot.state[:2] - np.asarray(near)[:2]) < 0.12 else None
            yaw, ctr, ext = P.fit_square(top[:, :2], self.cube_size, occ)
            cand = {"center": ctr, "yaw": float(yaw), "size": self.cube_size, "n": m["n"], "zmax": m["zmax"],
                    "ext": ext.tolist(), "cam": cam}
            if best is None or cand["n"] > best["n"]:
                best = cand
        return best

    def locate_cube(self, predicted, radius=0.09, excl_zmin=None):
        c = self.refine_cube(predicted, radius=radius, excl_zmin=excl_zmin)
        if c is None:
            c = self.refine_cube(predicted, radius=0.2, excl_zmin=excl_zmin)
        if c is not None:
            self.cube = c
        return c

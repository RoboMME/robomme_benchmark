"""Manner executors: pick-place, push with gripper, push/pull with a tool."""
import math

import numpy as np

import geom
import perception as P

# ---- tunables (untested live: R=1, chosen from Panda/ManiSkill geometry) ----
LIFT_CLEAR = 0.08        # m above cube top for transit
GRASP_MIN_W = 0.005      # finger sum above this after closing = something held
PLACE_TOL = 0.012        # m, cube-centre to target-centre acceptance
PUSH_TOL = 0.008
FINGER_HALF = 0.013      # m, closed finger outer face distance from TCP along closing axis
PUSH_SPEED = 0.008       # m/step during contact
CHUNK = 0.024            # m advanced between re-perceptions while pushing


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def nearest_equiv(target, cur, period):
    k = round((cur - target) / period)
    return target + k * period


class Executor:
    def __init__(self, robot, scene, log=print):
        self.R, self.S, self.log = robot, scene, log

    # --------------------------------------------------------------- helpers
    @property
    def tz(self):
        return self.S.table_z

    @property
    def s(self):
        return self.S.cube_size

    def target_xy(self):
        return np.asarray(self.S.target["center"], float)[:2]

    def cube_xy(self):
        return np.asarray(self.S.cube["center"], float)[:2]

    def up(self, z, grip, yaw=None):
        p = self.R.pos.copy()
        if p[2] < z - 0.004:
            p[2] = z
            self.R.move_to(p, yaw, grip, tol=0.008, label="up")

    def relocate(self, near=None, excl_zmin=None):
        near = self.cube_xy() if near is None else near
        c = self.S.locate_cube(near, excl_zmin=excl_zmin)
        return c

    # --------------------------------------------------------------- pick
    def pick_place(self, attempts=3):
        R, S = self.R, self.S
        for att in range(attempts):
            if R.steps_left() < 40:
                self.log("pick: not enough steps")
                return False
            self.log(f"[phase] pick attempt {att}")
            self.up(self.tz + self.s + LIFT_CLEAR, -1)
            c = self.relocate() or S.cube
            cxy = np.asarray(c["center"])[:2]
            yaw = nearest_equiv(c.get("yaw", 0.0), R.yaw(), math.pi / 2)
            hz = self.tz + self.s + LIFT_CLEAR
            R.move_to([cxy[0], cxy[1], hz], yaw, -1, tol=0.006, label="above cube")
            # refine from the new viewpoint (wrist sees the cube from above)
            c2 = self.S.locate_cube(cxy, radius=0.06)
            if c2 is not None and np.linalg.norm(np.asarray(c2["center"]) - cxy) < 0.04:
                cxy = np.asarray(c2["center"])[:2]
                yaw = nearest_equiv(c2.get("yaw", yaw), R.yaw(), math.pi / 2)
                R.move_to([cxy[0], cxy[1], hz], yaw, -1, tol=0.005, label="above cube 2")
            gz = self.tz + float(np.clip(self.s * 0.5, 0.012, 0.03))
            R.move_to([cxy[0], cxy[1], self.tz + self.s + 0.03], yaw, -1, tol=0.005, label="pre-grasp")
            R.move_to([cxy[0], cxy[1], gz], yaw, -1, tol=0.004, vmax=0.012, label="grasp")
            w = R.gripper(1.0, 12)
            self.log(f"pick: closed width {w:.4f}")
            if w < GRASP_MIN_W:
                self.log("pick: missed grasp")
                R.gripper(-1.0, 6)
                self.up(self.tz + self.s + LIFT_CLEAR, -1)
                continue
            R.move_to([R.pos[0], R.pos[1], self.tz + self.s + LIFT_CLEAR], None, 1.0, tol=0.008, label="lift")
            if R.grip_width < GRASP_MIN_W:
                self.log("pick: lost during lift")
                R.gripper(-1.0, 4)
                continue
            # re-detect target (it may have been occluded) only if not known
            t = self.target_xy()
            R.move_to([t[0], t[1], self.tz + self.s + LIFT_CLEAR], None, 1.0, tol=0.006, label="carry")
            R.move_to([t[0], t[1], self.tz + self.s * 0.5 + 0.012], None, 1.0, tol=0.004, vmax=0.012,
                      label="lower")
            R.gripper(-1.0, 8)
            self.up(self.tz + self.s + LIFT_CLEAR, -1)
            c3 = self.relocate(near=t)
            if c3 is None:
                self.log("pick: cube not seen after place; assume placed")
                return True
            err = float(np.linalg.norm(np.asarray(c3["center"])[:2] - t))
            self.log(f"pick: placed, cube-target error {err:.4f}")
            if err < PLACE_TOL:
                return True
        return False

    # --------------------------------------------------------------- push
    def push_loop(self, eff, label="push", attempts=5):
        """eff: callable(d) -> dict(yaw, along, lat, z_push, z_transit, grip, setup(d)->bool)."""
        R = self.R
        z_bias = 0.0
        strict = True
        prev_err = None
        for att in range(attempts):
            if R.steps_left() < 30:
                self.log(f"{label}: not enough steps")
                return False
            c = self.relocate(excl_zmin=self.tz + self.s + 0.008) or self.S.cube
            cxy = np.asarray(c["center"])[:2]
            t = self.target_xy()
            D = float(np.linalg.norm(t - cxy))
            self.log(f"[phase] {label} attempt {att}: cube {np.round(cxy, 3).tolist()} target {np.round(t, 3).tolist()} D={D:.3f}")
            if D < PUSH_TOL:
                return True
            d = unit(t - cxy)
            n = np.array([-d[1], d[0]])
            plan = eff(d, cxy)
            if plan is None:
                self.log(f"{label}: no plan")
                return False
            grip = plan["grip"]
            plan["z_push"] += z_bias
            # transit high, rotate, descend
            self.up(plan["z_transit"], grip)
            R.move_to([R.pos[0], R.pos[1], plan["z_transit"]], plan["yaw"], grip, tol=0.008, rtol=0.04, label="rotate")
            yerr = abs(geom.wrap(R.yaw() - plan["yaw"]))
            if yerr > 0.15 and plan.get("refine"):
                self.log(f"{label}: yaw off by {math.degrees(yerr):.0f} deg after rotate; refining plan at actual yaw")
                p2 = plan["refine"](d, cxy)
                if p2 is None:
                    self.log(f"{label}: no plan at actual yaw")
                    continue
                p2["z_push"] += z_bias
                plan = p2
            off = plan["along"] + self.s / 2
            e0 = cxy - d * (off + 0.025) - n * plan["lat"]
            R.move_to([e0[0], e0[1], plan["z_transit"]], plan["yaw"], grip, tol=0.006, label="pre-push")
            ok = R.move_to([e0[0], e0[1], plan["z_push"]], plan["yaw"], grip, tol=0.005, vmax=0.012, label="descend")
            if R.pos[2] > plan["z_push"] + 0.012:
                self.log(f"{label}: descent blocked at z {R.pos[2]:.3f}; raising push height")
                z_bias = min(z_bias + 0.008, 0.02)
                self.up(plan["z_transit"], grip)
                continue
            # contact pushing with re-perception
            lost = 0
            jumps = 0
            replan = False
            for k in range(40):
                if R.steps_left() < 12:
                    break
                e = R.pos[:2]
                c = self.S.locate_cube(cxy, radius=0.07, excl_zmin=self.tz + self.s + 0.008)
                if c is None:
                    lost += 1
                    # dead reckoning: the cube only moves once the effector face reaches it
                    face = e + d * off + n * plan["lat"]
                    push_in = float((face - cxy) @ d)
                    if push_in > 0:
                        cxy = cxy + d * push_in
                    if lost > 3:
                        self.log(f"{label}: cube lost")
                        replan = True
                        break
                else:
                    newc = np.asarray(c["center"])[:2]
                    face = e + d * off + n * plan["lat"]
                    pred = cxy + d * max(0.0, float((face - cxy) @ d))
                    if np.linalg.norm(newc - pred) > 0.03 and jumps < 1:
                        jumps += 1
                        self.log(f"  {label}: rejecting jump {np.round(newc, 3).tolist()} vs predicted {np.round(pred, 3).tolist()}")
                        cxy = pred
                    else:
                        jumps = 0
                        lost = 0
                        cxy = newc
                r = t - cxy
                ra, rl = float(r @ d), float(r @ n)
                rel = cxy - e
                gap = float(rel @ d) - off
                latoff = float(rel @ n) - plan["lat"]
                self.log(f"  {label} k={k} cube {np.round(cxy, 3).tolist()} ra={ra:.3f} rl={rl:.3f} gap={gap:.3f} latoff={latoff:.3f}")
                if math.hypot(ra, rl) < PUSH_TOL * 0.8 or (ra < 0.004 and abs(rl) < 0.012):
                    break
                contact = gap < 0.006 and k >= 1
                if ra < -0.004:
                    self.log(f"{label}: overshoot")
                    replan = True
                    break
                if strict and contact and abs(rl) > max(0.015, 0.5 * abs(ra)) and math.hypot(ra, rl) > PUSH_TOL:
                    self.log(f"{label}: lateral drift {rl:.3f}")
                    replan = True
                    break
                if strict and contact and abs(latoff) > self.s * 0.45:
                    self.log(f"{label}: contact slipping latoff {latoff:.3f}")
                    replan = True
                    break
                adv = max(0.0, gap) + min(ra, CHUNK)
                adv = min(adv, CHUNK + max(0.0, gap))
                lat_corr = float(np.clip(0.5 * latoff, -0.006, 0.006))
                p_before = R.pos.copy()
                R.linear(np.array([d[0] * adv + n[0] * lat_corr, d[1] * adv + n[1] * lat_corr,
                                   plan["z_push"] - R.pos[2]]), grip, speed=PUSH_SPEED, yaw=plan["yaw"])
                moved = float((R.pos - p_before)[:2] @ d)
                if adv > 0.008 and moved < 0.3 * adv:
                    self.log(f"{label}: effector blocked (moved {moved:.3f} of {adv:.3f})")
                    replan = True
                    break
            # retreat
            back = -d * 0.03
            try:
                R.linear(np.array([back[0], back[1], 0.0]), grip, speed=0.012, yaw=plan["yaw"])
            except Exception:
                raise
            self.up(plan["z_transit"], grip)
            c = self.relocate(near=cxy, excl_zmin=None)
            if c is not None:
                err = float(np.linalg.norm(np.asarray(c["center"])[:2] - t))
                self.log(f"{label}: after attempt {att} error {err:.4f}")
                if err < PUSH_TOL * 1.5 and not replan:
                    return True
                if err < PUSH_TOL:
                    return True
                if prev_err is not None and abs(prev_err - err) < 0.003:
                    strict = not strict
                    self.log(f"{label}: no progress; strict checks -> {strict}")
                prev_err = err
        return False

    def push_gripper(self):
        def eff(d, cxy):
            yaw = nearest_equiv(math.atan2(d[1], d[0]), self.R.yaw(), math.pi)
            return {"yaw": yaw, "along": FINGER_HALF, "lat": 0.0,
                    "z_push": self.tz + float(np.clip(0.45 * self.s, 0.013, 0.025)),
                    "z_transit": self.tz + self.s + 0.05, "grip": 1.0}
        self.R.gripper(1.0, 8)
        return self.push_loop(eff, "push")

    # --------------------------------------------------------------- tool
    def grasp_candidates(self, tool):
        pts = tool["pts"]
        xy = pts[:, :2]
        h = pts[:, 2] - self.tz
        ax = tool["axis"]
        perp = np.array([-ax[1], ax[0]])
        c = tool["center"]
        pa = (xy - c) @ ax
        pp = (xy - c) @ perp
        lo, hi = np.percentile(pa, 2), np.percentile(pa, 98)
        out = []
        for f in (0.12, 0.22, 0.35, 0.5, 0.65, 0.78, 0.88):
            sa = lo + f * (hi - lo)
            sec = np.abs(pa - sa) < 0.008
            if sec.sum() < 3:
                continue
            width = float(np.percentile(pp[sec], 95) - np.percentile(pp[sec], 5))
            if width > 0.055:
                continue
            ctr = c + ax * sa + perp * float(np.median(pp[sec]))
            loc = np.abs(pa - sa) < 0.03
            lx = xy[loc] - xy[loc].mean(0)
            if len(lx) >= 3:
                w, V = np.linalg.eigh(np.cov(lx.T) + 1e-12 * np.eye(2))
                lax = V[:, 1]
            else:
                lax = ax
            htop = float(np.percentile(h[sec], 90))
            out.append({"f": f, "xy": ctr, "axis": lax, "width": width, "htop": htop})
        return out

    def plan_tool_push(self, T_rel, d, yaw_g, yaw_now, s, max_dpsi_deg=120):
        """T_rel: Nx2 tool points relative to TCP at gripper yaw yaw_g. Choose rotation and lateral offset."""
        n = np.array([-d[1], d[0]])
        best = None
        for dpsi_deg in range(-max_dpsi_deg, max_dpsi_deg + 1, 10):
            dpsi = math.radians(dpsi_deg)
            Rm = geom.Rz(dpsi)[:2, :2]
            Tp = T_rel @ Rm.T
            al = Tp @ d
            la = Tp @ n
            for l0 in np.arange(np.min(la) + s / 2 - 0.01, np.max(la) - s / 2 + 0.0101, 0.005):
                band = np.abs(la - l0) < s / 2 + 0.008
                if band.sum() < 3:
                    continue
                front = float(np.max(al[band]))
                face = band & (al > front - 0.008) & (np.abs(la - l0) < s / 2)
                bins = np.unique(np.floor((la[face] - l0 + s / 2) / 0.004))
                cov = len(bins) / max(1, int(s / 0.004))
                if cov < 0.3:
                    continue
                # fingers (around TCP) must stay clear of the cube's swept band
                if abs(l0) < s / 2 + 0.035 and front < 0.04:
                    continue
                score = 2 * cov - 0.5 * abs(dpsi) / math.pi - 2.0 * max(0.0, abs(front) - 0.18) \
                    - 0.3 * abs(float(np.mean(la[face])) - l0) / s
                if best is None or score > best["score"]:
                    best = {"score": score, "dpsi": dpsi, "lat": float(l0), "along": front, "cov": cov}
        return best

    def tool_use(self, attempts=3):
        R, S = self.R, self.S
        tool = S.tool
        if tool is None:
            self.log("tool: no tool detected")
            return False
        cxy = self.cube_xy()
        t = self.target_xy()
        d = unit(t - cxy)
        cands = self.grasp_candidates(tool)
        self.log("tool: grasp candidates", [(c["f"], np.round(c["xy"], 3).tolist(), round(c["width"], 3), round(c["htop"], 3)) for c in cands])
        scored = []
        T = tool["pts"][:, :2]
        for c in cands:
            if np.linalg.norm(c["xy"] - cxy) < self.s / 2 + 0.05:
                continue
            ax = c["axis"]
            yaw_g = nearest_equiv(math.atan2(ax[1], ax[0]) + math.pi / 2, R.yaw(), math.pi)
            plan = self.plan_tool_push(T - c["xy"], d, yaw_g, R.yaw(), self.s)
            if plan is None:
                continue
            sc = plan["score"] - 1.0 * abs(c["f"] - 0.5)
            scored.append((sc, c, yaw_g, plan))
        if not scored:
            self.log("tool: no feasible grasp/plan")
            return False
        scored.sort(key=lambda x: -x[0])
        for sc, c, yaw_g, plan in scored[:3]:
            self.log(f"tool: try grasp f={c['f']} at {np.round(c['xy'], 3).tolist()} yaw {math.degrees(yaw_g):.0f} plan {plan}")
            yaws = (yaw_g,) if getattr(R, "closing_confirmed", False) else (yaw_g, yaw_g + math.pi / 2)
            for yaw_try in yaws:
                if R.steps_left() < 60:
                    return False
                ok = self.grasp_at(c, yaw_try)
                if ok:
                    break
            if not ok:
                continue
            g_pos = R.pos.copy()
            yaw_hold = R.yaw()
            T_rel = T - g_pos[:2]
            grasp_z = g_pos[2]
            z_low = grasp_z + 0.004
            z_tr = grasp_z + self.s + 0.035

            def eff(dd, cxy_now, T_rel=T_rel, yaw_hold=yaw_hold, fixed=False):
                yaw_cur = R.yaw()
                # current tool points relative to TCP: rotate by yaw change since grasp
                Tc = T_rel @ geom.Rz(yaw_cur - yaw_hold)[:2, :2].T
                p = self.plan_tool_push(Tc, dd, yaw_cur, yaw_cur, self.s, max_dpsi_deg=0 if fixed else 120)
                if p is None:
                    return None
                return {"yaw": yaw_cur + p["dpsi"], "along": p["along"], "lat": p["lat"],
                        "z_push": z_low, "z_transit": z_tr, "grip": 1.0,
                        "refine": None if fixed else (lambda d2, c2: eff(d2, c2, fixed=True))}

            self.up(z_tr, 1.0)
            if R.grip_width < GRASP_MIN_W * 0.6:
                self.log("tool: lost after lift")
                R.gripper(-1, 4)
                continue
            res = self.push_loop(eff, "toolpush")
            return res
        return False

    def grasp_at(self, c, yaw):
        R = self.R
        gxy = c["xy"]
        gz = self.tz + float(np.clip(0.5 * c["htop"], 0.009, 0.03))
        hz = self.tz + max(self.s, c["htop"]) + LIFT_CLEAR
        R.gripper(-1.0, 3)
        self.up(hz, -1.0)
        R.move_to([gxy[0], gxy[1], hz], yaw, -1.0, tol=0.006, label="above tool")
        R.move_to([gxy[0], gxy[1], gz + 0.03], yaw, -1.0, tol=0.005, label="pre-grasp tool")
        R.move_to([gxy[0], gxy[1], gz], yaw, -1.0, tol=0.004, vmax=0.01, label="grasp tool")
        w = R.gripper(1.0, 12)
        self.log(f"tool: closed width {w:.4f} at yaw {math.degrees(yaw):.0f}")
        if w < GRASP_MIN_W * 0.6:
            R.gripper(-1.0, 6)
            self.up(hz, -1.0)
            return False
        return True

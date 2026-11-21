"""Robot I/O wrapper: observe/step/pixel_to_position with defensive parsing and budget tracking."""
import time
import numpy as np
from mcp_util import (log, elapsed, parse_reply, find_key, find_all_keys, get_state, episode_ended,
                      task_complete, decode_image_block, read_manifest, frame_files, load_rgb, load_depth)
from geometry import parse_calibration, axang_to_R, R_to_axang


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, step_limit=None):
        self.ctx = ctx
        self.state = None
        self.step_id = None
        self.steps_used = 0
        self.step_limit = step_limit
        self.ended = False
        self.complete = False
        self.last_meta = {}
        self.cal = {}
        self.block_order = None   # e.g. ['front','wrist'] mapping for RGB image blocks
        self.block_idx = None
        self.last_progress = time.time()

    # ------------------------------------------------------------------ parsing
    def _ingest(self, meta):
        self.last_meta = meta
        s = get_state(meta)
        if s is not None:
            self.state = s
        sid = find_key(meta, 'step_id')
        if isinstance(sid, (int, float)):
            self.step_id = int(sid)
        if episode_ended(meta):
            self.ended = True
        if task_complete(meta):
            self.complete = True
        cal = parse_calibration(meta, find_key)
        if cal:
            self.cal = cal

    def observe(self):
        res = self.ctx.call('robomme_observe')
        meta, texts, imgs = parse_reply(res)
        self._ingest(meta)
        return meta, imgs

    def status(self):
        try:
            res = self.ctx.call('robomme_status')
            meta, _, _ = parse_reply(res)
            for k in ('step_limit', 'max_steps', 'max_episode_steps', 'episode_step_limit', 'limit'):
                v = find_key(meta, k)
                if isinstance(v, (int, float)) and v > 0:
                    self.step_limit = int(v)
                    break
            for k in ('steps_remaining', 'remaining_steps'):
                v = find_key(meta, k)
                if isinstance(v, (int, float)):
                    self.steps_remaining_reported = int(v)
            if episode_ended(meta):
                self.ended = True
            return meta
        except Exception as e:
            log('status failed', repr(e))
            return {}

    def remaining(self):
        if self.step_limit is None:
            return 10 ** 6
        return self.step_limit - self.steps_used

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        if elapsed() > 1080:
            log('wall-clock guard reached; stopping')
            raise EpisodeOver()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if not rows:
            return {}
        if self.step_limit is not None and self.remaining() - len(rows) < 2:
            rows = rows[:max(0, self.remaining() - 2)]
            if not rows:
                raise EpisodeOver()
        res = self.ctx.call('robomme_step', actions=rows)
        self.steps_used += len(rows)
        self.last_progress = time.time()
        meta, _, imgs = parse_reply(res)
        self._ingest(meta)
        if get_state(meta) is None and not self.ended:
            try:
                self.observe()
            except Exception as e:
                log('observe after step failed', repr(e))
        if isinstance(self.step_id, int) and self.step_id > self.steps_used:
            self.steps_used = self.step_id
        # authoritative step counter if present
        for k in ('steps_used', 'episode_step', 'elapsed_steps'):
            v = find_key(meta, k)
            if isinstance(v, (int, float)) and v >= self.steps_used:
                self.steps_used = int(v)
                break
        if self.ended:
            log('episode ended; complete=%s' % self.complete)
            raise EpisodeOver()
        return meta

    # ------------------------------------------------------------------ images
    def images_from_reply(self, meta, imgs):
        """Return {'front': rgb, 'wrist': rgb, 'front_depth':..., 'wrist_depth':...} best effort."""
        out = {}
        man, _ = read_manifest(meta)
        if man and man.get('frames'):
            frames = sorted(man['frames'], key=lambda f: f.get('frame_index', 0))
            fr = frames[-1]
            ff = frame_files(fr) if (fr.get('step_id') is None or self.step_id is None
                                     or int(fr.get('step_id')) == int(self.step_id)) else {}
            for cam in ('front', 'wrist'):
                try:
                    if (cam, 'rgb') in ff:
                        out[cam] = load_rgb(ff[(cam, 'rgb')][0])
                    if (cam, 'depth') in ff:
                        p, mpu = ff[(cam, 'depth')]
                        out[cam + '_depth'] = load_depth(p, mpu)
                except Exception as e:
                    log('frame load fail', cam, repr(e))
            if 'front' in out and 'wrist' in out:
                return out
        rgbs, depths = [], []
        for b in imgs:
            im = decode_image_block(b)
            if im is None:
                continue
            if im.mode in ('I;16', 'I', 'I;16B', 'F'):
                depths.append(np.array(im).astype(np.float32) * 0.001)
            else:
                rgbs.append(np.array(im.convert('RGB')))
        if self.block_idx:
            for cam, i in self.block_idx.items():
                if i < len(rgbs):
                    out.setdefault(cam, rgbs[i])
        else:
            order = self.block_order or ['front', 'wrist']
            for cam, im in zip(order, rgbs):
                out.setdefault(cam, im)
        out['_rgb_blocks'] = rgbs
        for cam, d in zip(self.block_order or ['front', 'wrist'], depths):
            out.setdefault(cam + '_depth', d)
        return out

    # ------------------------------------------------------------------ geometry
    def pixel_to_world(self, sels, radius=1):
        """sels: list of (camera, u, v). Returns list of np.array(3) or None per selection."""
        out = []
        for i in range(0, len(sels), 32):
            chunk = sels[i:i + 32]
            try:
                res = self.ctx.call('robomme_pixel_to_position', observation_step_id=int(self.step_id or 0),
                                    selections=[{'camera': c, 'u': float(np.clip(u, 0, 255)), 'v': float(np.clip(v, 0, 255))} for c, u, v in chunk],
                                    depth_window_radius=int(radius))
                meta, texts, _ = parse_reply(res)
                pts = _parse_points(meta, len(chunk))
            except Exception as e:
                log('pixel_to_position failed', repr(e)[:300])
                pts = [None] * len(chunk)
            out.extend(pts)
        return out


def _vec3(v):
    if isinstance(v, dict):
        if all(k in v for k in ('x', 'y', 'z')):
            try:
                return np.array([float(v['x']), float(v['y']), float(v['z'])])
            except Exception:
                return None
        return None
    if isinstance(v, (list, tuple)) and len(v) == 3:
        try:
            return np.array([float(a) for a in v])
        except Exception:
            return None
    return None


POINT_KEYS = ('world_position', 'position_world', 'world_point', 'position', 'point', 'world', 'xyz', 'world_xyz', 'position_m')


def _parse_points(meta, n):
    entries = None
    for k in ('results', 'points', 'selections', 'positions', 'items', 'data'):
        v = find_key(meta, k)
        if isinstance(v, list) and len(v) == n:
            entries = v
            break
    if entries is None:
        # collect dicts having a point key
        found = []

        def walk(o):
            if isinstance(o, dict):
                if any(k in o for k in POINT_KEYS):
                    found.append(o)
                    return
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(meta)
        entries = found if len(found) == n else None
    if entries is None:
        return [None] * n
    out = []
    for e in entries:
        p = None
        if isinstance(e, dict):
            if e.get('valid') is False:
                out.append(None)
                continue
            for k in POINT_KEYS:
                if k in e:
                    p = _vec3(e[k])
                    if p is not None:
                        break
        else:
            p = _vec3(e)
        if p is not None and not np.all(np.isfinite(p)):
            p = None
        out.append(p)
    return out

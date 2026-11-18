"""Robot wrapper around ctx.call with defensive parsing and step accounting."""
import time
import traceback

import numpy as np

from obsio import (apply_tf, Reply, log, find_key, find_dicts, files_of, load_manifest, manifest_paths,
                   frame_order_key, get_state, get_step_id, episode_ended, read_rgb, read_depth,
                   decode_image_block, image_similarity, is_vec, abspath)


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = 0
        self.ended = False
        self.end_key = None
        self.max_steps = None
        self.n_calls = 0
        self.ref_imgs = {}       # (cam, kind) -> array, labelled reference images
        self.last_reply = None
        self.t_last_step = time.time()
        self.success_seen = False
        self.last_step_reply = None
        self.frame_tf = {}      # camera -> dihedral transform for file-loaded frames
        self.last_step_reply_id = None

    # ------------------------------------------------------------------ calls
    def call(self, tool, **args):
        self.n_calls += 1
        raw = self.ctx.call(tool, **args)
        r = Reply(raw)
        self.last_reply = r
        for m in r.all():
            st = get_state(m)
            if st is not None:
                self.state = st
            sid = get_step_id(m)
            if sid is not None and tool != 'robomme_pixel_to_position':
                self.step_id = sid
            for key in ('episode_ended', 'terminated', 'truncated', 'task_complete'):
                if any(v is True for v in find_key(m, key)):
                    self.ended = True
                    self.end_key = key
                    if key == 'task_complete':
                        self.success_seen = True
        return r

    def status(self):
        try:
            r = self.call('robomme_status')
            meta = r.meta
            self._parse_limits(meta)
            return r
        except Exception as e:
            log('robomme_status failed', repr(e))
            return None

    def _parse_limits(self, meta):
        cands = []
        for path_key in ('episode_limits', 'limits', 'step_budget'):
            for v in find_key(meta, path_key):
                if isinstance(v, dict):
                    for k, x in v.items():
                        if isinstance(x, (int, float)) and not isinstance(x, bool) and 'step' in k.lower():
                            cands.append((k, int(x)))
                elif isinstance(v, (int, float)):
                    cands.append((path_key, int(v)))
        for k in ('max_steps', 'max_episode_steps', 'step_limit', 'horizon'):
            for v in find_key(meta, k):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cands.append((k, int(v)))
        good = [c for c in cands if 50 <= c[1] <= 100000 and ('max' in c[0].lower() or 'limit' in c[0].lower() or 'horizon' in c[0].lower())]
        if good:
            self.max_steps = max(x for _, x in good)
        elif cands:
            vals = [x for _, x in cands if 50 <= x <= 100000]
            if vals:
                self.max_steps = max(vals)
        log('limits candidates', cands, '-> max_steps', self.max_steps)

    def observe(self):
        return self.call('robomme_observe')

    def remaining(self):
        if self.max_steps is None:
            return 10 ** 6
        return self.max_steps - self.step_id

    def step(self, rows):
        if self.ended:
            raise EpisodeOver(self.end_key)
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if not rows:
            return None
        try:
            r = self.call('robomme_step', actions=rows)
        except Exception as e:
            log('robomme_step error', repr(e)[:500])
            # do not retry blindly: refresh state
            try:
                self.call('robomme_status')
                self.observe()
            except Exception as e2:
                log('refresh after step error failed', repr(e2)[:300])
            if self.ended:
                raise EpisodeOver(self.end_key)
            return None
        self.t_last_step = time.time()
        self.last_step_reply = r
        self.last_step_reply_id = self.step_id
        if self.ended:
            raise EpisodeOver(self.end_key)
        return r

    def pix2pos(self, step_id, selections, radius=0):
        r = self.call('robomme_pixel_to_position', observation_step_id=int(step_id),
                      selections=selections, depth_window_radius=int(radius))
        return r

    # --------------------------------------------------------- image access
    def frame_images(self, frame):
        """Load rgb/depth arrays of a manifest frame -> dict (cam, kind) -> array."""
        out = {}
        for (cam, kind), (p, mpu) in files_of(frame).items():
            try:
                if kind == 'rgb':
                    out[(cam, kind)] = apply_tf(read_rgb(p), self.frame_tf.get(cam))
                elif kind == 'depth':
                    out[(cam, kind)] = apply_tf(read_depth(p, mpu), self.frame_tf.get(cam))
            except Exception as e:
                log('read failed', p, repr(e)[:200])
        return out

    def current_images(self, reply, want_step=None):
        """Best-effort current rgb/depth for front & wrist from an observe/step reply."""
        imgs = {}
        src = {}
        # (1) labelled file dicts directly in the reply JSON (not manifests)
        for m in reply.all():
            fds = find_dicts(m, lambda d: 'camera' in d and 'path' in d)
            if fds:
                fr = {'files': fds}
                got = self.frame_images(fr)
                for k, v in got.items():
                    imgs[k] = v; src[k] = 'json-path'
        # (2) inline blocks
        blocks = [decode_image_block(b) for b in reply.images]
        labels = self._block_labels(reply, len(blocks))
        for i, (kind, arr) in enumerate(blocks):
            if arr is None:
                continue
            lab = labels[i] if labels and i < len(labels) else None
            if lab is None:
                continue
            k = (lab[0], kind if lab[1] is None else lab[1])
            if k[1] == 'depth' and kind == 'depth':
                mpu = self._mpu(reply)
                arr = arr * mpu
            if k[1] != kind:
                continue
            if k not in imgs:
                imgs[k] = arr; src[k] = 'inline'
        # unlabeled inline: match by similarity with references
        unl = [(kind, arr) for (kind, arr), l in zip(blocks, labels or [None] * len(blocks)) if arr is not None and l is None]
        if unl:
            rgbs = [a for k, a in unl if k == 'rgb']
            deps = [a * self._mpu(reply) for k, a in unl if k == 'depth']
            self._assign_by_similarity(imgs, src, rgbs, 'rgb')
            self._assign_by_similarity(imgs, src, deps, 'depth')
        # (3) manifest last frame
        need = [k for k in (('front', 'rgb'), ('front', 'depth'), ('wrist', 'rgb'), ('wrist', 'depth')) if k not in imgs]
        if need:
            for mp in manifest_paths(reply):
                man = load_manifest(mp)
                if not man:
                    continue
                frames = man.get('frames') or []
                if not frames:
                    continue
                frames = sorted(frames, key=frame_order_key)
                last = frames[-1]
                sid = last.get('step_id')
                if want_step is not None and isinstance(sid, int) and sid != want_step:
                    continue
                got = self.frame_images(last)
                for k in need:
                    if k in got and k not in imgs:
                        imgs[k] = got[k]; src[k] = 'manifest'
        return imgs, src

    def _mpu(self, reply):
        for m in reply.all():
            for v in find_key(m, 'meters_per_unit'):
                if isinstance(v, (int, float)) and 0 < v < 1:
                    return float(v)
            for v in find_key(m, 'depth_scale'):
                if isinstance(v, (int, float)) and 0 < v < 1:
                    return float(v)
        return 0.001

    def _block_labels(self, reply, n):
        if n == 0:
            return []
        # look for a list of descriptors with 'camera' whose length matches the block count
        for m in reply.all():
            lists = []
            for _, k, v in _walk_lists(m):
                if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) and 'camera' in x for x in v):
                    lists.append(v)
            for L in lists:
                labs = []
                for x in L:
                    cam = str(x.get('camera', '')).lower()
                    cam = 'wrist' if ('wrist' in cam or 'hand' in cam) else 'front'
                    kind = str(x.get('kind', x.get('type', ''))).lower()
                    kind = 'depth' if 'depth' in kind else ('rgb' if kind else None)
                    labs.append((cam, kind))
                return labs
        # text blocks between image blocks naming the camera
        labs = []
        content = reply.raw.get('content', []) or []
        last_txt = ''
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get('type') == 'text':
                last_txt = b.get('text', '')[-200:].lower()
            elif b.get('type') == 'image':
                cam = None
                if 'wrist' in last_txt:
                    cam = 'wrist'
                elif 'front' in last_txt:
                    cam = 'front'
                labs.append((cam, None) if cam else None)
        if labs and all(l is not None for l in labs) and len(set(l[0] for l in labs)) > 1:
            return labs
        return [None] * n

    def _assign_by_similarity(self, imgs, src, arrs, kind):
        """Unlabelled inline images: the front camera is static, so the image most similar to
        the latest front image is the front one; the other is the wrist."""
        cams = [c for c in ('front', 'wrist') if (c, kind) not in imgs]
        if not arrs or not cams:
            return
        if len(arrs) == 1 and len(cams) == 1:
            imgs[(cams[0], kind)] = arrs[0]; src[(cams[0], kind)] = 'inline-sim'
            return
        ref = self.ref_imgs.get(('front', kind))
        if len(arrs) >= 2 and ref is not None:
            a, b = arrs[0], arrs[1]
            sa, sb = image_similarity(a, ref), image_similarity(b, ref)
            f, w = (a, b) if sa >= sb else (b, a)
            imgs[('front', kind)], imgs[('wrist', kind)] = f, w
            src[('front', kind)] = src[('wrist', kind)] = 'inline-sim'
            self.ref_imgs[('front', kind)] = f
            self.sim_order = 'ab' if sa >= sb else 'ba'
        elif len(arrs) >= 2:
            imgs[('front', kind)], imgs[('wrist', kind)] = arrs[0], arrs[1]
            src[('front', kind)] = src[('wrist', kind)] = 'inline-order'


def _walk_lists(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from _walk_lists(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_lists(v, path + (i,))

import base64, io, json, time
import numpy as np
from PIL import Image
from geom import backproject

LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n"); LOG.flush()


def parse(reply):
    blocks = reply["content"]
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.asarray(im.convert("RGB"))
        else:
            imgs[(att["camera"], "depth")] = np.asarray(im).astype(np.float32) * 0.001
    return meta, imgs


class Obs:
    def __init__(self, meta, imgs):
        self.meta = meta
        self.step = meta.get("step_id")
        self.state = np.array(meta["state"])
        self.cams = {}
        for cam in ("front", "wrist"):
            if (cam, "rgb") in imgs and (cam, "depth") in imgs:
                K = np.array(meta["camera_intrinsics"][cam]["matrix"])
                pose = meta["camera_poses"][cam]
                dep = imgs[(cam, "depth")]
                P = backproject(dep, K, pose)
                P[dep <= 0] = np.nan
                self.cams[cam] = dict(rgb=imgs[(cam, "rgb")], depth=dep, P=P)
        self.ended = bool(meta.get("episode_ended", False))
        tr = meta.get("transition") or {}
        self.task_complete = bool(tr.get("task_complete")) or any(s.get("task_complete") for s in tr.get("steps", []))
        lim = meta.get("episode_limits") or {}
        self.remaining = lim.get("remaining_episode_steps")


def observe(ctx):
    return Obs(*parse(ctx.call("robomme_observe")))


def step(ctx, rows):
    rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
    return Obs(*parse(ctx.call("robomme_step", actions=rows)))

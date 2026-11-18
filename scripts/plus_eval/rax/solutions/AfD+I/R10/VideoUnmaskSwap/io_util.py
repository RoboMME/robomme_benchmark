import base64, io, json
import numpy as np
from PIL import Image

def parse(result):
    """MCP result -> (meta dict, {(camera,kind): ndarray})"""
    content = result["content"] if isinstance(result, dict) else result
    meta = None
    for b in content:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                continue
    imgs = {}
    if meta is not None:
        for a in meta.get("attachments", []) or []:
            idx = a.get("content_index")
            if idx is None or idx >= len(content):
                continue
            blk = content[idx]
            data = blk.get("data")
            if data is None:
                continue
            im = Image.open(io.BytesIO(base64.b64decode(data)))
            imgs[(a["camera"], a["kind"])] = np.asarray(im)
    return meta, imgs

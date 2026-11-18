import base64, io, json
import numpy as np
from PIL import Image

def decode(result):
    blocks = result.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        arr = np.array(im)
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs

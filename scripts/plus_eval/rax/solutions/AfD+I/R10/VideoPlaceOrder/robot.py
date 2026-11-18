import json, base64, io, numpy as np
from pathlib import Path
from PIL import Image

WS=Path('/workspace')

def parse(reply):
    """reply: MCP result (dict with content) or list of content blocks."""
    content = reply.get('content', reply) if isinstance(reply, dict) else reply
    meta = json.loads(content[0]['text'])
    imgs = {}
    for att in meta.get('attachments', []) or []:
        blk = content[att['content_index']]
        data = base64.b64decode(blk['data'])
        im = Image.open(io.BytesIO(data))
        if att['kind'] == 'rgb': imgs[(att['camera'], 'rgb')] = np.array(im.convert('RGB'))
        else: imgs[(att['camera'], 'depth')] = np.array(im).astype(np.float32) * 0.001
    return meta, imgs

def quat_rot(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def deproject(meta, depth, cam, u, v, r=1):
    K=np.array(meta['camera_intrinsics'][cam]['matrix']); fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    ui,vi=int(round(u)),int(round(v))
    win=depth[max(0,vi-r):vi+r+1, max(0,ui-r):ui+r+1]
    win=win[win>0]
    if len(win)==0: return None
    d=float(np.median(win))
    pc=np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    pose=meta['camera_poses'][cam]
    return quat_rot(pose['quaternion_wxyz'])@pc + np.array(pose['position_m'])

def deproject_mask(meta, depth, cam, mask):
    """median world point of mask pixels"""
    ys,xs=np.nonzero(mask)
    pts=[]
    K=np.array(meta['camera_intrinsics'][cam]['matrix']); fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    pose=meta['camera_poses'][cam]; R=quat_rot(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    d=depth[ys,xs]; ok=d>0
    xs,ys,d=xs[ok],ys[ok],d[ok]
    if len(d)==0: return None
    pc=np.stack([(xs-cx)*d/fx, -(ys-cy)*d/fy, -d],1)
    pw=pc@R.T+t
    return pw

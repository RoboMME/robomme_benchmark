import json, time, numpy as np
from PIL import Image
import io, base64

def parse(reply):
    meta=None; imgs=[]
    for b in reply.get('content',[]):
        if b.get('type')=='text' and meta is None:
            meta=json.loads(b['text'])
        elif b.get('type')=='image':
            imgs.append(b)
    return meta, reply.get('content',[])

def decode_obs(reply):
    """Return meta, dict camera-> {'rgb':arr,'depth':arr(m)}"""
    content=reply.get('content',[])
    meta=json.loads(next(b['text'] for b in content if b.get('type')=='text'))
    out={}
    for a in meta.get('attachments',[]):
        b=content[a['content_index']]
        if b.get('type')!='image': continue
        im=Image.open(io.BytesIO(base64.b64decode(b['data'])))
        arr=np.array(im)
        d=out.setdefault(a['camera'],{})
        if a['kind']=='rgb': d['rgb']=np.array(im.convert('RGB'))
        else: d['depth']=arr.astype(np.float32)*0.001
    return meta,out

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx=ctx; self.log=log; self.meta=None; self.cams=None
        self.ended=False; self.complete=False; self.ncalls=0
    def observe(self):
        r=self.ctx.call('robomme_observe'); self.meta,self.cams=decode_obs(r); return self.meta
    def step(self, rows):
        rows=[[float(np.clip(x,-1,1)) for x in r] for r in rows]
        r=self.ctx.call('robomme_step', actions=rows); self.ncalls+=1
        self.meta,self.cams=decode_obs(r)
        tr=self.meta.get('transition',{})
        for s in tr.get('steps',[]):
            if s.get('task_complete'): self.complete=True
        if tr.get('task_complete') or self.meta.get('task_complete'): self.complete=True
        if self.meta.get('episode_ended'): self.ended=True
        return self.meta
    @property
    def state(self): return np.array(self.meta['state'])
    @property
    def pos(self): return self.state[:3]
    @property
    def step_id(self): return self.meta['step_id']
    def remaining(self): return self.meta['episode_limits']['remaining_episode_steps']

    def move_to(self, target, grip=-1.0, tol=0.004, max_steps=80, vmax=0.8, chunk=4, rot_hold=True):
        """P-control on position; rot correction toward down orientation (axis-angle [pi,0,0])."""
        target=np.array(target,float); n=0
        while n<max_steps and not self.ended:
            err=target-self.pos
            if np.linalg.norm(err)<tol: break
            rows=[]
            e=err.copy()
            k=min(chunk, max_steps-n)
            for i in range(k):
                d=e/0.05
                m=np.max(np.abs(d))
                if m>vmax: d=d*vmax/m
                rows.append(list(d)+list(self.rot_corr() if (rot_hold and i==0) else [0,0,0])+[grip]); e=e-d*0.05
                if np.linalg.norm(e)<tol*0.5: break
            self.step(rows); n+=len(rows)
        return np.linalg.norm(target-self.pos)
    def rot_corr(self, gain=1.0):
        from scipy.spatial.transform import Rotation as Rot
        cur=Rot.from_rotvec(self.state[3:6]); tgt=Rot.from_rotvec(self.rot_target)
        e=(tgt*cur.inv()).as_rotvec()
        return np.clip(e*gain/0.5,-0.5,0.5)
    rot_target=np.array([np.pi,0,0])
    def hold(self, n, grip):
        rows=[[0,0,0,0,0,0,grip]]*n
        return self.step(rows)

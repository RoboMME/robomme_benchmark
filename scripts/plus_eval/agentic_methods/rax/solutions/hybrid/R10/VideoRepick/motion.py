import json, numpy as np

def meta(reply):
    if isinstance(reply,dict): reply=reply.get('content',[])
    for b in reply:
        if isinstance(b,dict) and b.get('type')=='text':
            try: return json.loads(b['text'])
            except Exception: pass
        t=getattr(b,'text',None)
        if t:
            try: return json.loads(t)
            except Exception: pass
    raise RuntimeError('no json in reply')

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx=ctx; self.log=log; self.ended=False; self.complete=False; self.last=None
        self.state=None; self.remaining=None
    def observe(self):
        m=meta(self.ctx.call('robomme_observe')); self._upd(m); return m
    def _upd(self,m):
        self.last=m
        if 'state' in m: self.state=np.array(m['state'])
        lim=m.get('episode_limits') or {}
        if 'remaining_episode_steps' in lim: self.remaining=lim['remaining_episode_steps']
        if m.get('episode_ended'): self.ended=True
        for s in m.get('steps',[]) or []:
            if isinstance(s,dict):
                if s.get('task_complete'): self.complete=True
                if s.get('episode_ended'): self.ended=True
        if m.get('task_complete'): self.complete=True
        tr=m.get('transition') or {}
        if tr.get('episode_ended'): self.ended=True
        if tr.get('task_complete'): self.complete=True
        for s in tr.get('steps',[]) or []:
            if isinstance(s,dict):
                if s.get('task_complete'): self.complete=True
                if s.get('episode_ended'): self.ended=True
    def step(self, rows):
        if self.ended: return None
        rows=[[float(np.clip(v,-1,1)) for v in r] for r in rows]
        m=meta(self.ctx.call('robomme_step',actions=rows)); self._upd(m); return m
    def move_to(self, target, grip, tol=0.004, max_steps=60, vmax=0.6, chunk=4):
        target=np.array(target,float); n=0
        while n<max_steps and not self.ended:
            cur=self.state[:3]; err=target-cur
            if np.linalg.norm(err)<tol: return True
            rows=[]
            # plan chunk of steps assuming partial tracking
            k=min(chunk,max_steps-n)
            for j in range(k):
                d=err/0.05
                mx=np.abs(d).max()
                if mx>vmax: d=d*vmax/mx
                rows.append([d[0],d[1],d[2],0,0,0,grip])
                err=err-d*0.05
                if np.linalg.norm(err)<tol*0.5: break
            self.step(rows); n+=len(rows)
        return np.linalg.norm(target-self.state[:3])<tol*2
    def hold(self, grip, n):
        return self.step([[0,0,0,0,0,0,grip]]*n)

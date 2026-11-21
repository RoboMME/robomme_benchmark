import json, numpy as np
from robot import parse

class Robot:
    def __init__(self, ctx, log=print):
        self.ctx=ctx; self.log=log; self.meta=None; self.imgs=None; self.ended=False; self.complete=False
    def observe(self):
        self.meta,self.imgs=parse(self.ctx.call('robomme_observe')); return self.meta
    @property
    def state(self): return np.array(self.meta['state'])
    def remaining(self):
        return self.meta.get('episode_limits',{}).get('remaining_episode_steps', 999)
    def step(self, rows):
        if self.ended: return False
        rows=[[float(np.clip(v,-1,1)) for v in r] for r in rows]
        rem=self.remaining()
        if rem<=0: self.ended=True; return False
        rows=rows[:max(1,min(100,rem))]
        reply=self.ctx.call('robomme_step', actions=rows)
        self.meta,self.imgs=parse(reply)
        tr=self.meta.get('transition') or {}
        if tr.get('task_complete') or self.meta.get('task_complete'): self.complete=True
        for s in tr.get('steps',[]) or []:
            if isinstance(s,dict) and s.get('task_complete'): self.complete=True
        if tr.get('episode_ended'): self.ended=True
        if self.meta.get('episode_ended'): self.ended=True
        return not self.ended
    def move_to(self, target, grip, max_step=0.025, tol=0.004, iters=6):
        target=np.array(target,float)
        for it in range(iters):
            pos=self.state[:3]; d=target-pos; dist=np.linalg.norm(d)
            if dist<tol: return True
            n=int(np.ceil(dist/max_step))
            dd=d/n/0.05
            rows=[[dd[0],dd[1],dd[2],0,0,0,grip]]*n
            if not self.step(rows): return False
        err=np.linalg.norm(target-self.state[:3])
        self.log(f'move_to {np.round(target,3)} residual {err:.4f}')
        return err<0.015
    def grip(self, g, n=12):
        pos=self.state[:3]
        return self.step([[0,0,0,0,0,0,g]]*n)

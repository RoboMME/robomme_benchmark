import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    """Constant-velocity tracker with Hungarian association; gate grows with time unseen."""
    def __init__(self, init_xy, t0=0, base_gate=0.025, vmax=0.007):
        self.pos=[np.array(p,float) for p in init_xy]
        self.vel=[np.zeros(2) for _ in init_xy]
        self.last_seen=[t0]*len(init_xy); self.t=t0
        self.base_gate=base_gate; self.vmax=vmax
        self.history=[]
    def update(self, t, dets):
        dt=max(1,t-self.t); n=len(self.pos)
        pred=[self.pos[i]+self.vel[i]*dt for i in range(n)]
        matched=[]
        if len(dets):
            dets=[np.asarray(d,float) for d in dets]
            D=np.array([[np.linalg.norm(pred[i]-d) for d in dets] for i in range(n)])
            gates=np.array([self.base_gate+self.vmax*min(t-self.last_seen[i],60) for i in range(n)])
            C=np.where(D<gates[:,None],D,1e3)
            r,c=linear_sum_assignment(C)
            usedd=set()
            for i,j in zip(r,c):
                if C[i,j]<1e3:
                    self._set(i,dets[j],t,dt); matched.append(i); usedd.add(j)
            # re-associate long-lost tracks with leftover detections (any distance)
            lost=[i for i in range(n) if i not in matched and t-self.last_seen[i]>15]
            left=[j for j in range(len(dets)) if j not in usedd]
            if lost and left:
                D2=np.array([[np.linalg.norm(pred[i]-dets[j]) for j in left] for i in lost])
                r2,c2=linear_sum_assignment(D2)
                for a,b in zip(r2,c2):
                    if D2[a,b]<0.25:
                        i=lost[a]; self.pos[i]=dets[left[b]].copy(); self.vel[i]=np.zeros(2); self.last_seen[i]=t; matched.append(i)
        for i in range(n):
            if i not in matched:
                self.pos[i]=pred[i]
                self.vel[i]=self.vel[i]*(0.9**dt)
        self.t=t
        self.history.append((t,[p.copy() for p in self.pos],list(matched)))
        return matched
    def _set(self,i,d,t,dt):
        gap=t-self.last_seen[i]
        newv=(d-self.pos[i])/max(1,dt) if gap<=dt else self.vel[i]
        sp=np.linalg.norm(newv)
        if sp>self.vmax: newv=newv*self.vmax/sp
        self.vel[i]=0.5*self.vel[i]+0.5*newv
        self.pos[i]=d.copy(); self.last_seen[i]=t

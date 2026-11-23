"""Transfer each memory task's destination and steering dynamics to other tasks.

The shared collision controller adapts these dynamics to the receiving scene.
Source rules: permanence/utils/moving.py, reference/utils/repick_motion.py,
and permanence/utils/routes.py (circulation adapted to receiving workspace).
"""
import numpy as np
from .motion import rng_for

class SourceStyle:
    def __init__(self,name,points,radii,seed,bounds):
        self.name=name;self.rng=rng_for(seed,'transferred-style/'+name)
        self.bounds=np.asarray(bounds);self.count=len(points);self.radii=np.asarray(radii)
        self.goals=np.asarray(points).copy();self.next_goal=0
        self.motions=[None]*self.count
        self.orders=[self.rng.permutation(4) for _ in points];self.cursors=np.zeros(self.count,int)
        self.names=('travel','s_curve','loop','reversal')
        self.events=[]
        # Scale compact goals to the receiving objects, including hollow covers.
        self.diameter=max(.30,2*float(max(radii))*np.sqrt(self.count)*1.65)
        low=self.bounds[:,0]+max(radii)+.015;high=self.bounds[:,1]-max(radii)-.015
        middle=(low+high)/2;half=(high-low)/2
        self.loop=middle+np.array([[-.9,-.8],[-.9,.8],[.25,.8],[.85,0],[.25,-.8]])*half
        self.loop_index=np.array([np.linalg.norm(self.loop-p,axis=1).argmin() for p in points])
        self.roles=set(int(i) for i in self.rng.permutation(self.count)[:max(2,min(3,self.count//2))])
        self.signs=self.rng.choice([-1,1],self.count)

    def step(self,points,tick):
        lo=self.bounds[:,0]+self.radii[:,None];hi=self.bounds[:,1]-self.radii[:,None]
        if tick>=self.next_goal:
            if self.name=='VideoRepick':
                center=self.rng.uniform(self.bounds[:,0]+max(self.radii),self.bounds[:,1]-max(self.radii))
                angle=self.rng.uniform(-np.pi,np.pi,self.count)
                distance=(self.diameter/2-self.radii)*np.sqrt(self.rng.uniform(size=self.count))
                self.goals=np.clip(center+distance[:,None]*np.c_[np.cos(angle),np.sin(angle)],lo,hi)
                self.goals=self.goals[self.rng.permutation(self.count)]
                self.next_goal=tick+int(self.rng.integers(20,51))
            else:
                self.goals=self.rng.uniform(lo,hi)
                self.next_goal=tick+80
            self.events.append(dict(step=tick,source_task=self.name,destinations=self.goals.round(4).tolist()))
        direction=self.goals-points
        if self.name=='ButtonUnmaskSwap':
            for i in self.roles:
                if np.linalg.norm(points[i]-self.loop[self.loop_index[i]])<.035:
                    self.loop_index[i]=(self.loop_index[i]+self.signs[i])%len(self.loop)
                direction[i]=self.loop[self.loop_index[i]]-points[i]
            # Counterflow on the front side, with several circulating visitors.
            for i in set(range(self.count))-self.roles:
                direction[i,0]+=.04
                direction[i,1]+=.08*self.signs[i]
        elif self.name=='VideoUnmaskSwap':
            for i in range(self.count):
                if np.linalg.norm(direction[i])<.04:
                    self.goals[i]=self.rng.uniform(lo[i],hi[i]);direction[i]=self.goals[i]-points[i]
        else:
            for i in range(self.count):
                preferred=direction[i]/max(np.linalg.norm(direction[i]),1e-9)
                motion=self.motions[i]
                if motion is None or tick>=motion['end']:
                    style=self.names[self.orders[i][self.cursors[i]%4]];self.cursors[i]+=1
                    side=int(self.rng.choice([-1,1]));normal=side*np.array([-preferred[1],preferred[0]])
                    radius=float(self.rng.uniform(.025,.045))
                    motion=dict(style=style,start=tick,end=tick+int(self.rng.integers(60,101)),
                                axis=preferred,normal=normal,center=points[i]+radius*normal,radius=radius,side=side)
                    self.motions[i]=motion
                phase=(tick-motion['start'])/(motion['end']-motion['start'])
                axis,normal=motion['axis'],motion['normal']
                if motion['style']=='s_curve':preferred=axis+1.2*np.sin(2*np.pi*phase)*normal
                elif motion['style']=='reversal':
                    angle=np.pi*(.5-.5*np.cos(2*np.pi*phase));preferred=np.cos(angle)*axis+np.sin(angle)*normal
                elif motion['style']=='loop':
                    radial=points[i]-motion['center'];distance=max(np.linalg.norm(radial),1e-9);radial/=distance
                    preferred=motion['side']*np.array([-radial[1],radial[0]])+np.clip((motion['radius']-distance)/motion['radius'],-1,1)*radial
                direction[i]=preferred
        rate=.65+.35*np.sin(tick/29+np.arange(self.count))**2
        return direction,rate

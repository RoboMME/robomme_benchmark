"""Shared seeded continuous travel and recurring physical swaps, only."""
import numpy as np
from .motion import rng_for,swap_plan,path_clear,minimum_gap,RateSchedule

PATTERNS=('wandering','circling','weaving','pause_reverse')

class Traffic:
    def __init__(self,points,radii,seed,*,speed,bounds,circles=(),valid=None,max_diameter=None,source_style=None):
        self.rng=rng_for(seed,'shared-traffic')
        self.speed=float(speed);self.bounds=np.asarray(bounds);self.circles=list(circles)
        self.valid=valid;self.max_diameter=max_diameter
        self.count=len(points);self.tick=0;self.next_swap=35;self.active=None
        self.centroid=np.asarray(points).mean(axis=0)
        self.goals=np.asarray(points).copy();self.heading=np.zeros_like(points)
        self.phases=self.rng.uniform(0,2*np.pi,self.count)
        self.orbit_sign=self.rng.choice([-1,1])
        self.orbit_radius=self.rng.uniform(.07,.12,self.count)
        self.schedule=RateSchedule(seed,'shared-speed')
        self.pattern=None;self.next_pattern=0
        self.source_style=None
        if source_style:
            from .styles import SourceStyle
            self.source_style=SourceStyle(source_style,points,radii,seed,bounds)
        self.audit=dict(kind='shared motion patterns with recurring physical swaps',
                        pattern_pool=list(PATTERNS),patterns=[],events=[],samples=[],
                        swaps_have_no_episode_count_limit=True)
        if self.source_style:
            self.audit.update(source_task=source_style,style_goal_events=self.source_style.events,
                              kind='transferred task motion plus simultaneous recurring swaps')

    def refresh(self,points,radii):
        # Independent seeded proposals; successful/desired actor identity is unknown.
        self.goals=self.rng.uniform(self.bounds[:,0]+radii[:,None]+.01,
                                    self.bounds[:,1]-radii[:,None]-.01)

    def clear(self,before,after,radii,present,parked,margin=.005):
        indices=list(present)
        moving=[i for i in indices if np.linalg.norm(after[i]-before[i])>1e-10]
        if not moving:return True
        others=[i for i in indices if i not in moving]
        circles=[*self.circles,*parked,*[(before[i],radii[i]) for i in others]]
        paths=np.asarray([before[moving],after[moving]])
        if not path_clear(paths,radii[moving],bounds=self.bounds,circles=circles,valid=self.valid,margin=margin):return False
        if self.max_diameter and len(indices)>1 and np.linalg.norm(after[indices,None]-after[None,indices],axis=-1).max()>self.max_diameter:return False
        # Check simultaneous sweeps, including an earlier actor moved this step.
        return minimum_gap(np.array([before[indices],after[indices]]),radii[indices],moving_only=True)>=margin

    def step(self,points,radii,*,frozen=(),parked=(),scale=1.,present=None,hint=None):
        points=np.asarray(points);radii=np.asarray(radii);self.tick+=1
        present=set(range(len(points))) if present is None else set(present)
        frozen=set(frozen);eligible=present-frozen
        source_directions,source_rates=None,None
        if self.source_style:
            source_directions,source_rates=self.source_style.step(points,self.tick)
            self.pattern='from '+self.source_style.name
        elif self.tick>=self.next_pattern:
            choices=[p for p in PATTERNS if p!=self.pattern]
            self.pattern=str(self.rng.choice(choices));self.next_pattern=self.tick+int(self.rng.integers(100,181))
            self.refresh(points,radii)
            self.audit['patterns'].append(dict(start_step=self.tick,pattern=self.pattern))
        if not self.source_style and self.tick%75==0:self.refresh(points,radii)
        if self.active is not None:
            path,pair,cursor=self.active
            expected=path[int(cursor)]*(1-cursor%1)+path[min(int(cursor)+1,len(path)-1)]*(cursor%1)
            if set(pair)&frozen or not set(pair)<=present or np.linalg.norm(points[pair]-expected,axis=1).max()>.004:
                self.audit['events'][-1].update(completed=False,cancelled_step=self.tick,reason='pickup or physical displacement of swap participant')
                self.active=None;self.next_swap=self.tick+25
        if self.active is None and self.tick>=self.next_swap and len(eligible)>=2:
            indices=sorted(present)
            local_allowed=[k for k,i in enumerate(indices) if i in eligible]
            candidate=swap_plan(points[indices],radii[indices],self.rng,steps=72,speed=self.speed,
                                bounds=self.bounds,circles=[*self.circles,*parked],valid=self.valid,
                                allowed=local_allowed,max_diameter=self.max_diameter,margin=.006)
            self.next_swap=self.tick+8
            if candidate:
                paths,event=candidate;local_pair=event['pair'];pair=[indices[k] for k in local_pair]
                event.update(pair=pair,start_step=self.tick,completed=False)
                self.audit['events'].append(event);self.active=(paths[:,local_pair].copy(),pair,0.)
        following=points.copy()
        swap_pair=[]
        reserved=[]
        if self.active is not None:
            path,pair,cursor=self.active
            cursor=min(cursor+scale,len(path)-1)
            following[pair]=path[int(cursor)]*(1-cursor%1)+path[min(int(cursor)+1,len(path)-1)]*(cursor%1)
            if self.clear(points,following,radii,present,parked,margin=.004):
                swap_pair=pair
                # Reserve the remaining exchange corridor for this pair. Other
                # objects continue along collision-checked paths around it.
                future=np.r_[np.arange(int(cursor),len(path),4),len(path)-1]
                reserved=[(point,radii[pair[k]]+.004) for frame in path[future] for k,point in enumerate(frame)]
                if cursor==len(path)-1:
                    self.audit['events'][-1].update(completed=True,end_step=self.tick)
                    self.active=None;self.next_swap=self.tick+int(self.rng.integers(25,56))
                else:self.active=(path,pair,cursor)
            else:
                self.audit['events'][-1].update(cancelled_step=self.tick,completed=False,reason='physical obstruction')
                self.active=None;self.next_swap=self.tick+20
                following=points.copy()
        rate=self.schedule.rate(self.tick) if self.pattern=='pause_reverse' else .75+.25*np.sin(self.tick/31)**2
        offsets=np.array([0,.35,-.35,.7,-.7,1.1,-1.1,1.6,-1.6,2.2,-2.2,np.pi])
        for i in self.rng.permutation(sorted(eligible-set(swap_pair))):
            direction=self.goals[i]-points[i]
            if hint is not None and np.linalg.norm(hint[i]-points[i])>1e-9:
                direction=hint[i]-points[i]
            if self.pattern=='circling':
                radial=points[i]-self.centroid;norm=max(np.linalg.norm(radial),1e-6)
                tangent=self.orbit_sign*np.array([-radial[1],radial[0]])/norm
                direction=tangent+2*(self.orbit_radius[i]-norm)*radial/norm/.1
            norm=max(np.linalg.norm(direction),1e-9);direction=direction/norm
            if self.pattern=='weaving':direction+=.85*np.sin(self.tick/14+self.phases[i])*np.array([-direction[1],direction[0]])
            if source_directions is not None:direction=source_directions[i]
            if rate<0:direction=-direction
            actor_rate=source_rates[i] if source_rates is not None else abs(rate)
            angle=np.arctan2(direction[1],direction[0]);distance=self.speed*actor_rate*scale
            if distance<1e-10:continue
            for offset in offsets:
                candidate=following.copy();candidate[i]+=distance*np.array([np.cos(angle+offset),np.sin(angle+offset)])
                if reserved and not path_clear(np.array([[points[i]],[candidate[i]]]),radii[[i]],
                                               bounds=self.bounds,circles=reserved,margin=.004):continue
                if self.clear(points,candidate,radii,present,parked):
                    following=candidate;break
        moving=np.linalg.norm(following-points,axis=1)>1e-8
        if moving.any():
            ix=sorted(present)
            gap=minimum_gap(np.array([points[ix],following[ix]]),radii[ix],moving_only=True)
            self.audit['minimum_commanded_moving_pair_gap_m']=min(self.audit.get('minimum_commanded_moving_pair_gap_m',float('inf')),gap)
        background=int(sum(moving[i] for i in eligible-set(swap_pair)))
        self.audit['samples'].append([self.tick,self.pattern,int(moving.sum()),len(swap_pair),background])
        if swap_pair:
            event=self.audit['events'][-1]
            event['concurrent_background_motion_steps']=event.get('concurrent_background_motion_steps',0)+int(background>0)
            event['background_objects_moving_max']=max(event.get('background_objects_moving_max',0),background)
        return following

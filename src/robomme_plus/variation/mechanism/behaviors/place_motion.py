"""alternating continuous-motion and global-swap phases for rings."""
import numpy as np
from .motion import rng_for
from .traffic import Traffic
from .ring_dynamics import Breathing,GAP,PADDING,install_safe_swaps,NoSeparatedSwap


class Phases:
    def __init__(self,seed):
        self.rng=rng_for(seed,'ring-phase-alternation');self.events=[]
        self.kind='continuous';self.start=0;self.rounds=0;self.enter('continuous',0)

    def enter(self,kind,clock):
        if self.events:self.events[-1].update(end_clock=clock,completed_swaps=self.rounds)
        self.kind=kind;self.start=clock;self.rounds=0
        self.duration=int(self.rng.integers(50,81)) if kind=='continuous' else None
        self.required=int(self.rng.integers(2,4)) if kind=='swaps' else 0
        self.events.append(dict(phase=kind,start_clock=clock,duration_steps=self.duration,
                                requested_swaps=self.required))


def install(task,seed,config):
    from robomme_plus.tasks.reference.utils import place_swaps as module
    from robomme_plus.common.sim import position,scalar,set_position,step_number
    install_safe_swaps(module,seed)
    config.update(style='continuous gather/spread, then repeated global swaps; alternating exclusive phases',
                  execution_dodge_limit=0,minimum_ring_edge_gap_m=GAP,
                  continuous_phase_steps=[50,80],swaps_per_phase=[2,3],global_swap_partners=True,
                  swap_control_steps=40,max_concurrent_pairs=2)
    config['execution_motion_stop']='permanently on first grasp of requested cube'
    original=module.ClockedSwaps.update
    gate=module.LevelSwaps.start_enabled.fget
    module.LevelSwaps.start_enabled=property(lambda self:gate(self) and
         getattr(getattr(self,'_ring_phases',None),'kind','continuous')=='swaps')

    def update(self):
        if not hasattr(self,'_ring_phases'):
            self._ring_phases=Phases(seed);self._ring_execution_path=None
        phases=self._ring_phases
        before=np.array([position(a,False)[:2] for a in self.actors])
        prior_pairs={i for p in self.pairs for i in p}
        prior_rounds=self.metadata['completed_rounds']
        blocked_correction=False
        try:original(self)
        except NoSeparatedSwap:
            blocked_correction=True
            self.metadata.setdefault('correction_spacing_wait_clocks',[]).append(self.clock)
        points=np.array([position(a,False)[:2] for a in self.actors])
        if not hasattr(self,'_ring_traffic'):
            engine=Traffic(points,np.full(len(points),self.radius+PADDING),seed,speed=.003,
                bounds=np.array([module.LAYOUT_LOW-self.radius,module.LAYOUT_HIGH+self.radius]).T,
                circles=[(position(self.env.button,False)[:2],.055)])
            self._ring_traffic=engine;engine.source_style=Breathing(points,seed);engine.next_swap=10**9
            self.metadata['mixed_motion']=engine.audit
            engine.audit.update(execution_dodges=[],native_demonstration_swaps=True,
                closing_corrections_preserved=True,stage_history=[],minimum_ring_edge_gap_m=GAP,
                kind=config['style'],gather_spread_events=engine.source_style.events,
                phase_events=phases.events,phase_samples=[],global_swap_partners=True,
                background_motion_during_swaps=False)
            self.metadata['strategy']=config['style']
        engine=self._ring_traffic
        if self.stage=='execution':
            execution=self.metadata['execution']
            if execution.get('stopped_on_grasp') or scalar(self.env.agent.is_grasping(self.env.target_cube)):
                if not execution.get('stopped_on_grasp'):
                    execution.update(stopped_on_grasp=True,stop_control_step=step_number(self.env),
                        stop_clock=self.clock,stop_cube=self.env.target_cube.name,
                        frozen_target_xy=points.tolist(),movement_steps_after_grasp=0)
                    if self._ring_execution_path is not None:
                        engine.audit['events'][-1].update(completed=False,cancelled_clock=self.clock,
                            reason='requested cube grasped')
                    self._ring_execution_path=None
                assert np.max(np.abs(points-np.asarray(execution['frozen_target_xy'])))<1e-7
                self.active=False
                execution['continuous_motion']=False
                self.history.append(points.copy())
                engine.audit['stage_history'].append([engine.tick,self.clock,self.stage])
                engine.audit['phase_samples'].append([self.clock,self.stage,'stopped after grasp',0,0])
                return
        frozen=set(self.pinned) if self.stage=='demonstration' else set()
        tcp=position(self.env.agent.tcp,False)
        if not hasattr(self,'_ring_nearby'):self._ring_nearby=set()
        if tcp[2]>=.145:self._ring_nearby.clear()
        elif tcp[2]<.13:
            self._ring_nearby.update(int(i) for i in np.flatnonzero(np.linalg.norm(points-tcp[:2],axis=1)<.085))
        frozen|=self._ring_nearby
        engine.audit['local_placement_pause']=dict(tcp_height_m=.13,radius_m=.085,release_height_m=.145,
            same_rule_in_demonstration_and_execution=True)
        obstacles=[]
        for cube in self.env.all_cubes:
            p=position(cube,False)
            if self.stage=='demonstration':obstacles.append((position(cube,True)[:2],.025))
            if p[2]<.055 and not scalar(self.env.agent.is_grasping(cube)):
                distances=np.linalg.norm(points-p[:2],axis=1);close=np.flatnonzero(distances<.055)
                if len(close):frozen.add(int(close[np.argmin(distances[close])]))
                obstacles.append((p[:2],.025))
        after=points.copy();closing=self.stage=='demonstration' and self.demonstration_closing
        pair=prior_pairs|{i for p in self.pairs for i in p}
        actual_mode=phases.kind
        if closing:
            actual_mode='final swaps' if pair or self.corrections_complete else 'clearance spreading'
            if blocked_correction and not getattr(self,'_ring_spreading_for_correction',False):
                self._ring_spreading_for_correction=True
                engine.source_style.compact=True;engine.source_style.next_change=0
        if (not closing and phases.kind=='continuous') or (closing and not pair and not self.corrections_complete):
            engine.next_swap=10**9
            after=engine.step(points,np.full(len(points),self.radius+PADDING),frozen=frozen,parked=obstacles)
        elif not closing and phases.kind=='swaps':
            if self.stage=='demonstration':
                phases.rounds+=self.metadata['completed_rounds']-prior_rounds
            else:
                active=self._ring_execution_path
                if active is not None and {i for p in active[1] for i in p}&frozen:
                    engine.audit['events'][-1].update(completed=False,cancelled_clock=self.clock,reason='placement approach')
                    self._ring_execution_path=None;active=None
                if active is None:
                    paths,pairs,_=module.plan_neighbor_pairs(points,phases.rng,frozen,maximum=module.sample_pair_count(phases.rng),radius=self.radius)
                    if pairs:
                        pair={i for p in pairs for i in p};active=(paths,pairs,0);self._ring_execution_path=active
                        self.counts[list(pair)]+=1;self.metadata['target_swap_counts']=self.counts.tolist()
                        engine.audit['events'].append(dict(pair=sorted(pair),pairs=[list(p) for p in pairs],start_step=self.clock,completed=False,
                            duration_steps=len(paths)-1,style='concurrent one-to-one opposite arcs',concurrent_background_motion_steps=0,
                            pair_distances_m=[float(np.linalg.norm(points[i]-points[j])) for i,j in pairs]))
                if active is not None:
                    paths,pairs,cursor=active;pair={i for p in pairs for i in p};cursor+=1;after=paths[cursor].copy()
                    if cursor==len(paths)-1:
                        phases.rounds+=1;self._ring_execution_path=None
                        engine.audit['events'][-1].update(completed=True,end_step=self.clock)
                    else:self._ring_execution_path=(paths,pairs,cursor)
        for i,a in enumerate(self.actors):
            if np.linalg.norm(after[i]-points[i])>1e-10:set_position(a,[*after[i],self.z[i]],self.q[i])
        self.history.append(after.copy())
        engine.audit['stage_history'].append([engine.tick,self.clock,self.stage])
        moved=np.linalg.norm(after-before,axis=1)>1e-8
        background=sum(bool(moved[i]) for i in range(len(points)) if i not in pair)
        if actual_mode in ('swaps','final swaps'):assert background==0,(actual_mode,background)
        engine.audit['phase_samples'].append([self.clock,self.stage,actual_mode,int(moved.sum()),int(background)])
        gaps=np.linalg.norm(after[:,None]-after[None,:],axis=-1)+np.eye(len(after))*10-2*self.radius
        minimum=float(gaps.min());assert minimum>=GAP-1e-6,minimum
        engine.audit['minimum_actual_ring_edge_gap_m']=min(minimum,engine.audit.get('minimum_actual_ring_edge_gap_m',10.))
        engine.audit.setdefault('spread_samples',[]).append([self.clock,float(np.sqrt(np.square(after-after.mean(axis=0)).sum(axis=1).mean()))])
        if not closing:
            if phases.kind=='continuous' and self.clock-phases.start>=phases.duration:
                phases.enter('swaps',self.clock)
            elif phases.kind=='swaps' and not self.pairs and self._ring_execution_path is None and (
                 phases.rounds>=phases.required or self.clock-phases.start>480):
                phases.events[-1]['ended_for_clearance']=phases.rounds<phases.required
                phases.enter('continuous',self.clock)
                engine.source_style.next_change=0
        self.metadata['execution']['continuous_motion']=True
    module.ClockedSwaps.update=update

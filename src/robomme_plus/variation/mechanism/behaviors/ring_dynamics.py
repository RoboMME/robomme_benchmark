"""Seeded gathering/spreading and separated ring swaps for VideoPlace."""
from itertools import combinations
import numpy as np
from .motion import rng_for,path_clear

GAP=.035
PADDING=GAP/2
SWAP_STEPS=40
SWAP_BOUNDS=np.array([[-.40,.20],[-.39,.39]])

class NoSeparatedSwap(Exception):pass


class Breathing:
    name='seeded gathering and spreading'
    def __init__(self,points,seed):
        self.rng=rng_for(seed,'ring-gather-spread');self.next_change=0
        self.compact=False;self.events=[];self.goals=np.array(points).copy()
        self.center=np.array([-.105,0.]);self.order=np.arange(len(points));self.axes=None

    def step(self,points,tick):
        if tick>=self.next_change:
            self.compact=not self.compact
            self.center=np.array([-.105,0.])+self.rng.uniform([-.01,-.015],[.01,.015])
            angle=np.arctan2(points[:,1]-self.center[1],points[:,0]-self.center[0])
            self.order=np.argsort(angle)
            phase=float(self.rng.uniform(-np.pi,-np.pi+2*np.pi/len(points)))
            theta=phase+2*np.pi*np.arange(len(points))/len(points)
            self.axes=self.rng.uniform([.14,.175],[.155,.20]) if self.compact else self.rng.uniform([.185,.265],[.20,.29])
            # More rings require a sufficiently large compact formation.
            self.axes=np.maximum(self.axes,(.08+GAP+.012)/(2*np.sin(np.pi/len(points))))
            self.goals[self.order]=self.center+self.axes*np.c_[np.cos(theta),np.sin(theta)]
            duration=int(self.rng.integers(140,221));self.next_change=tick+duration
            self.events.append(dict(step=tick,phase='gather' if self.compact else 'spread',duration_steps=duration,
                center_xy=self.center.tolist(),axes_m=self.axes.tolist(),goals=self.goals.tolist()))
        direction=self.goals-points
        rates=np.minimum(1.,np.linalg.norm(direction,axis=1)/.018)
        return direction,rates


def install_safe_swaps(module,seed):
    """Keep native identity/correction bookkeeping, replace overlapping arcs."""
    context={};cache={};original_init=module.ClockedSwaps.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs);context['routes']=self
        self.metadata.update(target_collision_avoidance=True,minimum_ring_edge_gap_m=GAP,
                             swap_path='direct opposite half-ellipse pairs, up to two concurrent pairs',round_control_steps=SWAP_STEPS)
    module.ClockedSwaps.__init__=init;module.SWAP_STEPS=SWAP_STEPS
    original_level_init=module.LevelSwaps.__init__
    def level_init(self,env,*args,**kwargs):
        env._limit_config=dict(env._limit_config,swap_start_interval_steps=SWAP_STEPS)
        original_level_init(self,env,*args,**kwargs)
    module.LevelSwaps.__init__=level_init

    def layout(count,radius,rng,obstacles=()):
        for _ in range(120):
            points=[]
            for _ in range(count):
                for _ in range(10000):
                    point=rng.uniform(module.LAYOUT_LOW+.025,module.LAYOUT_HIGH-.025)
                    if all(np.linalg.norm(point-p)>=2*radius+GAP+.008 for p in points) and all(
                        np.linalg.norm(point-p)>=radius+r+.025 for p,r in obstacles):
                        points.append(point);break
                else:break
            if len(points)==count:rng.shuffle(points);return np.asarray(points)
        raise RuntimeError('Cannot fit separated target rings')
    module.sample_random_layout=layout

    def pair_path(points,pair,radius):
        from robomme_plus.common.sim import position,scalar
        points=np.asarray(points)
        routes=context['routes'];env=routes.env
        circles=[(position(env.button,False)[:2],.055)]
        # Rings are noncolliding visual targets. Native swaps may pass beneath
        # a stationary cube; only its occupied ring is pinned by ClockedSwaps.
        # Treating every cube as an inflated routing obstacle can deadlock the
        # final swaps even though the task itself permits these trajectories.
        key=(points.round(7).tobytes(),tuple(pair),radius,
             tuple((tuple(np.round(p,6)),r) for p,r in circles))
        if key in cache:return cache[key]
        if len(cache)>256:cache.clear()
        bounds=SWAP_BOUNDS
        radii=np.full(len(points),radius+PADDING)
        i,j=pair;delta=points[j]-points[i];distance=np.linalg.norm(delta)
        if distance<2*radius+GAP:return None
        axis=delta/distance;normal=np.array([-axis[1],axis[0]]);center=(points[i]+points[j])/2
        u=np.linspace(0,1,SWAP_STEPS+1);angle=np.pi*u*u*(3-2*u)
        for width in [.16,.14,.11,.09,.075,.06]:
            for sign in [1,-1]:
                offset=-.5*distance*np.cos(angle)[:,None]*axis+sign*width*np.sin(angle)[:,None]*normal
                path=np.repeat(points[None],len(u),axis=0);path[:,i]=center+offset;path[:,j]=center-offset
                path[0]=points;path[-1,i]=points[j];path[-1,j]=points[i]
                if path_clear(path,radii,bounds=bounds,circles=circles,margin=.002,moving_only=True):
                    cache[key]=path;return path
        cache[key]=None;return None

    def pair_paths(points,pairs,radius):
        path=np.repeat(np.asarray(points)[None],SWAP_STEPS+1,axis=0)
        for pair in pairs:
            candidate=pair_path(points,pair,radius)
            if candidate is None:raise RuntimeError('Selected pair has no separated arc')
            path[:,list(pair)]=candidate[:,list(pair)]
        return path

    def ordinary(points,rng,pinned=(),maximum=2,*,radius=.04):
        from robomme_plus.common.sim import position
        eligible=[i for i in range(len(points)) if i not in pinned]
        pairs=list(combinations(eligible,2));rng.shuffle(pairs)
        counts=context['routes'].counts
        pairs.sort(key=lambda pair:(int(counts[list(pair)].sum()),
                    -float(np.linalg.norm(points[pair[0]]-points[pair[1]]))))
        valid=[pair for pair in pairs if pair_path(points,pair,radius) is not None]
        selected=[]
        if maximum>=2:
            for first,second in combinations(valid,2):
                if set(first)&set(second):continue
                paths=pair_paths(points,[first,second],radius)
                if path_clear(paths,np.full(len(points),radius+PADDING),bounds=SWAP_BOUNDS,
                    circles=[(position(context['routes'].env.button,False)[:2],.055)],margin=.002,moving_only=True):
                    selected=[first,second];break
        if not selected and valid:selected=[valid[0]]
        return pair_paths(points,selected,radius),selected,[dict(pair=list(pair),separation_checked=True) for pair in selected]

    def correction(points,goal_index,reference_xy,rng,pinned=()):
        from robomme_plus.tasks.reference.utils.place_corrections import GOAL_LOCATION_TOLERANCE_M
        routes=context['routes']
        if getattr(routes,'_ring_retry_clock',0)>routes.clock:raise NoSeparatedSwap()
        pairs=list(combinations([i for i in range(len(points)) if i not in pinned],2));rng.shuffle(pairs)
        for pair in pairs:
            i,j=pair;endpoint=points[j] if goal_index==i else points[i] if goal_index==j else points[goal_index]
            if np.linalg.norm(endpoint-reference_xy)<=GOAL_LOCATION_TOLERANCE_M:continue
            if pair_path(points,pair,context['routes'].radius) is not None:
                return pair,dict(pair=list(pair),separation_checked=True,goal_involved=goal_index in pair)
        routes._ring_retry_clock=routes.clock+8
        raise NoSeparatedSwap()
    module.pair_paths=pair_paths;module.plan_neighbor_pairs=ordinary;module.plan_correction_pair=correction

    # A separated exchange may need the formation to spread before it starts.
    # Keep that physical waiting/motion visible in the demo; retain both swaps.
    from robomme_plus.tasks.reference.utils import place_corrections
    def finish(wrapped,planner):
        import sapien
        from robomme_plus.common.sim import position,scalar
        from robomme_plus.control.planning import solve_pickup
        env=wrapped.unwrapped;routes=env._limit_routes
        assert not any(scalar(env.agent.is_grasping(c)) for c in env.all_cubes)
        # A long moving demo can nudge an unused cube onto a goal ring. Restore
        # displaced cubes physically before the final swaps, rather than leaving
        # the goal permanently occupied or changing the native pinning rule.
        for cube in env.all_cubes:
            home=position(cube,True).copy()
            if np.linalg.norm(position(cube,False)[:2]-home[:2])<=.025:continue
            event=dict(cube=cube.name,before=position(cube,False).tolist(),home=home.tolist())
            routes.metadata.setdefault('physical_cube_return_repairs',[]).append(event)
            for attempt in range(3):
                solve_pickup(wrapped,planner,obj=cube)
                if scalar(env.agent.is_grasping(cube)):break
            assert scalar(env.agent.is_grasping(cube)),'Cannot recover displaced demonstration cube'
            q=env.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1,4)[0]
            offset=position(cube,False)-position(env.agent.tcp,False)
            goal=home-offset;above=goal.copy();above[2]=.20
            for point in (above,goal):
                if planner.move_to_pose_with_screw(sapien.Pose(point,q))==-1:return -1
            planner.open_gripper(t=12)
            if planner.move_to_pose_with_screw(sapien.Pose(above,q))==-1:return -1
            event['after']=position(cube,False).tolist()
            assert np.linalg.norm(position(cube,False)[:2]-home[:2])<.025,'Displaced cube return missed'
        q=env.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1,4)[0]
        if planner.move_to_pose_with_screw(sapien.Pose(place_corrections.FINAL_HOLD_TCP_POSITION_M,q))==-1:return -1
        start=routes.clock
        if routes.pairs:planner.open_gripper(t=SWAP_STEPS-routes.phase)
        routes.begin_final_corrections()
        for _ in range(900):
            if routes.corrections_complete:break
            planner.open_gripper(t=1)
        if not routes.corrections_complete:raise RuntimeError('Separated final swaps did not finish')
        routes.metadata['demonstration_swap_wait_steps']=routes.clock-start
        env._limit_diagnostics['demonstration_swap_wait_steps']=routes.clock-start
    place_corrections.solve_final_corrections=finish
    import sys
    environment=sys.modules.get('robomme_plus.tasks.reference.utils.place_environment')
    if environment is not None:environment.solve_final_corrections=finish

    # A moving demonstration can leave a cube at a difficult pickup pose.
    # Retry the physical pickup before advancing the recorded task sequence.
    from robomme_plus.tasks.reference.utils import place_demonstration
    attach=place_demonstration.attach_checks
    def attach_checked(env):
        cube=None
        for entry in env.task_list:
            if not entry.get('demonstration'):continue
            if entry['name']=='pick up the cube':cube=entry['segment']
            elif entry['name']=='drop the cube onto target':
                from .ring_reference import place
                def drop(wrapped,planner,_cube=cube,_target=entry['segment'],_entry=entry):
                    result=place(wrapped,planner,_cube,_target)
                    from robomme_plus.common.sim import scalar
                    if not scalar(_entry['func']()):raise RuntimeError('Demonstration missed its requested target')
                    return result
                entry['_native_solve']=entry['solve']=drop
                continue
            else:continue
            original_pick=entry.get('_native_solve',entry['solve'])
            def pick(wrapped,planner,_entry=entry,_solve=original_pick):
                import sapien
                from robomme_plus.common.sim import position,scalar,step_number
                result=_solve(wrapped,planner)
                for attempt in range(2):
                    if scalar(_entry['func']()):return result
                    env._limit_routes.metadata.setdefault('physical_pickup_retries',[]).append(
                        dict(step=step_number(env),cube=_entry['segment'].name,attempt=attempt+1))
                    planner.open_gripper(t=12)
                    point=position(env.agent.tcp,False).copy();point[2]=.22
                    q=env.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1,4)[0]
                    planner.move_to_pose_with_screw(sapien.Pose(point,q))
                    planner.move_to_pose_with_screw(sapien.Pose([-.12,0.,.25],[0.,1.,0.,0.]))
                    result=_solve(wrapped,planner)
                if not scalar(_entry['func']()):
                    from .cube_reference import pickup
                    pickup(wrapped,planner,_entry['segment'])
                if not scalar(_entry['func']()):raise RuntimeError('Physical demonstration pickup failed after retries')
                return result
            entry['_native_solve']=entry['solve']=pick
        attach(env)
    place_demonstration.attach_checks=attach_checked
    if environment is not None:environment.attach_checks=attach_checked

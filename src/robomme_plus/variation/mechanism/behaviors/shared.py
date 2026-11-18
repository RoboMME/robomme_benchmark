"""Adapters for one motion library across the three memory tasks."""
import inspect
import numpy as np
from .traffic import Traffic,PATTERNS


def install(task,seed,config):
    source_style=config["source_motion_task"]
    config.update(style='transferred '+source_style+' motion with simultaneous recurring swaps',source_motion_task=source_style,
                  motion_patterns=list(PATTERNS),swap_patterns=['ellipse','bow','asymmetric_bow'],
                  continues_during_execution=True,pickup_proximity_pause=True)
    if task=='VideoRepick':return repick(seed,source_style)
    return permanence(task,seed,source_style)


def permanence(task,seed,source_style):
    from robomme_plus.tasks.permanence.utils import tasks,workspace,routes,moving
    from robomme_plus.common.sim import step_number
    original_mode=workspace.resolve_continuous_motion
    if task=='ButtonUnmaskSwap':
        workspace.resolve_continuous_motion=lambda t,d,s,value=None: True if t==task else original_mode(t,d,s,value)
    native_flow=routes.plan_routes;signature=inspect.signature(native_flow)
    native_moving=moving.plan_routes;moving_signature=inspect.signature(native_moving)

    def adapt(initial,metadata,*,speed,bounds,area,circles,barrier_data=None,source=None):
        def valid(path):
            pts=path.reshape(-1,2)
            if not area.contains(pts).all():return False
            if barrier_data is not None:
                a,b,widths,radius=barrier_data
                if not (routes.point_segment_distance(pts,a,b)>widths+radius+routes.MARGIN_M).all():return False
            delta=np.linalg.norm(np.diff(path,axis=0),axis=-1)>1e-9
            return all(area.segment(a,b) for a,b in zip(path[:-1][delta],path[1:][delta]))
        engine=Traffic(initial,np.full(len(initial),metadata['radius_m']),seed,speed=speed,bounds=bounds,circles=circles,valid=valid,source_style=source_style)
        metadata['mixed_motion']=engine.audit
        def stream():
            feedback=yield np.asarray(initial).copy()
            while True:
                actual,frozen=feedback[:2];parked=feedback[2] if len(feedback)>2 else []
                extra=feedback[3] if len(feedback)>3 else {}
                hint=source.send(feedback) if source is not None else None
                feedback=yield engine.step(actual,extra.get('actor_radii_m',np.full(len(initial),metadata['radius_m'])),
                                          frozen=frozen,parked=parked,scale=extra.get('speed_scale',1.),hint=hint)
        iterator=stream();next(iterator)
        return (initial,iterator),metadata

    def flow(*args,**kwargs):
        result=native_flow(*args,**kwargs)
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();p=bound.arguments
        if not p['continuous'] or p['geometry_only']:return result
        (initial,source),metadata=result
        circles=[*p['obstacles'],*p['extra_obstacles'],(routes.ROBOT_BASE_XY,routes.ROBOT_BASE_RADIUS_M)]
        circles += [(xy,float(np.linalg.norm(half))) for xy,half in p['rectangle_obstacles']]
        a,b,widths,_=routes.barriers(p['obstacles'])
        # The host's physical buttons/board remain obstacles. Its virtual
        # traffic lanes do not constrain a transferred roaming/swarm style.
        return adapt(initial,metadata,speed=p['speed'],bounds=metadata['bounds_xy_m'],area=p['reachable_area'],circles=circles)

    def unmask_plan(*args,**kwargs):
        filtered={k:v for k,v in kwargs.items() if k in moving_signature.parameters}
        paths,metadata=native_moving(*args,**filtered)
        if not kwargs.get('continuous'):return paths,metadata
        bound=moving_signature.bind(*args,**filtered);bound.apply_defaults();p=bound.arguments
        return adapt(paths[0],metadata,speed=p['speed'],bounds=[[-.31,.12],[-.32,.32]],
                     area=p['reachable_area'],circles=p['obstacles'])

    if task=='ButtonUnmaskSwap':routes.plan_routes=flow
    else:
        moving.plan_routes=unmask_plan
        old_routes,old_setup,old_update=tasks.swap_routes,tasks.setup,tasks.update
        def swap_routes(env,config,*args,**kwargs):
            # Use native initial packing and reachability, with the same cover
            # controller as ButtonUnmaskSwap. Retain the video reveal task.
            original={k:config.get(k) for k in ('continuous_motion','continuous_motion_version')}
            config.update(continuous_motion=True,continuous_motion_version=6)
            try:return old_routes(env,config,*args,**kwargs)
            finally:
                for k,v in original.items():
                    if v is None:config.pop(k,None)
                    else:config[k]=v
        def setup(env,config):
            config.pop('continuous_motion',None)
            config.pop('continuous_motion_version',None)
            old_setup(env,config)
            config.update(continuous_motion=True,continuous_motion_version=6)
            env.continuous_motion=True
        def update(env):
            old_update(env)
            route=env._limit_routes
            if route.execution_start is not None and not route.active and not env._limit_failed:
                route.active=True;route.update()
        tasks.swap_routes,tasks.setup,tasks.update=swap_routes,setup,update


def repick(seed,source_style):
    from robomme_plus.tasks.reference.utils import repick_tasks,repick_controller
    from robomme_plus.common.sim import position,scalar,set_position
    original_setup=repick_tasks.repick_setup
    original_update=repick_controller.Routes.update
    def setup(env,config):
        config['motion_tail_steps']=60
        original_setup(env,config)
        # Set after construction so the finite native setup plan remains small.
        config['motion_tail_steps']=100000
    def update(self):
        if not self.active:return
        before=np.array([position(a,False)[:2] for a in self.actors])
        if not hasattr(self,'_shared_traffic'):
            self._shared_traffic=Traffic(before,np.full(len(before),self.radius),seed,
                 speed=self.metadata['max_speed_m_per_control_step'],bounds=self.metadata['workspace_xy_bounds_m'],
                 circles=self.obstacles,max_diameter=None,source_style=source_style)
            self.metadata['mixed_motion']=self._shared_traffic.audit
        tcp=position(self.env.agent.tcp,False)
        tabletop={i for i,a in enumerate(self.actors) if position(a,False)[2]<=self.z[i]+.02}
        frozen=set(self.pinned)
        frozen|={i for i,a in enumerate(self.actors) if scalar(self.env.agent.is_grasping(a)) or i not in tabletop}
        if self.execution_start is not None and tcp[2]<.13:
            frozen|={i for i in tabletop if np.linalg.norm(before[i]-tcp[:2])<.09}
        scale=.5 if frozen else 1.
        after=self._shared_traffic.step(before,np.full(len(before),self.radius),frozen=frozen,present=tabletop,scale=scale)
        from .motion import minimum_gap
        if len(tabletop)>1:
            ix=sorted(tabletop);gap=minimum_gap(np.array([before[ix],after[ix]]),np.full(len(ix),self.radius))
            self.metadata['minimum_actual_continuous_pair_gap_m']=min(self.metadata.get('minimum_actual_continuous_pair_gap_m',float('inf')),gap)
        for i,actor in enumerate(self.actors):
            if i in frozen:continue
            k=(self._shared_traffic.tick-1)%(len(self.angles)-1)
            # Integrate native yaw increments continuously when the table moves.
            delta=float(self.angles[k+1,i]-self.angles[k,i])*scale
            q=actor.pose.q.detach().cpu().numpy().reshape(-1,4)[0]
            set_position(actor,[*after[i],self.z[i]],repick_controller.yaw_quaternion(q,delta))
        self.positions=after;self.history.append(after.copy());self.cursor+=1
        self.metadata['movement_steps']=self.cursor
        if self.execution_start is not None:self.metadata['motion_steps_after_reset']=self.metadata.get('motion_steps_after_reset',0)+1
        self.metadata['simultaneously_moving_max']=max(self.metadata.get('simultaneously_moving_max',0),int((np.linalg.norm(after-before,axis=1)>1e-8).sum()))
    repick_tasks.repick_setup=setup;repick_controller.Routes.update=update

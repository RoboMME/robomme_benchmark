"""Longer RouteStick activation, demonstrated with the same contact rule."""
import numpy as np
from .motion import rng_for


def install(task,seed,config):
    from robomme_plus.tasks.imitation import route_stick as module
    from robomme_plus.tasks.imitation.utils import controllers
    from robomme_plus.common.sim import step_number,scalar
    dwell_steps=int(rng_for(seed,'tracing-activation-or-size').integers(12,19))
    scale=1.
    config.update(style='slow_activation',activation_steps=dwell_steps,
        activation_seconds=dwell_steps/20,visible_target_radius_m=.02,
        target_scale=scale,contact_radius_m=.03,demonstration_has_same_requirement=True)

    original_touch=module.is_obj_swing_onto
    def touch(env,obj,target,*args,**kwargs):
        kwargs['distance_threshold']=.03*scale
        geometric=dict(kwargs);geometric.pop('judge_direction_list',None)
        inside=bool(scalar(original_touch(env,obj,target,*args,**geometric)))
        phase=getattr(getattr(env,'_drawing',None),'phase',None)
        step=int(step_number(env));key=(phase,id(target))
        state=getattr(env,'_slow_route_contacts',None)
        if state is None:state=env._slow_route_contacts={}
        item=state.setdefault(key,dict(last=None,steps=0,inside=False,lit=False))
        if item['last']!=step:
            item['steps']=item['steps']+1 if inside and item['inside'] and item['last']==step-1 else 0
            item.update(last=step,inside=inside)
            if not inside:item['lit']=False
        ready=inside and item['steps']>=dwell_steps
        if ready and not item['lit']:
            item['lit']=True
            env._limit_diagnostics.setdefault('slow_activation_events',[]).append(
                dict(step=step,phase=phase,target=target.name,held_intervals=item['steps'],required=dwell_steps))
        return bool(ready and scalar(original_touch(env,obj,target,*args,**kwargs)))
    module.is_obj_swing_onto=touch

    def hold_target(wrapped,planner,target):
        env=wrapped.unwrapped
        qpos=env.agent.robot.qpos.cpu().numpy()[0].copy()
        path={'position':qpos[None],'velocity':np.zeros_like(qpos)[None]}
        for _ in range(dwell_steps+25):
            if touch(env,env.agent.tcp,target) or scalar(env.failureflag):return
            planner.follow_path(path)
        raise RuntimeError('RouteStick reference did not hold inside its target')
    for name in ('solve_swingonto','solve_swingonto_withDirection'):
        original=getattr(module,name)
        def solve(wrapped,planner,*args,_original=original,**kwargs):
            result=_original(wrapped,planner,*args,**kwargs)
            target=kwargs.get('target',args[0] if args else None)
            hold_target(wrapped,planner,target)
            return result
        setattr(module,name,solve)
    original_start=controllers.start_drawing
    def start(env,planner):
        result=original_start(env,planner)
        if env.task_name=='RouteStick':hold_target(env,planner,env.selected_buttons[0])
        return result
    controllers.start_drawing=start

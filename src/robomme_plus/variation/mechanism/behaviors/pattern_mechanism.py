"""seeded PatternLock mechanisms, identical in demo and execution."""
import numpy as np
from .motion import rng_for

PRESETS=(('neighbor',),('slow',),('small',),('neighbor','slow'),('neighbor','small'),('slow','small'),('neighbor','slow','small'))
DIRECTIONS={'left':(0,1),'right':(0,-1),'forward':(1,0)}


def configuration(seed):
    rng=rng_for(seed,'pattern-mechanism-v1')
    factors=PRESETS[int(rng.integers(len(PRESETS)))]
    direction=str(rng.choice(list(DIRECTIONS))) if 'neighbor' in factors else None
    return dict(factors=list(factors),direction=direction,
        dwell_steps=int(rng.integers(12,19)) if 'slow' in factors else 6,
        target_scale=float(rng.uniform(.45,.60)) if 'small' in factors else 1.,
        rule_scope='one direction per episode',boundary_rule='cyclic wraparound within the same grid row or column',
        phases=['demonstration','execution'])


def mapping(size,direction):
    dr,dc=DIRECTIONS[direction] if direction else (0,0)
    return {r*size+c:((r+dr)%size)*size+(c+dc)%size for r in range(size) for c in range(size)}


def install(seed,config):
    from robomme_plus.tasks.imitation import pattern_lock
    from robomme_plus.tasks.imitation.utils import pattern_dwell,tracing
    spec=configuration(seed);scale=spec['target_scale'];direction=spec['direction']
    config.update(style='seeded PatternLock mechanism mix',pattern_mechanism=spec,
        activation_steps=spec['dwell_steps'],activation_seconds=spec['dwell_steps']/20,
        visible_target_radius_m=.02*scale,contact_radius_m=.01*scale,
        demonstration_has_same_requirement=True)
    if scale!=1:
        build=pattern_lock.build_gray_white_target
        def smaller(**kwargs):
            kwargs['radius']*=scale
            return build(**kwargs)
        pattern_lock.build_gray_white_target=smaller
    Original=pattern_dwell.PatternDwell
    class MechanismDwell(Original):
        def __init__(self,*args,**kwargs):
            kwargs['dwell_seconds']=spec['dwell_steps']/20
            super().__init__(*args,**kwargs)
            self.radius_m*=scale
            self.activation_map=mapping(int(round(len(self.xy)**.5)),direction)
            self.required_activated=[self.activation_map[i] for i in self.nodes]
            self.activated=[]
        @property
        def success(self):
            return super().success and self.activated==self.required_activated
        def update(self,*args,**kwargs):
            event=super().update(*args,**kwargs)
            if event is not None:
                touched=event['node'];lit=self.activation_map[touched]
                self.activated.append(lit)
                event.update(touched_node=touched,activated_node=lit)
                # Physical line/dwell checks retain the input path; feedback names the output.
                return dict(event,node=lit)
            return None
        def summary(self):
            return dict(super().summary(),mechanism=spec,activated_nodes=list(self.activated),
                required_activated_nodes=list(self.required_activated),input_to_output=dict(self.activation_map))
    pattern_dwell.PatternDwell=MechanismDwell
    def held(env,_env,obj,target,**kwargs):
        node=env._pattern_dwell.held_node
        return node is not None and target is env.buttons_grid[env._pattern_dwell.activation_map[node]]
    pattern_lock.PatternLock._pattern_target_is_held=held
    native_red=tracing.final_target_is_red
    def final_red(env):
        if getattr(env,'task_name',None)!='PatternLock':return native_red(env)
        dwell=env._pattern_dwell
        target=env.buttons_grid[dwell.required_activated[-1]]
        highlight=getattr(env,'_highlight_cache',{}).get(id(target),{})
        return bool(highlight.get('is_highlighted') and highlight.get('highlight_disk') is not None
                    and highlight.get('highlight_color_key')==tracing.RED_TARGET)
    tracing.final_target_is_red=final_red

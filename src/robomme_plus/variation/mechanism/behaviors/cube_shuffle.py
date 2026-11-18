"""ButtonUnmaskSwap: visible cluster shuffles after the first press."""
from itertools import combinations
import numpy as np
from .motion import rng_for,path_clear,minimum_gap


def plan_round(points,radius,rng,bounds,circles,duration_steps=64):
    from robomme_plus.tasks.reference.utils.highlight_motion import group_path
    # Several disjoint local clusters move at once, as in PickHighlight.
    # Endpoints are existing free slots, so covers still fit after each round.
    n=len(points);best=None
    for attempt in range(18):
        path=np.repeat(points[None],41,axis=0);groups=[];unused=set(range(n))
        candidates=list(combinations(range(n),3))+list(combinations(range(n),2))
        rng.shuffle(candidates)
        for group in candidates:
            if not set(group)<=unused:continue
            ids=list(group);p=points[ids]
            if np.linalg.norm(p[:,None]-p[None],axis=-1).max()>.29:continue
            angular=np.argsort(np.arctan2(*(p-p.mean(axis=0)).T[::-1]))
            assignment=np.empty(len(ids),int)
            assignment[angular]=np.roll(angular,int(rng.choice([-1,1])))
            goal=p[assignment]
            if np.linalg.norm(goal-p,axis=1).min()<.065:continue
            for k in range(36):
                trial=path.copy();trial[:,ids]=group_path(p,goal,rng,arc=k%2==0)
                if path_clear(trial,np.full(n,radius),bounds=bounds,circles=circles,margin=.008):
                    path=trial;groups.append(ids);unused-=set(ids);break
        score=sum(map(len,groups))
        time=np.linspace(0,40,duration_steps+1);lo=np.floor(time).astype(int);hi=np.minimum(lo+1,40)
        sampled=path[lo]*(1-(time-lo)[:,None,None])+path[hi]*(time-lo)[:,None,None]
        if not path_clear(sampled,np.full(n,radius),bounds=bounds,circles=circles,margin=.008):continue
        if best is None or score>best[0]:best=(score,sampled,groups)
        if score>=n-1 and len(groups)>=2:break
    if best is None:raise RuntimeError('No collision-free shuffle at requested speed')
    score,path,groups=best
    if len(groups)<2 or score<min(6,n-1):
        raise RuntimeError(f'Insufficient safe visible shuffle: {score}/{n} cubes in {len(groups)} clusters')
    return path,dict(clusters=groups,moving_cubes=score,duration_steps=duration_steps,
        minimum_pair_surface_gap_m=minimum_gap(path,np.full(n,radius)),
        maximum_step_m=float(np.linalg.norm(np.diff(path,axis=0),axis=-1).max()))


def install(seed,config):
    from robomme_plus.tasks.permanence.utils import tasks
    from robomme_plus.common.sim import position,set_position,step_number
    original_setup,original_update=tasks.setup,tasks.update
    is_button=config['task']=='ButtonUnmaskSwap'
    config.update(visible_shuffle_rounds=[2,3],first_press_starts_shuffle=is_button,
                  cover_after_shuffle=True,pre_mask_shuffle=True,
                  shuffle_speed_multiplier=2 if is_button else 1,
                  shuffle_control_steps=32 if is_button else 64)

    def setup(env,settings):
        original_setup(env,settings)
        if is_button:
            from .button_deadline import apply
            apply(env,config)
        route=env._limit_routes;initial=np.asarray(route.paths[0]).copy();points=initial.copy()
        rng=rng_for(seed,'visible-before-cover');count=int(rng.integers(2,4))
        obstacles=[(position(b,False)[:2],.055) for b in tasks.buttons(env)]
        bounds=np.asarray(route.metadata.get('initial_bounds_xy_m',[[-.31,.12],[-.32,.32]]))
        paths=[];events=[]
        for r in range(count):
            path,event=plan_round(points,np.sqrt(2)*env.cube_half_size/1.2,rng,bounds,obstacles,
                                  duration_steps=32 if is_button else 64)
            paths.extend(path[1:]);events.append(dict(round=r+1,**event));points=path[-1]
            paths.extend([points.copy() for _ in range(4 if is_button else 8)])
        audit=dict(kind='visible multi-cluster swaps before masking',rounds=count,events=events,
            started_step=None,covered_step=None,phase='waiting_for_first_button' if is_button else 'initial_reveal',
            cube_identities_preserved=True,container_start_slots_preserved=True)
        route.metadata['pre_mask_shuffle']=audit
        env._visible_shuffle=dict(path=np.asarray(paths),cursor=0,audit=audit,initial=initial,
                                  start_after=int(step_number(env))+env._limit_reveal_steps)
        route.active=False
        for cube,i,z in env._limit_hidden:
            set_position(cube,[*initial[i],z]);env._permanence_cube_spawned.add(cube.name)

    def update(env):
        state=env._visible_shuffle;audit=state['audit']
        if audit['covered_step'] is not None:return original_update(env)
        if is_button:tasks.observe_buttons(env)
        route=env._limit_routes;route.active=False;now=int(step_number(env))
        if env._limit_failed:return
        triggered=bool(env._permanence_presses) if is_button else now>=state['start_after']
        if triggered and audit['started_step'] is None:audit.update(started_step=now,phase='visible_cluster_swaps')
        points=state['initial']
        if audit['started_step'] is not None and state['cursor']<len(state['path']):
            points=state['path'][state['cursor']];state['cursor']+=1
        for cube,i,z in env._limit_hidden:set_position(cube,[*points[i],z])
        if state['cursor']==len(state['path']):
            # Cubes finish in existing container slots. Change the truthful
            # cube-to-cover mapping, retaining native container trajectories.
            distance=np.linalg.norm(points[:,None]-state['initial'][None],axis=-1)
            permutation=distance.argmin(axis=1)
            assert len(set(permutation.tolist()))==len(points) and distance[np.arange(len(points)),permutation].max()<1e-6
            env._limit_hidden=[(cube,int(permutation[i]),z) for cube,i,z in env._limit_hidden]
            env._permanence_permutation=[i for cube,i,z in env._limit_hidden]
            mapping={route.actors[index].name:color for (cube,index,z),color in zip(env._limit_hidden,env._repeat_target_colors)}
            env._repeat_progress.colors_by_actor.clear();env._repeat_progress.colors_by_actor.update(mapping)
            for i,actor in enumerate(route.actors):
                set_position(actor,[*state['initial'][i],route.z[i]],route.q[i]);env._permanence_cover_spawned.add(actor.name)
            audit.update(covered_step=now,phase='covered_motion',final_slot_permutation=permutation.tolist(),
                         colors_by_container=dict(mapping),container_motion_mode=config.get('container_motion','current transferred style'))
            env._limit_reveal_start=now-env._limit_reveal_steps
        env._limit_diagnostics['routes']=route.metadata

    tasks.setup,tasks.update=setup,update

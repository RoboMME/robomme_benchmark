"""fast visible shuffles for the two stationary Unmask tasks."""
from itertools import combinations
import numpy as np
from .motion import rng_for, path_clear, minimum_gap


def round_path(points, radius, rng, bounds, circles, boards, indices, duration_steps=40):
    from robomme_plus.tasks.reference.utils.highlight_motion import group_path
    n=len(points)
    def valid(path):
        return all(np.all(np.linalg.norm(np.maximum(np.abs(path-center)-extent,0),axis=-1)>radius+.006) for center,extent in boards)
    base=np.repeat(points[None],2,axis=0)
    if not path_clear(base,np.full(n,radius),bounds=bounds,circles=circles,valid=valid,margin=.008):
        raise RuntimeError(f'Initial shuffle geometry: gap={minimum_gap(base,np.full(n,radius))}, bounds={bounds.tolist()}, extents={[points.min(0).tolist(),points.max(0).tolist()]}, boards_valid={valid(base)}, circles={[(np.asarray(p).tolist(),r) for p,r in circles]}')
    best=None
    for attempt in range(32):
        path=np.repeat(points[None],41,axis=0);unused=set(indices);groups=[]
        candidates=list(combinations(indices,3))+list(combinations(indices,2));rng.shuffle(candidates)
        for group in candidates:
            if not set(group)<=unused:continue
            ids=list(group);p=points[ids]
            if np.linalg.norm(p[:,None]-p[None],axis=-1).max()>.34:continue
            angular=np.argsort(np.arctan2(*(p-p.mean(axis=0)).T[::-1]))
            assignment=np.empty(len(ids),int);assignment[angular]=np.roll(angular,int(rng.choice([-1,1])))
            for k in range(12):
                trial=path.copy();trial[:,ids]=group_path(p,p[assignment],rng,arc=k%2==0)
                if path_clear(trial,np.full(n,radius),bounds=bounds,circles=circles,valid=valid,margin=.008):
                    path=trial;groups.append(ids);unused-=set(ids);break
        score=sum(map(len,groups))
        sampled=path[::40//duration_steps]
        if not path_clear(sampled,np.full(n,radius),bounds=bounds,circles=circles,valid=valid,margin=.008):continue
        if not valid((sampled[:-1]+sampled[1:])/2):continue
        if best is None or score>best[0]:best=(score,sampled,groups)
        if score>=len(indices)-2 and len(groups)>=2:break
    if best is None:raise RuntimeError('No collision-free shuffle at requested speed')
    score,path,groups=best
    if len(groups)<2:raise RuntimeError(f'Cannot find two collision-free cube clusters ({score}/{n})')
    return path,dict(clusters=groups,moving_cubes=score,duration_steps=duration_steps,
        minimum_pair_surface_gap_m=minimum_gap(path,np.full(n,radius)),
        maximum_step_m=float(np.linalg.norm(np.diff(path,axis=0),axis=-1).max()))


def install(task,seed,config):
    from robomme_plus.tasks.permanence.utils import tasks
    from robomme_plus.common.sim import position,set_position,step_number
    original_setup,original_finished=tasks.setup,tasks.motion_finished
    is_button=task=='ButtonUnmask'
    config.update(style='fast visible cube cluster swaps then stationary covers',
        visible_shuffle_rounds=[1,2] if is_button else [2,3],first_press_starts_shuffle=is_button,
        pre_mask_shuffle=True,quick_mask_steps=1,shuffle_control_steps=20 if is_button else 40,
        shuffle_speed_multiplier=2 if is_button else 1)

    def setup(env,settings):
        original_setup(env,settings)
        route=env._limit_routes;initial=np.asarray(route.paths[0]).copy();points=initial.copy()
        rng=rng_for(seed,'static-unmask-fast-shuffle');count=int(rng.integers(1,3) if is_button else rng.integers(2,4))
        bounds=np.asarray(env._limit_diagnostics['static_layout']['bounds_xy_m']).copy()
        # Native stationary bounds constrain centers; path_clear expects physical edges.
        bounds+=np.array([-1.,1.])[None,:]*(np.sqrt(2)*env.cube_half_size/1.2)
        circles=[(position(b,False)[:2],.045) for b in tasks.buttons(env)]
        boards=[]
        if settings.get('blocking'):
            for description in env._limit_diagnostics['occluder']['board_descriptions']:
                center=np.asarray(description['center_xyz_m'])[:2].copy()
                extent=np.asarray(description['size_xyz_m'])[:2].copy()/2
                if is_button:extent[1]+=abs(center[1]);center[1]=0
                boards.append((center,extent))
        indices=[i for cube,i,z in env._limit_hidden]
        paths=[];events=[]
        for r in range(count):
            path,event=round_path(points,np.sqrt(2)*env.cube_half_size/1.2,rng,bounds,circles,boards,indices,
                                  duration_steps=20 if is_button else 40)
            paths.extend(path[1:]);points=path[-1];events.append(dict(round=r+1,**event))
            if r<count-1:paths.extend([points.copy() for _ in range(2 if is_button else 3)])
        audit=dict(kind='fast visible multi-cluster swaps before stationary masking',rounds=count,events=events,
            started_step=None,covered_step=None,phase='waiting_for_button' if is_button else 'initial_reveal',
            native_button_deadline_steps=settings.get('button_deadline_steps'),cube_identities_preserved=True,
            container_start_slots_preserved=True)
        route.metadata['pre_mask_shuffle']=audit
        env._static_shuffle=dict(path=np.asarray(paths),cursor=0,audit=audit,initial=initial,
            start_after=int(step_number(env))+36)
        for cube,i,z in env._limit_hidden:
            set_position(cube,[*initial[i],z]);env._permanence_cube_spawned.add(cube.name)

    def finished(env):
        state=getattr(env,'_static_shuffle',None)
        return state['audit']['covered_step'] is not None if state is not None else original_finished(env)

    def update(env):
        tasks.observe_buttons(env);tasks.check_early_opening(env)
        state=env._static_shuffle;audit=state['audit'];now=int(step_number(env));route=env._limit_routes
        if audit['covered_step'] is not None or env._limit_failed:return
        triggered=bool(env._permanence_presses) if is_button else now>=state['start_after']
        if triggered and audit['started_step'] is None:audit.update(started_step=now,phase='visible_cluster_swaps')
        points=state['initial']
        if audit['started_step'] is not None and state['cursor']<len(state['path']):
            points=state['path'][state['cursor']];state['cursor']+=1
        for cube,i,z in env._limit_hidden:set_position(cube,[*points[i],z])
        if state['cursor']==len(state['path']):
            distance=np.linalg.norm(points[:,None]-state['initial'][None],axis=-1);permutation=distance.argmin(axis=1)
            assert len(set(permutation.tolist()))==len(points) and distance[np.arange(len(points)),permutation].max()<1e-6
            env._limit_hidden=[(cube,int(permutation[i]),z) for cube,i,z in env._limit_hidden]
            env._permanence_permutation=[i for cube,i,z in env._limit_hidden]
            mapping={route.actors[index].name:color for (cube,index,z),color in zip(env._limit_hidden,env._repeat_target_colors)}
            env._repeat_progress.colors_by_actor.clear();env._repeat_progress.colors_by_actor.update(mapping)
            for i,actor in enumerate(route.actors):
                set_position(actor,[*state['initial'][i],route.z[i]],route.q[i]);env._permanence_cover_spawned.add(actor.name)
            env._permanence_cube_released=True
            audit.update(covered_step=now,phase='masked_stationary',final_slot_permutation=permutation.tolist(),colors_by_container=dict(mapping))
        env._limit_diagnostics['routes']=route.metadata
        env._limit_diagnostics['cube_release_complete']=env._permanence_cube_released

    tasks.setup,tasks.update,tasks.motion_finished=setup,update,finished

"""Color-only task materials; activation state and scoring stay native."""
import colorsys

import numpy as np

from .utils import rng_for,surfaces


CUE_TASKS=('SwingXtimes','PatternLock','RouteStick')


def color(seed,category):
    rng=rng_for(seed,'object-color/'+category)
    if category=='container':
        shade=float(rng.uniform(.12,.45))
        return [shade]*3
    return list(colorsys.hsv_to_rgb(float(rng.random()),float(rng.uniform(.55,.9)),float(rng.uniform(.45,.9))))


def cube_color(seed,original,*,preserve_hue=False):
    """Blend native and varied shades when distractors require stable color names."""
    key=','.join(f'{float(c):.5f}' for c in original)
    rng=rng_for(seed,'cube-shade/'+key)
    h,s,v=colorsys.rgb_to_hsv(*original)
    if s<.15:
        shade=float(rng.uniform(.82,.98) if v>.7 else rng.uniform(.28,.5))
        return [shade]*3
    varied=list(colorsys.hsv_to_rgb((h+float(rng.uniform(-.012,.012)))%1,
                                    float(rng.uniform(.65,.95)),float(rng.uniform(.65,.95))))
    if preserve_hue:
        return ((np.asarray(original)+varied)*.5).tolist()
    return varied


def tool_color(seed,original):
    # A common channel permutation and complement are an RGB isometry:
    # distinct parts retain exactly their original pairwise color distances.
    rng=rng_for(seed,'tool-color-isometry')
    order=rng.permutation(3)
    return (1.-np.asarray(original)[order]).tolist()


def tool_link_surfaces(articulation_name,link):
    if not articulation_name.startswith('panda_split_stick'):
        return list(surfaces(link))
    name=link.get_name()
    if name.startswith('stick_'):
        return list(surfaces(link))
    if name!='panda_hand':
        return []
    # The upper stick cylinder shares panda_hand with the hand mesh.
    import sapien.render as render
    selected=[]
    for body in link._objs:
        component=body.entity.find_component_by_type(render.RenderBodyComponent)
        if component is not None:
            selected.extend(shape for shape in component.render_shapes
                            if isinstance(shape,render.RenderShapeCylinder) and abs(shape.radius-.008)<1e-6)
    return selected


def cue_palette(seed):
    rng=rng_for(seed,'target-feedback-colors')
    h=float(rng.random())
    active=list(colorsys.hsv_to_rgb(h,float(rng.uniform(.7,.95)),float(rng.uniform(.85,1))))
    return dict(active=active,secondary=[1.,1.,1.],preserve_native_idle=True)


def category(name):
    lower=name.lower()
    if lower.startswith(('highlight','position_highlight','target_','temp_target','goal_site','home_site_','counting_target_')):
        return None
    if 'cube' in lower:
        return 'cube'
    if any(word in lower for word in ('cover','container')) or lower.startswith(('bin_','swarm_extra')):
        return 'container'
    if 'board' in lower or 'box' in lower:
        return 'board_or_bin'
    if any(word in lower for word in ('peg','ball','hammer','stick','tool','rod')):
        return 'tool'
    return None


def actor_category(native,name,actor):
    if native.task_name=='StopCube' and actor is getattr(native,'cube',None):
        return 'moving_cube'
    if native.task_name=='RouteStick' and any(actor is a for a in getattr(native,'cubes_on_targets',[])):
        return 'board_or_bin'
    return category(name)


def has_other_cube_colors(native,baselines):
    """Inspect native cube colors, including cubes hidden until later reveals."""
    for name,actor in native.scene.actors.items():
        if actor_category(native,name,actor)!='cube':
            continue
        for index,surface in enumerate(surfaces(actor)):
            rgb=baselines.get((name,index),np.asarray(surface.material.base_color)[:3])
            primary=np.eye(3)[int(np.argmax(rgb))]*max(rgb)
            if max(rgb)<=0 or not np.allclose(rgb,primary,atol=1e-6,rtol=0):
                return True
    return False


def paint_ring(actor,primary,secondary):
    bands=list(surfaces(actor))
    assert len(bands)==5,(actor.name,len(bands))
    for index,surface in enumerate(bands):
        surface.material.base_color=[*(primary if index%2==0 else secondary),1.]
        surface.material.base_color_texture=None


def cue_targets(native):
    if native.task_name in ('PatternLock','RouteStick'):
        return list(native.buttons_grid)
    if native.task_name=='SwingXtimes':
        return [native.target_right,native.target_left]
    return []


def target_rings(native):
    """Resolve task roles, excluding hidden helpers and non-ring targets."""
    import sapien.render as render
    hidden={id(actor) for actor in getattr(native,'_hidden_objects',[])}
    found={}
    for role in ('target','targets','buttons_grid','targets_grid','target_left','target_right'):
        value=getattr(native,role,None)
        values=value if isinstance(value,(list,tuple)) else [value]
        for actor in values:
            if actor is None or not hasattr(actor,'_objs') or id(actor) in hidden:
                continue
            bands=list(surfaces(actor))
            if len(bands)!=5 or not all(isinstance(b,render.RenderShapeCylinder) for b in bands):
                continue
            radii=np.asarray([b.radius for b in bands])
            if np.allclose(radii/radii.max(),[1.,.8,.6,.4,.2],atol=1e-5):
                found[id(actor)]=actor
    return list(found.values())


def button_articulations(native):
    for name,articulation in sorted(native.scene.articulations.items()):
        links=articulation.get_links()
        if name.startswith('button') or any(l.get_name()=='button_cap' for l in links):
            yield name,articulation


def apply(native,config):
    if not config['enabled']:
        return
    baselines=getattr(native,'_perceptual_material_baselines',None)
    if baselines is None:
        baselines=native._perceptual_material_baselines={}
    state=getattr(native,'_perceptual_object_colors',None)
    if state is None:
        state=native._perceptual_object_colors=dict(seen=set(),changes={},cue_ids=set(),ring_changes={},
                                                   preserve_cube_hues=has_other_cube_colors(native,baselines))
    for name,actor in sorted(native.scene.actors.items()):
        group=actor_category(native,name,actor)
        if group is None or group not in config['groups'] or id(actor) in state['seen']:
            continue
        state['seen'].add(id(actor));changes=[]
        original=[(surface,np.asarray(surface.material.base_color)[:3].copy()) for surface in surfaces(actor)]
        for index,(surface,rgb) in enumerate(original):
            # BinFill's separate black floor panel is not part of the wall palette.
            if native.task_name=='BinFill' and actor is getattr(native,'board_with_hole',None) and np.allclose(rgb,0.,atol=1e-8):
                continue
            baseline=baselines.setdefault((name,index),rgb.copy())
            new=(cube_color(config['appearance_seed'],baseline,preserve_hue=state['preserve_cube_hues']) if group=='cube' else
                 tool_color(config['appearance_seed'],baseline) if group=='tool' else config['colors'][group])
            surface.material.base_color=[*new,1.];surface.material.base_color_texture=None
            if group=='container':
                surface.material.normal_texture=None
                surface.material.roughness_texture=None;surface.material.metallic_texture=None
                surface.material.roughness=.85;surface.material.metallic=0.;surface.material.specular=.15
            changes.append(dict(original=rgb.tolist(),color=new))
        state['changes'][name]=dict(category=group,surfaces=changes)
    buttons={id(articulation) for _,articulation in button_articulations(native)}
    for name,articulation in sorted(native.scene.articulations.items()):
        is_button=id(articulation) in buttons
        is_tool=category(name)=='tool'
        if not is_button and not is_tool:
            continue
        for link in articulation.get_links():
            if id(link) in state['seen']:
                continue
            group=('button_cap' if 'cap' in link.get_name() else 'button_body') if is_button else 'tool'
            if group not in config['groups']:
                continue
            state['seen'].add(id(link))
            changes=[]
            selected=tool_link_surfaces(name,link) if is_tool else list(surfaces(link))
            originals=[(surface,np.asarray(surface.material.base_color)[:3].copy()) for surface in selected]
            for index,(surface,original) in enumerate(originals):
                baseline=baselines.setdefault((name,link.get_name(),index),original.copy())
                new=tool_color(config['appearance_seed'],baseline) if is_tool else config['colors'][group]
                surface.material.base_color=[*new,1.]
                surface.material.base_color_texture=None
                changes.append(dict(original=original.tolist(),color=new))
            if changes:state['changes'][name+'/'+link.get_name()]=dict(category=group,surfaces=changes)
    if native.task_name in CUE_TASKS and 'feedback' in config['groups']:
        palette=config['feedback']
        targets=cue_targets(native)
        # Native idle-ring materials are intentionally left untouched.
        for target in targets:
            entry=getattr(native,'_highlight_cache',{}).get(id(target),{})
            if entry.get('is_highlighted') and entry.get('highlight_disk') is not None:
                # Color is presentation only. Preserve the native cache key,
                # timestamps, visibility and live-actor identity used by scoring.
                paint_ring(entry['highlight_disk'],palette['active'],palette['secondary'])
                state['ring_changes'][target.name]=dict(mode='active_only',active=palette['active'],idle_unchanged=True)
    elif 'rings' in config['groups']:
        palette=config['rings']
        for target in target_rings(native):
            if id(target) not in state['cue_ids']:
                original=[np.asarray(s.material.base_color).tolist() for s in surfaces(target)]
                paint_ring(target,palette['primary'],palette['secondary']);state['cue_ids'].add(id(target))
                state['ring_changes'][target.name]=dict(mode='static',original=original,palette=palette)



STYLES=('original','teal_ivory','amber_charcoal','blue_gray','rose_cream','seeded')

def sample_palette(seed,style):
    if style not in STYLES:
        raise ValueError(style)
    if style=='original':
        return dict(style=style,primary=[160/255,32/255,240/255],secondary=[1.,1.,1.])
    selected=style
    if style=='seeded':
        selected=str(rng_for(seed,'ring-style').choice(STYLES[1:-1]))
    rng=rng_for(seed,'ring-palette/'+selected)
    hue_range=dict(teal_ivory=(.45,.51),amber_charcoal=(.07,.12),blue_gray=(.57,.65),rose_cream=(.91,.98))[selected]
    primary=list(colorsys.hsv_to_rgb(float(rng.uniform(*hue_range)),float(rng.uniform(.65,.9)),float(rng.uniform(.70,.88))))
    if selected=='amber_charcoal':
        shade=float(rng.uniform(.10,.19));secondary=[shade,shade,shade]
    else:
        secondary=list(rng.uniform([.85,.84,.79],[.98,.96,.92]))
    return dict(style=selected,primary=primary,secondary=secondary)


def configuration(seed):
    return dict(enabled=True,appearance_seed=int(seed),
                groups=['container','board_or_bin','cube','moving_cube','tool','button_body','button_cap','rings','feedback'],
                colors={k:color(seed,k) for k in ('container','board_or_bin','moving_cube','button_cap','button_body')},
                feedback=cue_palette(seed),rings=sample_palette(seed,'seeded'))

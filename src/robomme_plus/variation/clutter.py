"""Seeded, render-only tabletop props, clear of StopCube's complete route."""
import numpy as np

from .utils import rng_for


def sample_kinds(seed):
    """Choose a coaster, a block, or one of each, independently of layout RNG."""
    choices=(('coaster',),('desk_block',),('coaster','desk_block'))
    return choices[int(rng_for(seed,'stopcube-clutter-selection').integers(len(choices)))]


def sample_layout(seed,kinds,route_xy,button_xy,target_xy,cube_half_size):
    """Place distinct rectangular props without consuming the task's RNG."""
    rng=rng_for(seed,'stopcube-tabletop-clutter')
    route_xy=np.asarray(route_xy)
    props=[]
    for index,kind in enumerate(kinds):
        coaster=kind=='coaster'
        half=np.array([rng.uniform(.044,.058),rng.uniform(.029,.038),rng.uniform(.006,.009)]
                      if coaster else [rng.uniform(.034,.046),rng.uniform(.018,.022),rng.uniform(.015,.022)])
        radius=float(np.linalg.norm(half[:2]))
        for _ in range(10000):
            xy=rng.uniform([-.44+radius,-.40+radius],[.24-radius,.40-radius])
            if np.min(np.linalg.norm(route_xy-xy,axis=1))<=radius+np.sqrt(2)*cube_half_size+.045:
                continue
            if np.linalg.norm(xy-button_xy)<=radius+.10 or np.linalg.norm(xy-target_xy)<=radius+.075:
                continue
            if any(np.linalg.norm(xy-np.asarray(p['xy']))<=radius+p['radius']+.018 for p in props):
                continue
            break
        else:
            raise RuntimeError('Cannot place StopCube clutter clear of its route')
        # Elongated blocks and rectangular coasters are distinct from task cubes,
        # circular targets and push buttons, even when their colors are similar.
        color=(rng.uniform([.26,.15,.07],[.60,.43,.27]) if coaster else
               rng.uniform([.12,.18,.18],[.48,.50,.53]))
        props.append(dict(name=f'coaster_{index}' if coaster else f'desk_block_{index}',
                          xy=xy.tolist(),half_size=half.tolist(),radius=radius,
                          yaw=float(rng.uniform(-np.pi,np.pi)),color=[*color.tolist(),1.]))
    return props


def add_stopcube_clutter(native,config):
    """Return owned render entities; no actors, colliders or scoring state added."""
    import sapien
    import sapien.render as render
    from robomme_plus.common.sim import position

    route=native._counting.route
    # Four passes cover every native route shape, including both figure-eight
    # and double-ellipse lobes. Dense sampling retains a 4.5 cm safety margin.
    route_xy=np.asarray(route.center)+route.local_points(np.linspace(0,4,2049))
    props=sample_layout(config['seed'],config['clutter_kinds'],route_xy,
                        position(native.button,False)[:2],position(native.target,False)[:2],
                        native.cube_half_size)
    refs=[]
    for prop in props:
        entity=sapien.Entity();entity.name='perceptual_'+prop['name']
        body=render.RenderBodyComponent()
        material=render.RenderMaterial(base_color=prop['color'],roughness=.85)
        body.attach(render.RenderShapeBox(np.asarray(prop['half_size']),material))
        entity.add_component(body)
        yaw=prop['yaw']
        entity.pose=sapien.Pose([*prop['xy'],prop['half_size'][2]],
                                [np.cos(yaw/2),0,0,np.sin(yaw/2)])
        native.scene.sub_scenes[0].add_entity(entity)
        refs.append(entity)
    return refs,props

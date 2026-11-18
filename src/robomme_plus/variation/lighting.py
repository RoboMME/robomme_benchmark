"""Seeded render lights and continuous motion, with no flashing."""
import copy
import hashlib
from pathlib import Path
import shutil
import colorsys
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from .utils import rng_for,surfaces
from .objects import target_rings

LIGHTING=('dim','static_color','moving_color','disco','ring_pattern','overexposure','moving_patches')
ANIMATED=('moving_color','disco','ring_pattern','overexposure','moving_patches')


def sample_lighting(seed,mode):
    """All setup/motion draws occur once, without touching the simulator's RNG."""
    rng=rng_for(seed,'lighting/'+mode)
    if mode=='normal':
        return dict(mode=mode,seed=int(seed),original_lighting=True,lights=[])
    dim=mode=='dim'
    ambient=float(rng.uniform(.045,.085) if dim else rng.uniform(.13,.20))
    key=float(rng.uniform(.18,.30) if dim else rng.uniform(.4,.65))
    config=dict(mode=mode,seed=int(seed),ambient=[ambient,ambient,ambient*1.04],
                key=[key,key,key],key_direction=[float(rng.uniform(.2,.7)),float(rng.uniform(-.8,-.3)),-1.],
                lights=[])
    if dim:
        return config
    # Native shader supports one textured projector, with many rings in its image.
    count=(1 if mode in ('ring_pattern','moving_patches') else 3 if mode in ('static_color','moving_color') else 2 if mode=='overexposure' else int(rng.integers(5,8)))
    hue_offset=float(rng.random())
    for i in range(count):
        hue=(hue_offset+i/count+float(rng.uniform(-.04,.04)))%1
        broad=mode in ('static_color','moving_color')
        outer=float(rng.uniform(.42,.65) if mode=='overexposure' else rng.uniform(1.3,1.55) if mode=='moving_patches' else rng.uniform(.65,.90) if broad or mode=='ring_pattern' else rng.uniform(.18,.28))
        intensity=float(rng.uniform(5,8) if mode=='ring_pattern' else rng.uniform(2.0,3.2))
        config['lights'].append(dict(
            kind='projector' if mode in ('ring_pattern','moving_patches') else 'spot',
            origin=[float(rng.uniform(-.12,.10)),float(rng.uniform(-.14,.14)),float(rng.uniform(.75,.95))],
            center=[float(rng.uniform(-.16,-.04)),float(rng.uniform(-.05,.05)),0.],
            radius=[float(rng.uniform(.20,.34)),float(rng.uniform(.23,.36))],
            periods=[float(rng.uniform(7,14)),float(rng.uniform(8,16))],
            phase=[float(rng.uniform(0,2*np.pi)),float(rng.uniform(0,2*np.pi))],
            roll_phase=float(rng.uniform(0,2*np.pi)),roll_speed=float(rng.uniform(-.3,.3)),
            color=[1.,1.,1.] if mode in ('ring_pattern','moving_patches') else list(colorsys.hsv_to_rgb(hue,float(rng.uniform(.7,.98)),1)),
            intensity=intensity,intensity_period=float(rng.uniform(5,12)),
            intensity_phase=float(rng.uniform(0,2*np.pi)),
            inner_fov=outer*float(rng.uniform(.55,.72)),outer_fov=outer,
            hue_speed=float(rng.uniform(-.06,.06)) if mode=='moving_color' else 0.,hue=hue))
        if mode=='moving_patches':
            light=config['lights'][-1]
            light['inner_fov']=outer*.82
            light['intensity']=float(rng.uniform(5.,8.))
            light['radius']=[float(rng.uniform(.12,.22)),float(rng.uniform(.16,.25))]
            light['patch_count']=int(rng_for(seed,'patch-count').integers(85,121))
        if mode=='overexposure':
            light=config['lights'][-1]
            light['color']=[1.,float(rng.uniform(.96,1.)),float(rng.uniform(.94,1.))]
            light['intensity']=float(rng.uniform(24,40))
            light['constant_intensity']=True
    return config


def light_values(plan,light,seconds):
    t=float(seconds) if plan['mode'] in ANIMATED else 0.
    center=np.asarray(light['center'])
    target=center+np.array([light['radius'][0]*np.sin(2*np.pi*t/light['periods'][0]+light['phase'][0]),
                            light['radius'][1]*np.sin(2*np.pi*t/light['periods'][1]+light['phase'][1]),0.])
    intensity=light['intensity']*(.85+.15*np.sin(2*np.pi*t/light['intensity_period']+light['intensity_phase']))
    if light.get('constant_intensity'):
        intensity=light['intensity']
    color=np.asarray(light['color'])
    if light['hue_speed']:
        color=np.asarray(colorsys.hsv_to_rgb((light['hue']+t*light['hue_speed'])%1,.85,1.))
    return target,(color*intensity),light['roll_phase']+t*light['roll_speed']


def projected_patch_image(path,seed,count,palette):
    """Many independently placed colored regions, with no background tint."""
    rng=rng_for(seed,'projected-color-patches')
    image=Image.new('RGB',(1024,1024),(0,0,0));draw=ImageDraw.Draw(image)
    palette=[tuple(round(c*255) for c in rgb) for rgb in palette]
    offset=int(rng.integers(len(palette)))
    for index in range(count):
        x,y=rng.integers(-50,1024,2)
        width,height=rng.integers(60,155,2)
        draw.ellipse((x,y,x+width,y+height),fill=palette[(index+offset)%len(palette)])
    image.filter(ImageFilter.GaussianBlur(8)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_palette(native,objects):
    from .objects import category,CUE_TASKS
    hidden={id(a) for a in getattr(native,'_hidden_objects',[])}
    ring_ids={id(a) for a in target_rings(native)}
    colors=set()
    for name,actor in native.scene.actors.items():
        if id(actor) in hidden or (id(actor) not in ring_ids and category(name) is None):
            continue
        for surface in surfaces(actor):
            rgb=np.asarray(surface.material.base_color)[:3]
            if np.ptp(rgb)>.2 and rgb.max()>.35:colors.add(tuple(np.round(rgb,6)))
    if native.task_name in CUE_TASKS:
        colors.add(tuple(objects['feedback']['active']))
    return [list(c) for c in sorted(colors)] or [objects['rings']['primary']]


def projector_shader(asset_dir):
    """Use an instance camera config; change only RGB projector sampling."""
    import sapien
    import sapien.render as render
    from mani_skill.render.shaders import PREBUILT_SHADER_CONFIGS
    source=Path(sapien.__file__).resolve().parent/'vulkan_shader'
    destination=asset_dir/'rgb_projector_shader'
    for folder in ('minimal','common'):
        shutil.copytree(source/folder,destination/folder)
    file=destination/'minimal/gbuffer.frag'
    old='float visibility = texture(samplerTexturedLightTextures[i], shadowMapCoord.xy).x;'
    text=file.read_text()
    if text.count(old)!=1:
        raise RuntimeError('Unsupported SAPIEN projector shader')
    file.write_text(text.replace(old,'vec3 visibility = texture(samplerTexturedLightTextures[i], shadowMapCoord.xy).rgb;'))
    config=copy.deepcopy(PREBUILT_SHADER_CONFIGS['minimal'])
    config.shader_pack=str(destination/'minimal')
    render.set_picture_format('Color','r8g8b8a8unorm')
    render.set_picture_format('ColorRaw','r8g8b8a8unorm')
    render.set_picture_format('PositionSegmentation','r16g16b16a16sint')
    return config


def projected_ring_image(path,seed,palette,*,white=False):
    rng=rng_for(seed,'ring-projector')
    image=Image.new('RGB',(1024,1024),(0,0,0));draw=ImageDraw.Draw(image)
    spacing=int(rng.integers(170,210));offset=rng.integers(-spacing//2,spacing//2,2)
    for row in range(-1,10):
        for col in range(-1,10):
            x=col*spacing+(row%2)*spacing//2+offset[0];y=row*spacing+offset[1]
            radius=float(rng.uniform(.28,.38)*spacing)
            primary=tuple(round(c*255) for c in palette[(row*11+col)%len(palette)])
            if white:
                primary=(255,255,255)
            for band in range(5):
                r=radius*(5-band)/5
                gap=(0,0,0) if white else (245,243,237)
                draw.ellipse((x-r,y-r,x+r,y+r),fill=primary if band%2==0 else gap)
    image.save(path)


def add_lights(native,plan,objects,asset_dir,seed):
    import sapien
    import sapien.render as render
    mode=plan['mode'];palette=scene_palette(native,objects)
    normalized=[(np.asarray(rgb)/max(rgb)).tolist() for rgb in palette]
    order=rng_for(seed,'scene-light-palette').permutation(len(normalized))
    normalized=[normalized[int(i)] for i in order]
    plan['source_palette_rgb']=palette
    if mode in ('moving_color','disco','moving_patches'):
        for i,params in enumerate(plan['lights']):
            if params['kind']!='projector':params['color']=normalized[i%len(normalized)];params['hue_speed']=0.
    projector_file=asset_dir/'projector.png'
    if mode=='moving_patches':
        projected_patch_image(projector_file,seed,plan['lights'][0]['patch_count'],normalized)
    elif mode=='ring_pattern':
        projected_ring_image(projector_file,seed,normalized,white=plan.get('white_rings',False))
    refs=[];lights=[]
    for params in plan['lights']:
        target,color,roll=light_values(plan,params,0.)
        if params['kind']=='projector':
            entity=sapien.Entity();entity.name='perceptual_projector'
            component=render.RenderTexturedLightComponent()
            component.texture=render.RenderTexture2D(str(projector_file));refs.append(component.texture)
            component.inner_fov=params['inner_fov'];component.outer_fov=params['outer_fov']
            component.color=color;component.shadow=True
            component.shadow_near=.05;component.shadow_far=2.;component.shadow_map_size=512
            entity.add_component(component);native.scene.sub_scenes[0].add_entity(entity)
            refs.append(entity)
        else:
            component=native.scene.sub_scenes[0].add_spot_light(
                params['origin'],target-np.asarray(params['origin']),inner_fov=params['inner_fov'],
                outer_fov=params['outer_fov'],color=color,shadow=True,
                shadow_near=.05,shadow_far=2.,shadow_map_size=512)
        lights.append((component,params))
    return lights,refs


def update_lights(plan,lights,seconds):
    import sapien
    for component,params in lights:
        target,color,roll=light_values(plan,params,seconds)
        direction=target-np.asarray(params['origin'])
        align=sapien.Pose(params['origin'],sapien.math.shortest_rotation([1,0,0],direction))
        twist=sapien.Pose(q=[np.cos(roll/2),np.sin(roll/2),0,0])
        component.pose=align*twist;component.color=color

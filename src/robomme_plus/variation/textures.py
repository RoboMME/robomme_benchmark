"""Natural table color maps; no geometry or physical material changes."""
import hashlib
import numpy as np
from PIL import Image
from .utils import rng_for,surfaces


def texture(path, kind, seed):
    """Neutral, seeded material color maps; no moving pigment or target palette."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:1024, :1024]
    if kind == 'steel':
        # Long fine parallel brushing with subtle neutral variation.
        grain = rng.normal(0, 2.3, (1024, 1024))
        streak = np.asarray(Image.fromarray(np.clip(128+grain*8,0,255).astype('uint8'))
                            .resize((32,1024)).resize((1024,1024),Image.Resampling.BILINEAR)).astype(float)-128
        value = 145 + .4*streak + grain
        rgb = np.stack([value*.98,value,value*1.025],axis=-1)
    elif kind == 'marble':
        # Warped grey veins in an off-white stone. Independent smooth fields
        # vary vein widths and keep the repeated texture from looking tiled.
        coarse = Image.fromarray(rng.integers(0,256,(24,24),dtype=np.uint8))
        field = np.asarray(coarse.resize((1024,1024),Image.Resampling.BICUBIC)).astype(float)/255
        phase = x/75 + y/260 + 1.8*np.sin(y/170) + 1.6*field
        veins = np.exp(-(np.sin(phase)/.11)**2)
        thin = np.exp(-(np.sin(phase*2.7+.8)/.045)**2)
        value = 218 + 7*(field-.5) - 80*veins - 18*thin + rng.normal(0,.8,x.shape)
        rgb = np.stack([value*1.015,value,value*.97],axis=-1)
    elif kind == 'concrete':
        coarse = Image.fromarray(rng.integers(0,256,(32,32),dtype=np.uint8))
        clouds = np.asarray(coarse.resize((1024,1024),Image.Resampling.BICUBIC)).astype(float)-128
        fine = rng.normal(0,2.2,x.shape)
        pores = (rng.random(x.shape)<.018)*rng.uniform(5,24,x.shape)
        value = 155+.1*clouds+fine-pores
        rgb = np.stack([value,value,value*.985],axis=-1)
    else:
        raise ValueError(kind)
    Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_table(native,material,asset_dir,seed):
    if material=='wood':
        return None
    import sapien.render as render
    table=native.scene.actors['table-workspace']
    file=asset_dir/(material+'.png')
    if material=='light_oak':
        sources=[s.material.base_color_texture for s in surfaces(table) if s.material.base_color_texture is not None]
        data=max(sources,key=lambda t:t.width*t.height).download()[:,:,:3].astype(float)
        luma=data@np.array([.2126,.7152,.0722])
        tone=rng_for(seed,'material/light_oak').uniform([197,175,136],[214,191,150])
        rgb=np.clip((.6+.4*luma/max(float(np.median(luma)),1.))[...,None]*tone,0,255).astype(np.uint8)
        Image.fromarray(rgb).save(file)
    else:
        texture(file,material,int(rng_for(seed,'material/'+material).integers(0,2**32)))
    tex=render.RenderTexture2D(str(file))
    for surface in surfaces(table):
        m=surface.material;m.base_color=[1.,1.,1.,1.];m.base_color_texture=tex
        if material!='light_oak':m.normal_texture=None
        m.roughness_texture=None;m.metallic_texture=None
        m.roughness=dict(light_oak=.68,steel=.38,marble=.48,concrete=.95)[material]
        m.metallic=.7 if material=='steel' else 0.
        m.specular=dict(light_oak=.18,steel=.45,marble=.3,concrete=.05)[material]
    return tex

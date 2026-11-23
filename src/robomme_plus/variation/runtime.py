"""Instance-local rendering lifecycle; no task/controller monkey patches."""
from pathlib import Path
from tempfile import TemporaryDirectory

from . import configuration
from . import objects,lighting
from .sensors import SensorNoise
from .textures import apply_table
from .utils import rng_for,surfaces


class PerceptualVariation:
    def __init__(self,task,seed,profile):
        self.config=configuration(task,seed,profile)
        self.objects=objects.configuration(seed)
        self.atomic=isinstance(profile,dict)
        if self.atomic:
            self.objects['groups']=self.config['object_groups']
            self.objects['enabled']=bool(self.objects['groups'])
            if 'container_color' in self.config:
                self.objects['colors']['container']=self.config['container_color']
            self.objects['colors'].update(self.config.get('object_colors', {}))
        self.plan=lighting.sample_lighting(seed,self.config['lighting'])
        if self.config.get('white_rings'):
            self.plan['white_rings']=True
        self.noise=None if self.atomic else SensorNoise(seed)
        self.directory=TemporaryDirectory(prefix='robomme-plus-variation-')
        self.assets=Path(self.directory.name)
        self.scene_token=None
        self.lights=[];self.refs=[]
        self.clutter=[]

    def sensor_options(self,kwargs):
        if self.config['lighting'] not in ('moving_patches','ring_pattern'):
            return kwargs
        kwargs=dict(kwargs)
        configs=dict(kwargs.get('sensor_configs') or {})
        configs['shader_config']=lighting.projector_shader(self.assets)
        kwargs['sensor_configs']=configs
        return kwargs

    def reset(self,native):
        if self.noise is not None:
            self.noise.counter=0
        if hasattr(native,'_perceptual_object_colors'):
            del native._perceptual_object_colors

    def load_lighting(self,native):
        if self.config['lighting']=='normal':
            return False
        native.scene.set_ambient_light(self.plan['ambient'])
        native.scene.add_directional_light(self.plan['key_direction'],self.plan['key'],
                                          shadow=native.enable_shadow,shadow_scale=5,shadow_map_size=1024)
        return True

    def apply(self,native):
        objects.apply(native,self.objects)
        token=id(native.scene.sub_scenes[0])
        if self.scene_token!=token:
            self.scene_token=token
            self.refs=[]
            table=apply_table(native,self.config['table'],self.assets,self.config['seed'])
            if table is not None:self.refs.append(table)
            if native.task_name=='StopCube' and not self.atomic:
                from .clutter import add_stopcube_clutter
                refs,self.clutter=add_stopcube_clutter(native,self.config)
                self.refs.extend(refs)
            if native.task_name=='BinFill' and not self.atomic:
                bases=dict(wood=[.42,.28,.17],light_oak=[.72,.62,.46],steel=[.50,.51,.53],
                           marble=[.79,.78,.76],concrete=[.56,.56,.55])
                rng=rng_for(self.config['seed'],'conveyor-table-color')
                color=(rng.uniform(.9,1.1,size=3)*bases[self.config['table']]).tolist()
                for name,actor in native.scene.actors.items():
                    if name=='counting_conveyor_belt' or name.startswith('belt_tread_'):
                        for surface in surfaces(actor):
                            material=surface.material
                            material.base_color=[*color,1.];material.base_color_texture=None
                            material.normal_texture=None;material.roughness_texture=None;material.metallic_texture=None
                            material.roughness=.85;material.metallic=0.;material.specular=.15
            self.lights,refs=lighting.add_lights(native,self.plan,self.objects,self.assets,self.config['seed'])
            self.refs.extend(refs)
        seconds=float(native.elapsed_steps.reshape(-1)[0].item())*float(native.control_timestep)
        lighting.update_lights(self.plan,self.lights,seconds)

    def close(self):
        self.directory.cleanup()

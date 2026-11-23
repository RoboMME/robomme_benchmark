"""Perception selection and public sensor contracts without a simulator."""
import ast
from collections import Counter
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from robomme_plus import BenchmarkEnvBuilder
from robomme_plus.evaluation.metadata import normalize_episode,replay_identity
from robomme_plus.variation import TASKS,configuration,validate_profile
from robomme_plus.variation.sensors import SensorNoise


ROOT=Path(__file__).resolve().parents[2]


def test_selected_test_counts_and_atomic_training():
    from robomme_plus.variation.atomic import TASK_PRESETS
    selected=0
    training=0
    for task in BenchmarkEnvBuilder.get_task_list():
        builder=BenchmarkEnvBuilder(task)
        records=[builder.get_episode_metadata(i) for i in range(builder.get_episode_num())]
        marked=[r for r in records if r.get('perceptual_variation')]
        expected={'medium':5,'hard':8} if task in TASKS else {}
        assert Counter(r['difficulty'] for r in marked)==expected
        selected+=len(marked)
        train=BenchmarkEnvBuilder(task,dataset='train')
        marked=[train.get_episode_metadata(i) for i in range(train.get_episode_num())
                if train.get_episode_metadata(i).get('perceptual_variation')]
        assert len(marked)==(1 if task in TASK_PRESETS else 0)
        for row in marked:
            assert row['perceptual_variation']==dict(version=1,atomic=TASK_PRESETS[task])
            assert not row.get('mechanism_variation')
        training+=len(marked)
    assert selected==182
    assert training==13


def test_seeded_sampling_does_not_consume_global_rng():
    random.seed(5);np.random.seed(5)
    py_state=random.getstate();np_state=np.random.get_state()
    for task in TASKS:
        a=configuration(task,123,'non_lighting')
        assert a==configuration(task,123,'non_lighting')
        assert (a['lighting']!='normal')==(task=='VideoUnmask')
        b=configuration(task,456,'full')
        assert b==configuration(task,456,'full')
        if task=='PickHighlight':assert b['lighting'] in ('normal','dim')
        else:assert b['lighting']!='normal'
    assert random.getstate()==py_state
    after=np.random.get_state()
    assert np_state[0]==after[0] and np.array_equal(np_state[1],after[1]) and np_state[2:]==after[2:]


def test_video_unmask_medium_lighting_keeps_the_selected_seeds_and_table():
    builder=BenchmarkEnvBuilder('VideoUnmask')
    medium=[]
    for episode in range(builder.get_episode_num()):
        record=builder.get_episode_metadata(episode)
        if record['difficulty']!='medium' or not record.get('perceptual_variation'):continue
        assert record['perceptual_variation']=='non_lighting'
        config=configuration('VideoUnmask',record['seed'],record['perceptual_variation'])
        full=configuration('VideoUnmask',record['seed'],'full')
        assert config['lighting']==full['lighting']!='normal'
        assert config['table']==full['table']
        medium.append(episode)
    assert medium==[14,16,20,21,22]


def test_stopcube_clutter_is_limited_to_the_existing_selected_episodes():
    from robomme_plus.variation.clutter import sample_kinds
    builder=BenchmarkEnvBuilder('StopCube')
    counts=Counter()
    for episode in range(builder.get_episode_num()):
        record=builder.get_episode_metadata(episode)
        if not record.get('perceptual_variation'):continue
        config=configuration('StopCube',record['seed'],record['perceptual_variation'])
        assert config['clutter_kinds']==list(sample_kinds(record['seed']))
        assert config['clutter_kinds'] in [['coaster'],['desk_block'],['coaster','desk_block']]
        assert config['clutter_count']==len(config['clutter_kinds']) in (1,2)
        counts[record['difficulty']]+=1
    assert counts=={'medium':5,'hard':8}
    assert all('clutter_count' not in configuration(task,123,'full') for task in TASKS-{'StopCube'})


@pytest.mark.parametrize('shape',['line','ellipse','circle','figure8','double_ellipse'])
@pytest.mark.parametrize('kinds',[('coaster',),('desk_block',),('coaster','desk_block')])
def test_stopcube_clutter_preserves_clearance_for_the_complete_route(shape,kinds):
    from robomme_plus.tasks.counting.utils.motion import CurvedRoute
    from robomme_plus.variation.clutter import sample_layout
    for seed in range(10):
        route=CurvedRoute(shape,[50],center=(-.05,0),axes=(.16,.06),rotation_degrees=seed*37)
        points=np.asarray(route.center)+route.local_points(np.linspace(0,4,2049))
        button=np.array([-.32,.24])
        props=sample_layout(seed,kinds,points,button,route.target,.02)
        assert props==sample_layout(seed,kinds,points,button,route.target,.02)
        assert len(props)==len(kinds)
        for i,prop in enumerate(props):
            xy=np.asarray(prop['xy']);radius=prop['radius']
            assert np.min(np.linalg.norm(points-xy,axis=1))>radius+np.sqrt(2)*.02+.045
            assert np.linalg.norm(xy-button)>radius+.10
            assert np.linalg.norm(xy-route.target)>radius+.075
            assert np.all(xy-radius>=[-.44,-.40]) and np.all(xy+radius<=[.24,.40])
            for previous in props[:i]:
                assert np.linalg.norm(xy-previous['xy'])>radius+previous['radius']+.018


def test_pick_highlight_never_adds_spotlights_or_projected_patterns():
    from robomme_plus.variation.lighting import sample_lighting
    builder=BenchmarkEnvBuilder('PickHighlight')
    for episode in range(builder.get_episode_num()):
        record=builder.get_episode_metadata(episode)
        if not record.get('perceptual_variation'):continue
        config=configuration('PickHighlight',record['seed'],record['perceptual_variation'])
        assert config['lighting'] in ('normal','dim')
        assert sample_lighting(record['seed'],config['lighting'])['lights']==[]
    record=builder.get_episode_metadata(32)
    assert configuration('PickHighlight',record['seed'],'full')['lighting']=='normal'


@pytest.mark.parametrize('task,level,profile',[
    ('MoveCube',None,'full'),('StopCube','easy','full'),('StopCube','medium','full'),
    ('StopCube','hard','non_lighting'),('StopCube','medium',True),('StopCube','hard','high')])
def test_invalid_profiles_rejected(task,level,profile):
    with pytest.raises(ValueError):validate_profile(task,level,profile)


def test_variation_does_not_change_physical_replay_identity():
    record=BenchmarkEnvBuilder('StopCube').get_episode_metadata(12)
    assert record['perceptual_variation']=='non_lighting'
    plain=dict(record);plain.pop('perceptual_variation')
    assert replay_identity('StopCube',record)==replay_identity('StopCube',plain)
    bad=dict(record,perceptual_variation='full')
    with pytest.raises(ValueError):normalize_episode('StopCube',bad,12)


def test_optional_low_sensors_and_input_arrays_unchanged():
    rgb=np.full((64,64,3),128,dtype=np.uint8)
    obs=dict(front_rgb_list=rgb,eef_state_list=np.zeros(6))
    info={}
    for camera in ('front','wrist'):
        obs[camera+'_depth_list']=np.full((256,256,1),1000,dtype=np.int16)
        obs[camera+'_camera_extrinsic_list']=np.eye(4,dtype=np.float32)[:3].copy()
        info[camera+'_camera_intrinsic']=np.array([[128.,0.,128.],[0.,128.,128.],[0.,0.,1.]],dtype=np.float32)
    noise=SensorNoise(123)
    changed,changed_info=noise.transform(obs,info)
    repeated,repeated_info=SensorNoise(123).transform(obs,info)
    assert changed['front_rgb_list'] is rgb
    assert changed['eef_state_list'] is obs['eef_state_list']
    for camera in ('front','wrist'):
        key=camera+'_depth_list';depth=changed[key];valid=depth>0
        assert depth.dtype==np.int16 and depth.shape==obs[key].shape
        assert np.array_equal(depth,repeated[key]) and np.all(obs[key]==1000)
        assert .001<np.mean(~valid)<.004
        assert .9<np.std(depth[valid].astype(float)-1000)<1.15
        delta=noise.biases[camera]['delta']
        degrees=np.rad2deg(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
        assert np.isclose(degrees,.25) and np.isclose(np.linalg.norm(delta[:3,3]),.001)
        key=camera+'_camera_intrinsic'
        assert np.array_equal(changed_info[key],repeated_info[key])
        assert np.allclose(np.abs(np.diag(changed_info[key])[:2]/128.-1),.0025,atol=1e-7)
        assert np.allclose(np.abs(changed_info[key][:2,2]-128),.25)
        assert np.array_equal(info[key],np.array([[128.,0.,128.],[0.,128.,128.],[0.,0.,1.]]))
    minimal=dict(front_rgb_list=rgb)
    assert SensorNoise(123).transform(minimal,{})[0]['front_rgb_list'] is rgb


def test_per_task_switches_and_builder_override():
    switches=set()
    for path in (ROOT/'src/robomme_plus/tasks').glob('*/*.py'):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node,ast.ClassDef):continue
            for item in node.body:
                if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PERCEPTUAL_VARIATIONS' for t in item.targets):
                    assert ast.literal_eval(item.value) is True
                    switches.add(node.name)
    assert switches==TASKS|{'InsertPeg'}
    assert BenchmarkEnvBuilder('StopCube').perceptual_variations is True
    assert BenchmarkEnvBuilder('StopCube',perceptual_variations=False).perceptual_variations is False
    with pytest.raises(TypeError):BenchmarkEnvBuilder('StopCube',perceptual_variations='false')


@pytest.mark.parametrize('name',['cube_0','stick_0'])
def test_material_transform_is_stable_across_repeated_resets(monkeypatch,name):
    from robomme_plus.variation import objects
    material=SimpleNamespace(base_color=np.array([.9,.1,.1,1.]),base_color_texture=None)
    surface=SimpleNamespace(material=material)
    actor=SimpleNamespace(name=name)
    native=SimpleNamespace(task_name='PatternLock',scene=SimpleNamespace(actors={name:actor},articulations={}),buttons_grid=[])
    monkeypatch.setattr(objects,'surfaces',lambda a:[surface])
    config=objects.configuration(123)
    objects.apply(native,config);first=list(material.base_color)
    del native._perceptual_object_colors
    objects.apply(native,config)
    assert material.base_color==first


@pytest.mark.parametrize('distractor',[False,True])
def test_cube_hue_protection_depends_on_native_episode_palette(monkeypatch,distractor):
    from robomme_plus.variation import objects
    palette={'cube_red':[1.,0.,0.], 'cube_blue':[0.,0.,1.], 'cube_green':[0.,1.,0.],
             'board':[.55,.1,.85]}
    if distractor:palette['cube_violet']=[.55,.1,.85]
    actors={name:SimpleNamespace(name=name) for name in palette}
    materials={name:SimpleNamespace(base_color=[*rgb,1.],base_color_texture=None) for name,rgb in palette.items()}
    monkeypatch.setattr(objects,'surfaces',lambda actor:[SimpleNamespace(material=materials[actor.name])])
    native=SimpleNamespace(task_name='VideoPlaceOrder',scene=SimpleNamespace(actors=actors,articulations={}))
    config=objects.configuration(251502031)
    objects.apply(native,config)
    assert native._perceptual_object_colors['preserve_cube_hues'] is distractor
    blue=list(materials['cube_blue'].base_color)
    if distractor:
        assert np.allclose(blue[:3],[.10153034773058702,.09022059618354726,.8735057027013753])
    else:
        assert np.allclose(blue[:3],[.20306069546117403,.18044119236709452,.7470114054027505])
    del native._perceptual_object_colors
    objects.apply(native,config)
    assert native._perceptual_object_colors['preserve_cube_hues'] is distractor
    assert materials['cube_blue'].base_color==blue

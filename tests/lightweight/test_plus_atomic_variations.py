"""Atomic appearance changes must not silently enable the full test profile."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from robomme_plus.variation import configuration, validate_profile
from robomme_plus.variation.atomic import TASK_PRESETS
from robomme_plus.variation.mechanism import validate_profile as validate_mechanism


@pytest.mark.parametrize('task,atom',sorted(TASK_PRESETS.items()))
def test_atomic_profiles_are_isolated_and_have_exact_sensors(task,atom):
    from robomme_plus.variation.runtime import PerceptualVariation
    profile=dict(version=1,atomic=atom)
    original=copy.deepcopy(profile)
    for level in ('easy','medium','hard'):
        validate_profile(task,level,profile)
    variation=PerceptualVariation(task,123,profile)
    try:
        assert variation.noise is None
        assert profile==original
        config=variation.config
        assert 'clutter_kinds' not in config
        if atom.startswith('light_'):
            assert config['table']=='wood' and not variation.objects['enabled']
            assert config['lighting']!='normal'
        elif atom.startswith('table_'):
            assert config['lighting']=='normal' and not variation.objects['enabled']
            assert config['table']==atom.removeprefix('table_')
        else:
            assert config['table']=='wood' and config['lighting']=='normal'
            assert set(variation.objects['groups'])=={
                'ring_colors':{'rings'},'button_colors':{'button_body','button_cap'},
                'board_light_gray_containers':{'board_or_bin','container'},
            }[atom]
        assert configuration(task,123,profile)==config
    finally:
        variation.close()


def test_board_and_container_colors_leave_cubes_unchanged(monkeypatch):
    from robomme_plus.variation.runtime import PerceptualVariation
    from robomme_plus.variation import objects
    palette={'cube_red':[1.,0.,0.],'board':[.55,.1,.85],'cover_0':[.1,.1,.1]}
    actors={name:SimpleNamespace(name=name) for name in palette}
    materials={name:SimpleNamespace(base_color=[*rgb,1.],base_color_texture=None) for name,rgb in palette.items()}
    monkeypatch.setattr(objects,'surfaces',lambda actor:[SimpleNamespace(material=materials[actor.name])])
    native=SimpleNamespace(task_name='VideoUnmaskSwap',scene=SimpleNamespace(actors=actors,articulations={}))
    variation=PerceptualVariation('VideoUnmaskSwap',123,dict(version=1,atomic='board_light_gray_containers'))
    try:
        objects.apply(native,variation.objects)
        assert materials['cube_red'].base_color==[1.,0.,0.,1.]
        assert materials['cover_0'].base_color==[.75,.75,.75,1.]
        assert not np.allclose(materials['board'].base_color[:3],palette['board'])
    finally:variation.close()


def test_training_ring_projection_is_white_with_dark_gaps(tmp_path):
    from PIL import Image
    from robomme_plus.variation.lighting import projected_ring_image
    from robomme_plus.variation.runtime import PerceptualVariation
    variation=PerceptualVariation('VideoUnmask',1497627596,dict(version=1,atomic='light_ring_pattern'))
    try:
        assert variation.plan['white_rings']
        path=tmp_path/'rings.png'
        projected_ring_image(path,1497627596,[[1.,0.,0.]],white=variation.plan['white_rings'])
        rgb=np.asarray(Image.open(path))
        assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,1],rgb[:,:,2])
        assert set(np.unique(rgb))=={0,255}
    finally:variation.close()


def test_training_button_palette_is_separate_from_test_colors():
    from robomme_plus import BenchmarkEnvBuilder
    from robomme_plus.variation.runtime import PerceptualVariation
    from robomme_plus.variation.objects import configuration as object_colors
    variation=PerceptualVariation('ButtonUnmask',1878721550,dict(version=1,atomic='button_colors'))
    try:
        for task in BenchmarkEnvBuilder.get_task_list():
            builder=BenchmarkEnvBuilder(task)
            for i in range(builder.get_episode_num()):
                row=builder.get_episode_metadata(i)
                if not row.get('perceptual_variation'):continue
                test=object_colors(row['seed'])['colors']
                for key in ('button_body','button_cap'):
                    assert np.linalg.norm(np.array(test[key])-variation.objects['colors'][key])>.2
    finally:variation.close()


@pytest.mark.parametrize('profile',[{'version':True,'atomic':'light_dim'},
    {'version':1,'atomic':'light_dim','lighting':'disco'},
    {'version':1,'atomic':'table_steel'}, {'version':1,'atomic':['light_dim']}])
def test_invalid_atomic_appearance_rejected(profile):
    with pytest.raises(ValueError):validate_profile('BinFill','easy',profile)


@pytest.mark.parametrize('task,atom',[('RouteStick','longer_activation'),('ButtonUnmaskSwap','container_speed_2x')])
def test_atomic_mechanisms_work_on_each_tier_but_cannot_mix_appearance(task,atom):
    for level in ('easy','medium','hard'):
        validate_mechanism(task,level,dict(version=1,atomic=atom))
        with pytest.raises(ValueError,match='disjoint'):
            validate_mechanism(task,level,dict(version=1,atomic=atom),'full')
    with pytest.raises(ValueError):validate_mechanism('PickXtimes','hard',dict(version=1,atomic=atom))

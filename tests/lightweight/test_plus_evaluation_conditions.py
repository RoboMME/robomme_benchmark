"""Baseline and variation runs cannot overwrite or be grouped into each other."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/plus_eval'))
from evaluation import select_cases
from run_storage import check_configuration


@pytest.mark.parametrize('enabled',[False,True])
def test_condition_reaches_each_episode_builder(enabled,tmp_path):
    class Builder:
        @staticmethod
        def get_task_list():return ['StopCube']
        def __init__(self,task,**options):self.options=options
        def get_episode_num(self):return 50
    builders,_=select_cases(Builder,['StopCube'],[12],perceptual_variations=enabled)
    assert builders['StopCube'].options['perceptual_variations'] is enabled



@pytest.mark.parametrize('mode',['resume','rerun'])
def test_opposite_condition_requires_a_separate_directory_even_when_rerunning(mode):
    original={'method':'embodied-astra','benchmark':'robomme_plus','arguments':{},'perceptual_variations':False}
    changed={**original,'perceptual_variations':True}
    with pytest.raises(ValueError,match='separate output directory'):
        check_configuration(original,changed,mode)
    with pytest.raises(ValueError,match='separate output directory'):
        check_configuration(changed,original,mode)
    check_configuration(original,original,mode)


def test_legacy_baseline_is_protected_from_the_new_variation_default():
    original={'method':'embodied-astra','arguments':{}}
    changed={**original,'arguments':{'perceptual_variations':True}}
    with pytest.raises(ValueError,match='separate output directory'):
        check_configuration(original,changed)

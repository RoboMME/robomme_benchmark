"""Mechanism selection, opt-out budgets, and isolation from native task code."""
import inspect
import sys

import pytest

from robomme_plus import BenchmarkEnvBuilder, register_envs
from robomme_plus.evaluation.metadata import normalize_episode, replay_identity
from robomme_plus.registry import episode_budget
from robomme_plus.variation.mechanism import TASKS, validate_profile


def selected(task):
    builder = BenchmarkEnvBuilder(task)
    return [builder.get_episode_metadata(i) for i in range(builder.get_episode_num())
            if builder.get_episode_metadata(i).get('mechanism_variation')]


def test_selection_and_atomic_training():
    total = 0
    atoms = {'ButtonUnmaskSwap': 'container_speed_2x', 'RouteStick': 'longer_activation'}
    training = 0
    for task in BenchmarkEnvBuilder.get_task_list():
        rows = selected(task)
        assert len(rows) == (8 if task in TASKS else 0)
        assert all(r['difficulty'] == 'hard' and not r.get('perceptual_variation') for r in rows)
        assert len({r['seed'] for r in rows}) == len(rows)
        total += len(rows)
        train = BenchmarkEnvBuilder(task, dataset='train')
        marked = [train.get_episode_metadata(i) for i in range(train.get_episode_num())
                  if train.get_episode_metadata(i).get('mechanism_variation')]
        assert len(marked) == (1 if task in atoms else 0)
        disabled = BenchmarkEnvBuilder(task, dataset='train', mechanism_variations=False)
        for row in marked:
            assert row['mechanism_variation'] == dict(version=1, atomic=atoms[task])
            assert not row.get('perceptual_variation')
            assert disabled.get_episode_metadata(row['episode'])['oracle_policy_step'] == row['base_oracle_policy_step']
        training += len(marked)
    assert total == 104
    assert training == 2
    assert {r['episode'] for r in selected('ButtonUnmaskSwap')} == {27, 36, 38, 39, 40, 46, 48, 49}


@pytest.mark.parametrize('task', sorted(TASKS))
def test_opt_out_restores_original_timing_and_preserves_other_records(task):
    enabled = BenchmarkEnvBuilder(task)
    disabled = BenchmarkEnvBuilder(task, mechanism_variations=False)
    for i in range(enabled.get_episode_num()):
        varied, plain = enabled.get_episode_metadata(i), disabled.get_episode_metadata(i)
        if varied.get('mechanism_variation'):
            assert plain['oracle_policy_step'] == varied['base_oracle_policy_step']
            assert disabled.episode_max_steps(i) == episode_budget(varied['base_oracle_policy_step'])
            assert disabled.get_episode_metadata(i, mechanism_variations=True) == varied
            assert replay_identity(task, varied)['mechanism_variation'] == varied['mechanism_variation']
        else:
            assert varied == plain
    assert enabled.perceptual_variations and disabled.perceptual_variations


def test_bad_markers_and_switches_are_rejected():
    row = selected('PickXtimes')[0]
    for task, level, profile in [('MoveCube', 'hard', {'version': 1}),
                                 ('PickXtimes', 'medium', {'version': 1}),
                                 ('PickXtimes', 'hard', True),
                                 ('PickXtimes', 'hard', {'version': True}),
                                 ('PickXtimes', 'hard', {'version': 2})]:
        with pytest.raises(ValueError):
            validate_profile(task, level, profile)
    with pytest.raises(ValueError, match='disjoint'):
        normalize_episode('PickXtimes', {**row, 'perceptual_variation': 'full'}, row['episode'])
    with pytest.raises(ValueError, match='base_oracle_policy_step'):
        normalize_episode('PickXtimes', {**row, 'base_oracle_policy_step': None}, row['episode'])
    with pytest.raises(TypeError):
        BenchmarkEnvBuilder('PickXtimes', mechanism_variations='false')


def _native_objects():
    result = []
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('robomme_plus.tasks.'):
            continue
        result.append((module, dict(vars(module))))
        for obj in vars(module).values():
            if inspect.isclass(obj) and obj.__module__ == name:
                result.append((obj, dict(vars(obj))))
    return result


@pytest.mark.parametrize('task', sorted(TASKS))
def test_installation_cannot_patch_normal_modules_or_gym_registry(task):
    # Importing source definitions needs no simulator scene or GPU.
    from mani_skill.utils.registration import REGISTERED_ENVS
    from robomme_plus.variation.mechanism.namespace import TaskNamespace
    from robomme_plus.variation.mechanism.runtime import install

    register_envs()
    row = selected(task)[0]
    objects = _native_objects()
    registered = {key: value.cls for key, value in REGISTERED_ENVS.items()}
    left, right = TaskNamespace(), TaskNamespace()
    try:
        install(left, task, row['seed'], row['mechanism_variation'])
        install(right, task, row['seed'], row['mechanism_variation'])
        for obj, attributes in objects:
            assert vars(obj).keys() == attributes.keys(), (task, obj)
            for key, value in attributes.items():
                assert vars(obj)[key] is value, (task, obj, key)
        assert registered == {key: value.cls for key, value in REGISTERED_ENVS.items()}
        assert left.load('tasks') is not right.load('tasks')
    finally:
        left.close()
        right.close()
    assert not any(name.startswith((left.prefix + '.', right.prefix + '.')) for name in sys.modules)

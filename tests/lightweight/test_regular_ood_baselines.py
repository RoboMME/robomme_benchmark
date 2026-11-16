"""The learned baselines select the same regular OOD split and native budget."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def baseline_modules(monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'scripts/plus_eval'
    monkeypatch.syspath_prepend(str(root))
    return (importlib.import_module('regular_benchmark'), importlib.import_module('evaluation'),
            importlib.import_module('run_storage'))


def test_regular_counter_excludes_reset_and_preserves_terminal_status(baseline_modules):
    regular, _, _ = baseline_modules

    class Native:
        count = 0
        def get_wrapper_attr(self, name):
            assert name == 'steps_without_demonstration'
            return self.count
        def reset(self):
            self.count = 2
            return {}, {'status': 'ongoing'}
        def step(self, action):
            self.count += 1
            return {}, 0, True, False, {'status': 'success'}
        def close(self):
            pass

    env = regular.CountedEnvironment(Native())
    assert env.reset()[1]['execution_control_steps'] == 0
    output = env.step([0] * 8)
    assert output[2:4] == (True, False)
    assert output[4]['status'] == 'success' and output[4]['execution_total_control_steps'] == 1


@pytest.mark.parametrize('split, max_steps', [('test', 1300), ('ood_test', 1500)])
def test_regular_builder_routes_explicit_split_and_pinned_source(baseline_modules, tmp_path, monkeypatch, split, max_steps):
    regular, evaluation, _ = baseline_modules
    source = tmp_path / 'robomme'
    metadata = source / f'env_metadata/{split}/record_dataset_BinFill_metadata.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({'records': [{'task': 'BinFill', 'episode': 0, 'seed': 12345, 'difficulty': 'ood'}]}))
    calls = []

    class Native:
        @classmethod
        def get_task_list(cls, dataset):
            assert dataset == split
            return ['BinFill']
        def __init__(self, task, **kwargs):
            calls.append((task, kwargs))
        def get_episode_num(self):
            return 1
        def resolve_episode(self, episode):
            assert episode == 0
            return 12345, 'ood'
        def make_env_for_episode(self, episode, **options):
            return SimpleNamespace(episode=episode, options=options)

    original = importlib.import_module
    modules = {'robomme': SimpleNamespace(__file__=str(source / '__init__.py')),
               'robomme.robomme_env': SimpleNamespace(),
               'robomme.env_record_wrapper': SimpleNamespace(BenchmarkEnvBuilder=Native)}
    monkeypatch.setattr(regular.importlib, 'import_module', lambda name: modules[name] if name in modules else original(name))
    builder = regular.builder_class(source, split)
    builders, document = evaluation.select_cases(builder, benchmark='robomme', dataset=split)
    assert document == {'benchmark': 'robomme', 'dataset': split, 'cases': [{'task': 'BinFill', 'episode': 0}]}
    instance = builders['BinFill']
    assert calls[0][1]['dataset'] == split
    assert calls[0][1]['max_steps'] == max_steps
    assert calls[0][1]['override_metadata_path'] == metadata.parent
    assert instance.episode_max_steps(0) == max_steps + 2
    assert instance.get_episode_metadata(0)['native_max_steps_argument'] == max_steps
    assert instance.get_episode_metadata(0)['seed'] == 12345
    assert instance.make_env_for_episode(0).env.episode == 0
    with pytest.raises(ValueError, match='split'):
        builder('BinFill', dataset='test' if split == 'ood_test' else 'ood_test')
    modules['robomme'].__file__ = '/another/robomme/__init__.py'
    with pytest.raises(ValueError, match='different RoboMME'):
        regular.builder_class(source, 'ood_test')


def test_regular_resume_rejects_changed_seed_or_split(baseline_modules, tmp_path):
    _, _, storage = baseline_modules
    case = {'task': 'BinFill', 'episode': 0}
    record = {**case, 'benchmark': 'robomme', 'dataset': 'ood_test', 'seed': 12345, 'difficulty': 'ood',
              'control_freq': 20, 'max_steps': 1302, 'native_max_steps_argument': 1300}
    target = tmp_path / 'BinFill_ep0/metadata.json'
    target.parent.mkdir()
    target.write_text(json.dumps(record))
    assert storage.compatible_metadata(tmp_path, case, record)
    assert not storage.compatible_metadata(tmp_path, case, {**record, 'seed': 99})
    assert not storage.compatible_metadata(tmp_path, case, {**record, 'dataset': 'test'})


def test_regular_summary_retains_split_identity(baseline_modules, tmp_path):
    _, _, storage = baseline_modules
    (tmp_path / 'config.json').write_text(json.dumps({
        'benchmark': 'robomme', 'dataset': 'ood_test', 'method': 'simplememvla',
        'cases': [{'task': 'BinFill', 'episode': 0}],
    }))
    report = storage.rebuild_task(tmp_path)
    assert report['benchmark'] == 'robomme' and report['dataset'] == 'ood_test'
    assert report['completed'] == 0 and report['pending'] == 1


def test_exact_case_selection_does_not_repeat_completed_episodes(baseline_modules):
    _, evaluation, _ = baseline_modules
    class Builder:
        @classmethod
        def get_task_list(cls):
            return ['BinFill', 'StopCube']
        def __init__(self, *args, **kwargs):
            pass
        def get_episode_num(self):
            return 20
    remaining = [{'task': 'BinFill', 'episode': 19}, {'task': 'StopCube', 'episode': 2}]
    _, selection = evaluation.select_cases(Builder, selected_cases=remaining)
    assert selection['cases'] == remaining
    for invalid in [[], remaining * 2, [{'task': 'StopCube', 'episode': 20}],
                    [{'task': 'Unknown', 'episode': 0}], [{'task': 'StopCube', 'episode': True}]]:
        with pytest.raises(ValueError, match='Case selection'):
            evaluation.select_cases(Builder, selected_cases=invalid)

import sys
from types import SimpleNamespace

import pytest

from benchmark_io.protocol import Study
from benchmark_io.evaluation import EvaluationStudy


def frozen(tmp_path, monkeypatch):
    records = {task: [dict(seed=10+i, max_steps=100, control_freq=20) for i in range(3)]
               for task in ('BinFill', 'StopCube')}
    study = Study(tmp_path / 'development', method='aspire', setting='demo-only', budget=0, records=records)
    class Builder:
        def __init__(self, task, dataset, action_space):
            assert dataset == 'test'
        def get_episode_metadata(self, i):
            return dict(seed=1000+i, max_steps=100, control_freq=20)
    monkeypatch.setitem(sys.modules, 'robomme_plus', SimpleNamespace(BenchmarkEnvBuilder=Builder))
    policy = tmp_path / 'source'
    policy.mkdir()
    (policy / 'program.py').write_text('result = 1\n')
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    (benchmark / '__init__.py').write_text('')
    evaluation = EvaluationStudy.freeze(tmp_path / 'evaluation', development_study=study.root,
        policy_directory=policy, episodes={'BinFill': [0]}, benchmark_root=benchmark)
    return study, evaluation


def test_frozen_evaluation_is_separate_from_demo_only_development(tmp_path, monkeypatch):
    study, evaluation = frozen(tmp_path, monkeypatch)
    assert evaluation.dataset == 'test'
    attempt = evaluation.reserve('BinFill', 0, purpose='frozen_test')
    assert attempt['seed'] == 1000
    evaluation.finish(attempt['id'], {'success': False})
    assert study.attempts() == []
    with pytest.raises(PermissionError, match='already reserved'):
        evaluation.reserve('BinFill', 0, purpose='rerun')
    with pytest.raises(PermissionError, match='not authorized'):
        evaluation.reserve('BinFill', 1, purpose='extra')


def test_modified_frozen_policy_is_rejected(tmp_path, monkeypatch):
    _, evaluation = frozen(tmp_path, monkeypatch)
    (evaluation.root / 'policy/program.py').write_text('result = 2\n')
    with pytest.raises(ValueError, match='policy files changed'):
        evaluation.reserve('BinFill', 0, purpose='test')


def test_single_task_test_keeps_development_task_identity(tmp_path, monkeypatch):
    study, _ = frozen(tmp_path, monkeypatch)
    evaluation = EvaluationStudy.freeze(tmp_path / 'stopcube_eval', development_study=study.root,
        policy_directory=tmp_path / 'source', episodes={'StopCube': [0]},
        benchmark_root=tmp_path / 'benchmark')
    assert list(evaluation.config['records']) == ['StopCube']
    # Subsetting test episodes must not relabel StopCube from task 1 to task 0.
    assert evaluation.config['task_order'].index('StopCube') == sorted(study.config['records']).index('StopCube')


def test_frozen_evaluation_refuses_symbolic_policy_files(tmp_path, monkeypatch):
    study, _ = frozen(tmp_path, monkeypatch)
    source = tmp_path / 'symlink_policy'
    source.mkdir()
    (source / 'program.py').symlink_to(tmp_path / 'source/program.py')
    with pytest.raises(ValueError, match='symlinks'):
        EvaluationStudy.freeze(tmp_path / 'invalid', development_study=study.root,
            policy_directory=source, episodes={'BinFill': [0]}, benchmark_root=tmp_path / 'benchmark')

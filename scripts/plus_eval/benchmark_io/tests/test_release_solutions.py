import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmark_io import solutions
from benchmark_io.runtime_source import assemble
from benchmark_io.artifacts import link

RELEASE=Path(__file__).resolve().parents[2]


def test_published_solutions_match_all_64_evaluated_r10_policies():
    for method in ('aspire','rax'):
        root=RELEASE/'agentic_methods'/method/'solutions'
        index=json.loads((root/'index.json').read_text())
        assert len(index['entries'])==32
        assert {e['budget'] for e in index['entries']} == {10}
        assert {e['setting'] for e in index['entries']} == {'interaction-only', 'hybrid'}
        for entry in index['entries']:
            manifest=solutions.verified_files(root/entry['path'])
            assert manifest['policy_files'] and manifest['status']=='frozen_policy'
            assert all('__pycache__' not in Path(name).parts and not name.endswith(('.pyc', '.pyo'))
                       for name in manifest['policy_files'])
            assert manifest['setting']==entry['setting'] and manifest['budget']==entry['budget']
            assert not manifest['test_feedback_to_development']


def test_solution_hash_rejects_changed_program(tmp_path):
    (tmp_path/'policy.py').write_text('original')
    (tmp_path/'manifest.json').write_text(json.dumps({'policy_files':{'policy.py':hashlib.sha256(b'original').hexdigest()}}))
    solutions.verified_files(tmp_path)
    (tmp_path/'policy.py').write_text('modified')
    with pytest.raises(ValueError,match='changed'):solutions.verified_files(tmp_path)


def test_solution_hash_rejects_changed_runner(tmp_path):
    (tmp_path/'policy.py').write_bytes(b'policy')
    (tmp_path/'runner.json').write_bytes(b'{}')
    (tmp_path/'manifest.json').write_text(json.dumps({
        'policy_files': {'policy.py': hashlib.sha256(b'policy').hexdigest()},
        'runner_files': {'runner.json': hashlib.sha256(b'{}').hexdigest()}}))
    solutions.verified_files(tmp_path)
    (tmp_path/'runner.json').write_text('{"workspace_path":"changed"}')
    with pytest.raises(ValueError,match='runner.json'):solutions.verified_files(tmp_path)


def test_worker_preserves_native_episode_field(tmp_path,monkeypatch):
    monkeypatch.setattr(solutions,'EvaluationStudy',lambda _:SimpleNamespace(attempts=lambda:[]))
    monkeypatch.setattr(sys,'argv',['eval','--setting','Hybrid','--budget','10','--output-dir',str(tmp_path),
        '--_worker','--tasks','BinFill','--episodes','0'])
    solutions.main('aspire',lambda *a:dict(episode=0,task_success=True),tmp_path)
    assert json.loads((tmp_path/'episodes/BinFill/episode_000/result.json').read_text())['task_success']


def test_runtime_patches_are_applied_without_changing_submodules(tmp_path):
    result=assemble(tmp_path/'runtime',source=RELEASE)
    assert result['methods']['rax']['revision'].startswith('d49c8ad')
    assert (tmp_path/'runtime/rax/services/robomme_plus/backend.py').exists()
    assert (tmp_path/'runtime/aspire/ASPIRE/src/cap_harness/robomme_plus/adapter.py').exists()
    original=(RELEASE/'agentic_methods/aspire/ASPIRE/src/cap_harness/artifacts.py').read_text()
    patched=(tmp_path/'runtime/aspire/ASPIRE/src/cap_harness/artifacts.py').read_text()
    assert '"controller_no_progress"' not in original
    assert '"controller_no_progress"' in patched
    with pytest.raises(FileExistsError):assemble(tmp_path/'runtime',source=RELEASE)


def test_artifact_links_never_replace_existing_results(tmp_path):
    study=tmp_path/'study';source=tmp_path/'source'
    for method in ['aspire','rax']:(study/method).mkdir(parents=True)
    link(study,source);link(study,source)
    assert (source/'agentic_methods/rax/runs').resolve()==study/'rax'
    (source/'agentic_methods/rax/runs').unlink();(source/'agentic_methods/rax/runs').mkdir()
    with pytest.raises(FileExistsError):link(study,source)

import base64
from io import BytesIO
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from benchmark_io.prompt_bundle import DEFAULT_BUNDLE, install
from benchmark_io.protocol_fixtures import generate
from benchmark_io.runtime_source import hashes


def packet(path):
    reply = json.loads(path.read_text())
    return reply, json.loads(reply['content'][0]['text'])


def test_checked_in_examples_match_actual_serializers_without_a_simulator(tmp_path, monkeypatch):
    from services.robomme_plus.backend import RoboMMEPlusBackend
    def forbidden(*args, **kwargs):
        raise AssertionError('A public format example cannot start a simulator')
    monkeypatch.setattr(RoboMMEPlusBackend, '__init__', forbidden)
    generate(tmp_path)
    expected = DEFAULT_BUNDLE/'fixtures'
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (expected/path.relative_to(tmp_path)).read_bytes()


def test_decode_both_views_units_calibration_phase_boundary_limits_and_terminal():
    fixtures = DEFAULT_BUNDLE/'fixtures'
    reply, meta = packet(fixtures/'rax_observe.json')
    checks = json.loads((fixtures/'rax_checks.json').read_text())
    for descriptor in meta['attachments']:
        block = reply['content'][descriptor['content_index']]
        array = np.asarray(Image.open(BytesIO(base64.b64decode(block['data']))))
        camera, kind = descriptor['camera'], descriptor['kind']
        if kind == 'rgb':
            assert np.all(array == checks['expected_rgb'][camera])
        else:
            assert np.allclose(array*meta['depth'][camera]['meters_per_unit'], checks['expected_depth_m'][camera])
        assert len(meta['camera_intrinsics'][camera]['matrix']) == 3
        assert set(meta['camera_poses'][camera]) == {'position_m','quaternion_wxyz','frame','convention'}
    workspace = fixtures/checks['workspace_root']
    manifest = json.loads((workspace/meta['frame_sequence']['manifest_path']).read_text())
    assert [f['phase'] for f in manifest['frames']] == ['video_demo','video_demo','execution']
    assert [f['step_id'] for f in manifest['frames']] == [0, 0, 0]
    assert [f['timestamp_s'] for f in manifest['frames']] == [0., .05, 0.]
    for frame in manifest['frames']:
        assert {(f['camera'],f['kind']) for f in frame['files']} == {
            ('front','rgb'),('wrist','rgb'),('front','depth'),('wrist','depth')}
        assert all((workspace/f['path']).is_file() for f in frame['files'])
    _, pixel = packet(fixtures/'rax_pixel_to_position.json')
    assert pixel['selections'][0]['position_m'] == checks['front_center_pixel_world_m']
    assert meta['episode_limits'] == dict(episode_step_limit=100, remaining_episode_steps=100)
    for outcome in ('success', 'failure'):
        _, terminal = packet(fixtures/f'rax_terminal_{outcome}.json')
        assert terminal['episode_ended']
        step = terminal['transition']['steps'][-1]
        assert step == dict(step_id=1,episode_ended=True,task_complete=outcome=='success')
    rejected = json.loads((fixtures/'rax_rejected_action.json').read_text())
    assert rejected['error']['data']['stage'] == 'validation'
    assert not rejected['error']['data']['episode_ended']


def test_current_install_is_immutable_and_does_not_replace_v2(tmp_path, monkeypatch):
    source = tmp_path/'source'
    shutil.copytree(DEFAULT_BUNDLE, source)
    (source/'fixtures/__pycache__').mkdir(exist_ok=True)
    (source/'fixtures/__pycache__/unwanted.pyc').write_bytes(b'not source')
    monkeypatch.setenv('ROBOMME_AGENTIC_PROMPTS', str(source))
    workspace = tmp_path/'workspace'
    version = json.loads((source/'bundle.json').read_text())['version']
    assert f'robomme-protocol-v{version}' in install(workspace, 'rax')
    manifest = json.loads((workspace/f'docs/robomme-protocol-v{version}/manifest.json').read_text())
    assert not any(name.endswith('.pyc') for name in manifest['sha256'])
    before = hashes(workspace)
    (source/'RAX.md').write_text('changed')
    with pytest.raises(ValueError):
        install(workspace, 'rax')
    assert hashes(workspace) == before
    descriptor = json.loads((source/'bundle.json').read_text())
    descriptor['version'] = 2
    (source/'bundle.json').write_text(json.dumps(descriptor))
    monkeypatch.setenv('ROBOMME_AGENTIC_PROMPTS', str(source))
    with pytest.raises(ValueError, match='fresh workspace'):
        install(workspace, 'rax')


def test_future_protocol_does_not_mutate_the_legacy_bundle(tmp_path, monkeypatch):
    from benchmark_io.unified_study import environment
    assert environment(tmp_path)['ROBOMME_AGENTIC_PROMPTS'] == str(tmp_path/'protocol')
    (tmp_path/'protocol_v3').mkdir()
    shutil.copyfile(DEFAULT_BUNDLE/'bundle.json', tmp_path/'protocol_v3/bundle.json')
    (tmp_path/'future_development_protocol.json').write_text(json.dumps({'relative_directory':'protocol_v3'}))
    assert environment(tmp_path)['ROBOMME_AGENTIC_PROMPTS'] == str(tmp_path/'protocol_v3')

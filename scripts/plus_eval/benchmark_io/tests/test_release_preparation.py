import json
from pathlib import Path
import sqlite3

import h5py
import pytest

from benchmark_io import prepare_unified
from benchmark_io.unified_study import configurations


def test_fresh_prepare_needs_no_old_study_or_simulator(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[4]
    monkeypatch.setattr(prepare_unified,'REPO',source)
    monkeypatch.setattr(prepare_unified,'IMAGE','sha256:'+'0'*64)
    monkeypatch.setenv('ROBOMME_AGENTIC_REPO',str(source))
    demos=tmp_path/'recordings';demos.mkdir()
    for path in (source/'src/robomme_plus/env_metadata/train').glob('*.json'):
        task=path.name.removeprefix('record_dataset_').removesuffix('_metadata.json')
        with h5py.File(demos/f'record_dataset_{task}.h5','w') as f:
            for i,record in enumerate(json.loads(path.read_text())['records']):
                f.create_dataset(f'episode_{i}/setup/episode_metadata',data=json.dumps(record))
    root=tmp_path/'study'
    prepare_unified.prepare(root,demos,protocol=source/'scripts/plus_eval/benchmark_io/prompts/current')
    assert json.loads((root/'preparation.json').read_text())['simulator_starts']==0
    assert (root/'runtime/private/relay-token').stat().st_mode & 0o077 == 0
    for key,method,setting,budget in configurations():
        with sqlite3.connect(root/key/'study/attempts.sqlite3') as db:
            assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0
    assert (root/'runtime_source/rax/services/robomme_plus/server.py').is_file()
    with pytest.raises(FileExistsError):prepare_unified.prepare(root,demos)

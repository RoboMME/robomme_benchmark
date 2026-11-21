import json
from pathlib import Path
import sqlite3

import h5py
import pytest

from benchmark_io import prepare_unified
from benchmark_io.unified_study import configurations


@pytest.mark.parametrize('settings,case_count', [(None,6),(['interaction-only','demo+interaction'],4)])
def test_fresh_prepare_needs_no_old_study_or_simulator(tmp_path,monkeypatch,settings,case_count):
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
    prepare_unified.prepare(root,demos,protocol=source/'scripts/plus_eval/benchmark_io/prompts/current',settings=settings)
    assert json.loads((root/'preparation.json').read_text())['simulator_starts']==0
    plan = json.loads((root/'study_plan.json').read_text())
    assert len(plan['configurations']) == case_count and plan['test_trials'] == 800*case_count
    assert plan['logs_only_budgets'] == []
    assert (root/'runtime/private/relay-token').stat().st_mode & 0o077 == 0
    for key,method,setting,budget in configurations(root):
        with sqlite3.connect(root/key/'study/attempts.sqlite3') as db:
            assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0
    assert (root/'runtime_source/rax/services/robomme_plus/server.py').is_file()
    with pytest.raises(FileExistsError):prepare_unified.prepare(root,demos)


@pytest.mark.parametrize('field,value', [('oracle_policy_step',999),('control_freq',100)])
def test_recordings_with_matching_seeds_but_stale_execution_metadata_are_rejected(tmp_path,field,value):
    metadata=tmp_path/'metadata';metadata.mkdir()
    demos=tmp_path/'demos';demos.mkdir()
    row=dict(seed=123,difficulty='easy',oracle_policy_step=20,control_freq=20)
    (metadata/'record_dataset_BinFill_metadata.json').write_text(json.dumps({'records':[row]}))
    with h5py.File(demos/'record_dataset_BinFill.h5','w') as h5:
        h5.create_dataset('episode_0/setup/episode_metadata',data=json.dumps({**row,field:value}))
    target=tmp_path/'pinned'
    with pytest.raises(ValueError,match=field):
        prepare_unified.pin_demonstrations(demos,target,metadata)
    assert not target.exists()

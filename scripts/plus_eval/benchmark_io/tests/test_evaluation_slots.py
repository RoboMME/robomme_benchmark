from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmark_io.evaluation_slots import EvaluationSlots, active_slots
from benchmark_io.full_study import Pipeline


def test_both_methods_share_eight_slots_with_four_per_gpu(tmp_path):
    pools = [EvaluationSlots(tmp_path, ['0','1'], 4) for _ in range(2)]
    release = threading.Event()
    reached_capacity = threading.Event()
    mutex = threading.Lock()
    running = Counter()
    peaks = Counter()

    def job(index):
        with pools[index % 2].acquire(f'method{index%2}/{index}') as (gpu, slot):
            with mutex:
                running[gpu] += 1
                peaks[gpu] = max(peaks[gpu], running[gpu])
                assert running[gpu] <= 4
                if sum(running.values()) == 8:
                    reached_capacity.set()
            assert release.wait(10)
            with mutex:
                running[gpu] -= 1
            return gpu, slot

    with ThreadPoolExecutor(max_workers=16) as executor:
        jobs = [executor.submit(job, index) for index in range(16)]
        try:
            assert reached_capacity.wait(5)
            rows = active_slots(tmp_path)
            assert len(rows) == 8
            assert Counter(row['gpu'] for row in rows) == {'0': 4, '1': 4}
            assert len({(row['gpu'], row['slot']) for row in rows}) == 8
        finally:
            release.set()
        assert len([future.result() for future in jobs]) == 16
    assert peaks == {'0': 4, '1': 4}
    assert active_slots(tmp_path) == []


def test_exception_releases_slot_and_stale_record_is_not_active(tmp_path):
    pool = EvaluationSlots(tmp_path, ['0'], 1)
    with pytest.raises(RuntimeError):
        with pool.acquire('failed_worker'):
            raise RuntimeError('worker failed')
    assert active_slots(tmp_path) == []
    with pool.acquire('next_worker'):
        assert active_slots(tmp_path)[0]['owner'] == 'next_worker'


def test_evaluation_parallelism_does_not_increase_development_workers(tmp_path):
    pipeline = Pipeline(tmp_path, ['0'], method='aspire', eval_gpus=['0','1'], eval_workers_per_gpu=4)
    assert pipeline.slots.qsize() == 1
    assert pipeline.eval_slots.capacity == 8
    assert pipeline.provider_port('0') == 18110
    assert pipeline.provider_port('1') == 18210


def test_queue_adopts_existing_frozen_episode_without_another_start(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'robomme_plus', SimpleNamespace())
    case = tmp_path/'aspire/demo-only/R0'
    evaluation = case/'evaluation'
    evaluation.mkdir(parents=True)
    (evaluation/'frozen_evaluation.json').write_text('{}')
    for index in range(50):
        output = evaluation/'episodes/BinFill'/f'episode_{index:03d}'
        output.mkdir(parents=True)
        if index:
            (output/'result.json').write_text(json.dumps(dict(task='BinFill',episode=index,task_success=False)))
    output = evaluation/'episodes/BinFill/episode_000'
    argv = Path(f'/proc/{os.getpid()}/cmdline').read_bytes().decode().strip('\0').split('\0')
    (output/'worker_process.json').write_text(json.dumps(dict(pid=os.getpid(),argv=argv)))
    (output/'execution_placement.json').write_text(json.dumps(dict(gpu='0')))
    monkeypatch.setattr('benchmark_io.full_study.EvaluationStudy', lambda root: SimpleNamespace(
        attempts=lambda: pytest.fail('A live episode must be adopted before any new reservation')))
    def finish_existing(_):
        (output/'result.json').write_text(json.dumps(dict(task='BinFill',episode=0,task_success=True)))
    monkeypatch.setattr('benchmark_io.full_study.time.sleep', finish_existing)
    pipeline = Pipeline(tmp_path, ['0'], method='aspire', eval_gpus=['0','1'], eval_workers_per_gpu=4)
    monkeypatch.setattr(pipeline,'run',lambda *a,**kw:pytest.fail('Must not relaunch a completed or live episode'))
    rows = pipeline.aspire_evaluation(case, SimpleNamespace(config={'setting':'demo-only'}), ['BinFill'], 0)
    assert len(rows) == 50
    assert sum(row['task_success'] for row in rows) == 1
    assert active_slots(tmp_path) == []

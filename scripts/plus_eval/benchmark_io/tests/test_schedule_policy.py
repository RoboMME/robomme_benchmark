from benchmark_io.full_study import write
from benchmark_io.schedule_policy import evaluation_gate, evaluation_slots, evaluation_placement
from benchmark_io.unified_study import Queue, configurations
import pytest


@pytest.mark.parametrize('total,per_gpu', [(0,9),(25,13),(18,8),(True,9),(18,9.0)])
def test_evaluation_slots_reject_invalid_limits(tmp_path,total,per_gpu):
    write(tmp_path/'schedule_policy.json',dict(evaluation_workers_total=total,evaluation_workers_per_gpu=per_gpu))
    with pytest.raises(ValueError,match='evaluation concurrency'):
        evaluation_slots(tmp_path)


@pytest.mark.parametrize('limits', [{'0':15}, {'0':15,'1':3,'2':1}, {'0':15,'1':True},
    {'0':15,'1':-1}, {'0':25,'1':0}, {'0':14,'1':3}, [15,3]])
def test_evaluation_placement_rejects_invalid_gpu_maps(tmp_path, limits):
    write(tmp_path/'schedule_policy.json',dict(evaluation_workers_total=18,evaluation_workers_by_gpu=limits))
    with pytest.raises(ValueError,match='evaluation concurrency'):
        evaluation_placement(tmp_path)


def test_gpu_limits_reload_and_allow_draining_a_device_completely(tmp_path):
    assert evaluation_placement(tmp_path)==(16,{'0':8,'1':8})
    write(tmp_path/'schedule_policy.json',dict(evaluation_workers_total=18,evaluation_workers_by_gpu={'0':15,'1':3}))
    assert evaluation_placement(tmp_path)==(18,{'0':15,'1':3})
    write(tmp_path/'schedule_policy.json',dict(evaluation_workers_total=18,evaluation_workers_by_gpu={'0':18,'1':0}))
    assert evaluation_placement(tmp_path)==(18,{'0':18,'1':0})
    write(tmp_path/'schedule_policy.json',dict(evaluation_workers_total=20,evaluation_workers_by_gpu={'0':10,'1':10}))
    assert evaluation_placement(tmp_path)==(20,{'0':10,'1':10})


def test_scheduler_override_does_not_change_episode_worker_runtime(tmp_path):
    from benchmark_io.unified_study import environment
    from benchmark_io.watch_unified import scheduler_environment
    import os
    study = tmp_path/'study'
    (study/'runtime_source').mkdir(parents=True)
    scheduler = tmp_path/'operator'
    (scheduler/'benchmark_io').mkdir(parents=True)
    (scheduler/'benchmark_io/unified_study.py').write_text('')
    worker_env = environment(study)
    override = scheduler_environment(study, scheduler)
    assert override['PYTHONPATH'].split(os.pathsep)[0] == str(scheduler)
    assert override['PYTHONPATH'].split(os.pathsep)[1:] == worker_env['PYTHONPATH'].split(os.pathsep)
    assert environment(study) == worker_env
    assert scheduler_environment(study) == worker_env
    with pytest.raises(ValueError, match='Scheduler code'):
        scheduler_environment(study, tmp_path/'missing')


def freeze(queue, key):
    write(queue.root/key/'development_complete.json',dict(policies_frozen=True))
    queue.ready(key,['BinFill'])


def test_first_finished_configuration_does_not_release_evaluation(tmp_path):
    queue=Queue(tmp_path);queue.seed()
    write(tmp_path/'schedule_policy.json',dict(evaluation_mode='after_all_development'))
    key=next(configurations())[0]
    freeze(queue,key);queue.update('develop/'+key,state='done')
    assert evaluation_gate(tmp_path)['development_completed']==1
    assert queue.next_job('evaluate') is None
    assert queue.next_job('develop') is not None
    assert len(queue.jobs('queued','evaluate'))==50
    # The hold persists through a scheduler/queue restart without deleting work.
    assert Queue(tmp_path).next_job('evaluate') is None


@pytest.mark.parametrize('optional_budgets', [False, True])
def test_all_configurations_must_finish_and_freeze_before_release(tmp_path, optional_budgets):
    if optional_budgets:
        write(tmp_path/'study_plan.json', dict(configurations=[
            dict(key=f'{method}/{setting}/R{budget}', method=method, setting=setting, budget=budget)
            for method in ('aspire', 'rax')
            for setting, budgets in (('demo-only', [0]), ('interaction-only', [1, 3, 5, 10]), ('demo&interaction', [1, 3, 5, 10]))
            for budget in budgets]))
    queue=Queue(tmp_path);queue.seed()
    write(tmp_path/'schedule_policy.json',dict(evaluation_mode='after_all_development'))
    keys=[row[0] for row in configurations(tmp_path)]
    assert evaluation_gate(tmp_path)['development_total'] == len(keys) == (18 if optional_budgets else 6)
    for key in keys[:-1]:
        freeze(queue,key);queue.update('develop/'+key,state='done')
    last=keys[-1]
    freeze(queue,last);queue.update('develop/'+last,state='failed')
    assert not evaluation_gate(tmp_path)['enabled']
    assert queue.next_job('evaluate') is None
    queue.update('develop/'+last,state='running')
    assert queue.next_job('evaluate') is None
    queue.update('develop/'+last,state='done')
    assert evaluation_gate(tmp_path)['development_completed']==len(keys)
    assert queue.next_job('evaluate') is not None


def test_absent_policy_preserves_legacy_schedule_and_explicit_hold_is_durable(tmp_path):
    queue=Queue(tmp_path);queue.seed()
    freeze(queue,next(configurations())[0])
    assert queue.next_job('evaluate') is not None
    write(tmp_path/'schedule_policy.json',dict(evaluation_mode='held'))
    assert Queue(tmp_path).next_job('evaluate') is None


def test_case_review_hold_is_reloadable_and_leaves_other_cases_running(tmp_path):
    queue=Queue(tmp_path);queue.seed()
    first,second=[row[0] for row in configurations()][:2]
    for key in (first,second):
        freeze(queue,key);queue.update('develop/'+key,state='done')
    policy=dict(evaluation_mode='asynchronous',evaluation_excluded_cases=[first])
    write(tmp_path/'schedule_policy.json',policy)
    assert queue.next_job('evaluate')['case_key']==second
    assert queue.next_job('develop') is not None
    policy['evaluation_excluded_cases']=[first,second]
    write(tmp_path/'schedule_policy.json',policy)
    assert Queue(tmp_path).next_job('evaluate') is None
    policy['evaluation_excluded_cases']=[]
    write(tmp_path/'schedule_policy.json',policy)
    assert queue.next_job('evaluate')['case_key']==first


def test_sanity_subset_stops_at_two_episodes_and_preserves_full_queue(tmp_path):
    queue=Queue(tmp_path);queue.seed()
    first,second=[row[0] for row in configurations()][:2]
    for key in (first,second):
        write(tmp_path/key/'development_complete.json',dict(policies_frozen=True))
        queue.ready(key,['BinFill','VideoPlaceOrder'])
    # A completed result is reused, not re-dispatched by the new subset.
    prior=f'evaluate/{first}/BinFill/000'
    queue.update(prior,state='done')
    policy=dict(evaluation_mode='asynchronous',evaluation_episode_indices=[1,0,1],
                evaluation_excluded_cases=[second])
    write(tmp_path/'schedule_policy.json',policy)
    seen=[]
    while job:=queue.next_job('evaluate'):
        assert job['case_key']==first and job['episode'] in (0,1)
        seen.append((job['task'],job['episode']))
        queue.update(job['id'],state='done')
    assert sorted(seen)==[('BinFill',1),('VideoPlaceOrder',0),('VideoPlaceOrder',1)]
    assert Queue(tmp_path).next_job('evaluate') is None
    assert queue.next_job('develop') is not None
    assert len(queue.jobs('queued','evaluate'))==196
    # Removing a case exclusion must not remove the independent episode cap.
    policy['evaluation_excluded_cases']=[]
    write(tmp_path/'schedule_policy.json',policy)
    assert queue.next_job('evaluate')['case_key']==second
    policy['evaluation_episode_indices']=[]
    write(tmp_path/'schedule_policy.json',policy)
    assert queue.next_job('evaluate') is None
    # Only an explicit scope change allows the remaining full evaluation.
    policy['evaluation_episode_indices']=None
    policy['evaluation_excluded_cases']=[second]
    write(tmp_path/'schedule_policy.json',policy)
    assert queue.next_job('evaluate')['episode']==2


@pytest.mark.parametrize('invalid', [[-1],[50],[True],'0,1',[1.0]])
def test_sanity_subset_rejects_invalid_indices(tmp_path,invalid):
    queue=Queue(tmp_path);queue.seed();freeze(queue,next(configurations())[0])
    write(tmp_path/'schedule_policy.json',dict(evaluation_mode='asynchronous',evaluation_episode_indices=invalid))
    with pytest.raises(ValueError,match='released test indices'):
        queue.next_job('evaluate')

import json

import pytest

from benchmark_io.publish_progress import public_progress
from benchmark_io.study_configurations import configurations
from benchmark_io.unified_study import Queue


def optional_budget_plan():
    return {'configurations': [dict(key=f'{method}/{setting}/R{budget}', method=method,
                                    setting=setting, budget=budget)
            for method in ('aspire', 'rax')
            for setting, budgets in (('demo-only', [0]), ('interaction-only', [1, 3, 5, 10]), ('demo+interaction', [1, 3, 5, 10]))
            for budget in budgets]}


def test_new_study_schedules_only_current_budgets(tmp_path):
    queue = Queue(tmp_path)
    queue.seed()
    jobs = queue.jobs(kind='develop')
    expected = {(method, setting, budget) for method in ('aspire', 'rax')
                for setting, budgets in (('demo-only', [0]), ('interaction-only', [10]), ('demo+interaction', [10]))
                for budget in budgets}
    assert len(jobs) == len(expected) == 6
    assert {(j['method'], j['setting'], j['budget']) for j in jobs} == expected
    progress = public_progress(tmp_path)
    assert len(progress['rows']) == 6 and progress['test_trials_planned'] == 4800


@pytest.mark.parametrize('keep_plan', [True, False])
def test_resume_retains_pinned_assignments_and_finished_jobs(tmp_path, keep_plan):
    path = tmp_path / 'study_plan.json'
    path.write_text(json.dumps(optional_budget_plan()))
    queue = Queue(tmp_path)
    queue.seed()
    queue.update('develop/rax/interaction-only/R1', state='done', finished=123)
    before = queue.jobs()
    if not keep_plan:
        path.unlink()  # Earlier studies may have only a durable queue.
    Queue(tmp_path).seed()
    assert queue.jobs() == before and len(before) == 18
    assert len(list(configurations(tmp_path))) == 18
    progress = public_progress(tmp_path)
    assert len(progress['rows']) == 18 and progress['test_trials_planned'] == 14400


@pytest.mark.parametrize('fault', ['empty', 'duplicate', 'path', 'zero_interaction'])
def test_invalid_pinned_assignment_never_falls_back_to_new_defaults(tmp_path, fault):
    plan = optional_budget_plan()
    if fault == 'empty':
        plan['configurations'] = []
    elif fault == 'duplicate':
        plan['configurations'].append(plan['configurations'][0])
    elif fault == 'path':
        plan['configurations'][0]['key'] = '../elsewhere'
    else:
        plan['configurations'][0].update(setting='interaction-only', key='aspire/interaction-only/R0')
    (tmp_path / 'study_plan.json').write_text(json.dumps(plan))
    queue = Queue(tmp_path)
    with pytest.raises(ValueError):
        queue.seed()
    assert queue.jobs() == []

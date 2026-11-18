"""Small allowlisted progress snapshot for the study result summary."""
import json
from pathlib import Path
import time
import uuid
from .evaluation_slots import active_slots
from .development_selection import rax_source, training_accounting


def read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def case_phase(case, status, method, setting, budget):
    """A method can develop one configuration while evaluating a different one."""
    development=read(case/'development_status.json',{})
    evaluation=read(case/'evaluation_status.json',{})
    if evaluation.get('phase') in ('evaluation','starting_providers','needs_attention','completed'):
        return evaluation
    if (case/'development_complete.json').is_file():
        return dict(phase='awaiting_evaluation')
    if development:
        return development
    if (status.get('method'),status.get('setting'),status.get('budget'))==(method,setting,budget):
        return status
    return dict(phase='queued')


def rax_live_results(case):
    """Read finished native episodes while their 50-episode batch is still active."""
    episode_results=list((case/'evaluation/episodes').glob('*/*/result.json'))
    if episode_results:
        return [read(p,{}) for p in episode_results]
    rows=[]
    for path in (case/'development').glob('*/state.json'):
        saved=read(case/'evaluation'/path.parent.name/'results.json',None)
        if saved is not None:
            rows.extend(saved)
        else:
            path=rax_source(case,path.parent.name)[0]/'state.json'
            rows.extend(dict(task_success=r.get('formal_outcome')=='success',native_result=r)
                for r in read(path,{}).get('results',[]) if r.get('mode')=='formal')
    return rows


def public_progress(root):
    root=Path(root)
    legacy_status=read(root/'status.json',{})
    rows=[]
    for method in ('aspire','rax'):
        status=read(root/method/'status.json',legacy_status)
        for setting,budgets in [('AfD',[0]),('AfI',[1,3,5,10]),('AfD+I',[1,3,5,10])]:
            for budget in budgets:
                case=root/method/setting/f'R{budget}'
                complete=read(case/'completed.json',{})
                stage=case_phase(case,status,method,setting,budget)
                phase='completed' if complete else stage.get('phase','queued')
                accounting=training_accounting(case)
                if method=='aspire':
                    programs=sum((case/'development/workspace/campaign/robomme_plus/robomme_plus'/f'task_{i}'/'fix_code.py').is_file() for i in range(16))
                else:
                    programs=sum(bool(read(rax_source(case,p.parent.name)[0]/'development_result.json',{}).get('submission'))
                                 for p in (case/'development').glob('*/development_result.json'))
                costs=read(case/'api_cost_summary.json',{}).get('groups',[])
                requests=sum(c.get('requests',0) for c in costs)
                cost=sum(c.get('known_estimated_cost_usd',0) for c in costs)
                pending=sum(c.get('pending_usage_requests',0) for c in costs)
                progress=read(case/'evaluation/progress.json',{})
                done=800 if complete else progress.get('completed',stage.get('completed',0) if phase=='evaluation' else 0)
                successes=complete.get('successes',progress.get('successes',stage.get('successes') if phase=='evaluation' else None))
                errors=0
                if method=='rax':
                    native=rax_live_results(case)
                    if not complete and len(native)>done:
                        done=len(native)
                        successes=sum(bool(r.get('task_success')) for r in native)
                    errors=sum(r.get('native_result',{}).get('formal_outcome')=='error' for r in native)
                rows.append(dict(method=method,setting=setting,budget=budget,
                    phase=phase,
                    programs_saved=programs,**accounting,training_ceiling=16*budget,
                    evaluation_completed=done,evaluation_total=800,successes=successes,
                    execution_errors=errors,
                    success_rate=complete.get('success_rate'),api_requests=requests,
                    estimated_development_api_cost_usd=round(cost,6),pending_usage_requests=pending,
                    evaluation_api_cost_usd=complete.get('evaluation_api_cost_usd'),
                    artifacts='full' if budget in (0,10) else 'results and logs',
                    results_provisional=bool(errors or complete.get("infrastructure_errors"))))
    placement = read(root/'runtime/evaluation_config.json',{})
    occupied = active_slots(root)
    shared_status=read(root/'status.json',{}) if placement.get('scope')=='shared_methods' else {}
    parallelism = dict(total_workers=placement.get('total_workers'),
        workers_per_gpu=placement.get('workers_per_gpu'),
        workers_by_gpu=placement.get('workers_by_gpu'),
        method_gpus=placement.get('method_gpus'),
        asynchronous=placement.get('asynchronous_development_evaluation',False),
        evaluation_gate=placement.get('evaluation_gate'),
        shared_methods=placement.get('scope')=='shared_methods',
        development_workers_total=placement.get('development_workers_total'),
        development_running=shared_status.get('development_running'),
        active_by_gpu=shared_status.get('active_evaluation_by_gpu',
            {gpu:sum(row.get('gpu')==gpu for row in occupied) for gpu in placement.get('gpus',[])}))
    return dict(updated=time.time(),model='Claude Opus 5.5',provider='AWS Bedrock',reasoning_effort='high',
        evaluation_parallelism=parallelism,
        external_gemini=False,rows=rows,configurations_completed=sum(r['phase']=='completed' for r in rows),
        test_trials_planned=14400,cost_basis='Estimated USD list prices from provider usage; pending requests are excluded from known cost.')


def publish(root, destination):
    destination=Path(destination)
    data=read(destination,{})
    data['full_study']=public_progress(root)
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(data,indent=2))
    temporary.replace(destination)

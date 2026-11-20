import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark_io.costs import estimate, record_usage, summarize
from benchmark_io.protocol import Study, BudgetExhausted
from benchmark_io.episode import Episode
from aspire.campaign import resolve_task
from aspire.lean_recording import LogRecorder
from cap_harness.contracts import CameraObservation, Observation, Pose, RobotState, StepResult


def test_cost_counts_each_cache_duration_once_and_separates_settings(tmp_path):
    usage=dict(input_tokens=100,output_tokens=200,cache_read_input_tokens=1000,
               cache_creation_input_tokens=300,
               cache_creation=dict(ephemeral_5m_input_tokens=100,ephemeral_1h_input_tokens=200))
    expected=(100*4.4+200*22+1000*.22+100*5.5+200*8.8)/1e6
    assert estimate(usage)['estimated_cost_usd']==pytest.approx(expected)
    for setting in ('AfD','AfI'):
        directory=tmp_path/setting;directory.mkdir()
        record_usage(directory,usage,complete=True,context=dict(method='aspire',setting=setting,budget=0,phase='development'))
    result=summarize(tmp_path)
    assert len(result['groups'])==2
    assert all(row['known_estimated_cost_usd']==pytest.approx(expected) for row in result['groups'])


def test_partial_response_or_missing_cache_duration_is_not_reported_as_exact_cost(tmp_path):
    result=record_usage(tmp_path,{'output_tokens':100},complete=False)
    assert result['estimated_cost_usd'] is None
    assert result['estimated_cost_range_usd'][0]>0
    unknown=estimate(dict(cache_creation_input_tokens=100))
    assert unknown['estimated_cost_usd'] is None
    assert unknown['estimated_cost_range_usd']==[.00055,.00088]


def test_campaign_cannot_silently_route_ambiguous_task_to_first_task():
    tasks=['BinFill','StopCube']
    assert resolve_task({'task_id':'1'},tasks)=='StopCube'
    with pytest.raises(ValueError,match='disagree'):
        resolve_task({'task':'BinFill','task_id':'1'},tasks)
    with pytest.raises(ValueError,match='requires'):
        resolve_task({},tasks)
    with pytest.raises(ValueError):
        resolve_task({'task_id':'-1'},tasks)


def test_rehearsal_and_verification_share_training_ceiling_and_fail_before_reset(tmp_path):
    rows={'BinFill':[dict(seed=i,max_steps=20,control_freq=20) for i in range(3)]}
    study=Study(tmp_path/'study',method='rax',setting='AfI',budget=3,records=rows)
    for purpose in ('exploration','frozen_bundle_rehearsal','verification'):
        attempt=study.reserve('BinFill',purpose=purpose)
        study.finish(attempt['id'],dict(success=False))
    constructed=[]
    def builder(*args,**kwargs):
        constructed.append(True)
        raise AssertionError('Must not reach the simulator')
    with pytest.raises(BudgetExhausted):
        Episode('BinFill',0,study=study,output=tmp_path/'forbidden',builder_factory=builder)
    assert not constructed
    assert len(study.attempts())==3


def test_lean_recorder_retains_tool_logs_without_image_or_array_payloads(tmp_path):
    recorder=LogRecorder(output_root=tmp_path,benchmark='robomme_plus',suite='robomme_plus',
                         task_id=0,task_name='BinFill',seed=0,max_steps=5)
    camera=CameraObservation(rgb=np.ones((4,4,3),np.uint8),depth_m=np.ones((4,4)),
        intrinsics=np.eye(3),frame='front',camera_pose=Pose(position=[0,0,0],quaternion_wxyz=[1,0,0,0],frame="robot_base"))
    with recorder.span('camera_read',category='tool',inputs={}) as span:
        assert span.output(camera) is camera
    recorder.keyframe('requested')
    recorder.finalize(program_ok=True,task_success=False,termination_reason='program_completed',program_result=None)
    assert not [p for p in recorder.root.rglob('*') if p.suffix in ('.png','.mp4','.npy','.npz')]
    logged=next(recorder.root.glob('trace/calls/*/output.json'))
    assert json.loads(logged.read_text())['rgb']['shape']==[4,4,3]


def test_rax_geometry_uses_pinned_benchmark_but_operator_adapter_path(monkeypatch,tmp_path):
    from services.robomme_plus.process import simulator_environment
    monkeypatch.setenv('ROBOMME_PLUS_ROOT',str(tmp_path))
    env=simulator_environment('GPU-example')
    assert str(tmp_path/'src') in env['PYTHONPATH'].split(':')
    import services.robomme_plus.process as process
    assert str(Path(process.__file__).resolve().parents[3]) in env['PYTHONPATH'].split(':')


def test_lean_episode_keeps_sensor_inputs_and_step_accounting(monkeypatch,tmp_path):
    metadata=dict(seed=1,max_steps=20,control_freq=20)
    completed=[]
    study=SimpleNamespace(config={'records':{'BinFill':[metadata]},'artifact_profile':'logs'},
        reserve=lambda *a,**kw:dict(id=1,episode=0,purpose='frozen_test'),
        finish=lambda ident,value:completed.append(value))
    obs={f'{key}_list':[np.zeros(size)] for key,size in [('joint_state',7),('eef_state',6),('gripper_state',2)]}
    for name in ('front','wrist'):
        obs.update({f'{name}_rgb_list':[np.full((4,4,3),50,np.uint8)],
                    f'{name}_depth_list':[np.full((4,4),1000)],
                    f'{name}_camera_extrinsic_list':[np.c_[np.eye(3),np.zeros(3)]]})
    info={'task_goal':'test goal','front_camera_intrinsic':np.eye(3),'wrist_camera_intrinsic':np.eye(3)}
    class Env:
        def reset(self):return obs,info
        def step(self,action):return obs,0,False,False,{**info,'execution_total_control_steps':1}
        def close(self):pass
    class Builder:
        def __init__(self,*a,**kw):pass
        def get_episode_metadata(self,index):return metadata
        def make_env_for_episode(self,*a,**kw):return Env()
    monkeypatch.setattr(Episode,'_robot_calibration',lambda self:None)
    episode=Episode('BinFill',0,study=study,output=tmp_path/'episode',builder_factory=Builder)
    frame=episode.step(np.zeros(8))
    np.testing.assert_array_equal(frame['cameras']['front']['rgb'],obs['front_rgb_list'][0])
    np.testing.assert_array_equal(frame['cameras']['wrist']['depth_m'],np.ones((4,4)))
    assert episode.steps==1
    episode.close()
    assert completed[0]['control_steps']==1
    assert not any((tmp_path/'episode'/name).exists() for name in ('rollout.mp4','actions.jsonl','observations.jsonl'))


def test_parallel_method_queues_keep_placement_status_and_relays_separate(tmp_path,monkeypatch):
    from benchmark_io.full_study import Pipeline
    from benchmark_io.publish_progress import public_progress
    launched=[]
    monkeypatch.setattr('benchmark_io.full_study.time.sleep',lambda _:None)
    for method,gpu,port in [('aspire','0',18177),('rax','1',18277)]:
        pipeline=Pipeline(tmp_path,[gpu],method=method,proxy_port=port)
        assert pipeline.slots.get()[0]==gpu
        pipeline.current=dict(method=method,setting='AfD',budget=0,case='PRIVATE_PATH')
        pipeline.status(phase='development')
        monkeypatch.setattr(pipeline,'launch',lambda cmd,*a,**kw:(launched.append(cmd) or SimpleNamespace(poll=lambda:None)))
        pipeline.relay(tmp_path/method/'AfD/R0',method,'AfD',0)
        assert launched[-1][launched[-1].index('--port')+1]==str(port)
    report=public_progress(tmp_path)
    active=[r for r in report['rows'] if r['phase']=='development']
    assert {r['method'] for r in active}=={'aspire','rax'}
    assert len(report['rows'])==10
    assert report['test_trials_planned']==8000
    assert 'PRIVATE_PATH' not in json.dumps(report)

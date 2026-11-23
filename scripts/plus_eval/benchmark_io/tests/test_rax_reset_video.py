import json
import socket
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from PIL import Image

from services.robomme_plus.backend import RoboMMEPlusBackend
from services.robomme_plus.session import RoboMMEPlusSession
from services.robomme_plus.reset_video import reset_frames
from services.simulator import protocol
from services.simulator.io import RejectedBeforeAction
from services.controller.recording import RunRecording
from services.controller.gateway import Gateway
from services.mcp_contract import Contract


def observation(index):
    return dict(step_id=0, phase='execution' if index==4 else 'video_demo',
        instruction='test goal', state=[float(index)]*8,
        front=np.full((16,16,3),index,np.uint8), wrist=np.full((16,16,3),index+10,np.uint8),
        front_depth_m=np.full((16,16),.125,np.float32),
        wrist_depth_m=np.full((16,16),.5,np.float32),
        camera_intrinsics={'front':{'matrix':np.eye(3).tolist()},'wrist':{'matrix':np.eye(3).tolist()}},
        camera_poses={'front':{'position_m':[1,2,3]},'wrist':{'position_m':[4,5,6]}})


def session():
    sim=object.__new__(RoboMMEPlusSession)
    sim.episode_id='fixed-episode';sim.step_id=0
    sim.initial_frames=[observation(i) for i in range(5)]
    return sim


def test_video_exceeding_rpc_cap_transfers_losslessly_without_reset_or_step(monkeypatch):
    source=session()
    monkeypatch.setattr('services.robomme_plus.reset_video.RESET_PAGE_FRAMES',2)
    monkeypatch.setattr('services.robomme_plus.session.RESET_PAGE_FRAMES',2)
    first=source.dispatch('reset_sequence_page',dict(episode_id=source.episode_id,step_id=0,start=0,count=2))
    cap=len(msgpack.packb(first,default=protocol._encode,use_bin_type=True))+512
    monkeypatch.setattr(protocol,'MAX_DECOMPRESSED_BYTES',cap)
    left,right=socket.socketpair()
    requests=[]
    try:
        with pytest.raises(ValueError,match='RPC message too large'):
            protocol.send_packet(left,dict(frames=source.initial_frames))
        def request(op,**args):
            requests.append((op,args))
            protocol.send_packet(left,source.dispatch(op,dict(episode_id=source.episode_id,step_id=0,**args)))
            return protocol.receive_packet(right)
        frames=list(reset_frames(SimpleNamespace(episode_id=source.episode_id,step_id=0,request=request)))
    finally:
        left.close();right.close()
    assert len(frames)==5
    assert [args['start'] for _,args in requests]==[0,2,4]
    assert all(op=='reset_sequence_page' for op,_ in requests)
    assert source.step_id==0
    for actual,expected in zip(frames,source.initial_frames):
        for key,value in expected.items():
            if isinstance(value,np.ndarray):
                assert actual[key].dtype==value.dtype
                np.testing.assert_array_equal(actual[key],value)
            else:
                assert actual[key]==value


@pytest.mark.parametrize('start,count',[(True,1),(-1,1),(5,1),(0,0),(0,17),(0,True)])
def test_page_bounds_cannot_return_unlimited_data(start,count):
    s=session()
    with pytest.raises(RejectedBeforeAction):
        s.dispatch('reset_sequence_page',dict(episode_id=s.episode_id,step_id=0,start=start,count=count))


@pytest.mark.parametrize('field,value',[
    ('episode_id','other'),('step_id',1),('start',1),('next_start',0),('total_frames',0)])
def test_client_rejects_inconsistent_pages(field,value):
    def request(*args,**kwargs):
        return {**dict(episode_id='fixed-episode',step_id=0,start=0,next_start=1,
                       total_frames=1,frames=[observation(0)]),field:value}
    with pytest.raises(ValueError):
        list(reset_frames(SimpleNamespace(episode_id='fixed-episode',step_id=0,request=request)))


def test_sequence_writes_frames_incrementally_and_keeps_existing_contract(tmp_path):
    backend=object.__new__(RoboMMEPlusBackend)
    backend.sim=SimpleNamespace(episode_id='fixed-episode',step_id=0,run_id='test-run')
    backend.config=SimpleNamespace(task='BinFill')
    backend.frame_root=tmp_path/'runtime/frames'
    backend.contract=SimpleNamespace(observation=SimpleNamespace(geometry=True,clean=lambda x:x))
    backend._environment_spec={'native_step_limit':100}
    def frames():
        for i in range(5):
            if i:
                assert len(list(backend.frame_root.glob('robomme_plus/test-run/*/frame_*_front_rgb.png')))==i
            yield observation(i)
    transition=dict(episode_id='fixed-episode',step_id=0,phase='conditioning_video_then_execution',frame_stride=1)
    packet,images=backend._sequence_packet(frames(),transition)
    sequence=packet['frame_sequence']
    assert sequence['frame_count']==5
    assert sequence['frame_index_range']==[0,4]
    assert sequence['returned_images']=='final_frame_only'
    assert len(images)==4
    assert packet['state']==[4.]*8
    manifest=json.loads((tmp_path/sequence['manifest_path']).read_text())
    assert manifest['transition']==transition
    assert [row['phase'] for row in manifest['frames']]==['video_demo']*4+['execution']
    for i,row in enumerate(manifest['frames']):
        assert row['frame_index']==i and row['step_id']==0
        assert row['camera_intrinsics']==observation(i)['camera_intrinsics']
        assert len(row['files'])==4
        for file in row['files']:
            with Image.open(tmp_path/file['path']) as image:
                a=np.asarray(image)
            expected=(i+(10 if file['camera']=='wrist' else 0)) if file['kind']=='rgb' else (500 if file['camera']=='wrist' else 125)
            assert np.all(a==expected)


def test_start_boundary_publishes_every_reset_frame_to_controller_mount(tmp_path):
    backend=object.__new__(RoboMMEPlusBackend)
    backend.config=SimpleNamespace(task='BinFill', benchmark='robomme_plus', observation_profile='rgbd', external_perception=False)
    backend.contract=Contract.from_config(backend.config, isolated=True)
    backend._environment_spec={'episode_step_limit':100}
    backend._initial_sequence=None
    backend.frame_workspace=tmp_path/'private'
    backend.frame_root=backend.frame_workspace/'runtime/frames'
    calls=[]
    def request(op):
        calls.append(op)
        assert op=='observe'
        return observation(4)
    backend.sim=SimpleNamespace(episode_id='fixed-episode',step_id=0,run_id='test-run',active=True,request=request)
    transition=dict(episode_id='fixed-episode',step_id=0,phase='conditioning_video_then_execution',frame_stride=1)
    packet,_=backend._sequence_packet(iter(observation(i) for i in range(5)),transition)
    backend._initial_sequence=(backend.sim.episode_id,packet['frame_sequence'])
    recording=RunRecording(tmp_path/'published','formal-'+'a'*32)
    gateway=Gateway(backend,None,audit=recording.event,charge=lambda *a:None,refund=lambda *a:None,
        budget=lambda:None,publish_sequence=lambda value:recording.sequence(value,backend.frame_workspace),
        contract=backend.contract)
    token=gateway.acquire()
    response=gateway.handle(token,{'jsonrpc':'2.0','id':'supervisor-start','method':'tools/call',
        'params':{'name':'robomme_observe','arguments':{}}})
    value=json.loads(response['result']['content'][0]['text'])
    descriptor=value['frame_sequence']
    public=recording.frames/'sequence_000000/manifest.json'
    manifest=json.loads(public.read_text())
    assert manifest['frame_count']==descriptor['frame_count']==5
    assert descriptor['manifest_path'].startswith('runtime/autonomous_controller/runs/formal-')
    assert len(list(public.parent.glob('*.png')))==20
    for record in manifest['frames']:
        for file in record['files']:
            image=public.parent/Path(file['path']).name
            assert image.is_file() and image.stat().st_mode & 0o444
    assert [r['phase'] for r in manifest['frames']]==['video_demo']*4+['execution']
    assert calls==['observe'] and backend.sim.step_id==0
    # Repeated observation does not duplicate publication or consume a step.
    gateway.handle(token,{'jsonrpc':'2.0','id':'again','method':'tools/call',
        'params':{'name':'robomme_observe','arguments':{}}})
    assert len(list(recording.frames.iterdir()))==1
    backend.sim.step_id=1
    next_packet,_=backend._call('robomme_observe',{})
    assert 'frame_sequence' not in next_packet

"""Generate public RAX format examples through serializers, without a simulator.

Operator-side documentation tooling only. No benchmark scene, dataset, policy or
API is opened. The agent receives JSON/PNG artifacts, not this implementation.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np

from services.controller.gateway import Gateway
from services.controller.recording import RunRecording
from services.mcp_contract import Contract
from services.robomme_plus.backend import RoboMMEPlusBackend
from services.robomme_plus import tools as T

from .prompt_bundle import DEFAULT_BUNDLE
from .runtime_policy import sample_reset_frames


def synthetic_observation(phase='execution', *, source=2, timestamp=0., step=0):
    """Invented public sensor values, with two distinct blank RGB-D views."""
    return dict(
        phase=phase, step_id=step, frame_index=source, source_frame_index=source, timestamp_s=timestamp,
        instruction='Synthetic format example; no task to solve.',
        state=[0.1, 0.2, 0.3, 0., 0., 0., 0.04, 0.04],
        front=np.full((256, 256, 3), [40, 50, 60], dtype=np.uint8),
        wrist=np.full((256, 256, 3), [80, 90, 100], dtype=np.uint8),
        front_depth_m=np.full((256, 256), 0.5, dtype=np.float32),
        wrist_depth_m=np.full((256, 256), 0.8, dtype=np.float32),
        camera_intrinsics={c: dict(matrix=[[128., 0., 128.], [0., 128., 128.], [0., 0., 1.]],
                                   width=256, height=256) for c in T.CAMERAS},
        camera_poses={
            'front': dict(position_m=[1., 2., 3.], quaternion_wxyz=[1., 0., 0., 0.],
                          frame='world', convention='OpenGL'),
            'wrist': dict(position_m=[-1., 1., 2.], quaternion_wxyz=[0., 0., 0., 1.],
                          frame='world', convention='OpenGL'),
        },
    )


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='robomme-protocol-fixture-') as temporary:
        # Constructors that could start a process are deliberately not invoked.
        backend = object.__new__(RoboMMEPlusBackend)
        backend.config = SimpleNamespace(task='SyntheticTask')
        backend.contract = Contract(phase='isolated', benchmark='robomme_plus', external_perception=False)
        backend.sim = SimpleNamespace(episode_id='synthetic-episode', run_id='synthetic',
                                      step_id=0, active=True, finished=False)
        backend._environment_spec = dict(episode_step_limit=100)
        backend._instruction = 'Synthetic format example; no task to solve.'
        backend.frame_workspace = Path(temporary)/'native'
        backend.frame_root = backend.frame_workspace/'runtime/frames'
        workspace = Path(temporary)/'workspace'
        recording = RunRecording(workspace/'runtime/autonomous_controller', 'rehearsal-'+'0'*32)
        frames = [synthetic_observation('video_demo', source=i, timestamp=i/20) for i in range(2)]
        frames.append(synthetic_observation())
        _, sampling = sample_reset_frames(frames, 512)

        # Deterministic opaque IDs keep checked-in examples reproducible.
        with patch('services.robomme_plus.backend.uuid.uuid4', return_value=UUID(int=0)):
            initial = backend._sequence_packet(iter(frames),
                dict(frame_stride=1, video_sampling=sampling))
            class PacketSource:
                contract = backend.contract
                validate = staticmethod(T.validate_request)
                pending = initial

                def call(self, name, arguments):
                    if name == T.STATUS:
                        return backend._status(), []
                    if name == T.PIXEL_TO_POSITION:
                        return backend._pixel_to_position(arguments), []
                    return self.pending

            packets = PacketSource()
            gateway = Gateway(packets, None, audit=lambda event: None,
                charge=lambda *args: None, refund=lambda *args: None, budget=lambda: {},
                publish_sequence=lambda value: recording.sequence(value, backend.frame_workspace),
                contract=backend.contract)
            token = gateway.acquire()
            fixtures = {}

            def capture(filename, tool, arguments=None):
                response = gateway.handle(token, dict(jsonrpc='2.0', id=1, method='tools/call',
                    params=dict(name=tool, arguments=arguments or {})))
                fixtures[filename] = response.get('result', response)

            capture('rax_observe.json', T.OBSERVE)
            capture('rax_status.json', T.STATUS)
            capture('rax_pixel_to_position.json', T.PIXEL_TO_POSITION,
                    dict(observation_step_id=0, selections=[dict(camera='front', u=128, v=128)]))
            capture('rax_rejected_action.json', T.STEP, dict(actions=[[0., 0.]]))
            # Ordinary execution observation: no conditioning sequence after step 0.
            backend.sim.step_id = 1
            packets.pending = backend._observation_packet(synthetic_observation(step=1, timestamp=.05))
            capture('rax_live_observe.json', T.OBSERVE)
            # True terminal outcomes are separate examples, not simulated rollouts.
            for success in (False, True):
                gateway.terminal = False
                transition = dict(step_id=1, steps=[dict(step_id=1, terminated=True,
                    truncated=False, success=success)])
                packets.pending = backend._observation_packet(
                    synthetic_observation(step=1, timestamp=.05), transition=transition)
                capture('rax_terminal_'+('success' if success else 'failure')+'.json',
                        T.STEP, dict(actions=[[0.]*6+[-1.]]))

        # Copy only the public synthetic frames, not native artifacts or tooling.
        for path in sorted(workspace.rglob('*')):
            if path.is_file() and (path.suffix == '.png' or path.name == 'manifest.json'):
                target = destination/'rax_workspace'/path.relative_to(workspace)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for name, value in fixtures.items():
            (destination/name).write_text(json.dumps(value, indent=2)+'\n')
        observation = json.loads(fixtures['rax_observe.json']['content'][0]['text'])
        checks = dict(synthetic=True, generator='benchmark_io.protocol_fixtures',
            workspace_root='rax_workspace', initial_manifest=observation['frame_sequence']['manifest_path'],
            expected_rgb={'front': [40, 50, 60], 'wrist': [80, 90, 100]},
            expected_depth_m={'front': .5, 'wrist': .8},
            front_center_pixel_world_m=[1., 2., 2.5],
            notes='Blank images and invented values; no simulator, demonstration or policy execution.')
        (destination/'rax_checks.json').write_text(json.dumps(checks, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_BUNDLE/'fixtures')
    generate(parser.parse_args().output)

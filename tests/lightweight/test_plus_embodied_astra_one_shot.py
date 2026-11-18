"""One-shot sample selection, phase labels, prompts and conversation accounting."""
import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'scripts/plus_eval/agentic_methods/astra/embodied'
sys.path[:0] = [str(FOLDER), str(FOLDER.parent)]
from one_shot_examples import TrainingExample, TrainingExamples, phase_sampling
from agentic_methods.astra.image_sampling import image_count


def sample(root, task='PatternLock', episode=1, difficulty='medium', seed=41, frames=12, demo=5):
    plan = root / 'train_plan.json'
    rows = json.loads(plan.read_text()) if plan.exists() else []
    rows.append(dict(task=task, episode=episode, seed=seed, difficulty=difficulty))
    plan.write_text(json.dumps(rows))
    (root / 'h5').mkdir(exist_ok=True)
    path = root / 'h5' / f'record_dataset_{task}.h5'
    with h5py.File(path, 'a') as f:
        f.attrs.update(split='train', task=task)
        e = f.create_group(f'episode_{episode}')
        e.attrs.update(success=True, frames=frames, demonstration_frames=demo)
        setup = e.create_group('setup')
        setup['task'] = task
        setup['difficulty'] = difficulty or 'null'
        setup['seed'] = seed
        setup['control_freq'] = 20
        setup['task_goal'] = ['Training-only goal']
        setup['front_camera_intrinsic'] = [[120., 0., 2.], [0., 121., 2.], [0., 0., 1.]]
        setup['wrist_camera_intrinsic'] = [[130., 0., 1.5], [0., 131., 1.5], [0., 0., 1.]]
        for i in range(frames):
            t = e.create_group(f'timestep_{i}')
            obs = t.create_group('obs')
            obs['front_rgb'] = np.full((4, 4, 3), i, np.uint8)
            obs['wrist_rgb'] = np.full((4, 4, 3), i + 50, np.uint8)
            obs['eef_state'] = [i / 100., .2, .3, -3.14159265, 0., .1]
            obs['gripper_state'] = [.01 + i / 1000., .02 + i / 1000.]
            obs['front_camera_extrinsic'] = np.c_[np.eye(3), [i / 1000., .2, .3]]
            obs['wrist_camera_extrinsic'] = np.c_[np.eye(3), [i / 100., .4, .5]]
            info = t.create_group('info')
            info['is_video_demo'] = i < demo
            info['native_control_step'] = i
            info['grounded_subgoal_online'] = 'PRIVILEGED_TRAIN_SUBGOAL'
    return path


class TrainingExamplesTest(unittest.TestCase):
    def test_image_only_context_preserves_images_and_labels_without_training_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root)
            example = TrainingExamples(root, max_images=6).get('PatternLock')
            harness = SimpleNamespace(
                text_item=lambda text: {'type': 'input_text', 'text': text},
                image_item=lambda frame: {'type': 'input_image', 'frame': int(frame[0, 0, 0])})
            visual = example.content(harness, include_state=False)
            states = example.content(harness, include_state=True)
            self.assertEqual([p for p in visual if p['type'] == 'input_image'],
                             [p for p in states if p['type'] == 'input_image'])
            text = json.dumps(visual)
            for field in ('tcp_pose', 'finger_positions', 'extrinsic', 'camera_K',
                          'TRAINING CAMERA INTRINSICS', 'TRAINING FRAME STATE'):
                self.assertNotIn(field, text)
            self.assertLess(text.index('TRAINING VIDEO DEMO'), text.index('TRAINING EXECUTION'))
            self.assertIn('END TRAINING EXAMPLE', text)

    def test_explicit_hard_example_keeps_its_actual_label_and_validates_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sample(root, episode=0, difficulty='hard', seed=40)
            sample(root, episode=1, difficulty='medium', seed=41)
            example = TrainingExamples(root, difficulty='hard').get('PatternLock')
            self.assertEqual(example.report['seed'], 40)
            self.assertEqual(example.report['difficulty'], 'hard')
            self.assertEqual(example.report['selection'], 'hard_difficulty')
            self.assertEqual(TrainingExamples(root).get('PatternLock').report['seed'], 41)
            with h5py.File(path, 'a') as source:
                source['episode_0/setup/difficulty'][()] = 'medium'
            with self.assertRaisesRegex(ValueError, 'difficulty disagree'):
                TrainingExamples(root, difficulty='hard').get('PatternLock')

    def test_selects_medium_by_label_and_uses_insert_peg_episode_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root, episode=0, difficulty='hard', seed=40)
            sample(root, episode=2, difficulty='medium', seed=42)
            sample(root, task='InsertPeg', difficulty=None, seed=43)
            examples = TrainingExamples(root, max_images=6)
            self.assertEqual(examples.get('PatternLock').report['episode'], 2)
            peg = examples.get('InsertPeg').report
            self.assertEqual((peg['episode'], peg['selection']), (1, 'user_selected_episode_1_unlevelled'))
            self.assertIsNone(examples.get('MoveCube'))  # No MoveCube file exists.
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                examples.get('MissingTask')

    def test_phases_keep_endpoints_and_rgb_views_without_privileged_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root)
            example = TrainingExamples(root, max_images=6).get('PatternLock')
            phases = example.report['phases']
            self.assertEqual(phases['video_demo']['frame_indices'][::2], [0, 4])
            self.assertEqual(phases['execution']['frame_indices'][::2], [5, 11])
            self.assertEqual(example.report['input_images'], 6)
            self.assertEqual(example.phases['video_demo'][0].shape, (4, 8, 3))
            self.assertTrue((example.phases['video_demo'][0][:, :4] == 0).all())
            self.assertTrue((example.phases['video_demo'][0][:, 4:] == 50).all())
            harness = SimpleNamespace(text_item=lambda s: {'type': 'input_text', 'text': s},
                                      image_item=lambda _: {'type': 'input_image'})
            content = example.content(harness)
            text = json.dumps(content)
            self.assertLess(text.index('TRAINING VIDEO DEMO'), text.index('TRAINING EXECUTION'))
            self.assertIn('END TRAINING EXAMPLE', text)
            self.assertNotIn('PRIVILEGED', text)
            self.assertEqual(image_count(content), 6)
            self.assertEqual(phases['execution']['times_seconds'][-1], 0.55)

    def test_execution_only_sample_is_explicit_and_covers_the_entire_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root, task='BinFill', demo=0)
            example = TrainingExamples(root, max_images=4).get('BinFill')
            self.assertEqual(list(example.phases), ['execution'])
            self.assertEqual(example.report['phases']['execution']['frame_indices'], [0, 3, 7, 11])
            h = SimpleNamespace(text_item=lambda s: {'text': s}, image_item=lambda _: {})
            self.assertIn('TRAINING VIDEO DEMO: absent', json.dumps(example.content(h)))

    def test_robot_state_and_both_camera_extrinsics_match_each_sampled_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root)
            example = TrainingExamples(root, max_images=6).get('PatternLock')
            harness = SimpleNamespace(
                text_item=lambda text: {'type': 'input_text', 'text': text},
                image_item=lambda image: {'type': 'input_image', 'source_frame': int(image[0, 0, 0])},
            )
            content = example.content(harness)
            calibration = example.report['camera_calibration']
            self.assertEqual(calibration['per_camera_image_size_wh'], [4, 4])
            # K is for each original camera, without an offset for the wrist half.
            self.assertEqual(calibration['front_camera_K'][0][2], 2.)
            self.assertEqual(calibration['wrist_camera_K'][0][2], 1.5)
            self.assertEqual(calibration['extrinsics_convention'], 'world_to_camera_opencv')
            self.assertEqual(sum(item.get('text', '').startswith('TRAINING CAMERA INTRINSICS:')
                                 for item in content), 1)
            seen = []
            for index, item in enumerate(content):
                if item['type'] != 'input_image':
                    continue
                preceding = content[index - 1]['text']
                prefix = 'TRAINING FRAME STATE AND CAMERA EXTRINSICS (for the next image): '
                self.assertTrue(preceding.startswith(prefix))
                state = json.loads(preceding[len(prefix):])
                i = item['source_frame']
                seen.append(i)
                self.assertEqual(state['source_frame'], i)
                self.assertEqual(state['time_seconds'], i / 20.)
                self.assertEqual(state['tcp_pose_xyz_rpy'], [i / 100., .2, .3, -3.1416, 0., .1])
                np.testing.assert_allclose(state['finger_positions_m'], [.01 + i / 1000., .02 + i / 1000.])
                np.testing.assert_allclose(state['front_camera_extrinsic_world_to_camera'],
                                           np.c_[np.eye(3), [i / 1000., .2, .3]])
                np.testing.assert_allclose(state['wrist_camera_extrinsic_world_to_camera'],
                                           np.c_[np.eye(3), [i / 100., .4, .5]])
            self.assertEqual(seen, [i for detail in example.report['phases'].values()
                                    for i in detail['frame_indices']])
            self.assertEqual(image_count(content), 6)
            self.assertEqual(example.report['observation_context_version'], 2)

    def test_state_timestamps_use_the_native_clock_across_the_phase_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sample(root)
            with h5py.File(path, 'a') as source:
                for i in range(5, 12):
                    source[f'episode_1/timestep_{i}/info/native_control_step'][()] = i + 60
            example = TrainingExamples(root, max_images=6).get('PatternLock')
            phase = example.report['phases']['execution']
            self.assertEqual(phase['frame_indices'][0], 5)
            self.assertEqual(phase['observations'][0]['time_seconds'], 3.25)
            self.assertEqual([o['time_seconds'] for o in phase['observations']], phase['times_seconds'])

    def test_missing_or_invalid_robot_state_and_calibration_are_rejected(self):
        corruptions = [
            ('setup/front_camera_intrinsic', None),
            ('setup/wrist_camera_intrinsic', np.full((3, 3), np.nan)),
            ('timestep_0/obs/eef_state', None),
            ('timestep_0/obs/gripper_state', np.zeros(3)),
            ('timestep_0/obs/front_camera_extrinsic', np.eye(4)),
            ('timestep_0/obs/wrist_camera_extrinsic', np.full((3, 4), np.inf)),
        ]
        for field, replacement in corruptions:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = sample(root)
                with h5py.File(path, 'a') as source:
                    episode = source['episode_1']
                    del episode[field]
                    if replacement is not None:
                        episode[field] = replacement
                with self.assertRaisesRegex(ValueError, 'training observation field'):
                    TrainingExamples(root).get('PatternLock')

    def test_sample_hash_covers_robot_state_and_calibration_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sample(root)
            original = TrainingExamples(root).get('PatternLock').report['sample_sha256']
            with h5py.File(path, 'a') as source:
                source['episode_1/timestep_0/obs/eef_state'][0] = .5
            changed_state = TrainingExamples(root).get('PatternLock').report['sample_sha256']
            self.assertNotEqual(original, changed_state)
            with h5py.File(path, 'a') as source:
                source['episode_1/setup/wrist_camera_intrinsic'][0, 0] = 140.
            changed_camera = TrainingExamples(root).get('PatternLock').report['sample_sha256']
            self.assertNotEqual(changed_state, changed_camera)

    def test_rejects_test_seed_overlap_unverified_sample_and_bad_phase_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sample(root)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                TrainingExamples(root, test_seeds={41})
            with h5py.File(path, 'a') as f:
                f['episode_1'].attrs['success'] = False
            with self.assertRaisesRegex(ValueError, 'not verified'):
                TrainingExamples(root).get('PatternLock')
            with h5py.File(path, 'a') as f:
                f['episode_1'].attrs['success'] = True
                f['episode_1/timestep_10/info/is_video_demo'][()] = True
            with self.assertRaisesRegex(ValueError, 'phase at frame 10'):
                TrainingExamples(root).get('PatternLock')

    def test_very_short_phase_is_preserved_under_strong_sampling(self):
        for frames, demo in [(1000, 999), (1000, 1), (2, 1), (1, 0)]:
            selected = phase_sampling(frames, demo, 4, 3)
            indices = [i for phase in selected.values() for i in phase]
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], frames - 1)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertLessEqual(len(indices), 4)
            if demo:
                self.assertIn(demo - 1, selected['video_demo'])
                self.assertIn(demo, selected['execution'])


HAS_API = importlib.util.find_spec('openai') is not None
HARNESS = FOLDER / 'embodied-astra-robomme/harness/astra_waypoint_eval_v3.py'


@unittest.skipUnless(HAS_API and HARNESS.exists(), 'Requires Astra submodule and SDK')
class OneShotEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('one_shot_adapter', FOLDER / 'eval_plus_fs.py')
        cls.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.adapter)

    def policy(self, root):
        policy = self.adapter.OneShotEmbodiedAstraEvaluation(SimpleNamespace(
            model='fixture', tasks=['PatternLock'], training_max_images=550))
        policy.examples = TrainingExamples(root)
        policy.harness = self.adapter.base.load_harness()
        policy.client = Mock()
        policy.adaptation_prompt = self.adapter.PROMPT_FILE.read_text()
        policy.training_content_key, policy.training_content = None, []
        return policy

    def run_policy(self, policy, env, output, replies, task='PatternLock'):
        policy.client.responses.create.side_effect = replies
        builder = SimpleNamespace(make_env_for_episode=lambda *a, **kw: env,
                                  episode_max_steps=lambda _: 1877)
        with patch('evaluation.VideoRecorder'), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return asyncio.run(policy.run_episode(builder, task, 0, output / f'{task}_ep0'))

    def test_training_demo_then_execution_then_current_episode_in_single_initial_request(self):
        from tests.lightweight.test_plus_embodied_astra_eval import FakeEnvironment, response
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root)
            policy = self.policy(root)
            configuration = policy.configuration()
            self.assertEqual(configuration['adaptation']['first_turn_image_limit'], 1100)
            self.assertEqual(configuration['adaptation']['version'], 2)
            self.assertEqual(configuration['demonstration_image_sampling']['max_initial_images'], 1100)
            self.assertFalse(configuration['original_prompts_preserved'])
            result = self.run_policy(policy, FakeEnvironment(), root, [response(0), response(1)])
            first, follow = [call.kwargs for call in policy.client.responses.create.call_args_list]
            self.assertTrue(result['success'])
            self.assertEqual(result['method'], 'embodied-astra-fs')
            texts = [p['text'] for p in first['input'][0]['content'] if p['type'] == 'input_text']
            text = '\n'.join(texts)
            self.assertLess(text.index('TRAINING VIDEO DEMO'), text.index('TRAINING EXECUTION'))
            self.assertLess(text.index('END TRAINING EXAMPLE'), text.index('CURRENT TEST EPISODE'))
            self.assertNotIn('PRIVILEGED_TRAIN_SUBGOAL', json.dumps(first))
            self.assertIn('One-shot adaptation protocol', first['instructions'])
            expected_prompt = policy.harness.system_prompt('PatternLock').replace(
                'EPISODE_MAX_STEPS', '1877') + '\n\n' + policy.adaptation_prompt
            self.assertEqual(first['instructions'], expected_prompt)
            self.assertEqual(follow['instructions'], expected_prompt)
            self.assertIn('after 1877 control steps.', first['instructions'])
            self.assertNotIn('after 1300 control steps.', first['instructions'])
            self.assertIn('TRAINING CAMERA INTRINSICS', text)
            self.assertIn('TRAINING FRAME STATE AND CAMERA EXTRINSICS', text)

    def test_image_only_request_retains_current_test_state(self):
        from tests.lightweight.test_plus_embodied_astra_eval import FakeEnvironment, response
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root)
            policy = self.policy(root)
            policy.args.training_context = 'images'
            policy.adaptation_prompt = self.adapter.IMAGE_ONLY_PROMPT_FILE.read_text()
            result = self.run_policy(policy, FakeEnvironment(), root, [response(0), response(1)])
            first, follow = [call.kwargs for call in policy.client.responses.create.call_args_list]
            content = first['input'][0]['content']
            text = '\n'.join(p.get('text', '') for p in content)
            training, current = text.split('CURRENT TEST EPISODE', 1)
            self.assertNotIn('TRAINING FRAME STATE', training)
            self.assertNotIn('TRAINING CAMERA INTRINSICS', training)
            self.assertIn('Current wrist camera', current)
            self.assertIn('No training robot states', first['instructions'])
            self.assertEqual(result['one_shot_adaptation']['training_context'], 'images')
            self.assertEqual(policy.configuration()['adaptation']['version'], 1)
            self.assertIn('world_to_camera', text)
            self.assertEqual(follow['previous_response_id'], 'response-0')
            self.assertNotIn('Training-only goal', json.dumps(follow['input']))
            self.assertNotIn('TRAINING FRAME STATE', json.dumps(follow['input']))
            self.assertEqual(result['one_shot_adaptation']['first_turn_images'], image_count(first['input']))
            self.assertEqual(result['astra_image_budget']['submitted_images'], image_count(first['input']) + image_count(follow['input']))
            self.assertTrue((root / 'PatternLock_ep0/one_shot_adaptation.json').exists())

    def test_first_turn_has_1100_images_and_counts_training_in_total_limit(self):
        from tests.lightweight.test_plus_embodied_astra_eval import FakeEnvironment, response
        for motion_frames, expected_status, expected_calls in [(4, 'success', 2), (1200, 'fail', 1)]:
            with self.subTest(motion_frames=motion_frames), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sample(root)
                policy = self.policy(root)
                # A training video already sampled to its default 550-image share.
                example = SimpleNamespace(report={'input_images': 550, 'sample_sha256': 'fixture'},
                    content=lambda h, **kw: [h.image_item(np.zeros((4, 8, 3), np.uint8)) for _ in range(550)])
                policy.examples = SimpleNamespace(get=lambda _: example)
                env = FakeEnvironment()
                env.reset = lambda: (env.observation(20000), env.information(0))
                original_step = env.step
                def step(action):
                    _, reward, done, truncated, info = original_step(action)
                    return env.observation(motion_frames), reward, done, truncated, info
                env.step = step
                result = self.run_policy(policy, env, root, [response(0), response(1)])
                first = policy.client.responses.create.call_args_list[0].kwargs
                self.assertEqual(image_count(first['input']), 1100)
                self.assertEqual(result['one_shot_adaptation']['first_turn_images'], 1100)
                self.assertEqual(result['status'], expected_status)
                self.assertEqual(policy.client.responses.create.call_count, expected_calls)
                self.assertEqual(result['astra_image_budget']['submitted_images'], 1103 if motion_frames == 4 else 1100)
                if expected_status == 'fail':
                    self.assertEqual(result['end_reason'], 'limit:1500_total_images')

    def test_move_cube_sends_no_training_example(self):
        from tests.lightweight.test_plus_embodied_astra_eval import FakeEnvironment, response
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample(root)
            policy = self.policy(root)
            result = self.run_policy(policy, FakeEnvironment(), root, [response(0), response(1)], task='MoveCube')
            payload = policy.client.responses.create.call_args_list[0].kwargs
            self.assertNotIn('BEGIN TRAINING EXAMPLE', json.dumps(payload['input']))
            self.assertIn('omitted for MoveCube', json.dumps(payload['input']))
            self.assertFalse(result['one_shot_adaptation']['applied'])
            self.assertEqual(image_count(payload['input']), 3)

    def test_resume_uses_one_shot_1100_limit_and_cannot_reuse_zero_shot_run(self):
        from run_storage import check_configuration, select_pending
        cases = [{'task': 'PatternLock', 'episode': i} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, count in zip(cases, [750, 1100, 1101]):
                directory = root / f"PatternLock_ep{case['episode']}"
                directory.mkdir()
                result = {**case, 'method': 'embodied-astra-fs', 'status': 'fail',
                          'one_shot_adaptation': {'first_turn_images': count}}
                (directory / 'result.json').write_text(json.dumps(result))
            with patch('run_storage.valid_outcome', return_value=True):
                self.assertEqual(select_pending(root, cases, 'resume'), cases[2:])
        with self.assertRaisesRegex(ValueError, 'different method'):
            check_configuration({'method': 'embodied-astra'}, {'method': 'embodied-astra-fs', 'arguments': {}})

    def test_resume_cannot_mix_training_observation_context_versions(self):
        from run_storage import check_configuration
        current = {'method': 'embodied-astra-fs',
                   'arguments': {'training_observation_context_version': 2}}
        for previous in [
            {'method': 'embodied-astra-fs', 'arguments': {}},
            {'method': 'embodied-astra-fs', 'arguments': {}, 'argument_identity': {}},
            {'method': 'embodied-astra-fs',
             'arguments': {'training_observation_context_version': 1}},
        ]:
            with self.subTest(previous=previous), self.assertRaisesRegex(ValueError, 'observation context changed'):
                check_configuration(previous, current)
        check_configuration(current, current)
        check_configuration({'method': 'embodied-astra', 'arguments': {}},
                            {'method': 'embodied-astra', 'arguments': {}})


if __name__ == '__main__':
    unittest.main()

"""Continuous motion is independent of button timing and per-cover exposure."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.permanence.utils.board import ButtonDeadline
from robomme_plus.tasks.permanence.utils.routes import (
    plan_routes, completed_passages, passage_intersects, central_membership, MARGIN_M,
)
from robomme_plus.tasks.permanence.utils.continuous import ContinuousRoutes, gripper_over_container
from robomme_plus.tasks.permanence.utils.workspace import resolve_continuous_motion, resolve_workspace
import test_plus_button_swap_motion as legacy


class ContinuousMotionTest(unittest.TestCase):
    def test_new_episode_mix_is_seeded_and_only_medium_hard(self):
        task = 'ButtonUnmaskSwap'
        self.assertFalse(any(resolve_continuous_motion(task, 'easy', seed) for seed in range(1000)))
        for level in ('medium', 'hard'):
            choices = [resolve_continuous_motion(task, level, seed) for seed in range(1000)]
            self.assertGreater(sum(choices), 450)
            self.assertLess(sum(choices), 550)
            self.assertEqual(choices[::-1], [resolve_continuous_motion(task, level, seed)
                                            for seed in reversed(range(1000))])
            for flag in (False, True):
                self.assertTrue(all(resolve_continuous_motion(task, level, seed, flag) is flag
                                    for seed in range(20)))
        # Recorded Easy continuous previews remain replayable explicitly.
        self.assertTrue(resolve_continuous_motion(task, 'easy', 0, True))
        self.assertFalse(resolve_continuous_motion('VideoUnmask', 'hard', 0))
        for invalid in ('true', 1, 0):
            with self.assertRaises(ValueError):
                resolve_continuous_motion(task, 'hard', 0, invalid)
        with self.assertRaises(ValueError):
            resolve_continuous_motion('VideoUnmask', 'hard', 0, True)

    def test_compact_workspace_bounds_and_recorded_replay_settings(self):
        workspace = resolve_workspace()
        np.testing.assert_allclose(np.diff(workspace['bounds_xy_m'], axis=1).ravel(), [.56, .58])
        self.assertEqual(workspace['board_x_m'], .20)
        row = dict(seed=0, difficulty='hard', oracle_policy_step=1000,
                   continuous_motion=True, motion_workspace=workspace)
        self.assertEqual(normalize_episode('ButtonUnmaskSwap', row, 0)['motion_workspace'], workspace)
        previous = dict(workspace, board_x_m=.14)
        self.assertEqual(normalize_episode('ButtonUnmaskSwap',
            {**row, 'motion_workspace': previous}, 0)['motion_workspace'], previous)
        self.assertIsNone(resolve_workspace('legacy'))
        finite = normalize_episode('ButtonUnmaskSwap', {**row, 'continuous_motion': False}, 0)
        self.assertFalse(finite['continuous_motion'])
        self.assertEqual(finite['motion_workspace'], workspace)
        # Legacy records do not acquire a newly sampled flag or new geometry.
        old = normalize_episode('ButtonUnmaskSwap', dict(seed=0, difficulty='hard', oracle_policy_step=1000), 0)
        self.assertNotIn('continuous_motion', old)
        self.assertNotIn('motion_workspace', old)
        bad = dict(workspace, central_bounds_xy_m=[[-.5, .16], [-.28, .28]])
        with self.assertRaises(ValueError):
            resolve_workspace(bad)

    def test_compact_stream_keeps_full_container_footprints_inside(self):
        workspace = resolve_workspace()
        radius = np.sqrt(2)*.055/2
        (p, stream), meta = plan_routes(11, radius, 1157842054+742931,
            continuous=True, speed=.00675, passage_count=3, workspace=workspace,
            initial_board_front_x=workspace['board_x_m']-.015,
            extra_obstacles=[([workspace['board_x_m'], 0], .0498121471)],
            obstacles=[(np.array([-.1711063917, -.1040932823]), .055),
                       (np.array([-.1828720766, .0608339912]), .055)],
            reachable_area=legacy.RectangularReachability())
        bounds = np.asarray(workspace['bounds_xy_m'])
        history = [p.copy()]
        for step in range(700):
            p = stream.send((p, set()))
            self.assertTrue(np.all(p-radius >= bounds[:, 0]-1e-9))
            self.assertTrue(np.all(p+radius <= bounds[:, 1]+1e-9))
            history.append(p.copy())
        self.assertGreaterEqual(len(completed_passages(history, meta)), 5)
        stream.close()

    def test_buttons_unlock_before_deadline_without_stopping_motion(self):
        d = ButtonDeadline(('left', 'right'), 10, 480, continuous_motion=True)
        self.assertFalse(d.observe(100, {'left': .01}, True, tcp_position=[0, 0, .03]))
        self.assertFalse(d.observe(200, {'right': .01}, True, tcp_position=[0, .2, .03]))
        self.assertEqual(set(d.presses), {'left', 'right'})
        self.assertIsNone(d.motion_stop)
        self.assertEqual(d.snapshot()['deadline_control_step'], 480)
        self.assertFalse(d.observe(1500, {}, True, tcp_position=[0, 0, .2]))

    def test_pickup_bypass_is_temporary_and_preserves_swept_clearance(self):
        from robomme_plus.common.motion import swept_gap
        workspace, radius = resolve_workspace(), .0389
        (p, stream), meta = plan_routes(11, radius, 3, continuous=True,
            speed=.00675, passage_count=3, pickup_bypass_margin=.08, workspace=workspace,
            obstacles=[(np.array([-.1711063917, -.1040932823]), .055),
                       (np.array([-.1828720766, .0608339912]), .055)],
            extra_obstacles=[([.20, 0], .0498121471)],
            reachable_area=legacy.RectangularReachability())
        bounds = np.asarray(workspace['bounds_xy_m'])
        history, frozen_index = [p.copy()], None
        for step in range(1000):
            if step == 100:
                frozen_index = int(np.argmax(p[:, 1]))
            frozen = {frozen_index} if 100 <= step < 600 else set()
            before = p.copy()
            p = stream.send((p, frozen, (), dict(pickup_blockers=frozen)))
            self.assertGreaterEqual(swept_gap(before, p, np.full(11, radius)), MARGIN_M-1e-10)
            self.assertTrue(np.all(p-radius >= bounds[:, 0]-.08-1e-10))
            self.assertTrue(np.all(p+radius <= bounds[:, 1]+.08+1e-10))
            if step < 100 or step >= 950:
                self.assertTrue(np.all(p-radius >= bounds[:, 0]-1e-10))
                self.assertTrue(np.all(p+radius <= bounds[:, 1]+1e-10))
            history.append(p.copy())
        history = np.asarray(history)
        np.testing.assert_array_equal(history[100:601, frozen_index],
            np.tile(history[100, frozen_index], (501, 1)))
        events = meta['pickup_bypass_events']
        self.assertTrue(any(e['event'] == 'outside' and 100 <= e['route_step'] < 600 for e in events))
        self.assertFalse(any(e['route_step'] < 100 for e in events))
        self.assertTrue(any(e['event'] == 'returned' and e['route_step'] >= 600 for e in events))
        # At least one moving neighbor gets from one side of the paused pair
        # to the other while remaining within 20 cm of it.
        blocker = history[100, frozen_index]
        near = np.linalg.norm(history[100:601]-blocker, axis=2) < .20
        dx = history[100:601, :, 0]-blocker[0]
        self.assertTrue(np.any(np.any(near & (dx < -.05), axis=0)
                               & np.any(near & (dx > .05), axis=0)))
        self.assertLessEqual(np.linalg.norm(np.diff(history, axis=0), axis=2).max(), .00675+1e-10)
        stream.close()

    def test_pickup_bypass_cannot_expand_finite_motion_or_exceed_table_margin(self):
        for margin, continuous in ((-.01, True), (.11, True), (.08, False)):
            with self.subTest(margin=margin, continuous=continuous), self.assertRaises(ValueError):
                plan_routes(11, .0389, 3, pickup_bypass_margin=margin, continuous=continuous)

    def test_exposed_cube_allows_neighbor_to_pass_without_expanding_area(self):
        import json
        from pathlib import Path
        from robomme_plus.tasks.permanence.utils.reachability import ReachableArea
        from robomme_plus.common.motion import swept_gap
        f = json.loads((Path(__file__).parent/'fixtures/button_swap_blocked_pickup.json').read_text())
        prefix = np.load(Path(__file__).parent/'fixtures/button_swap_pickup_prefix.npz')['positions']
        area = ReachableArea.__new__(ReachableArea)
        area.x = np.linspace(*f['x_range_count'])
        area.y = np.linspace(*f['y_range_count'])
        shape = (len(area.x)-1, len(area.y)-1)
        area.cells = np.unpackbits(np.frombuffer(bytes.fromhex(f['cell_bits_hex']), dtype=np.uint8))[:np.prod(shape)].reshape(shape).astype(bool)
        extra = f['extra_obstacles']
        rectangles = [(o['xy'], [.015, np.sqrt(o['radius_m']**2-.015**2)]) for o in extra]
        (p, stream), meta = plan_routes(f['actor_count'], f['radius_m'], f['seed'],
            continuous=True, speed=f['speed'], passage_count=3, workspace=f['workspace'],
            obstacles=[(np.array(o['xy']), o['radius_m']) for o in f['obstacles']],
            rectangle_obstacles=rectangles, reachable_area=area,
            pickup_passing=True, pickup_bypass_margin=0.)
        history = []
        for step in range(720):
            if step <= 301:
                # Preserve the recorded approach to the reported jam so the
                # test exercises its actual positions and passage handovers.
                p = prefix[step].copy()
            radii = np.full(len(p), f['radius_m'])
            if 340 <= step < 710: radii[0] = np.sqrt(2)*.02/1.2
            parked = [(p[0].copy(), np.sqrt(2)*.03)] if 660 <= step < 710 else ()
            before = p.copy()
            p = stream.send((p, {0} if 301 <= step < 710 else set(), parked,
                dict(pickup_blockers={0} if 301 <= step < 710 else set(), actor_radii_m=radii)))
            self.assertGreaterEqual(swept_gap(before, p, radii), MARGIN_M-1e-9)
            history.append(p.copy())
        history = np.asarray(history)
        anchor = history[340, 0]
        for i in (9, 10):
            self.assertGreater(np.max(history[301:660, i, 1])-anchor[1], .01)
            self.assertTrue(np.any(history[341:660, i, 1] < anchor[1]-.04))
            self.assertTrue(any(e['actor_index']==i and e['event']=='finished'
                                and e['route_step']<660 for e in meta['pickup_passing_events']))
        self.assertFalse(meta['pickup_bypass_events'])
        stream.close()

    def test_thin_board_box_keeps_real_clearance_and_segment_intersections(self):
        from robomme_plus.tasks.permanence.utils.routes import point_box_distance, segment_box_distance
        center, half = np.array([[.20, 0.]]), np.array([[.015, .0475]])
        self.assertAlmostEqual(point_box_distance([.12, 0], center, half)[0], .065)
        self.assertAlmostEqual(segment_box_distance([.12, -.15], [.12, .15], center, half)[0], .065)
        self.assertEqual(segment_box_distance([.1, 0], [.3, 0], center, half)[0], 0.)
        self.assertEqual(segment_box_distance([.2, 0], [.2, 0], center, half)[0], 0.)

    def test_stream_keeps_moving_past_old_stop_and_freezes_only_one_pair(self):
        (p, stream), meta = plan_routes(11, .0389, 3, continuous=True, speed=.00675,
            passage_count=3,
            obstacles=[(np.array([-.23, -.1]), .055), (np.array([-.23, .1]), .055)],
            reachable_area=legacy.RectangularReachability())
        history = [p.copy()]
        for step in range(1300):
            frozen = {0} if 500 <= step < 650 else set()
            p = stream.send((p, frozen))
            history.append(p.copy())
        history = np.array(history)
        np.testing.assert_array_equal(history[500:651, 0], np.tile(history[500, 0], (151, 1)))
        self.assertGreater(np.linalg.norm(np.diff(history[500:651, 1:], axis=0), axis=2).sum(), .1)
        self.assertGreater(np.linalg.norm(np.diff(history[650:, 0], axis=0), axis=1).sum(), .02)
        self.assertGreater(np.linalg.norm(np.diff(history[750:], axis=0), axis=2).sum(), .1)
        speed = np.linalg.norm(np.diff(history, axis=0), axis=2)
        self.assertGreater(speed.mean(), .003)
        self.assertLessEqual(speed.max(), .00675 + 1e-10)
        passages = completed_passages(history, meta)
        self.assertGreaterEqual(len({p['actor_index'] for p in passages}), 6)
        self.assertTrue(any(p['exit_step'] > 850 for p in passages))
        lane = meta['one_way_passage']
        intrusions = passage_intersects(history[:-1], history[1:], lane['x_max_m'], lane['y_range_m'])
        self.assertFalse(np.any(intrusions & central_membership(1300, 11, meta)))
        stream.close()

    def test_passage_roles_follow_handover_times(self):
        roles = central_membership(8, 3, dict(central_actor_indices=[0, 1, 2],
            passage_assignments=[dict(actor_index=0, step=0, event='assigned'),
                dict(actor_index=0, step=3, event='released'),
                dict(actor_index=2, step=3, event='assigned')]))
        np.testing.assert_array_equal(roles[:3], np.tile([False, True, True], (3, 1)))
        np.testing.assert_array_equal(roles[3:], np.tile([True, True, False], (5, 1)))

    def test_lift_freezes_and_only_recovering_the_same_cube_resumes(self):
        env = legacy.EarlyOpeningTest.scene(presses=('left', 'right'), parked=False)
        env._limit_setup = True
        env.agent = SimpleNamespace(is_grasping=lambda actor: False)
        cover, cube = env._limit_routes.actors[0], env._limit_hidden[0][0]
        cover.pose.p = np.array([0., 0., .054])
        cube.pose.p = np.array([0., 0., .02/1.2])
        r = ContinuousRoutes.__new__(ContinuousRoutes)
        r.env, r.actors, r.z = env, [cover], [.054]
        r.positions = np.zeros((1, 2))
        r.covered, r.events, r._recovering = np.array([True]), [], {}
        r.observe_covers()
        self.assertTrue(r.covered[0])
        cover.pose.p[2] = .15
        r.observe_covers()
        self.assertFalse(r.covered[0])
        cover.pose.p = np.array([.1, 0., .054])
        for step in range(20, 25):
            env.elapsed_steps = np.array(step)
            r.observe_covers()
        self.assertFalse(r.covered[0])
        cover.pose.p = np.array([0., 0., .054])
        for step in range(25, 28):
            env.elapsed_steps = np.array(step)
            r.observe_covers()
        self.assertTrue(r.covered[0])
        self.assertEqual([e['event'] for e in r.events], ['uncovered', 'recovered'])

    def test_metadata_preserves_explicit_mode_and_rejects_ambiguous_flags(self):
        row = dict(seed=0, difficulty='hard', oracle_policy_step=1000, continuous_motion=True)
        self.assertTrue(normalize_episode('ButtonUnmaskSwap', row, 0)['continuous_motion'])
        with self.assertRaises(ValueError):
            normalize_episode('ButtonUnmaskSwap', {**row, 'continuous_motion': 'true'}, 0)
        with self.assertRaises(ValueError):
            normalize_episode('ButtonUnmask', row, 0)

    def test_proximity_uses_height_above_top_and_rotated_xy_footprint(self):
        from transforms3d.euler import euler2quat
        center, half = np.array([.1, -.1, .052]), .02
        q = [0, 1, 0, 0]
        # Roof at .052, grasp block top at .072, stop boundary at .132 m.
        self.assertTrue(gripper_over_container([.1, -.1, .1319], center, q, half))
        self.assertFalse(gripper_over_container([.1, -.1, .1321], center, q, half))
        self.assertTrue(gripper_over_container([.1299, -.1, .10], center, q, half))
        self.assertFalse(gripper_over_container([.1301, -.1, .10], center, q, half))
        turned = euler2quat(np.pi, 0, np.pi/4)
        self.assertTrue(gripper_over_container([.139, -.1, .10], center, turned, half))
        self.assertFalse(gripper_over_container([.129, -.071, .10], center, turned, half))
        self.assertFalse(gripper_over_container([.1, -.1, -.02], center, q, half))

    @staticmethod
    def traffic_scene():
        import torch
        env = legacy.EarlyOpeningTest.scene(presses=('left', 'right'), parked=False)
        env._limit_setup = True
        held = set()
        tcp = SimpleNamespace(pose=SimpleNamespace(p=np.array([0., 0., .22])))
        env.agent = SimpleNamespace(tcp=tcp, is_grasping=lambda actor: actor.name in held)
        env.seed, env.control_freq = 3, 20
        cover, cube = env._limit_routes.actors[0], env._limit_hidden[0][0]
        cover.pose.p = np.array([0., 0., .052])
        cube.pose.p = np.array([0., 0., .02/1.2])
        other = SimpleNamespace(name='other', pose=SimpleNamespace(
            p=np.array([.2, 0., .052]), q=torch.tensor([0., 1., 0., 0.])))
        r = ContinuousRoutes.__new__(ContinuousRoutes)
        r.env, r.actors, r.z = env, [cover, other], [.052, .052]
        r.execution_start = int(env.elapsed_steps) - 150
        r.q = [[0, 1, 0, 0], [0, 1, 0, 0]]
        r.radius, r.active, r.cursor = .0425, True, 0
        r.positions = np.array([[0., 0.], [.2, 0.]])
        r.covered, r.events, r._recovering, r._paused = np.array([True, True]), [], {}, {}
        r._nearby, r._pause_ceiling = set(), .132
        r.history, r.cursors, r.lengths = [r.positions.copy()], np.zeros(2, int), np.zeros(2)
        r.metadata = dict(simultaneously_moving_max=0)
        from robomme_plus.tasks.permanence.utils.rotation import SlowYaw
        r._yaw_motion, r._cube_q = SlowYaw(3, 2, .05), None
        r.yaw_history = [np.zeros(2)]
        r.radius_history = [np.full(2, r.radius)]
        def stream():
            p = r.positions.copy()
            while True:
                feedback = yield p.copy()
                p, frozen, parked = feedback[:3]
                p = p.copy()
                for i in range(len(p)):
                    if i not in frozen: p[i, 0] += .001 * feedback[3].get('speed_scale', 1.)
        r._stream = stream()
        next(r._stream)
        def move(actor, p, q=None):
            actor.pose.p = np.asarray(p)
            if q is not None: actor.pose.q = torch.tensor(q)
        return env, r, held, cover, cube, move

    def test_button_grace_keeps_translation_and_yaw_at_normal_speed_until_step_151(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        # Use a large native clock offset so initialization/demo time cannot
        # accidentally consume the execution-only grace period.
        r.execution_start = 1000
        r._nearby = {1}
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            for completed in range(150):
                env.elapsed_steps = np.array(1000 + completed)
                env.agent.tcp.pose.p = np.r_[cover.pose.p[:2], .10]
                r.update()
                self.assertEqual(r.metadata['pickup_neighborhood_indices'], [])
                self.assertFalse(r.metadata['gripper_proximity_pauses_enabled'])
                self.assertEqual(r.metadata['traffic_speed_scale'], 1.)
                self.assertEqual(r._paused, {})
            np.testing.assert_allclose(r.positions, [[.150, 0], [.350, 0]])
            self.assertGreater(abs(r.yaw_history[-1][0]), .01)

            env.elapsed_steps = np.array(1150)
            env.agent.tcp.pose.p = np.r_[cover.pose.p[:2], .10]
            before = r.positions.copy()
            yaw = cover.pose.q.clone()
            r.update()
            self.assertTrue(r.metadata['gripper_proximity_pauses_enabled'])
            self.assertEqual(r.metadata['pickup_neighborhood_indices'], [0])
            np.testing.assert_allclose(r.positions[0], before[0])
            np.testing.assert_array_equal(cover.pose.q.numpy(), yaw.numpy())
            self.assertAlmostEqual(r.positions[1, 0] - before[1, 0], .0005)
            self.assertEqual(r.metadata['traffic_speed_scale'], .5)

            env.agent.tcp.pose.p[2] = .22
            r.update()
            self.assertEqual(r.metadata['pickup_neighborhood_indices'], [])
            self.assertEqual(r.metadata['traffic_speed_scale'], 1.)
            self.assertGreater(r.positions[0, 0], before[0, 0])

    def test_button_grace_does_not_expire_before_execution_starts(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        r.execution_start = None
        env.elapsed_steps = np.array(5000)
        env.agent.tcp.pose.p = np.array([0., 0., .10])
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            r.update()
        self.assertEqual(r.metadata['pickup_neighborhood_indices'], [])
        self.assertEqual(r.metadata['traffic_speed_scale'], 1.)

    def test_historical_v5_keeps_immediate_proximity_pauses(self):
        from robomme_plus.tasks.permanence.utils.continuous_v5 import ContinuousRoutes as V5Routes
        env, current, held, cover, cube, move = self.traffic_scene()
        old = V5Routes.__new__(V5Routes)
        old.__dict__.update(current.__dict__)
        old.execution_start = int(env.elapsed_steps)
        env.agent.tcp.pose.p = np.array([0., 0., .10])
        frozen, _ = old._traffic_state()
        self.assertEqual(frozen, {0: 'pickup_neighborhood'})

    def test_neighborhood_pauses_until_lift_and_distant_traffic_slows(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        # TCP is outside the cover footprint, inside the 10 cm neighborhood.
        env.agent.tcp.pose.p = np.array([-.08, 0., .13])
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            r.update()
            np.testing.assert_array_equal(r.positions[0], [0, 0])
            self.assertAlmostEqual(r.positions[1, 0], .2005)
            self.assertEqual(r.metadata['traffic_speed_scale'], .5)
            # Moving sideways while low does not restart the first pair, and
            # also stops the newly approached neighbor.
            env.agent.tcp.pose.p[:2] = [.2, 0.]
            r.update()
            self.assertEqual(set(r._paused), {0, 1})
            self.assertAlmostEqual(r.positions[1, 0], .2005)
            env.agent.tcp.pose.p[2] = .22
            r.update()
            self.assertGreater(r.positions[0, 0], 0)
            self.assertEqual(r.metadata['traffic_speed_scale'], 1.)
            self.assertAlmostEqual(r.positions[1, 0], .2015)
        self.assertEqual([e['event'] for e in r.events], ['paused', 'paused', 'resumed', 'resumed'])

    def test_paused_pair_is_not_teleported_after_physical_contact(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        env.agent.tcp.pose.p[2] = .10
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move) as teleport:
            r.update()
            # Contact moves both physical objects within the paused region.
            cover.pose.p[:2] += [.006, -.002]
            cube.pose.p[:2] += [.004, -.001]
            expected_cover, expected_cube = cover.pose.p.copy(), cube.pose.p.copy()
            r.update()
            np.testing.assert_array_equal(cover.pose.p, expected_cover)
            np.testing.assert_array_equal(cube.pose.p, expected_cube)
            env.agent.tcp.pose.p[2] = .22
            r.update()
            # Resuming preserves the small physical offset instead of forcing
            # the cube and cover to the same center.
            self.assertAlmostEqual(np.linalg.norm(cover.pose.p[:2]-cube.pose.p[:2]),
                                   np.linalg.norm(expected_cover[:2]-expected_cube[:2]))

    def test_partial_cover_overlap_does_not_slide_cube_through_wall(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        cover.pose.p[0] = .03
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            r.update()
            self.assertFalse(r.covered[0])
            self.assertEqual(r._paused[0], 'cover_overlaps_cube')
            np.testing.assert_array_equal(cube.pose.p[:2], [0, 0])
            cover.pose.p[0] = .09
            r.update()
            self.assertNotIn(0, r._paused)
            self.assertGreater(cube.pose.p[0], 0.)

    def test_real_traffic_keeps_paused_neighbors_and_caps_other_speed(self):
        from robomme_plus.common.motion import swept_gap
        speed, radius = .00675, .0389
        (p, stream), _ = plan_routes(11, radius, 3, continuous=True, speed=speed,
            passage_count=3, pickup_passing=True, pickup_bypass_margin=.08,
            workspace=resolve_workspace(),
            obstacles=[(np.array([-.23, -.1]), .055), (np.array([-.23, .1]), .055)],
            reachable_area=legacy.RectangularReachability())
        frozen = set(np.flatnonzero(np.linalg.norm(p-p[0], axis=1) <= .10))
        origin, distance = p.copy(), np.zeros(len(p))
        for _ in range(120):
            before = p.copy()
            p = stream.send((p, frozen, (), dict(pickup_blockers=frozen, speed_scale=.5)))
            step = np.linalg.norm(p-before, axis=1)
            distance += step
            np.testing.assert_array_equal(p[sorted(frozen)], origin[sorted(frozen)])
            self.assertLessEqual(step.max(), .5*speed+1e-10)
            self.assertGreaterEqual(swept_gap(before, p, np.full(len(p), radius)), MARGIN_M-1e-9)
        self.assertGreater(distance.sum(), .2)
        for _ in range(80):
            p = stream.send((p, set(), (), dict(speed_scale=1.)))
        self.assertGreater(np.linalg.norm(p[0]-origin[0]), .005)
        # Physical contact can consume some planner margin without intersecting
        # the covers. Frozen neighbors must stay put instead of throwing or
        # teleporting; all subsequent motion must preserve that clearance.
        distances = np.linalg.norm(p[:, None]-p[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        a, b = np.unravel_index(distances.argmin(), distances.shape)
        direction = (p[b]-p[a]) / distances[a, b]
        p[b] = p[a] + direction * (2*radius+.005)
        anchor = p.copy()
        for _ in range(20):
            before = p.copy()
            p = stream.send((p, {a, b}, (), dict(pickup_blockers={a, b},
                speed_scale=.5, allow_contact_recovery=True)))
            np.testing.assert_array_equal(p[[a, b]], anchor[[a, b]])
            self.assertGreaterEqual(swept_gap(before, p, np.full(len(p), radius)), .005-1e-9)
        stream.close()

    def test_held_cube_stops_released_aside_cube_moves_and_cover_stays(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        cover.pose.p[2] = .18
        held.add(cover.name)
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            r.update()
            self.assertFalse(r.covered[0])
            np.testing.assert_array_equal(cube.pose.p[:2], [0, 0])
            cover.pose.p = np.array([-.12, .12, .052])
            r.update()  # Still held at the side: cube must stay still.
            np.testing.assert_array_equal(cube.pose.p[:2], [0, 0])
            held.clear()
            parked_position = cover.pose.p.copy()
            for _ in range(5): r.update()
            self.assertGreater(cube.pose.p[0], .004)
            np.testing.assert_array_equal(cover.pose.p, parked_position)
            self.assertEqual(r.metadata['parked_cover_obstacles'], 1)
            self.assertTrue(any(e['event']=='resumed' and not e['covered'] for e in r.events))

    def test_returned_cover_waits_for_release_then_moves_with_same_cube(self):
        env, r, held, cover, cube, move = self.traffic_scene()
        cover.pose.p[2] = .18
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            r.update()
            cover.pose.p = np.array([-.12, .12, .052])
            r.update()
            cover.pose.p = np.r_[cube.pose.p[:2], .052]
            held.add(cover.name)
            for step in range(20, 24):
                env.elapsed_steps = np.array(step)
                r.update()
            self.assertFalse(r.covered[0])
            held.clear()
            for step in range(24, 28):
                env.elapsed_steps = np.array(step)
                r.update()
            self.assertTrue(r.covered[0])
            np.testing.assert_allclose(cover.pose.p[:2], cube.pose.p[:2])
            self.assertGreater(r.positions[0, 0], .002)

    def test_parked_cover_obstacle_is_respected_by_real_stream(self):
        radius = .0389
        (p, stream), _ = plan_routes(11, radius, 3, continuous=True, speed=.00675,
            passage_count=3,
            obstacles=[(np.array([-.23, -.1]), .055), (np.array([-.23, .1]), .055)],
            reachable_area=legacy.RectangularReachability())
        # An empty cover at the center, away from all initial containers.
        choices = np.array([[x, y] for x in (-.05, .02, .08) for y in (-.15, 0, .15)])
        center = max(choices, key=lambda c: np.linalg.norm(p-c, axis=1).min())
        self.assertGreater(np.linalg.norm(p-center, axis=1).min(), 2*radius+.004)
        history = [p.copy()]
        for _ in range(400):
            p = stream.send((p, set(), [(center, radius)]))
            self.assertGreaterEqual(np.linalg.norm(p-center, axis=1).min(), 2*radius+.004-1e-10)
            history.append(p.copy())
        self.assertGreater(np.linalg.norm(np.diff(history, axis=0), axis=2).sum(), .5)
        stream.close()

    def test_motion_version_preserved_and_invalid_versions_rejected(self):
        from robomme_plus.tasks.permanence.utils.workspace import resolve_motion_version
        row = dict(seed=0, difficulty='hard', oracle_policy_step=1000, continuous_motion=True)
        self.assertEqual(resolve_motion_version(), 6)
        self.assertNotIn('continuous_motion_version', normalize_episode('ButtonUnmaskSwap', row, 0))
        self.assertEqual(normalize_episode('ButtonUnmaskSwap',
            dict(row, continuous_motion_version=2), 0)['continuous_motion_version'], 2)
        self.assertEqual(normalize_episode('ButtonUnmaskSwap',
            dict(row, continuous_motion_version=3), 0)['continuous_motion_version'], 3)
        self.assertEqual(normalize_episode('ButtonUnmaskSwap',
            dict(row, continuous_motion_version=4), 0)['continuous_motion_version'], 4)
        self.assertEqual(normalize_episode('ButtonUnmaskSwap',
            dict(row, continuous_motion_version=5), 0)['continuous_motion_version'], 5)
        self.assertEqual(normalize_episode('ButtonUnmaskSwap',
            dict(row, continuous_motion_version=6), 0)['continuous_motion_version'], 6)
        for version in (None, True, '2', 0, 7):
            with self.assertRaises(ValueError):
                normalize_episode('ButtonUnmaskSwap', dict(row, continuous_motion_version=version), 0)
        with self.assertRaises(ValueError):
            normalize_episode('ButtonUnmaskSwap', dict(row, continuous_motion=False, continuous_motion_version=2), 0)

    def test_historical_v4_keeps_the_previous_five_cm_footprint_rule(self):
        from robomme_plus.tasks.permanence.utils.continuous_v4 import gripper_over_container as previous
        self.assertTrue(previous([0, 0, .1219], [0, 0, .052], [0, 1, 0, 0], .02))
        self.assertFalse(previous([0, 0, .1221], [0, 0, .052], [0, 1, 0, 0], .02))
        self.assertFalse(previous([.08, 0, .10], [0, 0, .052], [0, 1, 0, 0], .02))

    def test_slow_yaw_is_seeded_smooth_and_independent_of_other_pickups(self):
        from robomme_plus.tasks.permanence.utils.rotation import SlowYaw
        a, b = SlowYaw(125, 3, .05), SlowYaw(125, 3, .05)
        history = []
        for step in range(2000):
            da = a.advance([True, True, True])
            db = b.advance([True, not 200 <= step < 900, True])
            np.testing.assert_array_equal(da[[0, 2]], db[[0, 2]])
            if 200 <= step < 900: self.assertEqual(db[1], 0.)
            history.append(da/.05)
        degrees = np.rad2deg(history)
        self.assertLessEqual(abs(degrees).max(), 8.+1e-9)
        self.assertLessEqual(abs(np.diff(degrees, axis=0)).max(), 4.*.05+1e-9)
        self.assertTrue(np.all(degrees.min(axis=0) < -1))
        self.assertTrue(np.all(degrees.max(axis=0) > 1))
        self.assertGreater(np.linalg.norm(degrees[:, 0]-degrees[:, 1]), 1.)

    def test_rotation_pauses_on_approach_and_keeps_cube_inside_cover(self):
        from robomme_plus.tasks.permanence.utils.progress import cover_exposure
        env, r, held, cover, cube, move = self.traffic_scene()
        with patch('robomme_plus.tasks.permanence.utils.continuous.set_position', move):
            for _ in range(80):
                r.update()
                self.assertIsNone(cover_exposure(cover.pose.p, cover.pose.q.numpy(),
                    cube.pose.p, cube.pose.q.numpy(), env.cube_half_size))
            self.assertGreater(abs(r.yaw_history[-1][0]), .05)
            before = cover.pose.q.clone()
            env.agent.tcp.pose.p = np.r_[cover.pose.p[:2], .10]
            for _ in range(30): r.update()
            np.testing.assert_array_equal(cover.pose.q.numpy(), before.numpy())
            self.assertEqual(r._yaw_motion.velocity[0], 0.)
            env.agent.tcp.pose.p[2] = .22
            r.update()
            self.assertFalse(np.array_equal(cover.pose.q.numpy(), before.numpy()))


if __name__ == '__main__':
    unittest.main()

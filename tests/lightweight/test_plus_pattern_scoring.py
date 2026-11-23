"""PatternLock's native success must include dwell, visible drawing and continuity."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from robomme_plus.common.sim import position
from robomme_plus.tasks.imitation.pattern_lock import PatternLock


class PatternScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PatternLock(seed=97, difficulty='hard', stick_change_version=4,
                              obs_mode='state', sim_backend='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        env = self.env
        env.reset(seed=97)
        env.demonstration_record_traj = True
        env.demonstration_record_traj = False
        env.use_demonstrationwrapper = True
        env.current_task_demonstration = False
        env.timestep = next(i for i, task in enumerate(env.task_list) if not task['demonstration'])
        self.tip = SimpleNamespace(pose=SimpleNamespace(
            p=torch.tensor([[0., 0., .2]]), q=torch.tensor([[0., 1., 0., 0.]])))
        for name, value in [('agent', SimpleNamespace(tcp=self.tip)), ('_stick_short_tip', self.tip)]:
            patched = patch.object(env, name, value)
            patched.start(); self.addCleanup(patched.stop)
        self.nodes = [position(node, False)[:2] for node in env.selected_buttons]

    def sample(self, xy, z=.07, count=1):
        for _ in range(count):
            self.tip.pose.p[0] = torch.tensor([*xy, z])
            self.env._elapsed_steps += 1
            result = self.env.evaluate()
        return result

    def test_full_native_pattern_succeeds_only_after_final_dwell_and_red_target(self):
        result = self.sample(self.nodes[0], z=.15, count=400)
        self.assertFalse(result['fail'].item())
        self.assertFalse(result['success'].item())
        result = self.sample(self.nodes[0], count=7)
        self.assertFalse(result['success'].item())
        for start, end in zip(self.nodes, self.nodes[1:]):
            count = int(np.ceil(np.linalg.norm(end - start) / .005))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                result = self.sample(start + fraction * (end - start))
                self.assertFalse(result['fail'].item())
            self.assertFalse(result['success'].item())
            result = self.sample(end, count=6)
            self.assertFalse(result['fail'].item())
        self.assertTrue(result['success'].item())
        self.assertTrue(self.env._pattern_dwell.success)
        from robomme_plus.tasks.imitation.utils.tracing import final_target_is_red
        self.assertTrue(final_target_is_red(self.env))
        self.assertNotIn('execution', self.env._trace_completion_started)

    def test_lifting_between_correct_nodes_cannot_skip_the_drawn_segment(self):
        self.sample(self.nodes[0], count=7)
        self.sample(self.nodes[0], z=.15)
        self.sample(self.nodes[1], z=.15)
        result = self.sample(self.nodes[1])
        self.assertTrue(result['fail'].item())
        self.assertFalse(result['success'].item())
        self.assertEqual(self.env._failure_code, 'discontinuous')

    def test_visiting_final_target_first_fails_instead_of_completing(self):
        result = self.sample(self.nodes[-1], count=7)
        self.assertTrue(result['fail'].item())
        self.assertFalse(result['success'].item())
        self.assertEqual(self.env._failure_code, 'wrong_endpoint_stop')


if __name__ == '__main__':
    unittest.main()

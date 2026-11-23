"""Curve extent and endpoint projection rules for RouteStick white dots."""
import unittest
from unittest.mock import patch

import numpy as np

from robomme_plus.tasks.imitation.utils.drawing import Drawing
from robomme_plus.tasks.imitation.utils.route_trace import RouteTrace, project_dots


class RouteTraceTest(unittest.TestCase):
    def reference(self):
        trace = RouteTrace([[0, 0], [.16, 0], [.32, 0]])
        for step, (curve, xy) in enumerate(((0, [0, 0]), (0, [.08, .1]), (0, [.16, 0]),
                                           (1, [.16, 0]), (1, [.24, -.06]), (1, [.32, 0]))):
            trace.update("demonstration", curve, step, xy)
        trace.begin_execution([[0, 0], [.16, 0], [.32, 0]])
        return trace

    def test_projection_is_rotation_and_translation_invariant(self):
        for angle in (0, .3, 1.8, -2.):
            r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            start = np.array([.3, -.2])
            along, lateral, length = project_dots(
                np.array([[-.02, .1], [.19, -.08]]) @ r.T + start, start,
                np.array([.16, 0]) @ r.T + start)
            np.testing.assert_allclose(along, [-.02, .19], atol=1e-12)
            np.testing.assert_allclose(lateral, [.1, .08], atol=1e-12)
            self.assertAlmostEqual(length, .16)

    def test_each_curve_uses_its_own_oracle_maximum(self):
        trace = self.reference()
        trace.update("execution", 0, 10, [.08, .13])
        self.assertIsNone(trace.failure)
        trace.update("execution", 1, 11, [.24, -.079])
        self.assertEqual(trace.failure["reason"], "trace_lateral_extent")
        self.assertAlmostEqual(trace.failure["limit_m"], .06 * 1.3)

    def test_endpoint_overshoot_at_three_cm_and_beyond_both_ends(self):
        for curve, x, passes in ((0, -.03, True), (0, .19, True), (0, -.0301, False),
                                 (0, .1901, False), (1, .1299, False), (1, .3501, False)):
            trace = self.reference()
            trace.update("execution", curve, 10, [x, .01])
            self.assertEqual(trace.failure is None, passes)
            if not passes:
                self.assertEqual(trace.failure["reason"], "trace_endpoint_overshoot")

    def test_scaled_row_scales_oracle_extent_but_keeps_absolute_endpoint_margin(self):
        trace = self.reference()
        trace.begin_execution([[.05, 0], [.242, 0], [.434, 0]])
        trace.update("execution", 0, 10, [.146, .156])
        self.assertIsNone(trace.failure)
        self.assertAlmostEqual(trace.summary()["curves"][0]["lateral_limit_m"], .156)
        trace.update("execution", 0, 11, [.2721, .01])
        self.assertEqual(trace.failure["reason"], "trace_endpoint_overshoot")

    def test_lifts_approach_resets_and_duplicate_evaluations_are_excluded(self):
        trace = self.reference()
        for phase, curve, draws in (("execution", None, True), ("execution", 0, False),
                                    ("demonstration", None, True)):
            trace.update(phase, curve, 10, [100, 100], draws)
        trace.update("execution", 0, 10, [.08, .1])
        trace.update("execution", 0, 10, [100, 100])
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.summary()["curves"][0]["drawing_points"], 1)

    def test_missing_reference_and_undrawn_curves_cannot_succeed(self):
        trace = RouteTrace([[0, 0], [.16, 0]])
        trace.begin_execution([[0, 0], [.16, 0]])
        trace.update("execution", 0, 10, [0, 0])
        self.assertEqual(trace.failure["reason"], "trace_reference_missing")
        trace = self.reference()
        trace.update("execution", 0, 10, [.08, .1])
        trace.finish(11)
        self.assertEqual(trace.failure["reason"], "trace_incomplete")

    def test_native_success_is_overridden_by_curve_rule(self):
        from itertools import product
        import torch
        from robomme_plus.tasks.imitation.utils.controllers import setup_occlusion
        from robomme_plus.tasks.imitation.utils.route_motion import rotate_points

        class Scene:
            task_name = "RouteStick"
            elapsed_steps = torch.tensor([10])
            timestep = 0

            def _load_scene(self): pass
            def _initialize_episode(self): pass
            def _after_simulation_step(self): pass
            def _update_trace_feedback(self): pass

            def evaluate(self, solve_complete_eval=False):
                self.timestep = 1  # final node acceptance must retain the old curve index
                self.successflag = torch.tensor([True])
                return dict(success=self.successflag, fail=torch.tensor([False]))

        setup_occlusion(Scene)
        cases = (([.08, .12], None), ([.08, .131], "trace_lateral_extent"),
                 ([.191, .01], "trace_endpoint_overshoot"))
        for (xy, reason), angle in product(cases, (0., -.26, .26)):
            env = Scene()
            pivot = np.array([-.2, .2])
            env._route_to_rest = lambda point: rotate_points(point, pivot, -angle)
            world_xy = rotate_points(xy, pivot, angle)
            env._route_trace = trace = RouteTrace([[0, 0], [.16, 0]])
            trace.update("demonstration", 0, 0, [.08, .1])
            trace.begin_execution([[0, 0], [.16, 0]])
            env._route_curve_tasks = {"execution": {0: 0}}
            env._drawing = Drawing()
            env._drawing.begin("execution", 0)
            env._drawing.update(10, [*xy, .07])
            with patch("robomme_plus.tasks.imitation.utils.tracing.final_feedback_ready", return_value=True), \
                 patch("robomme_plus.tasks.imitation.utils.split_stick.tip_position", return_value=np.array([*world_xy, .07])):
                result = env.evaluate()
            self.assertEqual(bool(result["success"]), reason is None)
            self.assertEqual(bool(result["fail"]), reason is not None)
            if reason:
                self.assertEqual(env._failure_code, reason)

    def test_rotating_layout_scores_each_dot_in_its_own_time_frame(self):
        from robomme_plus.tasks.imitation.utils.route_motion import rotate_points
        trace = self.reference()
        pivot = np.array([-.2, .2])
        for step, angle in enumerate(np.linspace(-.26, .26, 30)):
            local_dot = np.array([.08, .12])
            world_dot = rotate_points(local_dot, pivot, angle)
            trace.update("execution", 0, step+10, rotate_points(world_dot, pivot, -angle))
        self.assertIsNone(trace.failure)
        self.assertAlmostEqual(trace.summary()["curves"][0]["execution_max_distance_m"], .12)
        world_dot = rotate_points([.08, .131], pivot, .26)
        trace.update("execution", 0, 50, rotate_points(world_dot, pivot, -.26))
        self.assertEqual(trace.failure["reason"], "trace_lateral_extent")


if __name__ == "__main__":
    unittest.main()

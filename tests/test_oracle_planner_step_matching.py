import importlib.util
from pathlib import Path
import unittest

import numpy as np


def _load_oracle_action_matcher_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/robomme/env_record_wrapper/oracle_action_matcher.py"
    )
    spec = importlib.util.spec_from_file_location("oracle_action_matcher", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_MATCHER_MODULE = _load_oracle_action_matcher_module()
find_exact_option_index = _MATCHER_MODULE.find_exact_option_index
select_target_with_point = _MATCHER_MODULE.select_target_with_point


class _Obj:
    def __init__(self, name):
        self.name = name


class TestOraclePlannerStepMatching(unittest.TestCase):
    def setUp(self):
        self.obj_a = _Obj("a")
        self.obj_b = _Obj("b")
        self.obj_c = _Obj("c")

        # seg ids: 1 -> a, 2 -> b, 0 -> background
        self.seg_raw = np.array(
            [
                [1, 1, 0, 2],
                [1, 1, 0, 2],
                [0, 0, 0, 2],
                [0, 0, 0, 2],
            ],
            dtype=np.int64,
        )
        self.seg_id_map = {1: self.obj_a, 2: self.obj_b}

    def test_choice_exact_match_success_and_fail(self):
        options = [{"label": "pick up the cube"}, {"label": "press button"}]
        self.assertEqual(find_exact_option_index("pick up the cube", options), 0)
        self.assertEqual(find_exact_option_index("pick up", options), -1)
        self.assertEqual(find_exact_option_index(1, options), -1)

    def test_point_missing_returns_none(self):
        selected = select_target_with_point(
            seg_raw=self.seg_raw,
            seg_id_map=self.seg_id_map,
            available=[self.obj_a, self.obj_b],
            point_like=None,
        )
        self.assertIsNone(selected)

    def test_direct_hit_preferred_when_available(self):
        selected = select_target_with_point(
            seg_raw=self.seg_raw,
            seg_id_map=self.seg_id_map,
            available=[self.obj_a, self.obj_b],
            point_like=[0, 0],
        )
        self.assertIsNotNone(selected)
        self.assertIs(selected["obj"], self.obj_a)

    def test_nearest_used_when_direct_hit_not_available(self):
        selected = select_target_with_point(
            seg_raw=self.seg_raw,
            seg_id_map=self.seg_id_map,
            available=[self.obj_a, self.obj_b],
            point_like=[3, 2],  # background pixel, closer to obj_b centroid
        )
        self.assertIsNotNone(selected)
        self.assertIs(selected["obj"], self.obj_b)

    def test_returns_none_when_no_available_mask(self):
        selected = select_target_with_point(
            seg_raw=self.seg_raw,
            seg_id_map=self.seg_id_map,
            available=[self.obj_c],
            point_like=[1, 1],
        )
        self.assertIsNone(selected)

    def test_point_is_clipped_before_match(self):
        selected = select_target_with_point(
            seg_raw=self.seg_raw,
            seg_id_map=self.seg_id_map,
            available=[self.obj_a, self.obj_b],
            point_like=[-10, 100],  # clip to [0, 3] -> direct hit on seg id 2
        )
        self.assertIsNotNone(selected)
        self.assertIs(selected["obj"], self.obj_b)


if __name__ == "__main__":
    unittest.main()

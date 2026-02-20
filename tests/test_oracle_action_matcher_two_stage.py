import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

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
select_target_with_point = _MATCHER_MODULE.select_target_with_point
normalize_and_clip_point_xy = _MATCHER_MODULE.normalize_and_clip_point_xy


class _Obj:
    def __init__(self, name):
        self.name = name


class TestOracleActionMatcherTwoStage(unittest.TestCase):
    def setUp(self):
        self.obj_a = _Obj("a")
        self.obj_b = _Obj("b")
        self.obj_c = _Obj("c")

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

    def test_mask_hit_priority_over_random_fallback(self):
        with patch.object(_MATCHER_MODULE.random, "choice", return_value=self.obj_a):
            selected = select_target_with_point(
                seg_raw=self.seg_raw,
                seg_id_map=self.seg_id_map,
                available=[self.obj_a, self.obj_b],
                point_like=[3, 1],  # hit obj_b mask
            )
        self.assertIsNotNone(selected)
        self.assertIs(selected["obj"], self.obj_b)
        self.assertEqual(selected["seg_id"], 2)

    def test_random_fallback_when_no_mask_hit(self):
        with patch.object(_MATCHER_MODULE.random, "choice", return_value=self.obj_b):
            selected = select_target_with_point(
                seg_raw=self.seg_raw,
                seg_id_map=self.seg_id_map,
                available=[self.obj_a, self.obj_b],
                point_like=[2, 2],  # background
            )
        self.assertIsNotNone(selected)
        self.assertIs(selected["obj"], self.obj_b)

    def test_fallback_uses_action_available_candidates(self):
        with patch.object(_MATCHER_MODULE.random, "choice", return_value=self.obj_c):
            selected = select_target_with_point(
                seg_raw=self.seg_raw,
                seg_id_map=self.seg_id_map,
                available={"group": [self.obj_a, self.obj_c]},
                point_like=[3, 1],  # obj_b is clicked but not in available
            )
        self.assertIsNotNone(selected)
        self.assertIs(selected["obj"], self.obj_c)

    def test_normalize_and_clip_point_xy_bounds_and_invalid_inputs(self):
        self.assertEqual(
            normalize_and_clip_point_xy(point_like=[-4.3, 9.1], width=4, height=3),
            (0, 2),
        )
        self.assertIsNone(
            normalize_and_clip_point_xy(point_like=["x", 1], width=4, height=3)
        )
        self.assertIsNone(
            normalize_and_clip_point_xy(point_like=[1], width=4, height=3)
        )


if __name__ == "__main__":
    unittest.main()

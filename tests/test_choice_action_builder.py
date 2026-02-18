import importlib.util
import json
from pathlib import Path
import unittest


def _load_choice_action_builder_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/robomme/env_record_wrapper/choice_action_builder.py"
    )
    spec = importlib.util.spec_from_file_location("choice_action_builder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BUILDER_MODULE = _load_choice_action_builder_module()
build_choice_dict_find_point = _BUILDER_MODULE.build_choice_dict_find_point


class TestChoiceActionBuilder(unittest.TestCase):
    def test_pickxtimes_single_point(self):
        payload = json.loads(
            build_choice_dict_find_point(
                "PickXtimes",
                "press the button to stop",
                "press the button at <12, 34> to stop",
            )
        )
        self.assertEqual(payload["choice"], "press the button to stop")
        self.assertEqual(payload["point"], [12, 34])

    def test_pickxtimes_multiple_points_uses_first_only(self):
        payload = json.loads(
            build_choice_dict_find_point(
                "PickXtimes",
                "place the cube onto the target",
                "foo <10, 20> bar <30, 40>",
            )
        )
        self.assertEqual(payload["choice"], "place the cube onto the target")
        self.assertEqual(payload["point"], [10, 20])
        self.assertNotIn("points", payload)

    def test_pickxtimes_without_point(self):
        payload = json.loads(
            build_choice_dict_find_point(
                "PickXtimes",
                "pick up the cube",
                "pick up the cube now",
            )
        )
        self.assertEqual(payload, {"choice": "pick up the cube"})

    def test_pickxtimes_missing_vqa_label_returns_empty_json(self):
        self.assertEqual(build_choice_dict_find_point("PickXtimes", None, "x <1, 2>"), "{}")
        self.assertEqual(build_choice_dict_find_point("PickXtimes", "", "x <1, 2>"), "{}")

    def test_non_pickxtimes_returns_empty_json(self):
        self.assertEqual(
            build_choice_dict_find_point("MoveCube", "pick up the cube", "pick at <1, 2>"),
            "{}",
        )


if __name__ == "__main__":
    unittest.main()

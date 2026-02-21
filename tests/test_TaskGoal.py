"""
test_TaskGoal.py

Coverage tests for all env_id branches in task_goal.get_language_goal.
Run with: uv run python -m pytest tests/test_TaskGoal.py -s
"""
from pathlib import Path
import importlib.util
import types


def _load_task_goal_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "robomme" / "robomme_env" / "utils" / "task_goal.py"
    spec = importlib.util.spec_from_file_location("task_goal_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_task_goal_module()
get_language_goal = mod.get_language_goal


class _Unwrapped:
    """Mock env.unwrapped; arbitrary attributes can be set as needed."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Env:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped


class _Self:
    """Mock self object for called methods."""
    def __init__(self, env, **kwargs):
        self.env = env
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_self(unwrapped_attrs=None, **self_attrs):
    unwrapped = _Unwrapped(**(unwrapped_attrs or {}))
    env = _Env(unwrapped)
    return _Self(env, **self_attrs)


def _call(env_id, mock_self):
    result = get_language_goal(mock_self, env_id)
    print()
    for idx, goal in enumerate(result, start=1):
        print(f"[{env_id}] goal{idx}: {goal}")
    return result


def test_unknown_env_returns_single_goal_when_equal():
    """Unknown env_id goes to the default empty-string branch; when two goals are equal it returns a single-element list."""
    s = _make_self()
    result = _call("UnknownEnv", s)
    assert result == [""]


def test_movecube_still_returns_two_goals():
    """For defined branches with different texts, still return a two-element list."""
    s = _make_self()
    result = _call("MoveCube", s)
    assert len(result) == 2



def test_binfill_one_color():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(
        red_cubes_target_number=3,
        blue_cubes_target_number=0,
        green_cubes_target_number=0,
    ))
    result = _call("BinFill", s)
    assert "three red cubes" in result[0]
    assert " and " not in result[0]
    assert "three red cubes" in result[1]


def test_binfill_two_colors():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(
        red_cubes_target_number=1,
        blue_cubes_target_number=2,
        green_cubes_target_number=0,
    ))
    result = _call("BinFill", s)
    assert "one red cube" in result[0]
    assert "two blue cubes" in result[0]
    assert " and " in result[0]
    assert "one red cube" in result[1]
    assert "two blue cubes" in result[1]


def test_binfill_three_colors():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(
        red_cubes_target_number=2,
        blue_cubes_target_number=3,
        green_cubes_target_number=1,
    ))
    result = _call("BinFill", s)
    assert "two red cubes" in result[0]
    assert "three blue cubes" in result[0]
    assert "one green cube" in result[0]
    assert "two red cubes" in result[1]
    assert "three blue cubes" in result[1]
    assert "one green cube" in result[1]


# ── PickXtimes: 2 branches ──

def test_pickxtimes_once():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(num_repeats=1, target_color_name="red"))
    result = _call("PickXtimes", s)
    assert "repeating" not in result[0]
    assert "red cube" in result[0]
    if len(result) > 1:
        assert "repeating" not in result[1]
        assert "red cube" in result[1]


def test_pickxtimes_multiple():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(num_repeats=3, target_color_name="blue"))
    result = _call("PickXtimes", s)
    assert "repeating this action three times" in result[0]
    assert "blue cube" in result[0]
    assert "pick-and-place action three times" in result[1]
    assert "blue cube" in result[1]


# ── SwingXtimes: 2 branches ──

def test_swingxtimes_once():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(num_repeats=1, target_color_name="green"))
    result = _call("SwingXtimes", s)
    assert "put it down on the left-side target" in result[0]
    assert "repeating" not in result[0]
    assert "left-side target" in result[1]
    assert "repeating" not in result[1]


def test_swingxtimes_multiple():
    """Branch behavior test case."""
    s = _make_self(unwrapped_attrs=dict(num_repeats=5, target_color_name="red"))
    result = _call("SwingXtimes", s)
    assert "repeating this back and forth motion five times" in result[0]
    assert "right-to-left swing motion five times" in result[1]


# ── VideoUnmask: 2 branches ──

def test_videounmask_pick_one():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["red", "blue", "green"],
            configs={"easy": {"pick": 1}},
        ),
        difficulty="easy",
    )
    result = _call("VideoUnmask", s)
    assert "red cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container" not in g


def test_videounmask_pick_two():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["red", "blue", "green"],
            configs={"hard": {"pick": 2}},
        ),
        difficulty="hard",
    )
    result = _call("VideoUnmask", s)
    assert "red cube" in result[0]
    assert "another container hiding the blue cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container hiding the blue cube" in g


# ── VideoUnmaskSwap: 2 branches ──

def test_videounmaskswap_pick_one():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["red", "blue", "green"]),
        pick_times=1,
    )
    result = _call("VideoUnmaskSwap", s)
    assert "red cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container" not in g


def test_videounmaskswap_pick_two():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["red", "blue", "green"]),
        pick_times=2,
    )
    result = _call("VideoUnmaskSwap", s)
    assert "red cube" in result[0]
    assert "another container hiding the blue cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "red cube" in g
    assert "another container hiding the blue cube" in g


# ── ButtonUnmask: 2 branches ──

def test_buttonunmask_pick_one():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["green", "red", "blue"],
            configs={"easy": {"pick": 1}},
        ),
        difficulty="easy",
    )
    result = _call("ButtonUnmask", s)
    assert "press the button" in result[0]
    assert "green cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press the button" in g
    assert "green cube" in g
    assert "another container" not in g


def test_buttonunmask_pick_two():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(
            color_names=["green", "red", "blue"],
            configs={"hard": {"pick": 2}},
        ),
        difficulty="hard",
    )
    result = _call("ButtonUnmask", s)
    assert "press the button" in result[0]
    assert "another container hiding the red cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press the button" in g
    assert "another container hiding the red cube" in g


# ── ButtonUnmaskSwap: 2 branches ──

def test_buttonunmaskswap_pick_one():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["blue", "green", "red"]),
        pick_times=1,
    )
    result = _call("ButtonUnmaskSwap", s)
    assert "press both buttons" in result[0]
    assert "blue cube" in result[0]
    assert "another container" not in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press both buttons" in g
    assert "blue cube" in g
    assert "another container" not in g


def test_buttonunmaskswap_pick_two():
    """Branch behavior test case."""
    s = _make_self(
        unwrapped_attrs=dict(color_names=["blue", "green", "red"]),
        pick_times=2,
    )
    result = _call("ButtonUnmaskSwap", s)
    assert "press both buttons" in result[0]
    assert "another container hiding the green cube" in result[0]
    g = result[1] if len(result) > 1 else result[0]
    assert "press both buttons" in g
    assert "another container hiding the green cube" in g


# ── VideoPlaceButton: 1 branch ──

def test_videoplacebutton():
    """Branch behavior test case."""
    s = _make_self(
        target_color_name="red",
        target_target_language="after",
    )
    result = _call("VideoPlaceButton", s)
    assert "red cube" in result[0]
    assert "right after the button was pressed" in result[0]
    assert "red cube" in result[1]
    assert "where it was placed immediately after the button was pressed" in result[1]


# ── VideoPlaceOrder: 1 branch ──

def test_videoplaceorder():
    """Case: place at the N-th target in order.Inputs: target_color_name=blue, which_in_subset=3."""
    s = _make_self(
        target_color_name="blue",
        which_in_subset=3,
    )
    result = _call("VideoPlaceOrder", s)
    assert "blue cube" in result[0]
    assert "third target" in result[0]
    assert "blue cube" in result[1]
    assert "third target" in result[1]
    assert "where it was placed" in result[1]


# ── PickHighlight: 1 branch ──

def test_pickhighlight():
    """Branch behavior test case."""
    s = _make_self()
    result = _call("PickHighlight", s)
    assert "press the button" in result[0]
    assert "highlighteted" in result[0]
    assert "highlighted cubes" in result[1]
    assert "press the button again to stop" in result[1]


# ── VideoRepick: 2 branches ──
# Inputs: self.num_repeats.

def test_videorepick_once():
    """Branch behavior test case."""
    s = _make_self(num_repeats=1)
    result = _call("VideoRepick", s)
    assert "pick up the same block" in result[0]
    assert "repeatedly" not in result[0]
    assert "pick up the same cube" in result[1]
    assert "repeatedly" not in result[1]


def test_videorepick_multiple():
    """Branch behavior test case."""
    s = _make_self(num_repeats=4)
    result = _call("VideoRepick", s)
    assert "repeatedly pick up and put down" in result[0]
    assert "four times" in result[0]
    assert "same cube" in result[1]
    assert "four times" in result[1]


# ── StopCube: 1 branch ──

def test_stopcube():
    """Case: press button to stop at the N-th arrival.Inputs: stop_time=2."""
    s = _make_self(unwrapped_attrs=dict(stop_time=2))
    result = _call("StopCube", s)
    assert "second time" in result[0]
    assert "second visit" in result[1]


# ── InsertPeg: 1 branch ──

def test_insertpeg():
    """Case: insert the same peg end into the same side.Inputs: none."""
    s = _make_self()
    result = _call("InsertPeg", s)
    assert "grasp the same end" in result[0]
    assert "grasp the same peg at the same end" in result[1]
    assert "as in the video" in result[1]


# ── MoveCube: 1 branch ──

def test_movecube():
    """Case: move the cube to target in the same way as before.Inputs: none."""
    s = _make_self()
    result = _call("MoveCube", s)
    assert "move the cube to the target" in result[0]
    assert "shown in the video" in result[1]


# ── PatternLock: 1 branch ──

def test_patternlock():
    """Case: retrace the same pattern with the stick.Inputs: none."""
    s = _make_self()
    result = _call("PatternLock", s)
    assert "retrace the same pattern" in result[0]
    assert "retrace the same pattern shown in the video" in result[1]


# ── RouteStick: 1 branch ──

def test_routestick():
    """Case: use the stick to follow the same path around the rods.Inputs: none."""
    s = _make_self()
    result = _call("RouteStick", s)
    assert "navigate around the sticks" in result[0]
    assert "following the same path shown in the video" in result[1]

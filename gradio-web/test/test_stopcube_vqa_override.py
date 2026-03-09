from __future__ import annotations

from robomme.robomme_env.utils import vqa_options as upstream_vqa_options


class _DummyBase:
    def __init__(self, steps_press, interval=30):
        self.steps_press = steps_press
        self.interval = interval
        self.button = object()


class _DummyEnv:
    def __init__(self, base, elapsed_steps=0):
        self.unwrapped = base
        self.elapsed_steps = elapsed_steps


def _get_stopcube_options(module, env):
    return module.get_vqa_options(env, planner=None, selected_target={"obj": None}, env_id="StopCube")


def _get_remain_static_solver(options):
    for option in options:
        if option.get("action") == "remain static":
            return option["solve"]
    raise AssertionError("Missing 'remain static' option")


def test_stopcube_remain_static_merges_short_tail(monkeypatch, reload_module):
    override = reload_module("vqa_options_override")

    hold_calls = []

    def _hold_spy(env, planner, absTimestep):
        _ = planner
        hold_calls.append(int(absTimestep))
        env.elapsed_steps = int(absTimestep)
        return None

    monkeypatch.setattr(override, "solve_hold_obj_absTimestep", _hold_spy)

    base = _DummyBase(steps_press=270, interval=30)
    env = _DummyEnv(base, elapsed_steps=0)
    options = _get_stopcube_options(override, env)

    actions = [option.get("action") for option in options]
    assert actions == [
        "move to the top of the button to prepare",
        "remain static",
        "press button to stop the cube",
    ]

    solve_remain_static = _get_remain_static_solver(options)
    for _ in range(3):
        solve_remain_static()

    assert hold_calls == [100, 240, 240]


def test_stopcube_remain_static_keeps_boundary_tail(monkeypatch, reload_module):
    override = reload_module("vqa_options_override")

    hold_calls = []

    def _hold_spy(env, planner, absTimestep):
        _ = planner
        hold_calls.append(int(absTimestep))
        env.elapsed_steps = int(absTimestep)
        return None

    monkeypatch.setattr(override, "solve_hold_obj_absTimestep", _hold_spy)

    base = _DummyBase(steps_press=280, interval=30)
    env = _DummyEnv(base, elapsed_steps=0)
    solve_remain_static = _get_remain_static_solver(_get_stopcube_options(override, env))

    for _ in range(4):
        solve_remain_static()

    assert hold_calls == [100, 200, 250, 250]


def test_stopcube_remain_static_resets_after_elapsed_steps_go_back(monkeypatch, reload_module):
    override = reload_module("vqa_options_override")

    hold_calls = []

    def _hold_spy(env, planner, absTimestep):
        _ = planner
        hold_calls.append(int(absTimestep))
        env.elapsed_steps = int(absTimestep)
        return None

    monkeypatch.setattr(override, "solve_hold_obj_absTimestep", _hold_spy)

    base = _DummyBase(steps_press=270, interval=30)
    env = _DummyEnv(base, elapsed_steps=0)
    solve_remain_static = _get_remain_static_solver(_get_stopcube_options(override, env))

    solve_remain_static()
    solve_remain_static()
    env.elapsed_steps = 0
    solve_remain_static()

    assert hold_calls == [100, 240, 100]


def test_non_stopcube_builders_passthrough_to_upstream(reload_module):
    override = reload_module("vqa_options_override")

    assert override.OPTION_BUILDERS["StopCube"] is override._options_stopcube_override
    assert override.OPTION_BUILDERS["StopCube"] is not upstream_vqa_options.OPTION_BUILDERS["StopCube"]
    assert override.OPTION_BUILDERS["BinFill"] is upstream_vqa_options.OPTION_BUILDERS["BinFill"]

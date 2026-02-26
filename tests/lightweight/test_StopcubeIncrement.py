"""
Unit tests for the 'remain static' option in the StopCube task.

Test increment and saturation logic of the 'remain static' option in vqa_options._options_stopcube:
- Each time 'remain static' is selected, solve_hold_obj_absTimestep(absTimestep) is called internally;
- absTimestep increases as 100 -> 200 -> ... and does not exceed final_target (computed from steps_press and interval);
- If env.elapsed_steps goes backward, internal state should reset and start from 100 next time.
"""
from pathlib import Path
import importlib.util
import sys
import types

from tests._shared.repo_paths import find_repo_root


# Keep symbols consistent with planner module dependencies in vqa_options, used to build stubs
PLANNER_SYMBOLS = [
    "grasp_and_lift_peg_side",
    "insert_peg",
    "solve_button",
    "solve_button_ready",
    "solve_hold_obj",
    "solve_hold_obj_absTimestep",
    "solve_pickup",
    "solve_pickup_bin",
    "solve_push_to_target",
    "solve_push_to_target_with_peg",
    "solve_putdown_whenhold",
    "solve_putonto_whenhold",
    "solve_putonto_whenhold_binspecial",
    "solve_swingonto",
    "solve_swingonto_withDirection",
    "solve_swingonto_whenhold",
    "solve_strong_reset",
]


def _load_vqa_options_module():
    """Inject a stub planner module and load vqa_options, returning (module, hold_calls list)."""
    hold_calls = []

    planner_stub = types.ModuleType("robomme.robomme_env.utils.subgoal_planner_func")

    def _noop(*args, **kwargs):
        return None

    for symbol in PLANNER_SYMBOLS:
        setattr(planner_stub, symbol, _noop)

    def _hold_spy(env, planner, absTimestep):
        """Record the actual absTimestep passed for each 'remain static' call for assertions."""
        hold_calls.append(int(absTimestep))
        return None

    planner_stub.solve_hold_obj_absTimestep = _hold_spy

    robomme_pkg = types.ModuleType("robomme")
    robomme_pkg.__path__ = []
    robomme_env_pkg = types.ModuleType("robomme.robomme_env")
    robomme_env_pkg.__path__ = []
    utils_pkg = types.ModuleType("robomme.robomme_env.utils")
    utils_pkg.__path__ = []

    logging_utils_pkg = types.ModuleType("robomme.logging_utils")
    import logging
    logging_utils_pkg.logger = logging.getLogger("dummy")

    robomme_pkg.robomme_env = robomme_env_pkg
    robomme_pkg.logging_utils = logging_utils_pkg
    robomme_env_pkg.utils = utils_pkg
    utils_pkg.subgoal_planner_func = planner_stub

    injected = {
        "robomme": robomme_pkg,
        "robomme.robomme_env": robomme_env_pkg,
        "robomme.robomme_env.utils": utils_pkg,
        "robomme.robomme_env.utils.subgoal_planner_func": planner_stub,
        "robomme.logging_utils": logging_utils_pkg,
    }
    previous = {key: sys.modules.get(key) for key in injected}
    sys.modules.update(injected)

    try:
        repo_root = find_repo_root(__file__)
        module_path = repo_root / "src" / "robomme" / "robomme_env" / "utils" / "vqa_options.py"
        spec = importlib.util.spec_from_file_location("robomme.robomme_env.utils.vqa_options", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old_module

    return module, hold_calls


class _DummyEnv:
    """Mock environment exposing only elapsed_steps for _options_stopcube."""
    def __init__(self, elapsed_steps=0):
        self.elapsed_steps = elapsed_steps


class _DummyBase:
    """Mock StopCube base providing steps_press and interval to compute final_target."""
    def __init__(self, steps_press, interval=30):
        self.steps_press = steps_press
        self.interval = interval
        self.button = object()


def _get_remain_static_solver(options):
    """Get the solve function whose label is 'remain static' from StopCube options."""
    for option in options:
        if option.get("action") == "remain static":
            return option["solve"]
    raise AssertionError("Missing 'remain static' option")


def test_stopcube_remain_static_increment_and_saturation():
    """Test: when selecting 'remain static' multiple times, absTimestep increments 100->200->... and saturates at final_target."""
    module, hold_calls = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=270, interval=30)  # final_target = 240
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)
    solve_remain_static = _get_remain_static_solver(options)

    for _ in range(4):
        solve_remain_static()

    assert hold_calls == [100, 200, 240, 240]


def test_stopcube_remain_static_small_final_target():
    """Test: when final_target is small (e.g., 60), first call hits the cap and later calls stay at 60 (saturation)."""
    module, hold_calls = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=90, interval=30)  # final_target = 60
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)
    solve_remain_static = _get_remain_static_solver(options)

    solve_remain_static()
    solve_remain_static()

    assert hold_calls == [60, 60]


def test_stopcube_remain_static_resets_when_elapsed_steps_go_back():
    """Test: when elapsed_steps is reduced (e.g., 150 back to 0), internal step state should reset and restart from 100."""
    module, hold_calls = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=270, interval=30)  # final_target = 240
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)
    solve_remain_static = _get_remain_static_solver(options)

    solve_remain_static()  # 100
    env.elapsed_steps = 150
    solve_remain_static()  # 200
    env.elapsed_steps = 0
    solve_remain_static()  # reset -> 100

    assert hold_calls == [100, 200, 100]


def test_stopcube_option_label_order_stays_stable():
    """Test: StopCube option label order must stay fixed: prepare first, then remain static, then press button."""
    module, _ = _load_vqa_options_module()
    env = _DummyEnv(elapsed_steps=0)
    base = _DummyBase(steps_press=270, interval=30)
    options = module._options_stopcube(env, planner=None, require_target=lambda: None, base=base)

    actions = [option.get("action") for option in options]
    assert actions == [
        "move to the top of the button to prepare",
        "remain static",
        "press button to stop the cube",
    ]

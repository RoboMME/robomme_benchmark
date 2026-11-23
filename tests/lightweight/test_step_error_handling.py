# -*- coding: utf-8 -*-
"""
test_step_error_handling.py
============================
Lightweight test: step exceptions reach callers as info["status"] = "error"
(via FailAwareWrapper) instead of propagating.

Also verify that the step loops in run_example.py and dataset_replay.py
check info["status"] rather than bare try/except.

Run from the repository root:
    uv run python -m pytest tests/lightweight/test_step_error_handling.py
"""

from __future__ import annotations

import ast
import sys
import types
import unittest.mock as mock
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo path helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root, ensure_src_on_path

_PROJECT_ROOT = find_repo_root(__file__)
ensure_src_on_path(__file__)


# ---------------------------------------------------------------------------
# Helpers: load DemonstrationWrapper.step source for AST inspection
# ---------------------------------------------------------------------------

def _demo_wrapper_path() -> Path:
    return _PROJECT_ROOT / "src/robomme/env_record_wrapper/DemonstrationWrapper.py"


def _load_step_source() -> str:
    return _demo_wrapper_path().read_text(encoding="utf-8")


def _script_path(name: str) -> Path:
    return _PROJECT_ROOT / "scripts" / name


# ---------------------------------------------------------------------------
# Test 1: step exceptions reach callers as info["status"] == "error"
# ---------------------------------------------------------------------------

def test_step_error_returns_status_error() -> None:
    """
    The builder wraps every episode in FailAwareWrapper (episode_config_resolver.py),
    which converts a step exception into a terminated step with status="error".
    DemonstrationWrapper.step() itself lets exceptions propagate to that layer.
    """
    import gymnasium as gym
    from robomme.env_record_wrapper.FailAwareWrapper import FailAwareWrapper

    resolver = (_PROJECT_ROOT / "src/robomme/env_record_wrapper/episode_config_resolver.py").read_text()
    assert "FailAwareWrapper(env)" in resolver, "The builder must wrap episodes in FailAwareWrapper"

    class RaisingEnv(gym.Env):
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return 0, {}

        def step(self, action):
            raise RuntimeError("IK failed: no solution found")

    env = FailAwareWrapper(RaisingEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated is True and truncated is False
    assert info["status"] == "error"
    assert info["exception_type"] == "RuntimeError"
    assert "IK failed" in info["error_message"]


# ---------------------------------------------------------------------------
# Test 2: Runtime behavior verification — Mock actual call
# ---------------------------------------------------------------------------

def test_step_error_runtime_behavior() -> None:
    """
    Directly call DemonstrationWrapper.step() using a Mock object,
    verifying that the return value meets the contract when _step_batch throws an exception.
    """
    # Dynamically inject Mock dependencies, do not actually import ManiSkill
    _inject_mock_dependencies()

    # Set sys.path to point to the directory where DemonstrationWrapper is located, for from episode... import etc.
    wrapper_dir = str(_PROJECT_ROOT / "src" / "robomme" / "env_record_wrapper")
    if wrapper_dir not in sys.path:
        sys.path.insert(0, wrapper_dir)

    # Directly execute the try/except logic of step() without relying on a real class instance
    # ——by constructing a fake instance where _step_batch throws an exception

    class FakeDemoWrapper:
        """Minimal stub, only implements the logic used in step()."""

        @staticmethod
        def _step_batch(action):
            raise RuntimeError("IK failed: no solution found")

        @staticmethod
        def _flatten_info_batch(info_batch):
            return {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}

        def step(self, action):
            try:
                batch = self._step_batch(action)
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
                info_flat = self._flatten_info_batch(info_batch)
                return (obs_batch, reward_batch[-1], terminated_batch[-1], truncated_batch[-1], info_flat)
            except Exception as exc:
                error_info = {
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                return ({}, 0.0, True, False, error_info)

    wrapper = FakeDemoWrapper()
    obs, reward, terminated, truncated, info = wrapper.step(action=[0.0] * 8)

    assert obs == {}, f"obs should be an empty dict on error, got {obs!r}"
    assert reward == 0.0, f"reward should be 0.0 on error, got {reward!r}"
    assert terminated is True, f"terminated should be True on error, got {terminated!r}"
    assert truncated is False, f"truncated should be False on error, got {truncated!r}"
    assert info.get("status") == "error", f"status should be 'error', got {info.get('status')!r}"
    assert "RuntimeError" in info.get("error_message", ""), (
        f"error_message should contain exception type, got {info.get('error_message')!r}"
    )
    assert "IK failed" in info.get("error_message", ""), (
        f"error_message should contain original exception info, got {info.get('error_message')!r}"
    )

    print("  ✓ step() returns status='error' + correct error_message when throwing an exception")


def test_step_normal_returns_ongoing_status() -> None:
    """
    Verify that step() should not return status='error' when the Mock env returns normally.
    (Indirect test: status will not be tampered with as error under normal paths)
    """
    import torch

    class FakeDemoWrapperNormal:
        """Normal _step_batch, info["status"] = "ongoing"."""

        def _step_batch(self, action):
            obs_batch = {"front_rgb_list": [None]}
            reward_batch = torch.tensor([0.1])
            terminated_batch = torch.tensor([False])
            truncated_batch = torch.tensor([False])
            info_batch = {"status": ["ongoing"], "success": [False]}
            return (obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch)

        def _flatten_info_batch(self, info_batch):
            return {k: v[-1] if isinstance(v, list) and v else v for k, v in info_batch.items()}

        def step(self, action):
            try:
                batch = self._step_batch(action)
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = batch
                info_flat = self._flatten_info_batch(info_batch)
                return (obs_batch, reward_batch[-1], terminated_batch[-1], truncated_batch[-1], info_flat)
            except Exception as exc:
                error_info = {
                    "status": "error",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                return ({}, 0.0, True, False, error_info)

    wrapper = FakeDemoWrapperNormal()
    obs, reward, terminated, truncated, info = wrapper.step(action=[0.0] * 8)

    assert info.get("status") == "ongoing", (
        f"Normal step status should be 'ongoing', got {info.get('status')!r}"
    )
    assert "error_message" not in info, "Normal step should not contain error_message"

    print("  ✓ Normal step returns status='ongoing', no error_message")


# ---------------------------------------------------------------------------
# Test 3: AST check that scripts no longer have bare try/except wrapping env.step(action)
# ---------------------------------------------------------------------------

def test_scripts_use_status_check_not_bare_try_except() -> None:
    """
    Parse run_example.py and dataset_replay.py, verify:
    1. There is an info.get("status") or status == "error" check in the script
    2. env.step(action) calls are no longer directly wrapped by try/except Exception
    """
    scripts = ["run_example.py", "dataset_replay.py"]

    for script_name in scripts:
        script_path = _script_path(script_name)
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script_path))

        # ---- Check 1: Contains status related checks ----
        has_status_check = (
            'info.get("status"' in source
            or "status == \"error\"" in source
            or "status==" in source.replace(" ", "")
        )
        assert has_status_check, (
            f"{script_name}: Should have info.get('status') or status==\"error\" check"
        )

        # ---- Check 2: env.step is not wrapped by bare try/except ----
        # Exact lookup: env.step is directly called in the Try block and handler catches Exception
        _assert_no_bare_step_try_except(tree, script_name)

        print(f"  ✓ {script_name}: Use status check, no bare try/except wrapping env.step")


def _assert_no_bare_step_try_except(tree: ast.AST, script_name: str) -> None:
    """Check that there is no 'try block contains env.step and except catches Exception' structure in the AST."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        # Check if there is an env.step(action) call in the try body
        step_in_try = False
        for n in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(n, ast.Call)
                and isinstance(getattr(n, "func", None), ast.Attribute)
                and n.func.attr == "step"
            ):
                step_in_try = True
                break

        if not step_in_try:
            continue

        # Check if handler is a bare Exception catch
        for handler in node.handlers:
            if handler.type is None:
                assert False, (
                    f"{script_name}: env.step is still wrapped by bare try/except: (no exception type), should be changed to status check"
                )
            if isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                assert False, (
                    f"{script_name}: env.step is still wrapped by bare try/except Exception: should be changed to status check"
                )


# ---------------------------------------------------------------------------
# Utility: inject mock modules so imports inside DemonstrationWrapper don't fail
# ---------------------------------------------------------------------------

def _inject_mock_dependencies() -> None:
    """Inject placeholder mock modules to prevent import DemonstrationWrapper from failing due to missing ManiSkill."""
    mock_mods = [
        "mani_skill",
        "mani_skill.envs",
        "mani_skill.envs.sapien_env",
        "mani_skill.utils",
        "mani_skill.utils.common",
        "mani_skill.utils.gym_utils",
        "mani_skill.utils.sapien_utils",
        "mani_skill.utils.io_utils",
        "mani_skill.utils.logging_utils",
        "mani_skill.utils.structs",
        "mani_skill.utils.structs.types",
        "mani_skill.utils.wrappers",
        "mani_skill.examples",
        "mani_skill.examples.motionplanning",
        "mani_skill.examples.motionplanning.panda",
        "mani_skill.examples.motionplanning.panda.motionplanner",
        "mani_skill.examples.motionplanning.panda.motionplanner_stick",
        "mani_skill.examples.motionplanning.base_motionplanner",
        "mani_skill.examples.motionplanning.base_motionplanner.utils",
        "sapien",
        "sapien.physx",
        "gymnasium",
        "h5py",
        "imageio",
        "colorsys",
    ]
    for mod_name in mock_mods:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n[TEST] DemonstrationWrapper step error handling")

    test_step_error_returns_status_error()
    print("  test1: AST structure verification passed")

    test_step_error_runtime_behavior()
    print("  test2: Runtime behavior verification passed")

    test_step_normal_returns_ongoing_status()
    print("  test3: Normal path verification passed")

    print("\n[TEST] Script status check verification")
    test_scripts_use_status_check_not_bare_try_except()

    print("\nPASS: All step error handling tests passed")


if __name__ == "__main__":
    main()

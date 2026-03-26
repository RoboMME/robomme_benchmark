from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root, ensure_src_on_path

_PROJECT_ROOT = find_repo_root(__file__)
ensure_src_on_path(__file__)

_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "dev"
_SCRIPT_DIR_STR = str(_SCRIPT_DIR)
if _SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_STR)

import snapshot as snapshot_utils  # noqa: E402


class _FakeActor:
    def __init__(self, name: str, position_xyz):
        self.name = name
        self.pose = SimpleNamespace(
            p=np.asarray(position_xyz, dtype=np.float32).reshape(1, 3)
        )


class _FakeScene:
    def __init__(self, env, colliding_steps: set[int] | None = None):
        self.env = env
        self.colliding_steps = set(colliding_steps or set())

    def get_pairwise_contact_forces(self, actor_a, actor_b):
        if self.env.elapsed_steps in self.colliding_steps:
            return np.asarray([[0.0, 0.0, 0.1]], dtype=np.float32)
        return np.zeros((1, 3), dtype=np.float32)


class _ErrorScene:
    def get_pairwise_contact_forces(self, actor_a, actor_b):
        raise RuntimeError("pairwise query failed")


class _FakeEnv:
    def __init__(self, base_env):
        self.unwrapped = base_env
        self._base_env = base_env

    def step(self, action):
        self._base_env.elapsed_steps += 1
        return ("obs", 0.0, False, False, {"action": action})


def _make_base_env(*, colliding_steps: set[int] | None = None):
    bin_0 = _FakeActor("bin_0", [0.10, -0.10, 0.052])
    bin_1 = _FakeActor("bin_1", [0.00, 0.15, 0.052])
    bin_2 = _FakeActor("bin_2", [0.13, 0.16, 0.052])
    bin_3 = _FakeActor("bin_3", [0.09, -0.03, 0.052])

    cube_blue = _FakeActor("target_cube_blue", [0.10, -0.10, 0.0167])
    cube_red = _FakeActor("target_cube_red", [0.13, 0.16, 0.0167])
    cube_green = _FakeActor("target_cube_green", [0.00, 0.15, 0.0167])

    base_env = SimpleNamespace(
        elapsed_steps=0,
        spawned_bins=[bin_0, bin_1, bin_2, bin_3],
        cube_bin_pairs=[
            (cube_blue, bin_0),
            (cube_red, bin_2),
            (cube_green, bin_1),
        ],
        color_names=["blue", "red", "green"],
        bin_to_color={0: "blue", 2: "red", 1: "green"},
    )
    base_env.scene = _FakeScene(base_env, colliding_steps=colliding_steps)
    return base_env


def test_collect_button_unmask_swap_snapshot_shape() -> None:
    base_env = _make_base_env()

    payload = snapshot_utils._collect_snapshot(
        base_env=base_env,
        env_id="ButtonUnmaskSwap",
        episode=1,
        seed=0,
        difficulty="hard",
        capture_elapsed_steps=33,
        collision=True,
    )

    assert payload["env_id"] == "ButtonUnmaskSwap"
    assert payload["inspect_this_timestep"] == 33
    assert payload["capture_elapsed_steps"] == 33
    assert payload["collision"] is True

    cubes = payload["cubes"]
    bins = payload["bins"]

    assert len(cubes) == len(base_env.cube_bin_pairs)
    assert len(bins) == len(base_env.spawned_bins)

    for cube_item in cubes:
        assert "color" in cube_item
        assert isinstance(cube_item["position_xyz"], list)
        assert len(cube_item["position_xyz"]) == 3

    for bin_item in bins:
        assert "color" not in bin_item
        assert isinstance(bin_item["position_xyz"], list)
        assert len(bin_item["position_xyz"]) == 3

    empty_bin = next(item for item in bins if item["index"] == 3)
    assert empty_bin["has_cube_under_bin"] is False


def test_snapshot_json_path_location() -> None:
    output_root = Path("/tmp/robomme_snapshot_test")
    json_path = snapshot_utils._snapshot_json_path(
        output_root=output_root,
        env_id="ButtonUnmaskSwap",
        episode=1,
        seed=7,
    )
    assert json_path == output_root / "snapshots" / "ButtonUnmaskSwap_ep1_seed7_after_drop.json"


def test_button_unmask_swap_inspect_this_timestep_value() -> None:
    assert snapshot_utils.SNAPSHOT_ENVS["ButtonUnmaskSwap"] == 33


def test_install_snapshot_tracks_collision_across_steps(tmp_path, monkeypatch) -> None:
    base_env = _make_base_env(colliding_steps={2})
    env = _FakeEnv(base_env)
    monkeypatch.setitem(snapshot_utils.SNAPSHOT_ENVS, "ButtonUnmask", 3)

    state = snapshot_utils.install_snapshot_for_step(
        env=env,
        env_id="ButtonUnmask",
        episode=4,
        seed=1580400,
        difficulty="hard",
        output_dir=tmp_path,
    )

    for _ in range(3):
        env.step(None)

    assert state["collision_detected"] is True
    assert state["snapshot_written"] is True

    payload = json.loads(state["snapshot_json_path"].read_text(encoding="utf-8"))
    assert payload["capture_elapsed_steps"] == 3
    assert payload["collision"] is True


def test_install_snapshot_rewrites_collision_when_hit_after_capture(
    tmp_path, monkeypatch
) -> None:
    base_env = _make_base_env(colliding_steps={4})
    env = _FakeEnv(base_env)
    monkeypatch.setitem(snapshot_utils.SNAPSHOT_ENVS, "ButtonUnmask", 2)

    state = snapshot_utils.install_snapshot_for_step(
        env=env,
        env_id="ButtonUnmask",
        episode=5,
        seed=11,
        difficulty="hard",
        output_dir=tmp_path,
    )

    for _ in range(2):
        env.step(None)

    initial_payload = json.loads(state["snapshot_json_path"].read_text(encoding="utf-8"))
    assert initial_payload["capture_elapsed_steps"] == 2
    assert initial_payload["collision"] is False

    for _ in range(2):
        env.step(None)

    final_payload = json.loads(state["snapshot_json_path"].read_text(encoding="utf-8"))
    assert state["collision_detected"] is True
    assert state["snapshot_collision_synced"] is True
    assert final_payload["capture_elapsed_steps"] == 2
    assert final_payload["collision"] is True


def test_install_snapshot_writes_false_when_no_collision(tmp_path, monkeypatch) -> None:
    base_env = _make_base_env()
    env = _FakeEnv(base_env)
    monkeypatch.setitem(snapshot_utils.SNAPSHOT_ENVS, "ButtonUnmask", 2)

    state = snapshot_utils.install_snapshot_for_step(
        env=env,
        env_id="ButtonUnmask",
        episode=1,
        seed=7,
        difficulty="easy",
        output_dir=tmp_path,
    )

    for _ in range(2):
        env.step(None)

    assert state["collision_detected"] is False
    assert state["snapshot_written"] is True

    payload = json.loads(state["snapshot_json_path"].read_text(encoding="utf-8"))
    assert payload["capture_elapsed_steps"] == 2
    assert payload["collision"] is False


def test_install_snapshot_does_not_rewrite_when_no_collision_after_capture(
    tmp_path, monkeypatch
) -> None:
    base_env = _make_base_env()
    env = _FakeEnv(base_env)
    monkeypatch.setitem(snapshot_utils.SNAPSHOT_ENVS, "ButtonUnmask", 2)

    state = snapshot_utils.install_snapshot_for_step(
        env=env,
        env_id="ButtonUnmask",
        episode=6,
        seed=12,
        difficulty="easy",
        output_dir=tmp_path,
    )

    for _ in range(2):
        env.step(None)

    snapshot_path = state["snapshot_json_path"]
    assert snapshot_path is not None
    initial_mtime_ns = snapshot_path.stat().st_mtime_ns

    for _ in range(3):
        env.step(None)

    final_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert state["collision_detected"] is False
    assert state["snapshot_collision_synced"] is False
    assert snapshot_path.stat().st_mtime_ns == initial_mtime_ns
    assert final_payload["capture_elapsed_steps"] == 2
    assert final_payload["collision"] is False


def test_install_snapshot_ignores_contact_query_errors(tmp_path, monkeypatch) -> None:
    base_env = _make_base_env()
    base_env.scene = _ErrorScene()
    env = _FakeEnv(base_env)
    monkeypatch.setitem(snapshot_utils.SNAPSHOT_ENVS, "ButtonUnmask", 2)

    state = snapshot_utils.install_snapshot_for_step(
        env=env,
        env_id="ButtonUnmask",
        episode=2,
        seed=9,
        difficulty="medium",
        output_dir=tmp_path,
    )

    step_result = None
    for _ in range(2):
        step_result = env.step(None)

    assert step_result == ("obs", 0.0, False, False, {"action": None})
    assert state["collision_detected"] is False

    payload = json.loads(state["snapshot_json_path"].read_text(encoding="utf-8"))
    assert payload["collision"] is False


def test_step_has_bin_collision_ignores_offscreen_staging_bins() -> None:
    base_env = _make_base_env(colliding_steps={0})
    for actor in base_env.spawned_bins[:3]:
        actor.pose.p = np.asarray([[10.0, 10.0, 10.0]], dtype=np.float32)

    assert snapshot_utils._step_has_bin_collision(base_env) is False

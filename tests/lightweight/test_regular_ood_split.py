"""OOD identity and routing checks; simulator construction is checked separately."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from typing import Literal

import pytest



ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/robomme"


@pytest.fixture
def resolver(monkeypatch):
    if not (SOURCE / "env_metadata/ood_test").exists():
        pytest.skip("RoboMME OOD source is not installed")
    monkeypatch.syspath_prepend(str(SOURCE.parent))
    monkeypatch.setitem(sys.modules, "gymnasium", ModuleType("gymnasium"))
    path = SOURCE / "env_record_wrapper/episode_config_resolver.py"
    spec = importlib.util.spec_from_file_location(
        "robomme.env_record_wrapper.episode_config_resolver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ood_is_twenty_test_only_identities_per_task(resolver):
    tasks = resolver.BenchmarkEnvBuilder.get_task_list("ood_test")
    assert set(tasks) == set(resolver._DEFAULT_TASK_LIST) and len(tasks) == 16
    existing = {
        int(row["seed"])
        for split in ("train", "val", "test")
        for path in (SOURCE / "env_metadata" / split).glob("*.json")
        for row in json.loads(path.read_text())["records"]
    }
    ood = []
    for task in tasks:
        rows = json.loads((SOURCE / f"env_metadata/ood_test/record_dataset_{task}_metadata.json").read_text())["records"]
        builder = resolver.BenchmarkEnvBuilder(task, dataset="ood_test")
        assert builder.get_episode_num() == len(rows) == 20
        assert [row["episode"] for row in rows] == list(range(20))
        for row in rows:
            assert row["task"] == task and row["difficulty"] == "ood"
            if row.get("seed_scheme") == "robomme-ood-expert-v1":
                key = f"robomme-ood-expert-v1/{task}/{row['episode']}/attempt/{row['seed_attempt']}"
                if row.get("seed_collision", 0):
                    key += f"/collision/{row['seed_collision']}"
                assert row["expert_success"] and row["ee_replay_success"]
                assert len(row["expert_h5_sha256"]) == 64
            else:
                key = f"robomme-ood-v1/{task}/{row['episode']}"
                if row.get("seed_attempt", 0):
                    key += f"/attempt/{row['seed_attempt']}"
            expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % 2**31
            assert row["seed"] == expected
            assert builder.resolve_episode(row["episode"]) == (expected, "ood")
            ood.append(expected)
        assert all(row["difficulty"] != "ood" for row in json.loads((SOURCE / f"env_metadata/train/record_dataset_{task}_metadata.json").read_text())["records"])
    assert len(ood) == len(set(ood)) == 320
    assert set(ood).isdisjoint(existing)
    assert not (SOURCE / "env_metadata/ood_train").exists()


def test_ood_never_falls_back_to_a_regular_episode(resolver, tmp_path):
    builder = resolver.BenchmarkEnvBuilder("BinFill", dataset="ood_test")
    for episode in (-1, 20, 500):
        with pytest.raises(ValueError, match="Missing or invalid ood_test"):
            builder.resolve_episode(episode)
    with pytest.raises(ValueError, match="No ood_test episodes"):
        resolver.BenchmarkEnvBuilder("BinFill", dataset="ood_test", override_metadata_path=tmp_path)
    path = tmp_path / "record_dataset_BinFill_metadata.json"
    for row in ({"seed": 123, "difficulty": "hard"}, {"difficulty": "ood"}):
        path.write_text(json.dumps({"env_id": "BinFill", "records": [{"episode": 0, **row}]}))
        invalid = resolver.BenchmarkEnvBuilder("BinFill", dataset="ood_test", override_metadata_path=tmp_path)
        with pytest.raises(ValueError, match="Missing or invalid ood_test"):
            invalid.resolve_episode(0)


def test_regular_split_resolution_is_unchanged(resolver):
    for task in resolver.BenchmarkEnvBuilder.get_task_list():
        for split in ("train", "test", "val"):
            builder = resolver.BenchmarkEnvBuilder(task, dataset=split)
            assert builder.max_steps_without_demonstration == 10002
            rows = json.loads((SOURCE / f"env_metadata/{split}/record_dataset_{task}_metadata.json").read_text())["records"]
            assert builder.get_episode_num() == len(rows)
            for row in rows:
                assert builder.resolve_episode(row["episode"]) == (row["seed"], row["difficulty"])


def test_ood_default_step_budget_and_explicit_override(resolver):
    assert resolver.BenchmarkEnvBuilder('BinFill', dataset='ood_test').max_steps_without_demonstration == 1502
    assert resolver.BenchmarkEnvBuilder('BinFill', dataset='ood_test', max_steps=1300).max_steps_without_demonstration == 1302


@pytest.mark.parametrize("task", ["VideoUnmaskSwap", "ButtonUnmaskSwap"])
def test_swap_schedule_extends_past_three_without_changing_old_timing(task):
    path = SOURCE / f"robomme_env/{task}.py"
    if not path.exists():
        pytest.skip("RoboMME source is not installed")
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_refresh_swap_schedule")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    for count in range(1, 9):
        env = SimpleNamespace(swap_times=count)
        pairs = [(object(), object()) for _ in range(count)]
        for i, (left, right) in enumerate(pairs, 1):
            setattr(env, f"swap_pair{i}_idx1", left)
            setattr(env, f"swap_pair{i}_idx2", right)
        scope["_refresh_swap_schedule"](env)
        assert env.swap_schedule == [
            (*pair, 64 + 50 * i, 114 + 50 * i) for i, pair in enumerate(pairs)
        ]


def test_example_accepts_exactly_twenty_ood_episodes():
    path = ROOT / "scripts/run_example.py"
    if not path.exists():
        pytest.skip("RoboMME source is not installed")
    names = {"DatasetType", "EPISODE_LIMITS"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name == "_validate_episode_index")]
    scope = {"Literal": Literal}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    assert "ood_test" in scope["DatasetType"].__args__
    for episode in (-1, 0, 19):
        scope["_validate_episode_index"](episode, "ood_test")
    for episode in (-2, 20):
        with pytest.raises(ValueError, match="Invalid episode_idx"):
            scope["_validate_episode_index"](episode, "ood_test")


def test_placement_swaps_preserve_regular_call_and_extend_ood_windows():
    path = SOURCE / "robomme_env/utils/statechange.py"
    if not path.exists():
        pytest.skip("RoboMME source is not installed")
    method = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "swap_target_sequence")
    calls = []
    scope = {"swap_flat_two_lane": lambda env, **kwargs: calls.append(kwargs)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    targets = [object() for _ in range(4)]
    env = SimpleNamespace(targets=targets)
    original = dict(cube_a=targets[0], cube_b=targets[1], start_step=100,
                    end_step=150, cur_step=130, other_cube=targets[2:], lane_offset=0.1)
    scope["swap_target_sequence"](env, **original)
    assert calls == [original]
    for count in (2, 3, 4):
        calls.clear()
        env.target_swap_pairs = [(targets[i % 4], targets[(i + 1) % 4]) for i in range(count)]
        scope["swap_target_sequence"](env, **original)
        assert len(calls) == count
        for i, call in enumerate(calls):
            assert (call["start_step"], call["end_step"]) == (100 + 50 * i, 150 + 50 * i)
            assert (call["cube_a"], call["cube_b"]) == env.target_swap_pairs[i]
            assert set(call["other_cube"]) == set(targets) - set(env.target_swap_pairs[i])


@pytest.mark.parametrize("task", ["ButtonUnmask", "VideoUnmask"])
def test_unmask_ood_preserves_two_pickups_and_language(task):
    path = SOURCE / f"robomme_env/{task}.py"
    if not path.exists():
        pytest.skip("RoboMME source is not installed")
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    names = {"config_easy", "config_medium", "config_hard", "config_ood", "configs"}
    assignments = [n for n in cls.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    scope = {}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), scope)
    assert scope["config_ood"]["pick"] == scope["config_hard"]["pick"] == 2
    assert scope["config_ood"]["region_half_size"] > 0.2
    goals = runpy.run_path(str(SOURCE / "robomme_env/utils/task_goal.py"))["get_language_goal"]
    native = SimpleNamespace(configs=scope["configs"], color_names=["red", "green", "blue"])
    wrapper = SimpleNamespace(env=SimpleNamespace(unwrapped=native), difficulty="hard")
    before = goals(wrapper, task)
    wrapper.difficulty = "ood"
    assert goals(wrapper, task) == before

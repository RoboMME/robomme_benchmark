from __future__ import annotations

import json


def _write_metadata(root, env_id: str, episodes: list[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_id": env_id,
        "records": [
            {"task": env_id, "episode": ep, "seed": 1000 + ep, "difficulty": "easy"}
            for ep in episodes
        ],
    }
    (root / f"record_dataset_{env_id}_metadata.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_fixed_users_login_and_random_task_pool(monkeypatch, reload_module, tmp_path):
    metadata_root = tmp_path / "metadata"
    _write_metadata(metadata_root, "EnvA", [0, 1, 2])
    _write_metadata(metadata_root, "EnvB", [10, 11])
    monkeypatch.setenv("ROBOMME_METADATA_ROOT", str(metadata_root))

    user_manager_mod = reload_module("user_manager")
    monkeypatch.setattr(user_manager_mod.random, "choice", lambda seq: seq[0])
    manager = user_manager_mod.UserManager()

    success, _msg, status = manager.init_session("uid1")
    assert success
    assert status["current_task"]["env_id"] in {"EnvA", "EnvB"}
    assert status["current_task"]["episode_idx"] in {0, 1, 2, 10, 11}
    assert status["is_done_all"] is False


def test_switch_env_and_next_episode_stays_in_same_env(monkeypatch, reload_module, tmp_path):
    metadata_root = tmp_path / "metadata"
    _write_metadata(metadata_root, "EnvA", [0, 1, 2])
    _write_metadata(metadata_root, "EnvB", [10, 11])
    monkeypatch.setenv("ROBOMME_METADATA_ROOT", str(metadata_root))

    user_manager_mod = reload_module("user_manager")
    monkeypatch.setattr(user_manager_mod.random, "choice", lambda seq: seq[-1])
    manager = user_manager_mod.UserManager()

    success, _msg, _status = manager.init_session("uid2")
    assert success

    switched = manager.switch_env_and_random_episode("uid2", "EnvA")
    assert switched is not None
    assert switched["current_task"]["env_id"] == "EnvA"
    assert switched["current_task"]["episode_idx"] in {0, 1, 2}

    nxt = manager.next_episode_same_env("uid2")
    assert nxt is not None
    assert nxt["current_task"]["env_id"] == "EnvA"
    assert nxt["current_task"]["episode_idx"] in {0, 1, 2}


def test_complete_current_task_increments_completed_count(monkeypatch, reload_module, tmp_path):
    metadata_root = tmp_path / "metadata"
    _write_metadata(metadata_root, "EnvA", [0, 1])
    monkeypatch.setenv("ROBOMME_METADATA_ROOT", str(metadata_root))

    user_manager_mod = reload_module("user_manager")
    monkeypatch.setattr(user_manager_mod.random, "choice", lambda seq: seq[0])
    manager = user_manager_mod.UserManager()

    success, _msg, status = manager.init_session("uid3")
    assert success
    assert status["completed_count"] == 0

    updated = manager.complete_current_task(
        "uid3",
        env_id=status["current_task"]["env_id"],
        episode_idx=status["current_task"]["episode_idx"],
        status="success",
    )
    assert updated is not None
    assert updated["completed_count"] == 1
    assert updated["is_done_all"] is False


def test_init_session_fails_when_metadata_root_missing(monkeypatch, reload_module, tmp_path):
    missing_root = tmp_path / "missing-metadata-root"
    monkeypatch.setenv("ROBOMME_METADATA_ROOT", str(missing_root))

    user_manager_mod = reload_module("user_manager")
    manager = user_manager_mod.UserManager()

    success, msg, status = manager.init_session("uid-missing")

    assert success is False
    assert "No available environments" in msg
    assert status is None


def test_env_choices_follow_task_name_list_order(monkeypatch, reload_module, tmp_path):
    metadata_root = tmp_path / "metadata"
    _write_metadata(metadata_root, "VideoPlaceButton", [0])
    _write_metadata(metadata_root, "BinFill", [0])
    _write_metadata(metadata_root, "PatternLock", [0])
    _write_metadata(metadata_root, "StopCube", [0])
    monkeypatch.setenv("ROBOMME_METADATA_ROOT", str(metadata_root))

    user_manager_mod = reload_module("user_manager")
    manager = user_manager_mod.UserManager()

    assert manager.env_choices == [
        "BinFill",
        "StopCube",
        "VideoPlaceButton",
        "PatternLock",
    ]

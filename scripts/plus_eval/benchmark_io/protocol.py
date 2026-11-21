"""Durable benchmark attempt accounting, independent of either learning method.

This module belongs to the trusted operator, never the coding-agent workspace.
Reservations commit before simulator construction, including failed constructions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

SETTINGS = ("zero-shot", "demo-only", "interaction-only", "demo&interaction")


def setting_key(setting):
    if setting not in SETTINGS:
        raise ValueError("Unknown adaptation setting")
    return setting


class BudgetExhausted(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Study:
    def __init__(self, root, *, method, setting, budget, records, accounting_version=None):
        if not isinstance(method, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", method):
            raise ValueError("Method must be a nonempty identifier")
        setting = setting_key(setting)
        if type(budget) is not int or not 0 <= budget <= 10:
            raise ValueError("The per-task budget must be an integer in 0..10")
        if (setting in ("zero-shot", "demo-only")) != (budget == 0):
            raise ValueError("zero-shot/demo-only require R0; interaction settings require R1..R10")
        if not records or any(len(rows) != (5 if task == "MoveCube" else 3)
                              for task, rows in records.items()):
            raise ValueError("Use the released training episode counts")
        for task, rows in records.items():
            if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", task):
                raise ValueError("Invalid task identifier")
            if any(type(row.get("seed")) is not int for row in rows) or len({row["seed"] for row in rows}) != len(rows):
                raise ValueError("Released seeds must be distinct integers within each task")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = dict(method=method, setting=setting, budget=budget, records=records)
        self.path = self.root / "attempts.sqlite3"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL)")
            saved = db.execute("SELECT json FROM config WHERE id=1").fetchone()
            if accounting_version is None:
                accounting_version = json.loads(saved[0]).get("accounting_version", 1) if saved else 2
            if type(accounting_version) is not int or accounting_version not in (1, 2):
                raise ValueError("Unsupported accounting version")
            if accounting_version == 2:
                self.config["accounting_version"] = 2
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, task TEXT NOT NULL, "
                       "episode INTEGER NOT NULL, seed INTEGER NOT NULL, purpose TEXT NOT NULL, "
                       "started REAL NOT NULL, ended REAL, outcome TEXT)")
            db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (canonical(self.config),))
            if db.execute("SELECT json FROM config WHERE id=1").fetchone()[0] != canonical(self.config):
                raise ValueError("Study configuration or training metadata changed; refusing to reset its budget")
            if accounting_version == 2:
                db.execute("CREATE TABLE IF NOT EXISTS auto_cursor (task TEXT PRIMARY KEY, next_episode INTEGER NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS attempt_provenance (attempt_id INTEGER PRIMARY KEY, "
                           "selection TEXT NOT NULL, policy_revision TEXT)")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.execute("PRAGMA synchronous=FULL")
        return db

    @classmethod
    def open(cls, root):
        path = Path(root).resolve() / "attempts.sqlite3"
        if not path.is_file():
            raise FileNotFoundError("Prepare an operator-owned study before starting a simulator")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            config = json.loads(db.execute("SELECT json FROM config WHERE id=1").fetchone()[0])
        return cls(root, **config)

    def select_episode(self, task, episode=None, *, seed=None):
        """Validate an explicit released index/seed; None requests automatic selection."""
        if task not in self.config["records"]:
            raise ValueError("Task is not in this study")
        rows = self.config["records"][task]
        if seed is not None:
            if episode is not None:
                raise ValueError("Choose a seed or an episode index, not both")
            if type(seed) is not int or seed not in [r["seed"] for r in rows]:
                raise ValueError("Only released training seeds for this task are allowed")
            episode = next(i for i, row in enumerate(rows) if row["seed"] == seed)
        if episode is not None and (type(episode) is not int or not 0 <= episode < len(rows)):
            raise ValueError("Only released training episode indices are allowed")
        return episode

    def reserve(self, task, episode=None, *, purpose, seed=None, policy_revision=None):
        selection = "seed" if seed is not None else "episode" if episode is not None else "automatic"
        episode = self.select_episode(task, episode, seed=seed)
        rows = self.config["records"][task]
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("Attempt purpose is required")
        if policy_revision is not None and (not isinstance(policy_revision, str) or not policy_revision.strip()):
            raise ValueError("Policy revision must identify saved code")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            from .development_deadline import TaskClock
            TaskClock(self.root, task).require_open()
            counts = dict(db.execute("SELECT episode, count(*) FROM attempts WHERE task=? GROUP BY episode", (task,)))
            count = sum(counts.values())
            if count >= self.config["budget"]:
                raise BudgetExhausted(f"{task}: {count}/{self.config['budget']} attempts already reserved")
            if episode is None:
                if self.config.get("accounting_version", 1) == 1:
                    # Preserve the behavior pinned by historical campaigns.
                    episode = min(range(len(rows)), key=lambda i: (counts.get(i, 0), i))
                else:
                    cursor = db.execute("SELECT next_episode FROM auto_cursor WHERE task=?", (task,)).fetchone()
                    episode = cursor[0] if cursor else 0
                    db.execute("INSERT OR REPLACE INTO auto_cursor VALUES (?, ?)",
                               (task, (episode + 1) % len(rows)))
            row = rows[episode]
            cursor = db.execute("INSERT INTO attempts(task,episode,seed,purpose,started) VALUES(?,?,?,?,?)",
                                (task, episode, row["seed"], purpose, time.time()))
            attempt = dict(id=cursor.lastrowid, task=task, episode=episode, seed=row["seed"],
                           ordinal=count + 1, purpose=purpose)
            if self.config.get("accounting_version", 1) == 2:
                db.execute("INSERT INTO attempt_provenance VALUES (?, ?, ?)",
                           (attempt["id"], selection, policy_revision))
                attempt.update(selection=selection, policy_revision=policy_revision,
                               remaining=self.config["budget"] - count - 1)
        return attempt

    def finish(self, ident, outcome):
        with self._connect() as db:
            changed = db.execute("UPDATE attempts SET ended=?, outcome=? WHERE id=? AND ended IS NULL",
                                 (time.time(), canonical(outcome), ident)).rowcount
            if changed != 1:
                raise ValueError("Unknown or already finalized attempt")

    def attempts(self):
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY id")]

    def require_demos(self):
        if self.config["setting"] not in ("demo-only", "demo&interaction"):
            raise PermissionError("This setting cannot access released demonstrations")


def source_fingerprint(root):
    root = Path(root)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.suffix in (".py", ".json") and p.is_file()}
    return dict(sha256=hashlib.sha256(canonical(files).encode()).hexdigest(), files=files)

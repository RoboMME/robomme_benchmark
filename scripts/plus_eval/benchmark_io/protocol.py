"""Durable benchmark attempt accounting, independent of either learning method.

This module belongs to the trusted operator, never the coding-agent workspace.
Reservations commit before simulator construction, including failed constructions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time

SETTINGS = ("AfD", "AfI", "AfD+I")
METHODS = ("rax", "aspire")


class BudgetExhausted(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Study:
    def __init__(self, root, *, method, setting, budget, records):
        if method not in METHODS or setting not in SETTINGS:
            raise ValueError("Unknown method or adaptation setting")
        if type(budget) is not int or not 0 <= budget <= 10:
            raise ValueError("The per-task budget must be an integer in 0..10")
        if (setting == "AfD") != (budget == 0):
            raise ValueError("AfD requires R0; interaction settings require R1..R10")
        if not records or any(len(rows) != (5 if task == "MoveCube" else 3)
                              for task, rows in records.items()):
            raise ValueError("Use the released training episode counts")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = dict(method=method, setting=setting, budget=budget, records=records)
        self.path = self.root / "attempts.sqlite3"
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, task TEXT NOT NULL, "
                       "episode INTEGER NOT NULL, seed INTEGER NOT NULL, purpose TEXT NOT NULL, "
                       "started REAL NOT NULL, ended REAL, outcome TEXT)")
            db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (canonical(self.config),))
            if db.execute("SELECT json FROM config WHERE id=1").fetchone()[0] != canonical(self.config):
                raise ValueError("Study configuration or training metadata changed; refusing to reset its budget")

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

    def reserve(self, task, episode=None, *, purpose):
        if task not in self.config["records"]:
            raise ValueError("Task is not in this study")
        rows = self.config["records"][task]
        if episode is not None and (type(episode) is not int or not 0 <= episode < len(rows)):
            raise ValueError("Only released training episode indices are allowed")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("Attempt purpose is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            from .development_deadline import TaskClock
            TaskClock(self.root, task).require_open()
            counts = dict(db.execute("SELECT episode, count(*) FROM attempts WHERE task=? GROUP BY episode", (task,)))
            count = sum(counts.values())
            if count >= self.config["budget"]:
                raise BudgetExhausted(f"{task}: {count}/{self.config['budget']} attempts already reserved")
            if episode is None:
                episode = min(range(len(rows)), key=lambda i: (counts.get(i, 0), i))
            row = rows[episode]
            cursor = db.execute("INSERT INTO attempts(task,episode,seed,purpose,started) VALUES(?,?,?,?,?)",
                                (task, episode, row["seed"], purpose, time.time()))
            attempt = dict(id=cursor.lastrowid, task=task, episode=episode, seed=row["seed"],
                           ordinal=count + 1, purpose=purpose)
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
        if self.config["setting"] == "AfI":
            raise PermissionError("AfI cannot access released demonstrations")


def source_fingerprint(root):
    root = Path(root)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.suffix in (".py", ".json") and p.is_file()}
    return dict(sha256=hashlib.sha256(canonical(files).encode()).hexdigest(), files=files)

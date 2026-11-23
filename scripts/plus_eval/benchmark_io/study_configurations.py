"""Current study budgets, preserving the assignments of prepared experiments."""

import json
from pathlib import Path
import sqlite3


def configurations(root=None):
    """Use a saved plan/queue when resuming; new studies use 0 and 10."""
    rows = None
    if root is not None:
        root = Path(root)
        plan = root / 'study_plan.json'
        if plan.is_file():
            rows = json.loads(plan.read_text())['configurations']
        elif (root / 'queue.sqlite3').is_file():
            with sqlite3.connect((root / 'queue.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
                db.row_factory = sqlite3.Row
                rows = [dict(row) for row in db.execute(
                    'SELECT key,method,setting,budget FROM cases ORDER BY ordinal')]
            if not rows:
                rows = None  # A new queue, before its first assignment.
    if rows is None:
        rows = [dict(key=f'{method}/{setting}/R{budget}', method=method, setting=setting, budget=budget)
                for setting, budget in (('demo-only', 0), ('interaction-only', 10), ('demo&interaction', 10))
                for method in ('aspire', 'rax')]
    if not isinstance(rows, list) or not rows:
        raise ValueError('A study must have explicit configuration assignments')
    result, seen = [], set()
    for row in rows:
        method, setting, budget = row['method'], row['setting'], row['budget']
        key = row['key']
        if (method not in ('aspire', 'rax') or setting not in ('demo-only', 'interaction-only', 'demo&interaction')
                or type(budget) is not int or not 0 <= budget <= 10
                or (setting == 'demo-only') != (budget == 0)
                or key != f'{method}/{setting}/R{budget}' or key in seen):
            raise ValueError('Invalid or duplicate study configuration')
        seen.add(key)
        result.append((key, method, setting, budget))
    yield from result

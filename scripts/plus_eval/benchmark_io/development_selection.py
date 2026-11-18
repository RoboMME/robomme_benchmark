"""Locate the frozen native submission and count its original training ledger."""
from pathlib import Path
import sqlite3


def rax_source(case, task):
    case = Path(case).resolve()
    return case/'development'/task, case/'study'


def training_accounting(case):
    ledger = Path(case)/'study/attempts.sqlite3'
    if not ledger.is_file():
        return dict(training_starts=0)
    with sqlite3.connect(f'file:{ledger}?mode=ro', uri=True) as db:
        count = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    return dict(training_starts=count)

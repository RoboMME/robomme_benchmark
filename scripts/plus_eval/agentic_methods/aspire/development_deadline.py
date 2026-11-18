"""Track native ASPIRE worker assignments without imposing a new curriculum."""
from datetime import datetime
import json
from pathlib import Path
import re

from benchmark_io.development_deadline import TaskClock


class CampaignClocks:
    def __init__(self, study, home, tasks):
        self.study, self.home, self.tasks = Path(study), Path(home), list(tasks)
        self.offsets, self.calls, self.seen, self.completed = {}, {}, set(), set()

    def task_index(self, text):
        match = re.search(r'(?:worker_robomme_plus_|robomme_plus/task_)(\d+)\b', text)
        if match and int(match[1]) < len(self.tasks):
            return int(match[1])
        return None

    def consume(self, row):
        if row.get('type') not in ('assistant', 'user'):
            return
        stamp = row.get('timestamp')
        if not stamp:
            return
        when = datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
        content = (row.get('message') or {}).get('content', [])
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use' and block.get('name') in ('Agent', 'Task'):
                ident = block.get('id')
                index = self.task_index(json.dumps(block.get('input', {})))
                if index is None or ident in self.seen:
                    continue
                self.seen.add(ident)
                task = self.tasks[index]
                self.calls[ident] = task
                self.completed.discard(task)
                clock = TaskClock(self.study, task)
                clock.start(when)
                clock.update(last_assignment=when, completed_at=None)
            elif block.get('type') == 'text':
                text = block.get('text', '')
                match = re.search(r'<tool-use-id>([^<]+)</tool-use-id>', text)
                if match and '<status>completed</status>' in text and match[1] in self.calls:
                    task = self.calls[match[1]]
                    self.completed.add(task)
                    TaskClock(self.study, task).update(completed_at=when)

    def refresh(self):
        # Only coordinator transcripts: nested verifier tasks must not reset a
        # task's first worker assignment or close another worker's clock.
        for path in sorted((self.home/'projects/-workspace').glob('*.jsonl')):
            with path.open() as stream:
                stream.seek(self.offsets.get(path, 0))
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith('\n'):
                        stream.seek(position)
                        break
                    try:
                        self.consume(json.loads(line))
                    except (ValueError, TypeError):
                        continue
                self.offsets[path] = stream.tell()

    def overdue(self):
        self.refresh()
        rows = []
        for task in self.tasks:
            clock = TaskClock(self.study, task)
            row = clock.read()
            if row and not row.get('completed_at') and not row.get('expired_at') and clock.remaining() <= 0:
                rows.append(task)
        return rows

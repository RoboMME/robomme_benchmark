#!/usr/local/bin/python
"""Benchmark command adapter; original ASPIRE programs and artifacts stay unchanged."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error


def request(path, value=None):
    req = urllib.request.Request('http://127.0.0.1:17777' + path,
        data=None if value is None else json.dumps(value).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=3700) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'error': exc.read().decode()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['status', 'run', 'run-batch'])
    p.add_argument('--program', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds')
    p.add_argument('--purpose', default='development')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--benchmark', default='robomme_plus')
    p.add_argument('--task')
    p.add_argument('--task-id')
    p.add_argument('--suite')
    p.add_argument('--init-mode', default='released')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--log-dir')
    p.add_argument('--flat-run-dir', action='store_true')
    p.add_argument('--no-videos', action='store_true')
    args = p.parse_args()
    if args.command == 'status':
        print(json.dumps(request('/api/status'), indent=2))
        return
    if args.program is None:
        p.error('--program is required')
    if args.benchmark != 'robomme_plus':
        p.error('This study contains only RoboMME-Plus')
    seeds = [args.seed]
    if args.command == 'run-batch':
        if not args.seeds:
            p.error('--seeds is required')
        seeds = []
        for part in args.seeds.split(','):
            if '-' in part:
                a, b = map(int, part.split('-'))
                seeds.extend(range(a, b + 1))
            else:
                seeds.append(int(part))
        if len(seeds) > 10:
            p.error('At most ten released attempts')
    code = args.program.read_text()
    rows = []
    for seed in seeds:
        result = request('/api/run', {'code': code, 'episode': seed,
            'source_name': args.program.name, 'purpose': args.purpose, 'task': args.task, 'task_id': args.task_id})
        rows.append(result)
        print(json.dumps(result, indent=2), flush=True)
        if args.output_root:
            args.output_root.mkdir(parents=True, exist_ok=True)
            (args.output_root / f'run-{len(rows):02d}-reply.json').write_text(json.dumps(result, indent=2))
        if result.get('rejected'):
            break


if __name__ == '__main__':
    main()

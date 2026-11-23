"""Hash test-time dependencies that live outside the frozen policy folder.

Before testing (no --case): write a pre-test dependency manifest, part of the freeze.
After testing (with --case): write a post-test supplement for a dependency the freeze missed.

Read-only on study evidence. For each evaluated case it re-verifies the frozen
policy copy against the original pre-test manifest and checks that no development
rollout started after the freeze. It then hashes each named dependency (files or
directories) and reports any file whose content or metadata changed at or after
the earliest test start. The output is labelled post-test: it supplements the
original freeze manifest and never replaces it.

Example:
  python -m benchmark_io.freeze_supplement --output supplement.json \
    --case <STUDY>/aspire/interaction-only/R10 --case '<STUDY>/aspire/demo&interaction/R10' \
    --dependency sam3_weights=<PATH>/sam3.pt --dependency sam3_env=<PATH>/.venv-sam3
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_only(path):
    return sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)


def check_manifest(manifest_path, study_db):
    """Check one frozen evaluation: its policy copy, test starts and later development."""
    manifest = json.loads(manifest_path.read_text())
    frozen = manifest_path.parent
    policy = frozen / 'policy'
    current = {str(p.relative_to(policy)): sha256(p) for p in sorted(policy.rglob('*')) if p.is_file()}
    with read_only(frozen / 'evaluation_attempts.sqlite3') as db:
        first_test, tests = db.execute('SELECT min(started), count(*) FROM attempts').fetchone()
    tasks = sorted(manifest.get('episodes') or {})
    with read_only(study_db) as db:
        query = 'SELECT count(*) FROM attempts WHERE started >= ?'
        args = [manifest['created']]
        if tasks:  # a per-task freeze only constrains development of its own tasks
            query += ' AND task IN (' + ','.join('?' * len(tasks)) + ')'
            args += tasks
        after_freeze = db.execute(query, args).fetchone()[0]
    return dict(manifest=str(manifest_path), sha256=sha256(manifest_path), created=manifest['created'], tasks=tasks,
                first_test_start=first_test, test_starts=tests, policy_files=len(current),
                policy_matches=current == manifest['policy_sha256'], development_starts_after_freeze=after_freeze)


def check_case(case):
    """Verify every frozen evaluation in a case (one manifest, or one per task)."""
    case = Path(case).resolve()
    paths = sorted(case.glob('evaluation/frozen_evaluation.json')) + sorted(case.glob('evaluation/*/frozen_evaluation.json'))
    if not paths:
        raise FileNotFoundError(f'No frozen_evaluation.json under {case}/evaluation')
    rows = [check_manifest(p, case / 'study' / 'attempts.sqlite3') for p in paths]
    return dict(case=str(case), manifests=rows,
                first_test_start=min(r['first_test_start'] for r in rows), test_starts=sum(r['test_starts'] for r in rows),
                policy_files=sum(r['policy_files'] for r in rows),
                policy_matches_pre_test_manifest=all(r['policy_matches'] for r in rows),
                development_starts_after_freeze=sum(r['development_starts_after_freeze'] for r in rows))


def inventory(path):
    """Hash every regular file under path; record symlinks and hash their file targets."""
    path = Path(path)
    roots = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() or p.is_symlink())
    rows = []
    for item in roots:
        stat = item.lstat()
        row = dict(path=str(item), size=stat.st_size, mtime=stat.st_mtime, ctime=stat.st_ctime)
        if item.is_symlink():
            row['symlink_target'] = os.readlink(item)
            target = item.resolve()
            if target.is_file():
                tstat = target.stat()
                row.update(target=str(target), target_sha256=sha256(target),
                           target_mtime=tstat.st_mtime, target_ctime=tstat.st_ctime)
        else:
            row['sha256'] = sha256(item)
        rows.append(row)
    return rows


def changed_since(rows, reference):
    return [r['path'] for r in rows
            if max(r['mtime'], r['ctime'], r.get('target_mtime', 0), r.get('target_ctime', 0)) >= reference]


def build(cases, dependencies, evidence=()):
    """With cases: a post-test supplement. Without cases: a pre-test dependency manifest."""
    checked = [check_case(case) for case in cases]
    reference = min((c['first_test_start'] for c in checked), default=None)
    if reference is None:
        reference = float('inf')  # before testing nothing can have changed "after the first test"
    deps = {}
    for name, path in dependencies.items():
        rows = inventory(path)
        tree = hashlib.sha256(json.dumps([(r['path'], r.get('sha256') or r.get('target_sha256'))
                                          for r in rows]).encode()).hexdigest()
        deps[name] = dict(path=str(Path(path).resolve()), files=len(rows), bytes=sum(r['size'] for r in rows),
                          tree_sha256=tree, changed_at_or_after_first_test=changed_since(rows, reference),
                          newest_change=max((max(r['mtime'], r['ctime']) for r in rows), default=None),
                          inventory=rows)
    ok = (all(c['policy_matches_pre_test_manifest'] and c['development_starts_after_freeze'] == 0 for c in checked)
          and all(not d['changed_at_or_after_first_test'] for d in deps.values()))
    if not checked:
        return dict(kind='pre_test_dependency_manifest', created=time.time(),
                    note='Written before testing: content hashes of test-time dependencies outside the policy folder.',
                    cases=[], supporting_evidence={str(p): sha256(p) for p in evidence},
                    dependencies=deps, all_checks_pass=True)
    return dict(kind='post_test_freeze_supplement', created=time.time(),
                note=('Written after testing. Supplements, and never replaces, the pre-test freeze manifests '
                      'listed below. File change times are compared with the earliest test start.'),
                reference_first_test_start=reference, cases=checked,
                supporting_evidence={str(p): sha256(p) for p in evidence},
                dependencies=deps, all_checks_pass=ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', action='append', default=[],
                        help='Evaluated case directory (repeatable). Omit before testing to write a pre-test '
                             'dependency manifest instead of a post-test supplement.')
    parser.add_argument('--dependency', action='append', required=True, metavar='NAME=PATH',
                        help='Test-time dependency file or directory (repeatable)')
    parser.add_argument('--evidence', action='append', default=[], help='Supporting record to hash (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing supplement')
    dependencies = {}
    for item in args.dependency:
        name, sep, path = item.partition('=')
        if not sep or not name or not Path(path).exists() or name in dependencies:
            parser.error(f'Invalid or duplicate dependency: {item}')
        dependencies[name] = path
    result = build(args.case, dependencies, args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=1))
    temporary.replace(args.output)
    summary = dict(all_checks_pass=result['all_checks_pass'],
                   cases=[{k: c[k] for k in ('case', 'policy_matches_pre_test_manifest',
                                             'development_starts_after_freeze', 'test_starts')} for c in result['cases']],
                   dependencies={n: dict(files=d['files'], changed=len(d['changed_at_or_after_first_test']))
                                 for n, d in result['dependencies'].items()})
    print(json.dumps(summary, indent=1))
    raise SystemExit(0 if result['all_checks_pass'] else 1)


if __name__ == '__main__':
    main()

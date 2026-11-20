"""Prepare a fresh pinned study without API calls or simulator starts."""
import argparse
from dataclasses import asdict
import hashlib
import json
import re
import secrets
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .full_study import REPO, IMAGE, Pipeline, write
from .protocol import source_fingerprint
from .runtime_policy import RuntimePolicy
from .runtime_source import assemble
from .prompt_bundle import DEFAULT_BUNDLE, bundle_version
from .unified_study import Queue, configurations, environment


def pin_runtime(root, *, source=None):
    root = Path(root).resolve()
    if (root/'scheduler_process.json').exists():
        raise RuntimeError('Do not replace runtime code after model development starts')
    manifest = assemble(root/'runtime_source', source=source or REPO/'scripts/plus_eval')
    write(root/'runtime_source_manifest.json', manifest)


def pin_demonstrations(source, target, metadata):
    """Validate seed/variation provenance before copying any released recording."""
    import h5py
    source, target = Path(source), Path(target)
    rows = []
    for path in sorted(Path(metadata).glob('record_dataset_*_metadata.json')):
        task = path.name.removeprefix('record_dataset_').removesuffix('_metadata.json')
        records = json.loads(path.read_text())
        # Metadata is a task header with an ordered episode array.
        records = records['records'] if isinstance(records, dict) and 'records' in records else records
        if isinstance(records, dict):
            records = list(records.values())
        recording = source/f'record_dataset_{task}.h5'
        with h5py.File(recording, 'r') as h5:
            for i, row in enumerate(records):
                saved = json.loads(h5[f'episode_{i}/setup/episode_metadata'][()].decode())
                for key in ('seed', 'difficulty', 'variant', 'perceptual_variation', 'mechanism_variation'):
                    if saved.get(key) != row.get(key):
                        raise ValueError(f'{task}/{i}: stale demonstration {key}')
        with recording.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        rows.append(dict(task=task, episodes=len(records), sha256=digest, file=recording.name))
    if len(rows) != 16 or sum(row['episodes'] for row in rows) != 50:
        raise ValueError('Expected all 50 released training episodes across 16 tasks')
    target.mkdir(parents=True, exist_ok=False)
    for row in rows:
        # Reflinks retain independent immutable snapshots without duplicating blocks.
        subprocess.run(['cp', '--reflink=auto', '--', str(source/row['file']), str(target/row['file'])], check=True)
    return dict(files=rows, episodes=50)


def prepare(root, demos, *, protocol=None):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh study directory')
    image = IMAGE
    if not re.fullmatch(r'(?:[A-Za-z0-9_./:-]+@)?sha256:[a-f0-9]{64}', image):
        image = subprocess.check_output(['docker','image','inspect','--format={{.Id}}',image], text=True).strip()
    root.mkdir(parents=True)
    token = root/'runtime/private/relay-token'
    token.parent.mkdir(parents=True, mode=0o700)
    with token.open('x') as stream:
        token.chmod(0o600)
        stream.write(secrets.token_hex(32)+'\n')
    write(root/'preparation.json', dict(phase='preparing', started=time.time()))
    benchmark = root/'benchmark'
    for package in ('robomme', 'robomme_plus'):
        shutil.copytree(REPO/'src'/package, benchmark/'src'/package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    demonstrations = pin_demonstrations(demos, benchmark/'h5', benchmark/'src/robomme_plus/env_metadata/train')
    write(benchmark/'demonstrations_manifest.json', demonstrations)
    write(benchmark/'source_manifest.json', source_fingerprint(benchmark/'src/robomme_plus'))
    protocol = Path(protocol or DEFAULT_BUNDLE).resolve()
    version = bundle_version(protocol)
    shutil.copytree(protocol, root/'protocol', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write(root/'runtime_policy.json', asdict(RuntimePolicy()))
    pin_runtime(root)
    selected = [dict(key=k,method=m,setting=s,budget=b) for k,m,s,b in configurations()]
    write(root/'study_plan.json', dict(configurations=selected, development_workers=6,development_per_gpu=3,
        evaluation_workers=16,evaluation_per_gpu=8,image=image,model='us.anthropic.claude-opus-5-5',
        reasoning_effort='high',optional_gemini=False,conditioning_timestamps=512,test_trials=800*len(selected),
        test_feedback_to_development=False,independent_budgets=True,full_artifact_budgets=[0,10],
        logs_only_budgets=[5],protocol_version=version))
    Queue(root).seed()
    subprocess.run([sys.executable,'-m','benchmark_io.prepare_unified','ledgers','--root',str(root)],
                   env=environment(root),check=True,cwd=REPO)
    write(root/'preparation.json',dict(phase='prepared',completed=time.time(),demonstrations_verified=50,
                                     simulator_starts=0,api_calls=0))


def ledgers(root):
    pipeline = Pipeline(root, ['0','1'])
    from robomme_plus import BenchmarkEnvBuilder
    for key, method, setting, budget in configurations(root):
        study,tasks = pipeline.prepare(Path(root)/key,method,setting,budget)
        if len(tasks)!=16 or sum(map(len,study.config['records'].values()))!=50 or study.attempts():
            raise ValueError('Expected a fresh complete study')
        for task in tasks:
            train = {r['seed'] for r in study.config['records'][task]}
            builder = BenchmarkEnvBuilder(task,dataset='test')
            if builder.get_episode_num()!=50 or train.intersection(
                    builder.get_episode_metadata(i)['seed'] for i in range(50)):
                raise ValueError('Training and testing seeds must be disjoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','ledgers','pin-runtime'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--demos',type=Path,help='Directory containing the 16 released training H5 files')
    parser.add_argument('--protocol',type=Path,help='Immutable public protocol bundle')
    args = parser.parse_args()
    if args.command=='prepare':
        if args.demos is None:parser.error('prepare requires --demos')
        prepare(args.root,args.demos,protocol=args.protocol)
    elif args.command=='pin-runtime':pin_runtime(args.root)
    else:ledgers(args.root)


if __name__=='__main__':
    main()

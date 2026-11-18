"""Assemble immutable baseline runtime copies from clean, pinned submodules.

Neither the submodule nor the adapter checkout is modified. Patches and added
backend modules are applied only inside a new operator-owned output directory.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


SOURCE = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.git')


def hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()
            and '__pycache__' not in p.parts}


def assemble_method(method, destination, *, source=SOURCE):
    outer = Path(source)/method
    manifest = json.loads((outer/'upstream.json').read_text())
    core = outer/manifest['core']
    if not (core/'.git').exists():
        raise FileNotFoundError(f'Initialize submodule scripts/plus_eval/{method}/{manifest["core"]}')
    revision = subprocess.check_output(['git', '-C', str(core), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != manifest['revision']:
        raise ValueError(f'{method} submodule does not match upstream.json')
    dirty = subprocess.check_output(['git', '-C', str(core), 'status', '--porcelain'], text=True)
    if dirty:
        raise ValueError(f'{method} upstream submodule must be clean; keep adaptations outside it')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    upstream = destination/'ASPIRE' if method == 'aspire' else destination
    upstream.mkdir(exist_ok=True)
    archive = subprocess.check_output(['git', '-C', str(core), 'archive', revision])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(upstream, filter='data')
    # A temporary Git directory makes patch paths independent of the parent repo.
    subprocess.run(['git', 'init', '-q', str(upstream)], check=True)
    try:
        for relative in manifest['patches']:
            patch = (outer/relative).resolve()
            if not patch.is_relative_to(outer.resolve()):
                raise ValueError('Patch must be inside the method directory')
            subprocess.run(['git', '-C', str(upstream), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(upstream), 'apply', str(patch)], check=True)
    finally:
        shutil.rmtree(upstream/'.git')
    shutil.copytree(outer/manifest['overlay'], upstream, dirs_exist_ok=True, ignore=IGNORE)
    for path in outer.glob('*.py'):
        shutil.copy2(path, destination/path.name)
    (destination/'runtime').mkdir(exist_ok=True)
    return dict(**manifest, patch_sha256={p: hashlib.sha256((outer/p).read_bytes()).hexdigest()
                                        for p in manifest['patches']})


def assemble(destination, *, source=SOURCE):
    destination = Path(destination).resolve()
    source = Path(source).resolve()
    if destination.exists():
        raise FileExistsError('Use a new runtime directory; existing snapshots are immutable')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.runtime-', dir=destination.parent))
    try:
        shutil.copytree(source/'benchmark_io', staging/'benchmark_io', ignore=IGNORE)
        methods = {m: assemble_method(m, staging/m, source=source) for m in ('aspire', 'rax')}
        manifest = dict(schema_version=1, methods=methods, sha256=hashes(staging))
        (staging/'runtime_source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = assemble(args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), files=len(manifest['sha256']),
                          upstream={k: v['revision'] for k, v in manifest['methods'].items()})))


if __name__ == '__main__':
    main()

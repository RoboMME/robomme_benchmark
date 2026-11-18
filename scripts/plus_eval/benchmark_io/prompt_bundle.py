"""Copy an immutable public protocol bundle into a fresh agent workspace."""
import hashlib
import json
import os
from pathlib import Path


DEFAULT_BUNDLE = Path(__file__).parent/'prompts/current'


def bundle_version(source):
    """Legacy bundles are v2; newer bundles declare their immutable version."""
    descriptor = Path(source)/'bundle.json'
    version = json.loads(descriptor.read_text())['version'] if descriptor.exists() else 2
    if type(version) is not int or version < 2:
        raise ValueError('Protocol version must be an integer >= 2')
    return version


def install(workspace, method):
    source = os.environ.get('ROBOMME_AGENTIC_PROMPTS')
    if not source:
        return ''
    source = Path(source)
    workspace = Path(workspace)
    version = bundle_version(source)
    relative = f'docs/robomme-protocol-v{version}'
    destination = workspace/relative
    for existing in (workspace/'docs').glob('robomme-protocol-v*/manifest.json'):
        if existing.parent != destination:
            raise ValueError('Do not replace an existing agent protocol version; use a fresh workspace')
    files = ['BENCHMARK.md', 'DEMOS.md', method.upper()+'.md', 'RUNTIME.md', 'PUBLIC_RULES.md']
    if (source/'bundle.json').exists():
        files.extend(['bundle.json', 'README.md'])
    payloads = {}
    for name in files:
        payloads[name] = (source/name).read_bytes()
    fixtures = source/'fixtures'
    if fixtures.is_dir():
        for p in sorted(fixtures.rglob('*')):
            if p.is_file() and (version == 2 or ('__pycache__' not in p.parts and p.suffix != '.pyc')):
                payloads['fixtures/'+str(p.relative_to(fixtures))] = p.read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    manifest = dict(version=version,method=method,sha256=hashes)
    old_manifest = destination/'manifest.json'
    if old_manifest.exists() and json.loads(old_manifest.read_text()) != manifest:
        raise ValueError('Do not change an existing agent protocol snapshot or fixture manifest')
    # Check every byte before writing any file, so rejected updates are atomic.
    for name, data in payloads.items():
        target = destination/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Do not change an existing agent protocol snapshot or fixture')
    for name, data in payloads.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    old_manifest.write_text(json.dumps(manifest,indent=2))
    return ('\n\nRead '+relative+'/BENCHMARK.md, DEMOS.md, '+method.upper()+'.md, '
            'PUBLIC_RULES.md and RUNTIME.md before development. These specify the public benchmark, '
            'data formats, timing rules and resource limits; this version takes precedence over older adapter descriptions. Synthetic offline fixtures are in '
            +relative+'/fixtures; they contain no training or test scene. '
            'Preserve your original method workflow within this protocol.\n')

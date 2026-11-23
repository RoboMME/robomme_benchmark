"""Restore the exact upstream RAX commit without fetching a moving branch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def restore(destination=None):
    root = Path(__file__).resolve().parent
    pin = json.loads((root/'upstream.json').read_text())
    destination = Path(destination or root/pin['core']).resolve()
    bundle = root/pin['archive']['file']
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != pin['archive']['sha256']:
        raise ValueError('RAX source archive checksum mismatch')
    if (destination/'.git').exists():
        revision = subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'], text=True).strip()
        if revision != pin['revision']:
            raise ValueError('Existing RAX checkout has a different revision; it was left unchanged')
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Refusing to replace a nonempty directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rax-restore-', dir=destination.parent) as temporary:
        checkout = Path(temporary)/'checkout'
        subprocess.run(['git','clone','--quiet','--no-checkout',str(bundle),str(checkout)], check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','--quiet','--detach',pin['revision']], check=True)
        subprocess.run(['git','-C',str(checkout),'remote','set-url','origin',pin['repository']], check=True)
        if destination.exists(): destination.rmdir()  # Empty directories only.
        checkout.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    print(restore(parser.parse_args().destination))

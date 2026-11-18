import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]/'agentic_methods/rax'
spec = importlib.util.spec_from_file_location('restore_rax_source', ROOT/'restore_core.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pinned_source_can_be_restored_without_upstream(tmp_path):
    destination = module.restore(tmp_path/'rax')
    pin = json.loads((ROOT/'upstream.json').read_text())
    assert subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip() == pin['revision']
    assert subprocess.check_output(['git','-C',str(destination),'status','--porcelain'],text=True) == ''
    assert subprocess.check_output(['git','-C',str(destination),'ls-tree','-r','HEAD']) == subprocess.check_output(['git','-C',str(ROOT/'rax'),'ls-tree','-r','HEAD'])
    assert module.restore(destination) == destination


def test_restore_does_not_overwrite_user_files(tmp_path):
    destination=tmp_path/'rax';destination.mkdir();(destination/'keep.txt').write_text('keep')
    with pytest.raises(ValueError,match='nonempty'):module.restore(destination)
    assert (destination/'keep.txt').read_text()=='keep'

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


def test_recursive_clone_skips_unreachable_upstream_and_restores_offline(tmp_path):
    pin = json.loads((ROOT/'upstream.json').read_text())
    parent=tmp_path/'parent';parent.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(parent),*args],check=True,capture_output=True,text=True)
    git('init','-q')
    (parent/'.gitmodules').write_text('[submodule "rax"]\n\tpath = rax\n'
        '\turl = file:///nonexistent-rax-upstream\n\tupdate = none\n')
    git('add','.gitmodules')
    git('update-index','--add','--cacheinfo',f'160000,{pin["revision"]},rax')
    git('-c','user.name=Source archive test','-c','user.email=test@example.invalid',
        'commit','-qm','Pinned source')
    clone=tmp_path/'clone'
    subprocess.run(['git','clone','--quiet','--recurse-submodules',str(parent),str(clone)],check=True)
    assert not (clone/'rax/.git').exists()
    module.restore(clone/'rax')
    assert subprocess.check_output(['git','-C',str(clone/'rax'),'rev-parse','HEAD'],text=True).strip()==pin['revision']

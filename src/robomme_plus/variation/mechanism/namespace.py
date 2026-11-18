"""Load task implementations without modifying their shared modules or classes.

Each mechanism environment receives a separate module namespace. Native task
source files are reused in place; only task imports and registration decorators
are redirected while compiling that private namespace. Normal environments
continue using the original modules and Gym registry, even when interleaved.
"""
import ast
import importlib
import importlib.abc
import importlib.util
import itertools
from pathlib import Path
import sys
import threading
import types

ROOT = Path(__file__).resolve().parents[2]
PREFIX = __package__ + '._instances'
_counter = itertools.count()
_lock = threading.RLock()


class _Imports(ast.NodeTransformer):
    def __init__(self, prefix):
        self.prefix = prefix

    def name(self, name):
        original = 'robomme_plus.tasks'
        return self.prefix + '.tasks' + name[len(original):] if name == original or name.startswith(original + '.') else name

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            node.module = self.name(node.module)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            rewritten = self.name(alias.name)
            if rewritten != alias.name and alias.asname is None:
                raise ImportError('Isolated task imports require an explicit alias: ' + alias.name)
            alias.name = rewritten
        return node

    def visit_Constant(self, node):
        # Task code also uses importlib.import_module and sys.modules by name.
        if isinstance(node.value, str):
            node.value = self.name(node.value)
        return node

    def visit_ClassDef(self, node):
        node.decorator_list = [d for d in node.decorator_list if not (
            isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'register_env')]
        return self.generic_visit(node)


class _Source(importlib.abc.Loader):
    def __init__(self, path, prefix):
        self.path, self.prefix = path, prefix

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        tree = _Imports(self.prefix).visit(ast.parse(self.path.read_text(), filename=str(self.path)))
        exec(compile(ast.fix_missing_locations(tree), str(self.path), 'exec'), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(PREFIX + '.'):
            return None
        parts = fullname[len(PREFIX) + 1:].split('.')
        if len(parts) < 2 or parts[1] not in ('tasks', 'behaviors'):
            return None
        prefix = PREFIX + '.' + parts[0]
        base = ROOT / 'tasks' if parts[1] == 'tasks' else ROOT / 'variation/mechanism/behaviors'
        source = base.joinpath(*parts[2:])
        package = source.is_dir()
        source = source / '__init__.py' if package else source.with_suffix('.py')
        if not source.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, source,
            loader=_Source(source, prefix), submodule_search_locations=[str(source.parent)] if package else None)


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__package__ = name
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = module


class TaskNamespace:
    def __init__(self):
        with _lock:
            if PREFIX not in sys.modules:
                _package(PREFIX)
                sys.meta_path.insert(0, _Finder())
            self.prefix = PREFIX + '.episode_' + str(next(_counter))
            _package(self.prefix)

    def load(self, name):
        return importlib.import_module(self.prefix + '.' + name)

    def close(self):
        with _lock:
            for name in tuple(sys.modules):
                if name == self.prefix or name.startswith(self.prefix + '.'):
                    del sys.modules[name]

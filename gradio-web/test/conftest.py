from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _find_repo_root(start_file: str | Path) -> Path:
    path = Path(start_file).resolve()
    cur = path if path.is_dir() else path.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root from {path}")


REPO_ROOT = _find_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
GRADIO_ROOT = REPO_ROOT / "gradio-web"
if not GRADIO_ROOT.exists():
    GRADIO_ROOT = REPO_ROOT / "gradio"

for p in (str(REPO_ROOT), str(SRC_ROOT), str(GRADIO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def reload_module():
    def _reload(name: str):
        module = importlib.import_module(name)
        return importlib.reload(module)

    return _reload

"""Hugging Face Spaces entrypoint for the RoboMME Gradio app."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
GRADIO_DIR = ROOT / "gradio-web"
SKIP_APP_BOOTSTRAP_ENV = "ROBOMME_SKIP_APP_BOOTSTRAP"

for path in (ROOT, SRC_DIR, GRADIO_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Disable SSR for HF Spaces compatibility (avoids gradio_api heartbeat 404).
os.environ.setdefault("GRADIO_SSR_MODE", "false")

from main import build_app, main  # noqa: E402


demo = None if os.getenv(SKIP_APP_BOOTSTRAP_ENV) == "1" else build_app()


if __name__ == "__main__":
    main(demo=demo)

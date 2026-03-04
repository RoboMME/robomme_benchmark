"""Hugging Face Spaces entrypoint for RoboMME Gradio app."""

import os
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
GRADIO_WEB_DIR = APP_DIR / "gradio-web"
SRC_DIR = APP_DIR / "src"
VIDEOS_DIR = GRADIO_WEB_DIR / "videos"
TEMP_DEMOS_DIR = APP_DIR / "temp_demos"
CWD_TEMP_DEMOS_DIR = Path.cwd() / "temp_demos"

# Ensure local modules are importable when running from repository root (HF Spaces).
for import_path in (GRADIO_WEB_DIR, SRC_DIR, APP_DIR):
    resolved = str(import_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

from state_manager import start_timeout_monitor
from ui_layout import create_ui_blocks


def ensure_media_dirs() -> None:
    """Create temp media directories before first write."""
    TEMP_DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    CWD_TEMP_DEMOS_DIR.mkdir(parents=True, exist_ok=True)


def build_allowed_paths() -> list[str]:
    """Build Gradio file access allowlist (absolute, deduplicated)."""
    candidates = [
        Path.cwd(),
        APP_DIR,
        GRADIO_WEB_DIR,
        SRC_DIR,
        VIDEOS_DIR,
        TEMP_DEMOS_DIR,
        CWD_TEMP_DEMOS_DIR,
        Path(tempfile.gettempdir()),
    ]

    deduped = []
    seen = set()
    for path in candidates:
        normalized = str(path.resolve())
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


_APP_BOOTSTRAPPED = False


def bootstrap_runtime() -> None:
    """Initialize runtime side effects once per process."""
    global _APP_BOOTSTRAPPED
    if _APP_BOOTSTRAPPED:
        return

    ensure_media_dirs()
    start_timeout_monitor()
    os.environ.setdefault("ROBOMME_TEMP_DEMOS_DIR", str(TEMP_DEMOS_DIR))
    _APP_BOOTSTRAPPED = True


# Force SSR off before Gradio reads the setting (env var is the only
# reliable way — attribute assignment on Blocks is ignored by launch()).
os.environ["GRADIO_SSR_MODE"] = "false"

bootstrap_runtime()

demo = create_ui_blocks()

_allowed_paths = build_allowed_paths()

# Wrap launch() so that HF Spaces (which calls demo.launch() directly)
# always gets our required kwargs.
_original_launch = demo.launch

def _patched_launch(**kwargs):
    kwargs.setdefault("ssr_mode", False)
    kwargs.setdefault("allowed_paths", _allowed_paths)
    kwargs.setdefault("show_error", True)
    return _original_launch(**kwargs)

demo.launch = _patched_launch


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )

"""Main entry for Gradio app (single-instance mode for Hugging Face Spaces)."""

import os
import tempfile
from pathlib import Path

from ui_layout import create_ui_blocks
from state_manager import start_timeout_monitor

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
VIDEOS_DIR = APP_DIR / "videos"
TEMP_DEMOS_DIR = PROJECT_ROOT / "temp_demos"
CWD_TEMP_DEMOS_DIR = Path.cwd() / "temp_demos"


def ensure_media_dirs():
    """Ensure media temp directories exist before first write."""
    TEMP_DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    CWD_TEMP_DEMOS_DIR.mkdir(parents=True, exist_ok=True)


def build_allowed_paths():
    """Build Gradio file access allowlist (absolute, deduplicated)."""
    candidates = [
        Path.cwd(),
        PROJECT_ROOT,
        APP_DIR,
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


def main():
    ensure_media_dirs()
    start_timeout_monitor()

    os.environ.setdefault("ROBOMME_TEMP_DEMOS_DIR", str(TEMP_DEMOS_DIR))
    allowed_paths = build_allowed_paths()

    demo = create_ui_blocks()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()

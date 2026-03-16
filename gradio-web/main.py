"""Main entry for Gradio app (single-instance mode for Hugging Face Spaces)."""

import logging
import os
import sys
import tempfile
from pathlib import Path




APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
VIDEOS_DIR = APP_DIR / "videos"
TEMP_DEMOS_DIR = PROJECT_ROOT / "temp_demos"
CWD_TEMP_DEMOS_DIR = Path.cwd() / "temp_demos"
CPU_ONLY_ENV_OVERRIDES = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "SAPIEN_RENDER_DEVICE": "cpu",
}
CPU_ONLY_ENV_CLEAR_KEYS = (
    "NVIDIA_DRIVER_CAPABILITIES",
    "VK_ICD_FILENAMES",
    "MUJOCO_GL",
)






if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def configure_cpu_only_runtime(logger: logging.Logger | None = None):
    """Force CPU-only execution before importing project modules."""
    cleared = {}
    for key, value in CPU_ONLY_ENV_OVERRIDES.items():
        os.environ[key] = value
    for key in CPU_ONLY_ENV_CLEAR_KEYS:
        previous = os.environ.pop(key, None)
        if previous is not None:
            cleared[key] = previous
    if logger is not None:
        logger.info(
            "Configured CPU-only runtime overrides=%s cleared=%s",
            CPU_ONLY_ENV_OVERRIDES,
            cleared,
        )
    return cleared


configure_cpu_only_runtime()


def setup_logging() -> logging.Logger:
    """Configure structured logging for Spaces runtime."""
    level_name = "DEBUG"
    os.environ["LOG_LEVEL"] = level_name
    level = logging.DEBUG
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "pid=%(process)d tid=%(threadName)s | %(message)s"
        ),
        stream=sys.stdout,
        force=True,
    )
    for noisy_logger in [
        "asyncio",
        "httpx",
        "httpcore",
        "urllib3",
        "matplotlib",
        "PIL",
        "h5py",
        "trimesh",
        "toppra",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.getLogger("robomme").setLevel(logging.DEBUG)
    logger = logging.getLogger("robomme.main")
    logger.info("Logging initialized with LOG_LEVEL=%s", level_name)
    return logger


LOGGER = setup_logging()


def log_runtime_graphics_env():
    """Log graphics-related runtime env so Spaces diagnostics are visible in stdout."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_ICD_FILENAMES",
        "OMP_NUM_THREADS",
        "SAPIEN_RENDER_DEVICE",
        "MUJOCO_GL",
    ]
    snapshot = {key: os.getenv(key) for key in keys}
    LOGGER.info("Runtime graphics env: %s", snapshot)


def ensure_media_dirs():
    """Ensure media temp directories exist before first write."""
    TEMP_DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    CWD_TEMP_DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(
        "Ensured media dirs: temp_demos=%s cwd_temp_demos=%s",
        TEMP_DEMOS_DIR,
        CWD_TEMP_DEMOS_DIR,
    )


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
    LOGGER.debug("Allowed paths resolved (%d): %s", len(deduped), deduped)
    return deduped


def main():
    configure_cpu_only_runtime(LOGGER)
    from ui_layout import CSS, create_ui_blocks

    LOGGER.info("Starting Gradio real environment entrypoint: %s", __file__)
    log_runtime_graphics_env()
    ensure_media_dirs()

    os.environ.setdefault("ROBOMME_TEMP_DEMOS_DIR", str(TEMP_DEMOS_DIR))
    allowed_paths = build_allowed_paths()
    server_port = int(os.getenv("PORT", "7860"))
    #server_port = 7862
    LOGGER.info(
        "Launching UI with server_name=%s server_port=%s ROBOMME_TEMP_DEMOS_DIR=%s",
        "0.0.0.0",
        server_port,
        os.environ.get("ROBOMME_TEMP_DEMOS_DIR"),
    )
    LOGGER.debug("Python path head entries: %s", sys.path[:5])

    demo = create_ui_blocks()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        allowed_paths=allowed_paths,
        ssr_mode=False,
        debug=True,
        show_error=True,
        quiet=False,
        theme=getattr(demo, "theme", None),
        css=CSS,
        head=getattr(demo, "head", None),
    )


if __name__ == "__main__":
    main()

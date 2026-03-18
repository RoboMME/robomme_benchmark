"""Main entry for the Gradio app across local and Hugging Face Spaces runtimes."""

from __future__ import annotations

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
DEFAULT_LLVMPipe_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
DEFAULT_CPU_RENDER_BACKEND = "pci:0"
CPU_ONLY_ENV_OVERRIDES = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
}
RENDER_ENV_OVERRIDES = {
    "ROBOMME_RENDER_BACKEND": DEFAULT_CPU_RENDER_BACKEND,
}
RENDER_ENV_CLEAR_KEYS = (
    "SAPIEN_RENDER_DEVICE",
    "MUJOCO_GL",
)
LOCAL_ONLY_ENV_CLEAR_KEYS = ("NVIDIA_DRIVER_CAPABILITIES",)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def is_spaces_runtime() -> bool:
    """Best-effort detection of Hugging Face Spaces runtime."""
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def configure_runtime(logger: logging.Logger | None = None):
    """
    Configure runtime defaults.

    Local runs default to CPU-only rendering to keep development portable.
    Hugging Face Spaces runs preserve the GPU environment so ZeroGPU can
    allocate hardware on decorated functions.
    """
    cleared = {}
    for key, value in RENDER_ENV_OVERRIDES.items():
        os.environ[key] = value

    if is_spaces_runtime():
        runtime_mode = "spaces"
        cpu_only = False
        if logger is not None:
            logger.info(
                "Detected Spaces runtime; preserving GPU visibility for ZeroGPU while forcing CPU Vulkan rendering"
            )
    else:
        runtime_mode = "local"
        cpu_only = True
        for key, value in CPU_ONLY_ENV_OVERRIDES.items():
            os.environ[key] = value

    clear_keys = RENDER_ENV_CLEAR_KEYS + (LOCAL_ONLY_ENV_CLEAR_KEYS if cpu_only else ())
    for key in clear_keys:
        previous = os.environ.pop(key, None)
        if previous is not None:
            cleared[key] = previous
    vk_icd_status = "preserved"
    if "VK_ICD_FILENAMES" not in os.environ:
        if DEFAULT_LLVMPipe_ICD.exists():
            os.environ["VK_ICD_FILENAMES"] = str(DEFAULT_LLVMPipe_ICD)
            vk_icd_status = "auto-set"
        else:
            vk_icd_status = "unavailable"
    if logger is not None:
        logger.info(
            "Configured runtime mode=%s cpu_only=%s render_overrides=%s compute_overrides=%s cleared=%s vk_icd_status=%s vk_icd=%s",
            runtime_mode,
            cpu_only,
            RENDER_ENV_OVERRIDES,
            CPU_ONLY_ENV_OVERRIDES if cpu_only else {},
            cleared,
            vk_icd_status,
            os.environ.get("VK_ICD_FILENAMES"),
        )
    return {"mode": runtime_mode, "cpu_only": cpu_only, "cleared": cleared}


def configure_cpu_only_runtime(logger: logging.Logger | None = None):
    """Backward-compatible alias for older tests and scripts."""
    return configure_runtime(logger)


def setup_logging() -> logging.Logger:
    """Configure structured logging for runtime diagnostics."""
    level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
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
    """Log graphics-related runtime env so diagnostics are visible in stdout."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_ICD_FILENAMES",
        "OMP_NUM_THREADS",
        "ROBOMME_RENDER_BACKEND",
        "SAPIEN_RENDER_DEVICE",
        "MUJOCO_GL",
        "SPACE_ID",
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


def build_app():
    """Create the Gradio demo after runtime setup."""
    configure_runtime(LOGGER)
    from ui_layout import CSS, create_ui_blocks

    LOGGER.info("Building Gradio UI entrypoint: %s", __file__)
    log_runtime_graphics_env()
    ensure_media_dirs()

    os.environ.setdefault("ROBOMME_TEMP_DEMOS_DIR", str(TEMP_DEMOS_DIR))
    demo = create_ui_blocks()
    demo.css = CSS
    return demo


def main(*, demo=None):
    """Launch the Gradio app."""
    if demo is None:
        demo = build_app()
    allowed_paths = build_allowed_paths()
    server_port = int(os.getenv("PORT", "7860"))
    LOGGER.info(
        "Launching UI with server_name=%s server_port=%s ROBOMME_TEMP_DEMOS_DIR=%s",
        "0.0.0.0",
        server_port,
        os.environ.get("ROBOMME_TEMP_DEMOS_DIR"),
    )
    LOGGER.debug("Python path head entries: %s", sys.path[:5])
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        allowed_paths=allowed_paths,
        ssr_mode=False,
        debug=True,
        show_error=True,
        quiet=False,
        theme=getattr(demo, "theme", None),
        css=getattr(demo, "css", None),
        head=getattr(demo, "head", None),
    )


if __name__ == "__main__":
    main()

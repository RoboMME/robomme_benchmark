import logging

logger = logging.getLogger("robomme")

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] [%(name)s] %(message)s',
        force=True,
    )


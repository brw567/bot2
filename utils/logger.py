"""Central logging configuration with a rotating file handler."""

import logging
from logging.handlers import RotatingFileHandler


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a configured logger with a rotating file handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=3)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

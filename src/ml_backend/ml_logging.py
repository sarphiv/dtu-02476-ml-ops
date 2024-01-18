import logging
import sys
import os
from logging import Logger
from pathlib import Path
from hydra import initialize, compose

log_level_mapping = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

def get_configs(cfg):
    logging_config = {
        "version": 1,
        "formatters": { #
            "minimal": {"format": "%(message)s"},
            "detailed": {
                "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            },
        },
        "handlers": { #
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "minimal",
                "level": logging.DEBUG,
            },
            "info": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(cfg.log_dir, "info.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.INFO,
            },
            "error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(cfg.log_dir, "error.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.ERROR,
            },
        },
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    }
    return logging_config

def get_logger() -> Logger:
    with initialize(config_path=os.environ.get("CONFIG_DIR", "../../configs"), version_base="1.3"):
        cfg = compose(config_name="logging")
        level = log_level_mapping[cfg.log_level]
        logging.basicConfig(stream=sys.stdout, level=level)
        # logging.config.dictConfig(config)
        return logging.getLogger(__name__)

if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    logger.debug("Example debug")
    logger.info("Example info")
    logger.warning("Example warning")
    logger.error("Example error")
    logger.critical("Example critical error")

import logging
import sys
from typing import Optional


def configure_logging(use_json: Optional[bool] = None):
    logging.basicConfig()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if use_json:
        import jsonlogging
        formatter = jsonlogging.JSONFormatter()

        try:
            handler = logger.handlers[0]
            handler.setFormatter(formatter)
        except IndexError:
            log_handler = logging.StreamHandler(sys.stdout)
            log_handler.setFormatter(formatter)
            logger.addHandler(log_handler)

    return logger

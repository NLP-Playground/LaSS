import logging


def get_logger():
    name = "logger"
    if hasattr(get_logger, name):
        return getattr(get_logger, name)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    setattr(get_logger, name, logger)
    return logger
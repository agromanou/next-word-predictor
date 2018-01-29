import logging
from logging import handlers


def setup_logger(name, level='INFO'):
    """

    :param name:
    :param level:
    :return:
    """

    # create a logging format
    formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)d - %(name)s.py:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)

    if level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # create INFO file handler
    file_handler = handlers.RotatingFileHandler('{}.log'.format(name),
                                                maxBytes=1024 * 1024 * 100,
                                                backupCount=20)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # create ERROR file handler
    error_file_handler = handlers.RotatingFileHandler('{}.error.log'.format(name),
                                                      maxBytes=1024 * 1024 * 100,
                                                      backupCount=20)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    logger.addHandler(error_file_handler)

    # create INFO stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

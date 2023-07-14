import datetime
import logging
from logging.handlers import RotatingFileHandler

loggers = {}


def myLogger(name: str) -> logging.Logger:
    """
    Create or return a logger object with the specified name.

    Arguments:
    name -- The name of the logger

    Returns:
    logger -- The logger object
    """
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        now = datetime.datetime.now()
        handler = logging.FileHandler(
            "/root/credentials/Logs/ProvisioningPython"
            + now.strftime("%Y-%m-%d")
            + ".log"
        )
        formatter = logging.Formatter(
            "[%(asctime)s] - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers.update(dict(name=logger))

        return logger


def setup_logger(logger_name: str, log_file: str, level=logging.DEBUG) -> logging.Logger:
    """
    Configure the logger with the specified name and log file.

    Arguments:
    logger_name -- The name of the logger
    log_file -- The path to the log file
    level -- The logging level (default: logging.DEBUG)

    Returns:
    logger -- The logger object
    """
    if loggers.get(logger_name):
        return loggers.get(logger_name)
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] - %(levelname)s - %(message)s")
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(level)

        my_handler = RotatingFileHandler(
            log_file,
            mode="a",
            maxBytes=7 * 1024 * 1024,
            backupCount=10,
            encoding="utf8",
            delay=0,
        )
        my_handler.setFormatter(formatter)
        my_handler.setLevel(level)

        app_log = logging.getLogger(logger_name)
        app_log.setLevel(level)
        app_log.addHandler(streamHandler)
        app_log.addHandler(my_handler)

        loggers.update({logger_name: app_log})
        return app_log


class Logger:
    def __init__(self, logger_name: str, log_file: str):
        """
        Initialize a Logger object with the specified name and log file.

        Arguments:
        logger_name -- The name of the logger
        log_file -- The path to the log file
        """
        self.__log = setup_logger(logger_name, log_file)

    def error(self, msg: str) -> None:
        """
        Log an error message to the log file with color formatting.

        Arguments:
        msg -- The error message content
        """
        self.__log.error('\033[91m' + msg + '\033[0m')

    def debug(self, msg: str) -> None:
        """
        Log a debug message to the log file with color formatting.

        Arguments:
        msg -- The debug message content
        """
        self.__log.debug('\033[94m' + msg + '\033[0m')

    def warning(self, msg: str) -> None:
        """
        Log a warning message to the log file with color formatting.

        Arguments:
        msg -- The warning message content
        """
        self.__log.warning('\033[93m' + msg + '\033[0m')

    def info(self, msg: str) -> None:
        """
        Log an info message to the log file with color formatting.

        Arguments:
        msg -- The info message content
        """
        self.__log.info('\033[92m' + msg + '\033[0m')

    def critical(self, msg: str) -> None:
        """
        Log a critical message to the log file with color formatting.

        Arguments:
        msg -- The critical message content
        """
        self.__log.critical('\033[95m' + msg + '\033[0m')

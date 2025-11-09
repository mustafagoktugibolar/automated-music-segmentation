import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

_logger = logging.getLogger("music-segmentation")
_logger.setLevel(logging.INFO)
_logger.propagate = False

if not _logger.handlers:
    log_format = logging.Formatter(
        "%(levelname)s: | %(asctime)s | Message: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    _logger.addHandler(stream_handler)

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "music-segmentation.log")

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight", 
        interval=1,           
        backupCount=7,        
        encoding="utf-8",
        utc=False             
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(log_format)
    _logger.addHandler(file_handler)


def info(message: str, *args, **kwargs):
    _logger.info(message, *args, **kwargs)

def debug(message: str, *args, **kwargs):
    _logger.debug(message, *args, **kwargs)

def warning(message: str, *args, **kwargs):
    _logger.warning(message, *args, **kwargs)

def error(message: str, exception: Exception = None, *args, **kwargs):
    if exception is not None:
        kwargs["exc_info"] = True
    _logger.error(message, *args, **kwargs)

def fatal(message: str, exception: Exception = None, *args, **kwargs):
    if exception is not None:
        kwargs["exc_info"] = True
    _logger.critical(message, *args, **kwargs)
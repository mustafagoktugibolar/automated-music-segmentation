import logging
import sys

# singleton logger instance
_logger = logging.getLogger("core")
_logger.setLevel(logging.INFO)
_logger.propagate = False

# TODO: Configure logger file output — logs should be stored in a specific directory, organized by date
if not _logger.hasHandlers():
    log_format = logging.Formatter(
        "%(levelname)s: | %(asctime)s | Message: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    stream_handler.flush = sys.stdout.flush
    _logger.addHandler(stream_handler)


def info(message: str, *args, **kwargs):
    _logger.info(message, *args, **kwargs)    

def debug(message: str, *args, **kwargs):
    _logger.debug(message, *args, **kwargs)
    
def warning(message: str, *args, **kwargs):
    _logger.warning(message, *args, **kwargs)
    
def error(message: str, exception: Exception = None, *args, **kwargs):
    if exception is not None:
        kwargs['exc_info'] = True
    _logger.error(message, *args, **kwargs)
    
def fatal(message: str, exception: Exception = None, *args, **kwargs):
    if exception is not None:
        kwargs['exc_info'] = True
    _logger.critical(message, *args, **kwargs)
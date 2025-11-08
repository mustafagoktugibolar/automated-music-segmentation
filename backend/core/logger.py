import logging
import sys

# singleton logger instance
_logger = logging.getLogger()

# TODO: Configure logger file output — logs should be stored in a specific directory, organized by date
if not _logger.hasHandlers():
    log_format = logging.Formatter("%(levelname)s: | %(asctime)s | Message: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    _logger.addHandler(stream_handler)


def info(message: str, *args, **kwargs):
    _logger.setLevel(logging.INFO)
    _logger.info(message, *args, **kwargs)    

def debug(message: str, *args, **kwargs):
    _logger.setLevel(logging.DEBUG)
    _logger.debug(message, *args, **kwargs)
    
def warning(message: str, *args, **kwargs):
    _logger.setLevel(logging.WARNING)
    _logger.warning(message, *args, **kwargs)
    
def error(message: str, exception: Exception = None, *args, **kwargs):
    if exception is not None:
        kwargs['exc_info'] = True
    _logger.setLevel(logging.ERROR)
    _logger.error(message, *args, **kwargs)
    
def fatal(message: str, exception: Exception = None, *args, **kwargs):
    if exception is not None:
        kwargs['exc_info'] = True
    _logger.setLevel(logging.CRITICAL)
    _logger.critical(message, *args, **kwargs)
import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

def get_logger(name: str = None):
    if not name:
        name = os.getenv("SERVICE_NAME", "app")
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_format = logging.Formatter(
            "%(levelname)s: | %(asctime)s | Message: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_format)
        logger.addHandler(stream_handler)

        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{name}.log")

        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight", 
            interval=1,           
            backupCount=7,        
            encoding="utf-8",
            utc=False             
        )
        file_handler.suffix = "%Y-%m-%d"
    
        def custom_namer(default_name):
            base_dir, filename = os.path.split(default_name)
            
            if default_name.endswith(".log") or len(filename.split('.')) < 3:
                 return default_name
                 
            parts = filename.split('.')
            date_part = parts[-1] 
            name_part = parts[0]
            
            new_filename = f"{date_part}-{name_part}.log"
            return os.path.join(base_dir, new_filename)

        file_handler.namer = custom_namer
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
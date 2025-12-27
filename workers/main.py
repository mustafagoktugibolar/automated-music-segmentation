import os
import sys
from shared.logger import get_logger
from workers.msaf_worker import MSAFWorker
from workers.custom_worker import CustomWorker

logger = get_logger()

def main():
    worker_type = os.getenv("WORKER_TYPE", "unknown")
    logger.info(f"Initializing worker of type: {worker_type}")

    worker = None

    if worker_type == "msaf_segmentation":
        worker = MSAFWorker()
    elif worker_type == "custom_segmentation":
        worker = CustomWorker()
    else:
        logger.error(f"Unknown WORKER_TYPE: {worker_type}")
        sys.exit(1)

    if worker:
        worker.start()

if __name__ == "__main__":
    main()

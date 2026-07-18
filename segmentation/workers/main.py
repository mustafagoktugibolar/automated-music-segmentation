import os
import sys
from segmentation.infrastructure.logging import get_logger

logger = get_logger()

def main():
    worker_type = os.getenv("WORKER_TYPE", "unknown")
    logger.info(f"Initializing worker of type: {worker_type}")

    # Worker classes are imported lazily, one per branch, so that a broken
    # dependency in one segmenter (e.g. msaf's fragile numpy/scipy stack)
    # can't crash every other worker type at container startup.
    worker = None

    if worker_type == "msaf_segmentation":
        from segmentation.workers.segmenters.msaf.msaf_worker import MSAFWorker
        worker = MSAFWorker()
    elif worker_type == "custom_segmentation":
        from segmentation.workers.segmenters.custom.custom_worker import CustomWorker
        worker = CustomWorker()
    elif worker_type == "fusion_segmentation":
        from segmentation.workers.segmenters.fusion.fusion_worker import FusionWorker
        worker = FusionWorker()
    else:
        logger.error(f"Unknown WORKER_TYPE: {worker_type}")
        sys.exit(1)

    if worker:
        worker.start()

if __name__ == "__main__":
    main()

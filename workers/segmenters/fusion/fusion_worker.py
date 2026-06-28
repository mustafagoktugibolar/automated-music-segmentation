import os

from shared.logger import get_logger
from workers.base_worker import BaseWorker
from workers.segmenters.fusion.fusion_service import fuse_algorithm_results

logger = get_logger()


class FusionWorker(BaseWorker):
    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.fusion")

        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "fusion-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code],
        )

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        logger.info("Received algorithm-level fusion task for %s", task_id)

        algorithm_results = task.get("algorithm_results") or {}
        if len(algorithm_results) < 2:
            raise ValueError("Fusion requires at least two completed base algorithm results")

        file_path = None
        try:
            file_path = self._resolve_file_path(task)
        except (ValueError, KeyError, RuntimeError):
            logger.debug("No audio file path in fusion task %s; labels will use fallback clustering", task_id)

        return fuse_algorithm_results(
            algorithm_results,
            task_id=task_id,
            params=task.get("params") or {},
            file_path=file_path,
        )

import os

from shared.logger import get_logger
from workers.BaseWorker import BaseWorker

logger = get_logger()


class CustomWorker(BaseWorker):
    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.custom")

        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "custom-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code],
        )

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        params = (task.get("params") or {}).get("custom") or {}

        file_path = self._resolve_file_path(task)
        logger.info(f"Received Custom segmentation task for file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File {file_path} does not exist")

        from workers.segmenters.segmentation_service import process_file_path

        result = process_file_path(file_path, params=params)
        logger.info(f"Custom segmentation finished for {task_id}")

        response = {
            "task_id": task_id,
            "status": "completed",
            "worker_type": "custom",
            "algorithm": "custom",
            "segments": result.get("segments", []),
        }
        for key in ("estimated_bpm", "candidate_boundaries", "diagnostics"):
            if key in result:
                response[key] = result[key]
        return response

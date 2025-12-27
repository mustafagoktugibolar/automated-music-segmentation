import os
import time
from workers.base import BaseWorker
from shared.logger import get_logger

logger = get_logger()

class CustomWorker(BaseWorker):
    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.custom")
        
        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "custom-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code]
        )

    def process_task(self, task: dict) -> dict:
        """
        Simulates Custom Segmentation processing.
        """
        file_path = task.get("file_path")
        task_id = task.get("task_id")
        
        logger.info(f"Received Custom segmentation task for file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File {file_path} does not exist")

        # Call the actual segmentation logic
        from workers.segmenters.segmentation_service import process_file_path
        result = process_file_path(file_path)
        
        logger.info(f"Custom segmentation finished for {task_id}")

        return {
            "task_id": task_id,
            "status": "completed",
            "worker_type": "custom",
            "segments": result.get("segments", [])
        }

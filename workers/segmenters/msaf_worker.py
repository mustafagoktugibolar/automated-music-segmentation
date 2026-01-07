import os
import msaf
from workers.BaseWorker import BaseWorker
from shared.logger import get_logger

logger = get_logger()

class MSAFWorker(BaseWorker):
    def __init__(self):
        # Use MESSAGE_CODE from environment variable as the routing key
        message_code = os.getenv("MESSAGE_CODE", "segmentation.request.msaf")
        self.algorithm = os.getenv("MSAF_ALGORITHM", "foote")
        
        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "msaf-worker"),
            queue_name=f"queue_{message_code}", # Unique queue per algorithm/worker
            routing_keys=[message_code]
        )

    def process_task(self, task: dict) -> dict:
        """
        Executes MSAF processing using the configured algorithm.
        """
        file_path = task.get("file_path")
        task_id = task.get("task_id")
        
        logger.info(f"Received MSAF task (Algo: {self.algorithm}) for: {file_path}")

        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            # Depending on policy, might want to return failed status or raise
            raise FileNotFoundError(error_msg)

        try:
            # MSAF processing
            # boundaries_id defaults to valid algo. labels_id can be same or specifically 'scluster', 'fmc2d' etc.
            # We let MSAF default the labeling (usually fmc2d) if we only provide boundaries_id
            est_times, est_labels = msaf.process(
                file_path, 
                boundaries_id=self.algorithm
            )
            
            # format the output
            segments = []
            for i in range(len(est_times) - 1):
                start = float(est_times[i])
                end = float(est_times[i+1])
                label = str(est_labels[i]) if i < len(est_labels) else "Unknown"
                
                segments.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "label": label
                })
            
            logger.info(f"MSAF ({self.algorithm}) finished for {task_id}. Found {len(segments)} segments.")

            # Return result
            return {
                "task_id": task_id,
                "status": "completed",
                "worker_type": "msaf",
                "algorithm": self.algorithm,
                "segments": segments
            }

        except Exception as e:
            logger.error(f"MSAF processing failed for {file_path}", exc_info=True)
            raise e

import os

import msaf

from shared.logger import get_logger
from workers.BaseWorker import BaseWorker

logger = get_logger()


class MSAFWorker(BaseWorker):
    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.request.msaf")
        self.algorithm = os.getenv("MSAF_ALGORITHM", "foote")

        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "msaf-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code],
        )

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        msaf_params = (task.get("params") or {}).get("msaf") or {}

        file_path = self._resolve_file_path(task)
        logger.info(f"Received MSAF task (Algo: {self.algorithm}) for: {file_path}")

        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            process_kwargs = {
                "boundaries_id": self.algorithm,
            }
            if msaf_params.get("labeling_id"):
                process_kwargs["labels_id"] = msaf_params["labeling_id"]
            if msaf_params.get("hier") is not None:
                process_kwargs["hier"] = bool(msaf_params["hier"])

            est_times, est_labels = msaf.process(file_path, **process_kwargs)

            segments = []
            for i in range(len(est_times) - 1):
                start = float(est_times[i])
                end = float(est_times[i + 1])
                label = str(est_labels[i]) if i < len(est_labels) else "Unknown"
                segments.append(
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "label": label,
                    }
                )

            logger.info(f"MSAF ({self.algorithm}) finished for {task_id}. Found {len(segments)} segments.")

            return {
                "task_id": task_id,
                "status": "completed",
                "worker_type": "msaf",
                "algorithm": self.algorithm,
                "segments": segments,
            }

        except Exception:
            logger.error(f"MSAF processing failed for {file_path}", exc_info=True)
            raise

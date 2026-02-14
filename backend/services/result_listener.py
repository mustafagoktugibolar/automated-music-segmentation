import threading

import requests

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.db.models import Base, SegmentationTask
from shared.config import DBSettings
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient

logger = get_logger()

# Callback registry for SSE notifications
sse_callbacks: dict[str, callable] = {}


def register_sse_callback(task_id: str, callback: callable):
    sse_callbacks[task_id] = callback


def unregister_sse_callback(task_id: str):
    sse_callbacks.pop(task_id, None)


class ResultListener:
    def __init__(self):
        self.rabbitmq = RabbitMQClient(service_name="result-listener")
        self.engine = create_engine(DBSettings.DB_URL)
        Base.metadata.create_all(bind=self.engine)

    def start(self):
        logger.info("Starting Result Listener...")
        t = threading.Thread(target=self._consume_loop, daemon=True)
        t.start()

    def _consume_loop(self):
        self.rabbitmq.consume(
            queue_name="segmentation_results_queue",
            routing_keys=["segmentation.result"],
            callback=self._process_result,
        )

    def _result_key(self, worker_type: str, algorithm: str | None) -> str:
        worker_type = (worker_type or "").lower().strip()
        if worker_type == "custom":
            return "custom"
        return (algorithm or "default").lower().strip()

    def _call_webhook(self, task: SegmentationTask):
        try:
            payload = {
                "task_id": task.task_id,
                "status": task.status,
                "filename": task.filename,
                "results": task.results,
            }
            
            # Push to SSE callback if exists
            logger.info(f"Checking SSE callbacks. Available: {list(sse_callbacks.keys())}")
            if task.task_id in sse_callbacks:
                try:
                    logger.info(f"Calling SSE callback for task {task.task_id}")
                    sse_callbacks[task.task_id](payload)
                    logger.info(f"Pushed result to SSE for task {task.task_id}")
                except Exception as e:
                    logger.error(f"SSE callback failed for task {task.task_id}: {e}")
                return
            
            # Fallback: call external webhook if URL is set
            if task.webhook_url:
                response = requests.post(task.webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                logger.info(f"Webhook called successfully for task {task.task_id}: {task.webhook_url}")
        except Exception as e:
            logger.error(f"Failed to notify for task {task.task_id}: {e}")

    def _process_result(self, ch, method, properties, body):
        try:
            data = body
            task_id = data.get("task_id")
            worker_type = data.get("worker_type")
            algorithm = data.get("algorithm")
            segments = data.get("segments", [])

            key = self._result_key(worker_type, algorithm)
            logger.info(f"Received result for task {task_id} from {key}")

            with Session(self.engine) as session:
                task = session.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()

                if not task:
                    task = SegmentationTask(task_id=task_id, status="processing")
                    session.add(task)

                current_results = dict(task.results) if task.results else {}
                current_results[key] = segments
                task.results = current_results

                expected = {str(a).lower().strip() for a in (task.expected_algorithms or [])}
                received = {str(a).lower().strip() for a in current_results.keys()}

                if expected and expected.issubset(received):
                    task.status = "completed"
                    logger.info(f"Task {task_id} COMPLETED. All expected results received: {received}")
                else:
                    task.status = "processing"
                    logger.info(f"Task {task_id} processing. Received: {received}, Expected: {expected}")
                
                # Push update to SSE (even for partial results)
                self._call_webhook(task)

                session.commit()
                logger.info(f"Updated DB for task {task_id}")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            logger.error("Failed to process result", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

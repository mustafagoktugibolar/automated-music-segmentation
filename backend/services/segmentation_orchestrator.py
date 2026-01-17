import os
import uuid
import aiofiles
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient
from backend.db.models import SegmentationTask
from backend.db.postgreSQL import SessionLocal

logger = get_logger()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "media/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SegmentationOrchestrator:
    def __init__(self):
        self.rabbitmq = RabbitMQClient(service_name="segmentation_orchestrator")
        self.algo_to_routing_key = {
            "custom": "segmentation.custom",
            "foote": "segmentation.foote",
            "cnmf": "segmentation.cnmf",
            "scluster": "segmentation.scluster"
        }

    async def process_upload(self, file, requested_algos: list[str]) -> str:
        """
        Orchestrates the upload process:
        1. Validates algorithms
        2. Saves file to disk or get file from azure blob storage from path TODO!
        3. Creates DB record
        4. Publishes tasks to RabbitMQ
        Returns: task_id (str)
        """
        target_keys = []
        for algo in requested_algos:
            key = self.algo_to_routing_key.get(algo.lower())
            if key:
                target_keys.append(key)
            else:
                logger.warning(f"Unknown algorithm requested: {algo}")

        if not target_keys:
            raise ValueError("No valid algorithms specified.")

        task_id = str(uuid.uuid4())
        filename = f"{task_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        try:
            logger.info(f"Saving uploaded file to {file_path}")
            async with aiofiles.open(file_path, 'wb') as out_file:
                while content := await file.read(1024 * 1024):
                    await out_file.write(content)
            
            try:
                db = SessionLocal()
                new_task = SegmentationTask(
                    task_id=task_id, 
                    filename=file.filename,
                    status="processing",
                    results={},
                    expected_algorithms=requested_algos
                )
                db.add(new_task)
                db.commit()
                db.close()
            except Exception as e:
                logger.error("Failed to save initial task to DB", exc_info=True)
                raise RuntimeError("Database error during task creation")

            task_payload = {
                "task_id": task_id,
                "original_filename": file.filename,
                "file_path": file_path,
                "content_type": file.content_type
            }
            
            logger.info(f"Distributing tasks for {task_id} to workers: {target_keys}")
            for key in target_keys:
                self.rabbitmq.publish(
                    exchange="segmentation_topic",
                    routing_key=key,
                    message=task_payload
                )

            return task_id

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            raise e

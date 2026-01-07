import threading
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from shared.rabbitmq import RabbitMQClient
from shared.logger import get_logger
from shared.config import DBSettings
from backend.db.models import SegmentationTask, Base

logger = get_logger()

class ResultListener:
    def __init__(self):
        self.rabbitmq = RabbitMQClient(service_name="result-listener")
        
        db_settings = DBSettings
        self.engine = create_engine(db_settings.DB_URL)
        
        Base.metadata.create_all(bind=self.engine)

    def start(self):
        """
        Starts consuming segmentation results.
        Running in a separate thread or process is essential if this blocks.
        For FastAPI integration, we might run this in a background thread.
        """
        logger.info("Starting Result Listener...")
        # We use a separate thread usually, but for consuming we can use the blocking consume in a thread

        t = threading.Thread(target=self._consume_loop, daemon=True)
        t.start()

    def _consume_loop(self):
        self.rabbitmq.consume(
            queue_name="segmentation_results_queue",
            routing_keys=["segmentation.result"],
            callback=self._process_result
        )

    def _process_result(self, ch, method, properties, body):
        try:
            data = body
            task_id = data.get("task_id")
            worker_type = data.get("worker_type") # 'msaf' or 'custom'
            algorithm = data.get("algorithm", "default") # 'foote', 'custom', etc.
            segments = data.get("segments", [])
            
            key = algorithm if worker_type == 'msaf' else 'custom'
            
            logger.info(f"Received result for task {task_id} from {key}")

            with Session(self.engine) as session:
                task = session.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()
                
                if not task:
                    task = SegmentationTask(task_id=task_id, status="processing")
                    session.add(task)
                
                current_results = dict(task.results) if task.results else {}
                current_results[key] = segments
                
                task.results = current_results
                
                expected = set(task.expected_algorithms or [])
                received = set(current_results.keys())
                
                if expected and expected.issubset(received):
                    task.status = "completed"
                    logger.info(f"Task {task_id} COMPLETED. All expected results received: {received}")
                    # TODO: Remove the audio file from the upload directory
                else:
                    task.status = "processing"
                    logger.info(f"Task {task_id} processing. Received: {received}, Expected: {expected}")
                
                session.commit()
                logger.info(f"Updated DB for task {task_id}")

            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error("Failed to process result", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

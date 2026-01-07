from abc import ABC, abstractmethod
import signal
import sys
from shared.rabbitmq import RabbitMQClient
from shared.logger import get_logger

logger = get_logger()

class BaseWorker(ABC):
    def __init__(self, service_name: str, queue_name: str, routing_keys: list[str]):
        self.service_name = service_name
        self.queue_name = queue_name
        self.routing_keys = routing_keys
        self.rabbitmq = RabbitMQClient(service_name=service_name)
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    @abstractmethod
    def process_task(self, task: dict) -> dict:
        """
        Process the incoming task and return the result.
        Must be implemented by subclasses.
        """
        pass

    def _callback(self, ch, method, properties, body):
        logger.info(f"[{self.service_name}] Processing task...")
        try:
            # Execute the actual worker logic
            result = self.process_task(body)
            
            # Publish result if any
            if result:
                self.rabbitmq.publish(
                    exchange='segmentation_topic',
                    routing_key='segmentation.result',
                    message=result
                )
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"[{self.service_name}] Task completed and acked.")
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Task processing failed", exc_info=True)
            # Negative acknowledgement TODO: (requeue=False -> dead letter or discard)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start(self):
        logger.info(f"[{self.service_name}] Starting worker...")
        self.rabbitmq.consume(
            queue_name=self.queue_name,
            routing_keys=self.routing_keys,
            callback=self._callback
        )

    def shutdown(self, signum, frame):
        logger.info(f"[{self.service_name}] Shutting down...")
        self.rabbitmq.close()
        sys.exit(0)

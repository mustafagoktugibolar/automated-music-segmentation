from abc import ABC, abstractmethod
import os
import signal
import sys

from shared.blob_helper import AzureBlobCacheHelper
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient

logger = get_logger()


class BaseWorker(ABC):
    def __init__(self, service_name: str, queue_name: str, routing_keys: list[str]):
        self.service_name = service_name
        self.queue_name = queue_name
        self.routing_keys = routing_keys
        self.rabbitmq = RabbitMQClient(service_name=service_name)
        self._blob_helper = None

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    @abstractmethod
    def process_task(self, task: dict) -> dict:
        pass

    def _get_blob_helper(self) -> AzureBlobCacheHelper:
        if self._blob_helper is None:
            self._blob_helper = AzureBlobCacheHelper()
        return self._blob_helper

    def _resolve_file_path(self, task: dict) -> str:
        source_type = str(task.get("source_type", "upload")).lower().strip()

        if source_type == "upload":
            file_path = task.get("file_path")
            if not file_path:
                raise ValueError("file_path is required for upload source")
            return file_path

        if source_type == "storage":
            blob_name = task.get("blob_name")
            if not blob_name:
                raise ValueError("blob_name is required for storage source")

            container = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "").strip()
            if not container:
                raise RuntimeError("AZURE_STORAGE_CONTAINER_RAW is not configured")

            helper = self._get_blob_helper()
            return helper.download_to_cache(container=container, blob_name=blob_name)

        raise ValueError(f"Unsupported source_type={source_type}")

    def _callback(self, ch, method, properties, body):
        logger.info(f"[{self.service_name}] Processing task...")
        try:
            result = self.process_task(body)

            if result:
                self.rabbitmq.publish(
                    exchange="segmentation_topic",
                    routing_key="segmentation.result",
                    message=result,
                )

            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"[{self.service_name}] Task completed and acked.")

        except Exception:
            logger.error(f"[{self.service_name}] Task processing failed", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start(self):
        logger.info(f"[{self.service_name}] Starting worker...")
        self.rabbitmq.consume(
            queue_name=self.queue_name,
            routing_keys=self.routing_keys,
            callback=self._callback,
        )

    def shutdown(self, signum, frame):
        logger.info(f"[{self.service_name}] Shutting down...")
        self.rabbitmq.close()
        sys.exit(0)

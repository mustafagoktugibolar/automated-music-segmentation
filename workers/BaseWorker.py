from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import sys
import time
import traceback

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
        self._concurrency = int(os.getenv("WORKER_CONCURRENCY", "1"))
        self._executor = ThreadPoolExecutor(max_workers=self._concurrency)

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    @abstractmethod
    def process_task(self, task: dict) -> dict:
        pass

    def _failure_result(self, task: dict, exc: Exception) -> dict:
        worker_type = os.getenv("WORKER_TYPE", self.service_name)
        algorithm = task.get("algorithm")
        if not algorithm:
            if worker_type == "custom_segmentation":
                algorithm = "custom_librosa"
            elif worker_type == "msaf_segmentation":
                algorithm = os.getenv("MSAF_ALGORITHM", "msaf")
            elif worker_type == "fusion_segmentation":
                algorithm = "fusion"
            elif worker_type == "llm_segmentation":
                algorithm = "llm"
        return {
            "task_id": task.get("task_id"),
            "status": "failed",
            "worker_type": worker_type,
            "algorithm": algorithm,
            "segments": [],
            "boundaries": [],
            "diagnostics": {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            },
        }

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
        # Return immediately so pika's event loop isn't blocked.
        # Ack/nack are posted back to pika's thread via add_callback_threadsafe,
        # which is the only thread-safe way to call channel methods from threads.
        def _run():
            try:
                t_start = time.perf_counter()
                result = self.process_task(body)
                processing_time = round(time.perf_counter() - t_start, 3)

                if result:
                    result["processing_time_seconds"] = processing_time

                def _ack_and_publish():
                    try:
                        if result:
                            self.rabbitmq.publish(
                                exchange="segmentation_topic",
                                routing_key="segmentation.result",
                                message=result,
                            )
                    finally:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        logger.info(f"[{self.service_name}] Task completed and acked.")

                self.rabbitmq.connection.add_callback_threadsafe(_ack_and_publish)

            except Exception as exc:
                logger.error(f"[{self.service_name}] Task processing failed", exc_info=True)
                failure_result = self._failure_result(body, exc)

                def _publish_failure_and_ack():
                    try:
                        self.rabbitmq.publish(
                            exchange="segmentation_topic",
                            routing_key="segmentation.result",
                            message=failure_result,
                        )
                    finally:
                        ch.basic_ack(delivery_tag=method.delivery_tag)

                self.rabbitmq.connection.add_callback_threadsafe(_publish_failure_and_ack)

        self._executor.submit(_run)

    def start(self):
        logger.info(
            f"[{self.service_name}] Starting worker (concurrency={self._concurrency})..."
        )
        self.rabbitmq.consume(
            queue_name=self.queue_name,
            routing_keys=self.routing_keys,
            callback=self._callback,
            prefetch_count=self._concurrency,
        )

    def shutdown(self, signum, frame):
        logger.info(f"[{self.service_name}] Shutting down...")
        self._executor.shutdown(wait=False)
        self.rabbitmq.close()
        sys.exit(0)

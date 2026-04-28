import json
import os
import time
from typing import Callable

import pika

from shared.logger import get_logger

logger = get_logger()


class RabbitMQClient:
    def __init__(self, service_name: str = "rabbitmq_client"):
        self.host = os.getenv("RABBITMQ_HOST", "music_segmentation_rabbitmq")
        self.user = os.getenv("RABBITMQ_DEFAULT_USER", "guest")
        self.password = os.getenv("RABBITMQ_DEFAULT_PASS", "guest")
        self.port = int(os.getenv("RABBITMQ_PORT", 5672))
        self.service_name = service_name
        self.connection = None
        self.channel = None

    def connect(self):
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )

        while True:
            try:
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                logger.info(f"[{self.service_name}] Connected to RabbitMQ at {self.host}:{self.port}")
                self.channel.exchange_declare(exchange="segmentation_topic", exchange_type="topic", durable=True)
                return
            except pika.exceptions.AMQPConnectionError:
                logger.error(f"[{self.service_name}] Connection failed, retrying in 5s...", exc_info=True)
                time.sleep(5)

    def publish(self, routing_key: str, message: dict, exchange: str = "segmentation_topic"):
        if not self.connection or self.connection.is_closed:
            self.connect()

        try:
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
            )
            logger.info(f"[{self.service_name}] Published to {routing_key}: {message.get('task_id', 'unknown')}")
        except Exception:
            logger.error(f"[{self.service_name}] Publish failed", exc_info=True)
            self.connect()
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
            )

    def consume(self, queue_name: str, routing_keys: list[str], callback: Callable, exchange: str = "segmentation_topic"):
        if not self.connection or self.connection.is_closed:
            self.connect()

        self.channel.queue_declare(queue=queue_name, durable=True)

        for key in routing_keys:
            self.channel.queue_bind(exchange=exchange, queue=queue_name, routing_key=key)
            logger.info(f"[{self.service_name}] Bound queue {queue_name} to key {key}")

        self.channel.basic_qos(prefetch_count=1)

        def on_message(ch, method, properties, body):
            try:
                message = json.loads(body)
                logger.info(f"[{self.service_name}] Received message: {method.routing_key}")
                callback(ch, method, properties, message)
            except Exception:
                logger.error(f"[{self.service_name}] Error processing message", exc_info=True)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_consume(queue=queue_name, on_message_callback=on_message)
        logger.info(f"[{self.service_name}] Waiting for messages in {queue_name}...")
        self.channel.start_consuming()

    def rpc_call(self, routing_key: str, message: dict, timeout: int = 30, exchange: str = "segmentation_topic") -> dict:
        import uuid
        corr_id = str(uuid.uuid4())

        # Create a fresh connection for RPC thread-safety
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        result = channel.queue_declare(queue='', exclusive=True)
        callback_queue = result.method.queue

        response = None

        def on_response(ch, method, props, body):
            nonlocal response
            if props.correlation_id == corr_id:
                try:
                    response = json.loads(body)
                except Exception:
                    response = {"error": "Failed to parse response JSON"}

        channel.basic_consume(
            queue=callback_queue,
            on_message_callback=on_response,
            auto_ack=True
        )

        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=corr_id,
                content_type="application/json"
            ),
            body=json.dumps(message)
        )

        # Wait for response
        start_time = time.time()
        while response is None:
            connection.process_data_events(time_limit=1)
            if time.time() - start_time > timeout:
                break

        connection.close()

        if response is None:
            raise TimeoutError(f"RPC call to {routing_key} timed out after {timeout}s")

        return response

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()

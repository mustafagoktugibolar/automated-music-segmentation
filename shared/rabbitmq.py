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

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()

import json
import logging


from backend.services.dataset_worker import get_available_songs
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient

logger = get_logger()

class DatasetWorker:
    def __init__(self):
        self.rabbitmq = RabbitMQClient(service_name="dataset_worker")
        self.rabbitmq.connect()

    def start(self):
        logger.info("[DatasetWorker] Starting to consume dataset queues...")
        
        # We need to bind to specific dataset routing keys
        self.rabbitmq.channel.queue_declare(queue="dataset_queue", durable=True)
        self.rabbitmq.channel.queue_bind(exchange="segmentation_topic", queue="dataset_queue", routing_key="dataset.list_musics")
        self.rabbitmq.channel.queue_bind(exchange="segmentation_topic", queue="dataset_queue", routing_key="dataset.get_music")
        
        self.rabbitmq.channel.basic_qos(prefetch_count=1)
        self.rabbitmq.channel.basic_consume(queue="dataset_queue", on_message_callback=self.on_message)
        
        logger.info("[DatasetWorker] Ready.")
        try:
            self.rabbitmq.channel.start_consuming()
        except KeyboardInterrupt:
            self.rabbitmq.close()

    def on_message(self, ch, method, properties, body):
        try:
            req_data = json.loads(body)
            routing_key = method.routing_key
            logger.info(f"[DatasetWorker] Received RPC request on {routing_key}")
            
            response = {"error": "Unknown request"}
            
            if routing_key == "dataset.list_musics":
                songs = get_available_songs()
                response = {"songs": [{"song_id": s.song_id, "url": s.archive_path} for s in songs]}
            elif routing_key == "dataset.get_music":
                song_id = req_data.get("song_id")
                if not song_id:
                    response = {"error": "song_id is missing"}
                else:
                    songs = get_available_songs()
                    target_song = next((s for s in songs if s.song_id == song_id), None)
                    if target_song:
                        response = {"song_id": target_song.song_id, "location": target_song.archive_path}
                    else:
                        response = {"error": f"Song {song_id} not found."}

            if properties.reply_to and properties.correlation_id:
                # Send RPC response
                ch.basic_publish(
                    exchange='',
                    routing_key=properties.reply_to,
                    properties=import_pika_props(properties.correlation_id),
                    body=json.dumps(response)
                )
                logger.debug(f"[DatasetWorker] Sent RPC response to {properties.reply_to}")
            
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"[DatasetWorker] Error processing message: {str(e)}", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def import_pika_props(correlation_id):
    import pika
    return pika.BasicProperties(correlation_id=correlation_id, content_type="application/json")

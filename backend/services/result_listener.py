import os
import threading

import requests

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.db.models import Base, SegmentationTask
from shared.config import DBSettings
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient
from shared.segmentation_utils import (
    BASELINE_ALGORITHMS,
    canonical_algorithm_name,
    extract_segments,
    normalize_algorithm_result,
)

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
            return "custom_librosa"
        return canonical_algorithm_name(algorithm or "default")

    def _visible_result_keys(self, current_results: dict) -> set[str]:
        return {
            str(key).lower().strip()
            for key, value in (current_results or {}).items()
            if "__" not in str(key)
            and isinstance(value, list)
        }

    def _normalized_result(self, data: dict, key: str) -> dict:
        if isinstance(data.get("segments"), list) and data.get("algorithm") and "boundaries" in data:
            normalized = dict(data)
            normalized["algorithm"] = canonical_algorithm_name(normalized.get("algorithm") or key)
            normalized["segments"] = extract_segments(normalized)
            normalized.setdefault("diagnostics", {})
            return normalized

        return normalize_algorithm_result(
            task_id=data.get("task_id"),
            status=data.get("status") or "completed",
            worker_type=data.get("worker_type") or key,
            algorithm=data.get("algorithm") or key,
            duration_seconds=data.get("duration_seconds"),
            boundaries=data.get("boundaries") or data.get("candidate_boundaries") or [],
            segments=data.get("segments") or [],
            diagnostics=data.get("diagnostics") or {},
        )

    def _maybe_dispatch_fusion(self, task: SegmentationTask, current_results: dict) -> bool:
        expected = {canonical_algorithm_name(a) for a in (task.expected_algorithms or [])}
        if "fusion" not in expected:
            return False
        if "fusion" in self._visible_result_keys(current_results):
            return False
        if current_results.get("fusion__dispatched"):
            return False

        base_results: dict[str, dict] = {}
        resolved_algorithms: set[str] = set()
        failed_algorithms: list[str] = []
        for algorithm in BASELINE_ALGORITHMS:
            full = current_results.get(f"{algorithm}__result")
            if full:
                resolved_algorithms.add(algorithm)
                if full.get("status") == "completed" and extract_segments(full):
                    base_results[algorithm] = full
                else:
                    failed_algorithms.append(algorithm)
            elif isinstance(current_results.get(algorithm), list):
                resolved_algorithms.add(algorithm)
                base_results[algorithm] = normalize_algorithm_result(
                    task_id=task.task_id,
                    status="completed",
                    worker_type="result_listener",
                    algorithm=algorithm,
                    duration_seconds=None,
                    boundaries=[],
                    segments=current_results[algorithm],
                    diagnostics={"warning": "Reconstructed fusion input from legacy segment list."},
                )

        if len(resolved_algorithms) < len(BASELINE_ALGORITHMS):
            logger.info(
                "Fusion for task %s waiting for all baseline results. Have=%s/%s resolved",
                task.task_id,
                len(resolved_algorithms),
                len(BASELINE_ALGORITHMS),
            )
            return False

        if len(base_results) < 2:
            failure = normalize_algorithm_result(
                task_id=task.task_id,
                status="failed",
                worker_type="fusion",
                algorithm="fusion",
                duration_seconds=None,
                boundaries=[],
                segments=[],
                diagnostics={
                    "error": "Fusion requires at least two successful base algorithm results.",
                    "successful_algorithms": sorted(base_results),
                    "failed_or_missing_algorithms": failed_algorithms,
                },
            )
            current_results["fusion"] = []
            current_results["fusion__result"] = failure
            current_results["fusion__diagnostics"] = failure["diagnostics"]
            current_results["fusion__dispatched"] = True
            logger.error("Fusion failed for %s: fewer than two base results succeeded", task.task_id)
            return True

        upload_dir = os.getenv("UPLOAD_DIR", "media/uploads")
        audio_source: dict = {}
        if task.source_type == "upload":
            audio_source = {
                "source_type": "upload",
                "file_path": os.path.join(upload_dir, f"{task.task_id}_{task.filename}"),
            }
        elif task.source_type == "storage" and task.source_song_id:
            audio_source = {
                "source_type": "storage",
                "blob_name": f"songs/{task.source_song_id}.mp3",
            }

        payload = {
            "task_id": task.task_id,
            "algorithm": "fusion",
            "worker_type": "fusion",
            "algorithm_results": base_results,
            "params": {
                **((task.requested_params or {}).get("fusion") or {}),
                "failed_or_missing_algorithms": failed_algorithms,
            },
            **audio_source,
        }
        self.rabbitmq.publish(
            exchange="segmentation_topic",
            routing_key="segmentation.fusion",
            message=payload,
        )
        current_results["fusion__dispatched"] = True
        logger.info("Dispatched fusion task for %s with base results %s", task.task_id, sorted(base_results))
        return True

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

            key = self._result_key(worker_type, algorithm)
            logger.info(f"Received result for task {task_id} from {key}")
            normalized = self._normalized_result(data, key)
            segments = normalized.get("segments", [])

            with Session(self.engine) as session:
                task = session.query(SegmentationTask).filter(SegmentationTask.task_id == task_id).first()

                if not task:
                    task = SegmentationTask(task_id=task_id, status="processing")
                    session.add(task)

                current_results = dict(task.results) if task.results else {}
                current_results[key] = segments
                current_results[f"{key}__result"] = normalized
                if normalized.get("diagnostics"):
                    current_results[f"{key}__diagnostics"] = normalized["diagnostics"]
                if normalized.get("boundaries"):
                    current_results[f"{key}__boundaries"] = normalized["boundaries"]
                if normalized.get("duration_seconds") is not None:
                    current_results[f"{key}__duration_seconds"] = normalized["duration_seconds"]
                # Preserve LLM-specific metadata so the frontend can display
                # the agent explanation alongside the standard segments list.
                if data.get("agent_explanation"):
                    current_results[f"{key}__explanation"] = data["agent_explanation"]
                if data.get("evaluation"):
                    current_results[f"{key}__evaluation"] = data["evaluation"]
                if data.get("processing_time_seconds") is not None:
                    current_results[f"{key}__processing_time"] = data["processing_time_seconds"]
                self._maybe_dispatch_fusion(task, current_results)
                task.results = current_results

                expected = {canonical_algorithm_name(a) for a in (task.expected_algorithms or [])}
                received = self._visible_result_keys(current_results)

                if expected and expected.issubset(received):
                    task.status = "completed"
                    logger.info(f"Task {task_id} COMPLETED. All expected results received: {received}")
                else:
                    task.status = "processing"
                    logger.info(f"Task {task_id} processing. Received: {received}, Expected: {expected}")
                
                session.commit()
                logger.info(f"Updated DB for task {task_id}")

                # Push update to SSE after commit so frontend refetches fresh data
                self._call_webhook(task)

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            logger.error("Failed to process result", exc_info=True)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiofiles

from backend.api.schemas import ALLOWED_ALGORITHMS, SegmentationParams
from backend.db.models import SegmentationTask
from backend.db.postgreSQL import SessionLocal
from shared.blob_helper import AzureBlobCacheHelper
from shared.logger import get_logger
from shared.rabbitmq import RabbitMQClient
from shared.segmentation_utils import BASELINE_ALGORITHMS, canonical_algorithm_name

logger = get_logger()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "media/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@dataclass
class SongInfo:
    song_id: str
    url: str


class SegmentationOrchestrator:
    def __init__(self):
        self.rabbitmq = RabbitMQClient(service_name="segmentation_orchestrator")
        self.algo_to_routing_key = {
            "custom_librosa": "segmentation.custom",
            "foote": "segmentation.foote",
            "cnmf": "segmentation.cnmf",
            "scluster": "segmentation.scluster",
            "fusion": "segmentation.fusion",
            "llm": "segmentation.llm",
        }
        self._blob_helper = None

    @property
    def azure_container(self) -> str:
        container = os.getenv("AZURE_STORAGE_CONTAINER_RAW", "").strip()
        if not container:
            raise RuntimeError("AZURE_STORAGE_CONTAINER_RAW is not configured")
        return container

    def _get_blob_helper(self) -> AzureBlobCacheHelper:
        if self._blob_helper is None:
            self._blob_helper = AzureBlobCacheHelper()
        return self._blob_helper

    def _normalize_algorithms(self, requested_algos: list[str]) -> list[str]:
        normalized: list[str] = []
        for algo in requested_algos:
            candidate = canonical_algorithm_name(algo)
            if candidate in ALLOWED_ALGORITHMS and candidate not in normalized:
                normalized.append(candidate)
            else:
                logger.warning(f"Unknown or duplicate algorithm requested: {algo}")

        if not normalized:
            raise ValueError("No valid algorithms specified.")

        return normalized

    def _expand_requested_algorithms(self, algorithms: list[str]) -> tuple[list[str], list[str]]:
        expected = list(algorithms)
        dispatch = [a for a in algorithms if a != "fusion"]

        if "fusion" in algorithms:
            for base_algo in BASELINE_ALGORITHMS:
                if base_algo not in expected:
                    expected.insert(0, base_algo)
                if base_algo not in dispatch:
                    dispatch.append(base_algo)

        return expected, dispatch

    def _validate_and_trim_params(self, params: SegmentationParams | None, algorithms: list[str]) -> dict:
        payload = params.model_dump(exclude_none=True) if params else {}
        if not payload:
            return {}

        custom_requested = "custom_librosa" in algorithms or "fusion" in algorithms
        if "custom" in payload and not custom_requested:
            logger.warning("Ignoring custom params because custom_librosa algorithm was not requested")
            payload.pop("custom", None)
        if "custom_librosa" in payload and not custom_requested:
            logger.warning("Ignoring custom_librosa params because custom_librosa algorithm was not requested")
            payload.pop("custom_librosa", None)

        msaf_requested = any(a in {"foote", "cnmf", "scluster"} for a in algorithms) or "fusion" in algorithms
        if "msaf" in payload and not msaf_requested:
            logger.warning("Ignoring msaf params because no MSAF algorithm was requested")
            payload.pop("msaf", None)

        llm_requested = "llm" in algorithms
        if "llm_segmentation" in payload and not llm_requested:
            logger.warning("Ignoring llm_segmentation params because llm algorithm was not requested")
            payload.pop("llm_segmentation", None)

        if "fusion" in payload and "fusion" not in algorithms:
            logger.warning("Ignoring fusion params because fusion algorithm was not requested")
            payload.pop("fusion", None)

        return payload

    def _create_task_record(
        self,
        *,
        task_id: str,
        filename: str,
        expected_algorithms: list[str],
        source_type: str,
        source_song_id: str | None,
        requested_params: dict,
        webhook_url: str | None = None,
    ) -> None:
        db = SessionLocal()
        try:
            new_task = SegmentationTask(
                task_id=task_id,
                filename=filename,
                status="processing",
                results={},
                expected_algorithms=[a.lower() for a in expected_algorithms],
                source_type=source_type,
                source_song_id=source_song_id,
                requested_params=requested_params or None,
                webhook_url=webhook_url,
            )
            db.add(new_task)
            db.commit()
        except Exception:
            db.rollback()
            logger.error("Failed to save initial task to DB", exc_info=True)
            raise RuntimeError("Database error during task creation")
        finally:
            db.close()

    def _publish_tasks(self, task_payload: dict, algorithms: list[str]) -> None:
        target_keys = [self.algo_to_routing_key[a] for a in algorithms]
        logger.info(f"Distributing tasks for {task_payload['task_id']} to workers: {target_keys}")
        for key in target_keys:
            self.rabbitmq.publish(
                exchange="segmentation_topic",
                routing_key=key,
                message=task_payload,
            )

    def list_available_songs(self) -> list[SongInfo]:
        from backend.services.dataset_worker import get_available_songs
        
        songs_data = get_available_songs()
        songs: list[SongInfo] = []

        for s in songs_data:
            songs.append(SongInfo(song_id=s.song_id, url=s.archive_path))

        songs.sort(key=lambda s: s.song_id)
        return songs

    async def process_upload(self, file, requested_algos: list[str], params: SegmentationParams | None = None, webhook_url: str | None = None) -> str:
        algorithms = self._normalize_algorithms(requested_algos)
        expected_algorithms, dispatch_algorithms = self._expand_requested_algorithms(algorithms)
        effective_params = self._validate_and_trim_params(params, algorithms)

        task_id = str(uuid.uuid4())
        filename = f"{task_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        try:
            logger.info(f"Saving uploaded file to {file_path}")
            async with aiofiles.open(file_path, "wb") as out_file:
                while content := await file.read(1024 * 1024):
                    await out_file.write(content)

            self._create_task_record(
                task_id=task_id,
                filename=file.filename,
                expected_algorithms=expected_algorithms,
                source_type="upload",
                source_song_id=None,
                requested_params=effective_params,
                webhook_url=webhook_url,
            )

            task_payload = {
                "task_id": task_id,
                "source_type": "upload",
                "original_filename": file.filename,
                "file_path": file_path,
                "content_type": file.content_type,
                "algorithms": algorithms,
                "params": effective_params,
            }
            self._publish_tasks(task_payload, dispatch_algorithms)
            return task_id

        except Exception:
            logger.error("Orchestration failed for upload flow", exc_info=True)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            raise

    async def process_from_storage(
        self,
        song_id: str,
        requested_algos: list[str],
        params: SegmentationParams | None = None,
    ) -> str:
        song_id = song_id.strip()
        if not song_id:
            raise ValueError("song_id is required")

        algorithms = self._normalize_algorithms(requested_algos)
        expected_algorithms, dispatch_algorithms = self._expand_requested_algorithms(algorithms)
        effective_params = self._validate_and_trim_params(params, algorithms)

        blob_name = f"songs/{song_id}.mp3"
        helper = self._get_blob_helper()
        exists = helper.blob_exists(self.azure_container, blob_name)
        if not exists:
            raise FileNotFoundError(f"Song not found in storage for song_id={song_id}")

        task_id = str(uuid.uuid4())
        self._create_task_record(
            task_id=task_id,
            filename=f"{song_id}.mp3",
            expected_algorithms=expected_algorithms,
            source_type="storage",
            source_song_id=song_id,
            requested_params=effective_params,
        )

        task_payload = {
            "task_id": task_id,
            "source_type": "storage",
            "song_id": song_id,
            "blob_name": blob_name,
            "content_type": "audio/mpeg",
            "algorithms": algorithms,
            "params": effective_params,
        }
        self._publish_tasks(task_payload, dispatch_algorithms)
        return task_id

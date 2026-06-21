import os
import time

import numpy as np

from shared.logger import get_logger
from shared.segmentation_utils import normalize_algorithm_result
from workers.BaseWorker import BaseWorker

logger = get_logger()

_SR = 22050


def _warmup_pipeline() -> None:
    """Force BLAS/scipy/librosa lazy-init before the first real task arrives.

    Without this, the first call to librosa.feature.rms (active-region step)
    triggers BLAS library loading inside Docker, which adds ~8-10s to that
    step only. A 5-second sine wave is enough to exercise the full path.
    """
    from workers.segmenters.segmentation_service import _detect_active_region, _extract_downsampled_features

    t0 = time.perf_counter()
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 5, _SR * 5, dtype=np.float32))
    _detect_active_region(y, _SR)
    _extract_downsampled_features(y, _SR)
    logger.info("[worker-custom] Warmup completed in %.2fs", time.perf_counter() - t0)


class CustomWorker(BaseWorker):
    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.custom")

        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "custom-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code],
        )
        _warmup_pipeline()

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        all_params = task.get("params") or {}
        params = all_params.get("custom_librosa") or all_params.get("custom") or {}

        file_path = self._resolve_file_path(task)
        logger.info(f"Received Custom segmentation task for file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File {file_path} does not exist")

        from workers.segmenters.segmentation_service import process_file_path

        result = process_file_path(file_path, params=params)
        logger.info(f"Custom segmentation finished for {task_id}")

        boundaries = result.get("boundaries") or result.get("candidate_boundaries") or []
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics.setdefault("segmenter", "custom_librosa")
        diagnostics.setdefault("fusion_level", "feature")
        diagnostics.setdefault("feature_candidate_boundaries", result.get("candidate_boundaries", []))
        if result.get("estimated_bpm") is not None:
            diagnostics.setdefault("estimated_bpm", result["estimated_bpm"])

        response = normalize_algorithm_result(
            task_id=task_id,
            status="completed",
            worker_type="custom",
            algorithm="custom_librosa",
            duration_seconds=result.get("duration_seconds"),
            boundaries=boundaries,
            segments=result.get("segments", []),
            diagnostics=diagnostics,
        )
        for key in ("estimated_bpm", "candidate_boundaries"):
            if key in result:
                response[key] = result[key]
        return response

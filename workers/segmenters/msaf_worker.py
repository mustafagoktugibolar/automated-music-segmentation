import os
import time
from typing import Any

import msaf

from shared.labeling import apply_two_layer_labels
from shared.logger import get_logger
from shared.segmentation_utils import (
    boundaries_to_segments,
    get_audio_duration,
    normalize_algorithm_result,
    normalize_boundaries,
)
from workers.BaseWorker import BaseWorker

logger = get_logger()


class MSAFWorker(BaseWorker):
    SUPPORTED_BOUNDARIES = {"foote", "cnmf", "scluster"}

    def __init__(self):
        message_code = os.getenv("MESSAGE_CODE", "segmentation.request.msaf")
        self.algorithm = os.getenv("MSAF_ALGORITHM", "foote").lower().strip()

        media_root = os.getenv("MEDIA_ROOT", "/app/media")
        os.makedirs(os.path.join(media_root, "features"), exist_ok=True)
        os.makedirs(os.path.join(media_root, "estimations"), exist_ok=True)

        super().__init__(
            service_name=os.getenv("SERVICE_NAME", "msaf-worker"),
            queue_name=f"queue_{message_code}",
            routing_keys=[message_code],
        )

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        msaf_params = (task.get("params") or {}).get("msaf") or {}

        t_total = time.perf_counter()
        file_path = self._resolve_file_path(task)
        logger.info("[%s] START algo=%s file=%s", task_id, self.algorithm, file_path)

        if self.algorithm not in self.SUPPORTED_BOUNDARIES:
            raise ValueError(f"Unsupported MSAF boundary algorithm: {self.algorithm}")
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        t0 = time.perf_counter()
        duration_seconds = get_audio_duration(file_path)
        logger.info("[%s][%.2fs] audio duration=%.1fs", task_id, time.perf_counter() - t0, duration_seconds)

        diagnostics: dict[str, Any] = {
            "msaf_boundaries_id": self.algorithm,
            "duration_seconds": round(duration_seconds, 3),
            "warnings": [],
        }

        try:
            process_kwargs: dict[str, Any] = {"boundaries_id": self.algorithm}
            labels_id = msaf_params.get("labeling_id")
            if labels_id:
                process_kwargs["labels_id"] = labels_id
                diagnostics["msaf_labels_id"] = labels_id
            if msaf_params.get("hier") is not None:
                process_kwargs["hier"] = bool(msaf_params["hier"])
                diagnostics["hier"] = bool(msaf_params["hier"])

            logger.info("[%s] msaf.process START kwargs=%s", task_id, process_kwargs)
            t0 = time.perf_counter()
            est_times, est_labels = msaf.process(file_path, **process_kwargs)
            t_msaf = time.perf_counter() - t0
            logger.info("[%s][%.2fs] msaf.process DONE boundaries=%d labels=%d",
                        task_id, t_msaf,
                        len(est_times) if est_times is not None else 0,
                        len(est_labels) if est_labels is not None else 0)

            raw_times = [float(t) for t in list(est_times) if t is not None] if est_times is not None else []
            raw_labels = list(est_labels) if est_labels is not None else []

            diagnostics["raw_est_times_count"] = len(raw_times)
            diagnostics["raw_est_labels_count"] = len(raw_labels)
            diagnostics["msaf_process_kwargs"] = process_kwargs
            diagnostics["timing_msaf_process_seconds"] = round(t_msaf, 3)

            t0 = time.perf_counter()
            boundaries = normalize_boundaries(
                raw_times,
                duration_seconds,
                min_gap_seconds=float(msaf_params.get("min_boundary_gap_seconds", 0.25)),
                include_edges=True,
            )
            if boundaries and boundaries[0] != 0.0:
                diagnostics["warnings"].append("Inserted missing 0.0 boundary.")
            if duration_seconds and boundaries and abs(boundaries[-1] - duration_seconds) > 0.25:
                diagnostics["warnings"].append("Inserted or corrected final duration boundary.")

            segment_count = max(0, len(boundaries) - 1)
            labels = _normalize_msaf_labels(raw_labels, segment_count, diagnostics)
            segments = boundaries_to_segments(
                boundaries,
                duration_seconds,
                labels=labels,
                min_gap_seconds=0.1,
                boundary_metadata=[
                    {
                        "time": t,
                        "source": self.algorithm,
                        "sources": [self.algorithm],
                        "confidence": 1.0,
                    }
                    for t in boundaries
                ],
            )
            for idx, seg in enumerate(segments):
                raw_label = labels[idx] if idx < len(labels) else None
                if raw_label is not None:
                    seg["raw_msaf_label"] = raw_label
            logger.info("[%s][%.2fs] boundary normalization + segment build segs=%d",
                        task_id, time.perf_counter() - t0, len(segments))

            logger.info("[%s] apply_two_layer_labels START (audio reload for descriptors)", task_id)
            t0 = time.perf_counter()
            segments = apply_two_layer_labels(
                segments,
                file_path=file_path,
                duration_seconds=duration_seconds,
                semantic_enabled=bool(msaf_params.get("semantic_labeling_enabled", True)),
                method_hint="feature_clustering",
            )
            t_labels = time.perf_counter() - t0
            logger.info("[%s][%.2fs] apply_two_layer_labels DONE", task_id, t_labels)
            diagnostics["timing_labeling_seconds"] = round(t_labels, 3)

            diagnostics["normalized_boundary_count"] = len(boundaries)
            diagnostics["normalized_segment_count"] = len(segments)
            diagnostics["label_policy"] = "MSAF labels retained as raw_msaf_label; structural labels assigned separately."

            t_total_elapsed = time.perf_counter() - t_total
            diagnostics["timing_total_seconds"] = round(t_total_elapsed, 3)
            logger.info("[%s] DONE algo=%s total=%.2fs (msaf=%.2fs labels=%.2fs)",
                        task_id, self.algorithm, t_total_elapsed, t_msaf, t_labels)

            return normalize_algorithm_result(
                task_id=task_id,
                status="completed",
                worker_type="msaf",
                algorithm=self.algorithm,
                duration_seconds=duration_seconds,
                boundaries=[
                    {
                        "time": t,
                        "confidence": 1.0,
                        "source": self.algorithm,
                        "sources": [self.algorithm],
                    }
                    for t in boundaries
                ],
                segments=segments,
                diagnostics=diagnostics,
            )

        except Exception:
            logger.error(f"MSAF processing failed for {file_path}", exc_info=True)
            raise


def _normalize_msaf_labels(raw_labels: list[Any], segment_count: int, diagnostics: dict) -> list[str]:
    labels: list[str] = []
    for label in raw_labels[:segment_count]:
        if label is None:
            labels.append("")
        else:
            labels.append(str(label))

    if len(labels) < segment_count:
        diagnostics.setdefault("warnings", []).append(
            f"MSAF label count {len(raw_labels)} did not match segment count {segment_count}; padded labels."
        )
        labels.extend(chr(65 + min(i, 25)) for i in range(len(labels), segment_count))

    cleaned: list[str] = []
    for idx, label in enumerate(labels):
        value = str(label).strip()
        if not value or value.lower() in {"none", "nan", "unknown"}:
            value = chr(65 + min(idx, 25))
        cleaned.append(value)
    return cleaned

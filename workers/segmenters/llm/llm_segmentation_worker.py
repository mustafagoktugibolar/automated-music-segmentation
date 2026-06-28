import os

from workers.base_worker import BaseWorker
from shared.logger import get_logger

logger = get_logger()


class LLMSegmentationWorker(BaseWorker):
    """
    Worker that processes 'segmentation.llm' tasks using the LangChain-based
    AI Music Segmentation Agent.

    Routing key : segmentation.llm
    Queue       : queue_segmentation.llm
    Result key  : llm  (stored in SegmentationTask.results["llm"])
    """

    ROUTING_KEY = "segmentation.llm"
    QUEUE_NAME = "queue_segmentation.llm"

    def __init__(self):
        super().__init__(
            service_name="llm-segmentation-worker",
            queue_name=self.QUEUE_NAME,
            routing_keys=[self.ROUTING_KEY],
        )
        # Lazy import so that missing langchain deps raise a clear error at startup
        # rather than when the first task arrives.
        from workers.segmenters.llm.music_segmentation_agent import SegmentationService  # noqa: PLC0415

        # LLM_PROVIDER and LLM_MODEL_NAME are read from the environment so the
        # same Docker image can be switched between Anthropic and OpenAI by
        # changing env vars in docker-compose.yml or the .env file.
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
        model_name = os.getenv("LLM_MODEL_NAME") or None
        mode = os.getenv("LLM_SEGMENTATION_MODE", "deterministic").lower().strip()

        self.service = SegmentationService(provider=provider, model_name=model_name, mode=mode)
        logger.info("LLMSegmentationWorker ready: provider=%s, model=%s, mode=%s", provider, model_name, mode)

    def process_task(self, task: dict) -> dict:
        task_id = task.get("task_id")
        logger.info(f"[llm-segmentation-worker] Processing task {task_id}")

        file_path = self._resolve_file_path(task)

        # Pull llm_segmentation sub-params; fall back to empty dict.
        params = (task.get("params") or {}).get("llm_segmentation") or {}
        mode = str(params.get("mode") or os.getenv("LLM_SEGMENTATION_MODE", "deterministic")).lower().strip()

        result = self.service.segment_audio_dict(
            file_path=file_path,
            track_id=task.get("track_id"),
            salami_annotation_path=task.get("salami_annotation_path"),
            timed_lyrics=task.get("timed_lyrics"),
            params={k: v for k, v in (task.get("params") or {}).items() if k != "llm_segmentation"},
            mode=mode,
        )

        # Convert PredictedSegment dicts to the standard format consumed by the
        # evaluation comparison endpoint (expects start/end as floats + label).
        segments = [
            {
                "start": s.get("start_seconds", 0.0),
                "end": s.get("end_seconds", 0.0),
                "label": s.get("label", ""),
                "section_type": s.get("label", ""),
                "confidence": s.get("confidence", 0.0),
                "source_features": s.get("source_features", []),
                "reason": s.get("reason", ""),
            }
            for s in (result.get("predicted_segments") or [])
        ]

        # "algorithm": "llm" must match the algorithm ID used in the UI so that
        # ResultListener stores the result under results["llm"] and the
        # allRequestedResultsPresent check in the frontend passes.
        return {
            "task_id": task_id,
            "status": "completed",
            "worker_type": "llm_segmentation",
            "algorithm": "llm",
            "segments": segments,
            "estimated_bpm": result.get("estimated_bpm"),
            "candidate_boundaries": result.get("candidate_boundaries", []),
            "evaluation": result.get("evaluation"),
            # Stored separately by ResultListener as results["llm__explanation"]
            "agent_explanation": result.get("agent_explanation", ""),
        }

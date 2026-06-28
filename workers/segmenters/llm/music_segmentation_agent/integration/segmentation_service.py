"""
Public integration wrapper for the LangChain-based segmentation agent.

This is the primary entry point for code outside the music_segmentation_agent
package. It wraps SegmentationAgent with a stable, JSON-serialisable interface
compatible with the existing worker infrastructure.

Usage
-----
    from workers.segmenters.llm.music_segmentation_agent import SegmentationService

    service = SegmentationService()
    result = service.segment_audio(
        file_path="/data/audio/track_123.mp3",
        track_id="salami_123",
        salami_annotation_path="/data/salami/123/annotations/annotator1.txt",
    )
    print(result.model_dump())

    # Or for plain-dict output (worker / RabbitMQ serialisation):
    payload = service.segment_audio_dict(file_path="/data/audio/track_123.mp3")
"""

from __future__ import annotations

from ..agent.segmentation_agent import SegmentationAgent
from ..core.models import SegmentationResult
from shared.logger import get_logger

logger = get_logger("segmentation_service")


class SegmentationService:
    """
    High-level wrapper around SegmentationAgent.

    Provides two public methods:
    - ``segment_audio``      → returns a SegmentationResult Pydantic model.
    - ``segment_audio_dict`` → returns a plain dict (JSON-serialisable).

    Parameters
    ----------
    provider       : LLM provider — ``"anthropic"`` (default) or ``"openai"``.
    model_name     : Model ID override. Defaults to provider's default model.
    default_params : Default pipeline parameters applied to every call
                     (can be overridden per-call via the *params* argument).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model_name: str | None = None,
        mode: str = "deterministic",
        default_params: dict | None = None,
    ) -> None:
        # Some older versions of SegmentationAgent did not accept a `mode`
        # keyword. Try constructing with `mode` first, and fall back to the
        # legacy signature if needed for backwards compatibility.
        try:
            self.agent = SegmentationAgent(provider=provider, model_name=model_name, mode=mode)
        except TypeError:
            self.agent = SegmentationAgent(provider=provider, model_name=model_name)
        self.default_params: dict = default_params or {}
        logger.info(
            "SegmentationService initialised: provider=%s, model=%s, mode=%s, default_params=%s",
            provider,
            model_name,
            mode,
            self.default_params,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def segment_audio(
        self,
        file_path: str,
        track_id: str | None = None,
        salami_annotation_path: str | None = None,
        timed_lyrics: list | None = None,
        params: dict | None = None,
        mode: str | None = None,
    ) -> SegmentationResult:
        """
        Segment an audio file and return a SegmentationResult.

        Parameters
        ----------
        file_path              : Absolute path to the audio file.
        track_id               : Optional track identifier (SALAMI ID, DB key, etc.).
        salami_annotation_path : Path to a SALAMI annotation file for evaluation.
                                 If None, evaluation metrics will be zero.
        timed_lyrics           : Optional list of {time_seconds, text} dicts.
        params                 : Pipeline parameter overrides (merged with
                                 self.default_params; call-level params win).

        Returns
        -------
        SegmentationResult
        """
        merged_params = {**self.default_params, **(params or {})}
        logger.info(
            "segment_audio: file=%s, track_id=%s", file_path, track_id
        )
        # Call the agent.run method. Some older agent versions do not accept
        # a `mode` keyword, so try with it first and fall back to the
        # legacy signature if a TypeError is raised.
        try:
            return self.agent.run(
                file_path=file_path,
                track_id=track_id,
                salami_annotation_path=salami_annotation_path,
                timed_lyrics=timed_lyrics,
                params=merged_params,
                mode=mode,
            )
        except TypeError:
            # Legacy SegmentationAgent.run(signature) without `mode`
            logger.info("SegmentationAgent.run() does not accept 'mode'; calling without it for compatibility.")
            return self.agent.run(
                file_path=file_path,
                track_id=track_id,
                salami_annotation_path=salami_annotation_path,
                timed_lyrics=timed_lyrics,
                params=merged_params,
            )

    def segment_audio_dict(
        self,
        file_path: str,
        track_id: str | None = None,
        salami_annotation_path: str | None = None,
        timed_lyrics: list | None = None,
        params: dict | None = None,
        mode: str | None = None,
    ) -> dict:
        """
        Segment an audio file and return a plain dict.

        Same signature as ``segment_audio`` but serialises the result to a
        regular Python dict, suitable for JSON encoding or message-queue dispatch.

        Returns
        -------
        dict — result of SegmentationResult.model_dump().
        """
        result = self.segment_audio(
            file_path=file_path,
            track_id=track_id,
            salami_annotation_path=salami_annotation_path,
            timed_lyrics=timed_lyrics,
            params=params,
            mode=mode,
        )
        return result.model_dump()

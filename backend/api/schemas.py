from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ALLOWED_ALGORITHMS = ("custom", "foote", "cnmf", "scluster", "llm")


class TimedLyricLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)


class CustomSegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_segment_duration_seconds: float | None = Field(default=None, gt=0, le=120)
    novelty_kernel_size_seconds: float | None = Field(default=None, gt=0, le=30)
    n_clusters: int | None = Field(default=None, ge=1, le=26)
    # Feature extraction
    use_mfcc: bool | None = None
    mfcc_n_components: int | None = Field(default=None, ge=1, le=40)
    # Novelty blending
    spectral_flux_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_weights: dict[str, float] | None = None
    timed_lyrics: list[TimedLyricLine] | None = None
    return_diagnostics: bool | None = None
    # Clustering
    auto_n_clusters: bool | None = None
    # New: beat-sync and SSM enhancement controls
    use_beat_sync: bool | None = None
    smoothing_L: int | None = Field(default=None, ge=1, le=100)
    transposition_invariant: bool | None = None


class MSAFSegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labeling_id: str | None = None
    hier: bool | None = None


class LLMSegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["deterministic", "ai_generated"] | None = Field(default=None)


class SegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    custom: CustomSegmentationParams | None = None
    msaf: MSAFSegmentationParams | None = None
    llm_segmentation: LLMSegmentationParams | None = None


class StorageSegmentationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song_id: str = Field(min_length=1)
    algorithms: list[Literal["custom", "foote", "cnmf", "scluster", "llm"]] = Field(min_length=1)
    params: SegmentationParams | None = None


class SongItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song_id: str
    blob_name: str

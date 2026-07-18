from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from segmentation.core.segmentation.utils import ALLOWED_ALGORITHMS


class TimedLyricLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)


class CustomSegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_segment_duration_seconds: float | None = Field(default=None, gt=0, le=120)
    novelty_kernel_size_seconds: float | None = Field(default=None, gt=0, le=30)
    target_fps: float | None = Field(default=None, gt=0, le=50)
    novelty_prominence: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_density_seconds: float | None = Field(default=None, gt=0, le=120)
    feature_fusion_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_fusion_merge_window_seconds: float | None = Field(default=None, gt=0, le=20)
    semantic_labeling_enabled: bool | None = None
    labeling_method: Literal["heuristic", "ml", "ml_sequence"] | None = None
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
    semantic_labeling_enabled: bool | None = None
    min_boundary_gap_seconds: float | None = Field(default=None, gt=0, le=10)


class FusionSegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merge_window_seconds: float | None = Field(default=None, gt=0, le=30)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_segment_duration_seconds: float | None = Field(default=None, gt=0, le=120)
    anchor_strategy: Literal["weighted_mean", "custom_snap"] | None = "weighted_mean"
    required_vote_count: int | None = Field(default=None, ge=1, le=4)
    semantic_labeling_enabled: bool | None = None
    weights: dict[str, float] | None = None


class SegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    custom: CustomSegmentationParams | None = None
    custom_librosa: CustomSegmentationParams | None = None
    msaf: MSAFSegmentationParams | None = None
    fusion: FusionSegmentationParams | None = None


class StorageSegmentationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song_id: str = Field(min_length=1)
    algorithms: list[Literal["custom", "custom_librosa", "foote", "cnmf", "scluster", "fusion"]] = Field(min_length=1)
    params: SegmentationParams | None = None


class SongItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    song_id: str
    blob_name: str

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class SegmentationTask(Base):
    __tablename__ = "segmentation_tasks"

    task_id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String, default="pending")
    source_type = Column(String, default="upload")
    source_song_id = Column(String, nullable=True)
    requested_params = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    results = Column(JSON, default={})
    expected_algorithms = Column(JSON, default=[])
    webhook_url = Column(String, nullable=True)


class Algorithm(Base):
    """User-created segmentation algorithm code, versioned by name."""

    __tablename__ = "algorithms"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_algorithm_name_version"),)

    algorithm_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    code = Column(Text, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    params_schema = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Dataset(Base):
    """Registry of datasets (SALAMI built-in or custom uploads)."""

    __tablename__ = "datasets"

    dataset_id = Column(String, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String, nullable=False)  # 'salami' | 'custom'
    track_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetTrack(Base):
    """Individual audio track within a dataset, optionally with ground truth."""

    __tablename__ = "dataset_tracks"

    track_id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"), nullable=False, index=True)
    song_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    artist = Column(String, nullable=True)
    audio_url = Column(String, nullable=True)
    audio_blob_name = Column(String, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    has_ground_truth = Column(Boolean, default=False, nullable=False)
    ground_truth = Column(JSON, nullable=True)  # [{start, end, label}, ...]
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EvaluationRun(Base):
    """Stored boundary detection evaluation result for an algorithm on a track."""

    __tablename__ = "evaluation_runs"

    eval_id = Column(String, primary_key=True, index=True)
    algorithm_name = Column(String, nullable=False, index=True)
    track_id = Column(String, ForeignKey("dataset_tracks.track_id"), nullable=True, index=True)
    task_id = Column(String, ForeignKey("segmentation_tasks.task_id"), nullable=True)
    tolerance_seconds = Column(Float, default=3.0, nullable=False)
    metrics = Column(JSON, nullable=False)  # {precision, recall, f_measure, n_ref, n_est}
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BatchEvalJob(Base):
    """Persisted batch evaluation run with aggregated results."""

    __tablename__ = "batch_eval_jobs"

    job_id = Column(String, primary_key=True, index=True)
    status = Column(String, default="running", nullable=False)  # running | completed | failed
    max_tracks = Column(Integer, nullable=False)
    tolerance_seconds = Column(Float, nullable=False)
    concurrency = Column(Integer, nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    summary = Column(Text, nullable=True)
    rows = Column(JSON, default=[])  # [{song_id, title, n_ref, n_est, precision, recall, f_measure, error}]
    error = Column(String, nullable=True)
    tracks_ok = Column(Integer, nullable=True)
    tracks_total = Column(Integer, nullable=True)
    avg_precision = Column(Float, nullable=True)
    avg_recall = Column(Float, nullable=True)
    avg_f1 = Column(Float, nullable=True)

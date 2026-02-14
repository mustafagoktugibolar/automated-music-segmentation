from sqlalchemy import JSON, Column, DateTime, String
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

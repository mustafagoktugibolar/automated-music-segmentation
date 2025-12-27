from sqlalchemy import Column, String, JSON, Integer, DateTime
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SegmentationTask(Base):
    __tablename__ = "segmentation_tasks"

    task_id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String, default="pending") # pending, processing, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Stores results from different workers:
    # {
    #   "custom": [...segments...],
    #   "foote": [...segments...],
    #   ...
    # }
    results = Column(JSON, default={})
    
    # List of algorithms we expect results from
    # e.g. ["custom", "foote"]
    expected_algorithms = Column(JSON, default=[])

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Boundary:
    time: float
    confidence: float = 1.0
    source: str | None = None
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["source"] is None:
            data.pop("source")
        if not data["sources"]:
            data.pop("sources")
        if not data["metadata"]:
            data.pop("metadata")
        return data


@dataclass
class Segment:
    start: float
    end: float
    label: str = "A"
    structural_label: str | None = None
    semantic_label: str | None = None
    semantic_confidence: float | None = None
    semantic_reason: str | None = None
    confidence: float | None = None
    source_features: list[str] = field(default_factory=list)
    cluster_id: int | None = None
    label_confidence: float | None = None
    label_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in list(data):
            if data[key] is None or data[key] == []:
                data.pop(key)
        data.setdefault("structural_label", data.get("label", "A"))
        data.setdefault("label", data["structural_label"])
        if data.get("semantic_label"):
            data.setdefault("section_type", data["semantic_label"])
        else:
            data.setdefault("section_type", "Unknown")
        return data


@dataclass
class AlgorithmResult:
    task_id: str
    status: str
    worker_type: str
    algorithm: str
    duration_seconds: float | None = None
    boundaries: list[Boundary] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "worker_type": self.worker_type,
            "algorithm": self.algorithm,
            "boundaries": [b.to_dict() for b in self.boundaries],
            "segments": [s.to_dict() for s in self.segments],
            "diagnostics": self.diagnostics or {},
        }
        if self.duration_seconds is not None:
            data["duration_seconds"] = self.duration_seconds
        return data

"""
Rule-based structural and semantic segment labeling.

Pure domain logic — no audio I/O, no heavy external frameworks.
numpy and sklearn (optional, for AgglomerativeClustering) are the only deps.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _letters_for_cluster_ids(cluster_ids: list[int]) -> dict[int, str]:
    counts = Counter(cluster_ids)
    ordered = [cid for cid, _ in counts.most_common()]
    return {cid: chr(65 + idx) for idx, cid in enumerate(ordered)}


def _segment_durations(segments: list[dict]) -> np.ndarray:
    return np.asarray(
        [max(0.0, float(s.get("end", 0.0) or 0.0) - float(s.get("start", 0.0) or 0.0))
         for s in segments],
        dtype=float,
    )


def _cluster_descriptors(descriptors: np.ndarray) -> tuple[np.ndarray, float, str] | None:
    if descriptors is None or len(descriptors) == 0:
        return None
    n = len(descriptors)
    if n < 3:
        return np.zeros(n, dtype=int), 0.55, "fallback"

    X = np.nan_to_num(descriptors.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(X, axis=0, keepdims=True)
    X = (X - np.mean(X, axis=0, keepdims=True)) / np.maximum(std, 1e-8)

    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        best_labels: np.ndarray | None = None
        best_score = -1.0
        upper = min(6, n - 1)
        for k in range(2, upper + 1):
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(X, labels))
            if score > best_score:
                best_score = score
                best_labels = labels
        if best_labels is not None:
            confidence = float(np.clip((best_score + 1.0) / 2.0, 0.35, 0.9))
            return best_labels.astype(int), round(confidence, 3), "feature_clustering"
    except Exception:
        pass

    labels: list[int] = []
    centroids: list[np.ndarray] = []
    threshold = 0.65
    for row in X:
        if not centroids:
            centroids.append(row.copy())
            labels.append(0)
            continue
        sims = []
        for c in centroids:
            denom = float(np.linalg.norm(row) * np.linalg.norm(c))
            sims.append(float(np.dot(row, c) / denom) if denom > 1e-8 else 0.0)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            labels.append(best_idx)
            members = [X[i] for i, lbl in enumerate(labels) if lbl == best_idx]
            centroids[best_idx] = np.mean(members, axis=0)
        else:
            labels.append(len(centroids))
            centroids.append(row.copy())
    return np.asarray(labels, dtype=int), 0.45, "fallback"


def _energy_profile(segments: list[dict], descriptors: np.ndarray | None = None) -> np.ndarray:
    if descriptors is not None and len(descriptors) == len(segments) and descriptors.shape[1] >= 3:
        values = np.linalg.norm(np.nan_to_num(descriptors), axis=1)
    else:
        values = np.asarray(
            [float(s.get("confidence", 0.5) or 0.5) for s in segments], dtype=float
        )
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_structural_labels(
    segments: list[dict],
    descriptors: np.ndarray | None = None,
    file_path: str | None = None,
    method_hint: str | None = None,
) -> list[dict]:
    """Assign stable A/B/C structural labels without semantic claims."""
    if not segments:
        return []
    out = [dict(s) for s in segments]

    if descriptors is None and file_path:
        # Lazy import: infrastructure is an allowed adapter for core convenience.
        from segmentation.infrastructure.audio.features import build_segment_descriptors
        descriptors = build_segment_descriptors(file_path, out)

    clustered = _cluster_descriptors(descriptors) if descriptors is not None else None
    if clustered is None:
        existing = [str(s.get("structural_label") or s.get("label") or "").strip() for s in out]
        if all(label and label.lower() not in {"unknown", "none"} for label in existing):
            counts = Counter(existing)
            if len(counts) > 1:
                order = {
                    label: chr(65 + idx)
                    for idx, (label, _) in enumerate(counts.most_common())
                }
                for seg, label in zip(out, existing):
                    structural = order.get(label, "A")
                    seg["structural_label"] = structural
                    seg["label"] = structural
                    seg.setdefault("label_confidence", 0.5)
                    seg.setdefault("label_method", method_hint or "existing_structural")
                return out

        for idx, seg in enumerate(out):
            structural = chr(65 + min(idx, 25))
            seg["structural_label"] = structural
            seg["label"] = structural
            seg.setdefault("label_confidence", 0.35)
            seg.setdefault("label_method", "fallback")
        return out

    cluster_ids, confidence, method = clustered
    id_to_letter = _letters_for_cluster_ids([int(x) for x in cluster_ids])
    for seg, cid in zip(out, cluster_ids):
        structural = id_to_letter[int(cid)]
        seg["structural_label"] = structural
        seg["label"] = structural
        seg["cluster_id"] = int(cid)
        seg["label_confidence"] = confidence
        seg["label_method"] = method_hint or method
    return out


def assign_semantic_labels(
    segments: list[dict],
    duration_seconds: float | None = None,
    descriptors: np.ndarray | None = None,
    enabled: bool = True,
) -> list[dict]:
    """Conservative semantic labels layered on top of structural labels."""
    out = [dict(s) for s in segments]
    if not out:
        return out
    if not enabled:
        for seg in out:
            seg["semantic_label"] = "Unknown"
            seg["section_type"] = "Unknown"
            seg["semantic_confidence"] = 0.0
            seg["semantic_reason"] = "Semantic labeling disabled."
        return out

    labels = [str(s.get("structural_label") or s.get("label") or "A") for s in out]
    counts = Counter(labels)
    durations = _segment_durations(out)
    energy = _energy_profile(out, descriptors)
    repeated = {label for label, count in counts.items() if count >= 2}
    total_duration = float(
        duration_seconds
        or (max(float(s.get("end", 0.0) or 0.0) for s in out) if out else 0.0)
    )

    _RMS_IDX = 50

    def _check_silence(i: int) -> bool:
        return (
            descriptors is not None
            and i < len(descriptors)
            and descriptors.shape[1] > _RMS_IDX
            and float(descriptors[i, _RMS_IDX]) < 0.005
        )

    _silence_flags = [_check_silence(i) for i in range(len(out))]
    _content_indices = [i for i, s in enumerate(_silence_flags) if not s]
    _first_content = _content_indices[0] if _content_indices else 0
    _last_content = _content_indices[-1] if _content_indices else len(out) - 1

    chorus_label: str | None = None
    if repeated:
        candidates = []
        for label in repeated:
            idxs = [i for i, value in enumerate(labels) if value == label]
            candidates.append((float(np.mean(energy[idxs])), len(idxs), label))

        body_candidates = [
            (e, c, lbl) for e, c, lbl in candidates
            if any(
                i != _first_content and i != _last_content
                for i, v in enumerate(labels)
                if v == lbl
            )
        ]
        if not body_candidates:
            body_candidates = candidates

        best_energy, best_count, best_label = max(body_candidates)
        if best_energy >= 0.40:
            chorus_label = best_label
        else:
            _, _, chorus_label = max(body_candidates, key=lambda c: (c[1], c[0]))

    for idx, seg in enumerate(out):
        structural = labels[idx]
        seg["structural_label"] = structural
        seg["label"] = structural

        duration = durations[idx] if idx < len(durations) else 0.0
        position_start = float(seg.get("start", 0.0) or 0.0) / max(total_duration, 1.0)
        position_end   = float(seg.get("end",   0.0) or 0.0) / max(total_duration, 1.0)
        semantic   = "Unknown"
        confidence = 0.2
        reason     = "Insufficient evidence for a semantic section name."

        if _silence_flags[idx]:
            semantic   = "Silence"
            confidence = 0.75
            reason     = "Very low absolute RMS energy; likely a near-silent section."
        elif idx == _first_content and position_end <= 0.20:
            semantic   = "Intro"
            confidence = 0.65 if counts[structural] == 1 else 0.52
            reason     = "First non-silent structural section near the beginning of the track."
        elif idx == _last_content and position_start >= 0.75:
            semantic   = "Outro"
            confidence = 0.65 if counts[structural] == 1 else 0.52
            reason     = "Last non-silent structural section near the end of the track."
        elif chorus_label is not None and structural == chorus_label:
            semantic   = "Chorus"
            confidence = 0.68
            reason     = "Repeated structural section with relatively high energy evidence."
        elif structural in repeated and chorus_label is not None and structural != chorus_label:
            semantic   = "Verse"
            confidence = 0.52
            reason     = "Repeated structural section distinct from the higher-energy repeated section."
        elif idx > 1 and idx < len(out) - 1 and counts[structural] == 1 and duration >= 8.0:
            semantic   = "Bridge"
            confidence = 0.50
            reason     = "Single distinct middle section after earlier material."

        if semantic == "Unknown" and idx not in {_first_content, _last_content}:
            if position_start < 0.33:
                semantic = "Early"
            elif position_start < 0.66:
                semantic = "Middle"
            else:
                semantic = "Late"
            confidence = 0.30
            reason = "No specific section pattern identified; position-based label."

        seg["semantic_label"]     = semantic
        seg["section_type"]       = semantic
        seg["semantic_confidence"] = round(float(confidence), 3)
        seg["semantic_reason"]    = reason

    return out


def apply_two_layer_labels(
    segments: list[dict],
    *,
    file_path: str | None = None,
    descriptors: np.ndarray | None = None,
    duration_seconds: float | None = None,
    semantic_enabled: bool = True,
    method_hint: str | None = None,
    method: str = "heuristic",
) -> list[dict]:
    """Apply structural (A/B/C) then semantic (Intro/Verse/Chorus …) labels.

    Parameters
    ----------
    method : ``"heuristic"`` (default) or ``"ml"``.
    """
    if descriptors is None and file_path:
        from segmentation.infrastructure.audio.features import build_segment_descriptors
        descriptors = build_segment_descriptors(file_path, segments)

    labeled = assign_structural_labels(
        segments,
        descriptors=descriptors,
        file_path=None,
        method_hint=method_hint,
    )

    if method == "ml":
        from segmentation.core.labeling.ml import predict_semantic_labels
        return predict_semantic_labels(
            labeled,
            descriptors=descriptors,
            file_path=None,
            duration_seconds=duration_seconds,
        )
    return assign_semantic_labels(
        labeled,
        duration_seconds=duration_seconds,
        descriptors=descriptors,
        enabled=semantic_enabled,
    )

"""
ML-based semantic label inference for music segments.

Loads a pre-trained ``HistGradientBoostingClassifier`` bundle from
``models/segment_label_clf.joblib`` and applies it to a list of segments.

If the model file is missing, cannot be loaded, or inference raises an
exception the function **silently falls back** to the heuristic
``assign_semantic_labels`` so the pipeline never breaks.

Bundle format (written by scripts/label_training/train_label_classifier.py)
-------------------------------------------------------------
{
    "clf"          : fitted HistGradientBoostingClassifier,
    "label_encoder": fitted sklearn.preprocessing.LabelEncoder,
    "feature_names": list[str],        # for diagnostics / feature-importance
    "classes"      : list[str],        # label_encoder.classes_
    "trained_at"   : str (ISO datetime),
    "dataset"      : str,
    "n_train"      : int,
    "cv_accuracy"  : float,
}
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("ml_labeling")

# ── Model path (resolved relative to the repo root) ──────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.environ.get(
    "LABEL_MODEL_PATH",
    os.path.join(_REPO_ROOT, "models", "segment_label_clf.joblib"),
)

# Lazy-load cache — populated once on first call.
_MODEL_BUNDLE: dict[str, Any] | None = None
_LOAD_ATTEMPTED: bool = False


def _load_bundle() -> dict[str, Any] | None:
    """Load (and cache) the model bundle; return None on any failure."""
    global _MODEL_BUNDLE, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL_BUNDLE
    _LOAD_ATTEMPTED = True

    if not os.path.exists(_MODEL_PATH):
        logger.warning(
            "ML label model not found at %s; using heuristic fallback.", _MODEL_PATH
        )
        return None

    try:
        import joblib  # ships with scikit-learn

        bundle = joblib.load(_MODEL_PATH)
        if not isinstance(bundle, dict) or "clf" not in bundle or "label_encoder" not in bundle:
            logger.warning(
                "Unexpected model bundle format at %s; using heuristic fallback.", _MODEL_PATH
            )
            return None

        classes = bundle.get("classes", [])
        logger.info(
            "ML label model loaded from %s  classes=%s  cv_acc=%.3f",
            _MODEL_PATH,
            classes,
            bundle.get("cv_accuracy", float("nan")),
        )
        _MODEL_BUNDLE = bundle
        return _MODEL_BUNDLE

    except Exception as exc:
        logger.warning(
            "Failed to load ML label model (%s); using heuristic fallback.", exc
        )
        return None


def reset_model_cache() -> None:
    """Force the next call to re-load the model file (useful for testing)."""
    global _MODEL_BUNDLE, _LOAD_ATTEMPTED
    _MODEL_BUNDLE = None
    _LOAD_ATTEMPTED = False


# ── Public API ────────────────────────────────────────────────────────────────

def predict_semantic_labels(
    segments: list[dict],
    descriptors: np.ndarray | None = None,
    file_path: str | None = None,
    duration_seconds: float | None = None,
) -> list[dict]:
    """Assign semantic labels to *segments* using the trained GBDT model.

    On any failure (model missing, inference error) the function falls back
    to the rule-based ``assign_semantic_labels`` without raising.

    Parameters
    ----------
    segments:
        List of segment dicts (at minimum ``start``, ``end``, and ideally
        ``structural_label`` for the repetition-count contextual feature).
    descriptors:
        Pre-computed acoustic descriptor matrix *(N, 54)* from
        ``shared.labeling.build_segment_descriptors``.  Passed directly
        to ``build_segment_label_vectors``; computed from *file_path* when
        omitted.
    file_path:
        Audio file path; used to compute descriptors when *descriptors*
        is *None*.
    duration_seconds:
        Track duration; used by the heuristic fallback only.

    Returns
    -------
    List of segment dicts with updated fields:
      ``semantic_label``, ``section_type``, ``semantic_confidence``,
      ``semantic_reason``, ``label_method``.
    """
    from workers.core.labeling.heuristic import assign_semantic_labels

    def _heuristic_fallback() -> list[dict]:
        return assign_semantic_labels(
            segments,
            duration_seconds=duration_seconds,
            descriptors=descriptors,
            enabled=True,
        )

    bundle = _load_bundle()
    if bundle is None:
        return _heuristic_fallback()

    try:
        from workers.core.labeling.features import build_segment_label_vectors

        clf = bundle["clf"]
        le  = bundle["label_encoder"]

        X, _ = build_segment_label_vectors(
            segments,
            descriptors=descriptors,
            file_path=file_path,
        )

        if X.shape[0] == 0:
            return list(segments)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)  # shape (N, n_classes)

        out = [dict(s) for s in segments]
        for i, seg in enumerate(out):
            label = str(le.inverse_transform([preds[i]])[0])
            conf  = float(np.max(proba[i]))
            seg["semantic_label"]     = label
            seg["section_type"]       = label
            seg["semantic_confidence"] = round(conf, 3)
            seg["semantic_reason"]    = f"ml:{label} p={conf:.2f}"
            seg["label_method"]       = "ml"
        return out

    except Exception as exc:
        logger.warning("ML inference failed (%s); falling back to heuristic.", exc)
        return _heuristic_fallback()

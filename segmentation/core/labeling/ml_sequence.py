"""
BiLSTM-CRF sequence label inference for music segments.

Unlike the GBDT classifier (``segmentation.core.labeling.ml``), which scores each
segment independently, this model decodes a *whole song's* ordered segment
list jointly via CRF Viterbi decoding — it can learn structural grammar
(Intro → Verse → Pre-Chorus → Chorus → …) that a per-segment classifier
cannot see.

Loads a pre-trained checkpoint from
``models/segment_label_seq_{SEQ_MERGE_MODE}.pt`` (default mode: ``other``,
matching the label vocabulary already served by the GBDT model).

If the model file is missing, cannot be loaded, the checkpoint expects
feature columns this module cannot reproduce (e.g. a GBDT-stacked
checkpoint — see ``scripts/label_training/train_sequence_model.py``
``--gbdt-stack``), or inference raises, the function **silently falls
back** to the heuristic ``assign_semantic_labels`` so the pipeline never
breaks.

Checkpoint format (written by scripts/label_training/train_sequence_model.py)
-------------------------------------------------------------------------
{
    "model_state":   state_dict for SegmentSequenceModel,
    "scaler_mean":   np.ndarray (D,)   — StandardScaler mean, base features only,
    "scaler_scale":  np.ndarray (D,)   — StandardScaler scale, base features only,
    "label_encoder": fitted sklearn.preprocessing.LabelEncoder,
    "feature_names": list[str],        # aug_feature_cols (base + optional gbdt_p_*)
    "classes":       list[str],
    "merge_mode":    str,
    "config":        {input_dim, hidden_dim, num_layers, dropout, num_tags,
                       use_attention, attn_heads, gbdt_stack, noise_sigma,
                       n_acoustic_feat},
    "trained_at":    str (ISO datetime),
    "val_macro_f1":  float,
    "test_macro_f1": float,
}
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("ml_sequence_labeling")

# segmentation/core/labeling/ml_sequence.py -> repo root (3 levels up)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_SEQ_MERGE_MODE = os.environ.get("SEQ_LABEL_MERGE_MODE", "other")
_MODEL_PATH = os.environ.get(
    "SEQ_LABEL_MODEL_PATH",
    os.path.join(_REPO_ROOT, "models", f"segment_label_seq_{_SEQ_MERGE_MODE}.pt"),
)

# Lazy-load cache — populated once on first call.
_CHECKPOINT: dict[str, Any] | None = None
_MODEL: Any | None = None  # SegmentSequenceModel instance, once built
_LOAD_ATTEMPTED: bool = False


def _load_checkpoint() -> tuple[dict[str, Any], Any] | None:
    """Load (and cache) the checkpoint + reconstructed model; None on any failure."""
    global _CHECKPOINT, _MODEL, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return (_CHECKPOINT, _MODEL) if _CHECKPOINT is not None else None
    _LOAD_ATTEMPTED = True

    if not os.path.exists(_MODEL_PATH):
        logger.warning(
            "Sequence label model not found at %s; using heuristic fallback.", _MODEL_PATH
        )
        return None

    try:
        import torch

        ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)

        config = ckpt.get("config", {})
        if config.get("gbdt_stack"):
            logger.warning(
                "Sequence model at %s was trained with --gbdt-stack; the stacking "
                "GBDT booster it depends on is not persisted, so this checkpoint "
                "cannot be served. Retrain without --gbdt-stack. Using heuristic fallback.",
                _MODEL_PATH,
            )
            return None

        from segmentation.core.labeling.sequence_arch import SegmentSequenceModel

        model = SegmentSequenceModel(
            input_dim     = config["input_dim"],
            hidden_dim    = config["hidden_dim"],
            num_tags      = config["num_tags"],
            num_layers    = config.get("num_layers", 1),
            dropout       = config.get("dropout", 0.0),
            use_attention = config.get("use_attention", False),
            attn_heads    = config.get("attn_heads", 4),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        logger.info(
            "Sequence label model loaded from %s  merge_mode=%s  classes=%s  test_f1=%.3f",
            _MODEL_PATH, ckpt.get("merge_mode"), ckpt.get("classes"),
            ckpt.get("test_macro_f1", float("nan")),
        )
        _CHECKPOINT = ckpt
        _MODEL = model
        return _CHECKPOINT, _MODEL

    except Exception as exc:
        logger.warning(
            "Failed to load sequence label model (%s); using heuristic fallback.", exc
        )
        return None


def reset_model_cache() -> None:
    """Force the next call to re-load the checkpoint (useful for testing)."""
    global _CHECKPOINT, _MODEL, _LOAD_ATTEMPTED
    _CHECKPOINT = None
    _MODEL = None
    _LOAD_ATTEMPTED = False


# ── Public API ────────────────────────────────────────────────────────────────

def predict_semantic_labels_sequence(
    segments: list[dict],
    descriptors: np.ndarray | None = None,
    file_path: str | None = None,
    duration_seconds: float | None = None,
) -> list[dict]:
    """Assign semantic labels to one song's *segments* via BiLSTM-CRF.

    *segments* must be the full, time-ordered segment list for a single
    song — the CRF decodes the sequence jointly, so scoring a subset or
    scrambled order will degrade quality silently.

    On any failure (model missing, incompatible checkpoint, inference
    error) falls back to the rule-based ``assign_semantic_labels``.
    """
    from segmentation.core.labeling.heuristic import assign_semantic_labels

    def _heuristic_fallback() -> list[dict]:
        return assign_semantic_labels(
            segments,
            duration_seconds=duration_seconds,
            descriptors=descriptors,
            enabled=True,
        )

    loaded = _load_checkpoint()
    if loaded is None:
        return _heuristic_fallback()
    ckpt, model = loaded

    try:
        import torch
        from segmentation.core.labeling.features import build_segment_label_vectors

        le = ckpt["label_encoder"]

        X_full, full_feature_names = build_segment_label_vectors(
            segments,
            descriptors=descriptors,
            file_path=file_path,
        )
        if X_full.shape[0] == 0:
            return list(segments)

        feature_names = list(ckpt["feature_names"])
        name_to_idx = {name: i for i, name in enumerate(full_feature_names)}
        missing = [name for name in feature_names if name not in name_to_idx]
        if missing:
            logger.warning(
                "Sequence model expects unknown feature columns %s; falling back to heuristic.",
                missing[:5],
            )
            return _heuristic_fallback()
        indices = [name_to_idx[name] for name in feature_names]
        X_model = X_full[:, indices]

        expected = ckpt["config"]["input_dim"]
        if X_model.shape[1] != expected:
            logger.warning(
                "Feature count mismatch: sequence model expects %d features but got %d. "
                "Falling back to heuristic.", expected, X_model.shape[1],
            )
            return _heuristic_fallback()

        mean  = ckpt["scaler_mean"]
        scale = ckpt["scaler_scale"]
        X_scaled = (X_model - mean) / scale

        x = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0)       # (1, L, D)
        mask = torch.ones(1, X_scaled.shape[0], dtype=torch.bool)            # (1, L)

        with torch.no_grad():
            emissions = model._emit(x, mask)                                 # (1, L, K)
            emission_probs = torch.softmax(emissions, dim=-1)[0]             # (L, K)
        pred_ids = model.predict(x, mask)[0]                                 # list[int], len L
        labels = le.inverse_transform(np.asarray(pred_ids))

        out = [dict(s) for s in segments]
        for i, seg in enumerate(out):
            label = str(labels[i])
            # Emission softmax at the chosen tag — an approximate per-segment
            # confidence; the CRF's actual path score is joint, not per-token.
            conf = float(emission_probs[i, pred_ids[i]])
            seg["semantic_label"]     = label
            seg["section_type"]       = label
            seg["semantic_confidence"] = round(conf, 3)
            seg["semantic_reason"]    = f"ml_sequence:{label} p={conf:.2f}"
            seg["label_method"]       = "ml_sequence"
        return out

    except Exception as exc:
        logger.warning("Sequence model inference failed (%s); falling back to heuristic.", exc)
        return _heuristic_fallback()

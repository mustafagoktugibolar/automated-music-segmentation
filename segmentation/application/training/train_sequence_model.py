#!/usr/bin/env python
"""
BiLSTM-CRF sequence model for music segment label classification.
==================================================================
Treats each song as an ordered sequence of segments and learns both
per-segment acoustics (BiLSTM) and structural grammar (CRF transitions).

Architecture:
  Input(D) → Linear(64) + LayerNorm + ReLU + Dropout(0.5)
            → BiLSTM(hidden=64, layers=1)
            → Linear(K) [emissions]
            → CRF (learned K×K transition matrix + L2 reg, from-scratch impl)

Loss = CRF negative log-likelihood + crf_reg * ||transitions||²
     + λ · class-weighted cross-entropy on emissions (label_smoothing=0.1)
     (λ=0.4 keeps rare class recall while CRF learns grammar)

Regularization (overfitting fixes):
  - dropout 0.5, weight_decay 1e-3, grad_clip 1.0, patience 12
  - CRF transition L2 penalty (crf_reg=0.01)
  - label smoothing 0.1 on aux CE loss
  - reduced capacity: hidden_dim=64, num_layers=1

Split: same GroupShuffleSplit(raw_track_id) as GBDT → honest comparison.
Seeds: same DEFAULT_SEEDS → multi-seed mean±std reported.

Usage:
    python scripts/label_training/train_sequence_model.py
    python scripts/label_training/train_sequence_model.py --merge-mode none
    python scripts/label_training/train_sequence_model.py --feature-set acoustic
    python scripts/label_training/train_sequence_model.py \\
        --extra-parquet /app/data/label_training/harmonix_segments.parquet

Outputs:
    models/segment_label_seq_{merge_mode}.pt   — best checkpoint per mode
    models/evaluation/sequence_model_{merge_mode}.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Re-use everything from GBDT trainer for identical split/data logic
from train_label_classifier import (  # noqa: E402
    load_dataset, apply_label_merge, build_features, select_feature_columns,
    make_grouped_split, _MERGE_MAPS, DEFAULT_SEEDS,
)
# Model architecture lives in workers/core so training and serving share one
# definition — see workers/core/labeling/sequence_arch.py.
from segmentation.core.labeling.sequence_arch import CRF, SegmentSequenceModel  # noqa: E402

MODELS_DIR   = os.path.join(_app_root, "models")
EVAL_DIR     = os.path.join(MODELS_DIR, "evaluation")
PARQUET_PATH = os.path.join(_app_root, "data", "label_training", "segments.parquet")


# ── Sequence dataset builder ──────────────────────────────────────────────────

def build_sequences(
    df: "pd.DataFrame",
    feature_cols: list[str],
    le,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(X_song (L, D), y_song (L,)), ...] sorted by segment_idx."""
    seqs: list[tuple[np.ndarray, np.ndarray]] = []
    for sid in df["song_id"].unique():
        sub = df[df["song_id"] == sid].sort_values("segment_idx")
        X   = sub[feature_cols].values.astype(np.float32)
        y   = le.transform(sub["label"].values)
        seqs.append((X, y))
    return seqs


def collate_sequences(
    batch: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (X,y) pairs → (X_pad, y_pad, mask), all on device."""
    Xs, ys = zip(*batch)
    lengths = [len(x) for x in Xs]
    L_max   = max(lengths)
    D       = Xs[0].shape[1]

    X_pad = torch.zeros(len(Xs), L_max, D, device=device)
    y_pad = torch.zeros(len(Xs), L_max, dtype=torch.long, device=device)
    mask  = torch.zeros(len(Xs), L_max, dtype=torch.bool, device=device)
    for i, (x, y) in enumerate(zip(Xs, ys)):
        L = len(x)
        X_pad[i, :L] = torch.from_numpy(x).to(device)
        y_pad[i, :L] = torch.from_numpy(y).to(device)
        mask[i, :L]  = True
    return X_pad, y_pad, mask


# ── Training helpers ──────────────────────────────────────────────────────────

def _eval_split(
    model: SegmentSequenceModel,
    seqs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, float, dict[int, float]]:
    """Return (accuracy, macro-F1, per_class_f1) on a list of sequences."""
    model.eval()
    y_true_all, y_pred_all = [], []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i: i + batch_size]
        Xb, yb, mb = collate_sequences(batch, device)
        preds = model.predict(Xb, mb)
        for j, (pred_seq, (_, y_seq)) in enumerate(zip(preds, batch)):
            y_true_all.extend(y_seq.tolist())
            y_pred_all.extend(pred_seq)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    acc    = float((y_true == y_pred).mean())
    macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pc     = {
        i: float(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0))
        for i in range(model.num_tags)
    }
    return acc, macro, pc


# ── GBDT stacking helper (OOF) ────────────────────────────────────────────────

def _train_stacking_gbdt_oof(
    df_train: "pd.DataFrame",
    feature_cols: list,
    le,
    K: int,
    args,
    seed: int,
):
    """OOF stacking: each training segment is predicted by a GBDT that was NOT
    trained on its song. This prevents leakage into BiLSTM-CRF training.
    Returns (final_booster, oof_proba array of shape (N_train, K)).
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    X_tr = df_train[feature_cols].values.astype(np.float32)
    y_tr = le.transform(df_train["label"].values)
    groups = df_train["raw_track_id"].values

    params = dict(
        objective        = "multiclass",
        num_class        = K,
        num_leaves       = args.gbdt_num_leaves,
        max_depth        = 5,
        learning_rate    = 0.05,
        min_child_samples= 20,
        reg_lambda       = 1.0,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        seed             = seed,
        verbose          = -1,
        n_jobs           = -1,
    )

    oof_proba = np.zeros((len(X_tr), K), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for fold_tr_idx, fold_va_idx in gkf.split(X_tr, y_tr, groups):
        bst_fold = lgb.train(
            params,
            lgb.Dataset(X_tr[fold_tr_idx], label=y_tr[fold_tr_idx]),
            num_boost_round = args.gbdt_n_rounds,
            valid_sets      = [lgb.Dataset(X_tr[fold_va_idx], label=y_tr[fold_va_idx])],
            callbacks       = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        oof_proba[fold_va_idx] = bst_fold.predict(X_tr[fold_va_idx])

    # Final model trained on ALL train data → used for val/test at inference
    bst_final = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round = args.gbdt_n_rounds,
        callbacks       = [lgb.log_evaluation(period=-1)],
    )
    return bst_final, oof_proba


# ── Main per-seed run ─────────────────────────────────────────────────────────

def run_seed(
    seed: int,
    df: "pd.DataFrame",
    feature_cols: list[str],
    le,
    merge_mode: str,
    args,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Full train/val/test pipeline for one seed. Returns eval dict."""
    import pandas as pd

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Grouped split ─────────────────────────────────────────────────────────
    group_col = "raw_track_id" if "raw_track_id" in df.columns else "song_id"
    groups    = df[group_col].values
    y_all     = le.transform(df["label"].values)

    train_idx, val_idx, test_idx = make_grouped_split(
        groups, val_size=args.val_size, test_size=args.test_size,
        random_state=seed,
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    # ── StandardScaler (fit on train only) ───────────────────────────────────
    K = len(le.classes_)
    scaler = StandardScaler()
    X_tr_raw = df_train[feature_cols].values.astype(np.float32)
    scaler.fit(X_tr_raw)

    # ── GBDT stacking (OOF predictions for train — no leakage) ──────────────
    aug_feature_cols = feature_cols
    if args.gbdt_stack:
        gbdt_bst, gbdt_oof_proba = _train_stacking_gbdt_oof(
            df_train     = df_train,
            feature_cols = feature_cols,
            le           = le,
            K            = K,
            args         = args,
            seed         = seed,
        )
        gbdt_prob_cols   = [f"gbdt_p_{c}" for c in le.classes_]
        aug_feature_cols = feature_cols + gbdt_prob_cols

    def _scale_df(sub: "pd.DataFrame") -> "pd.DataFrame":
        sub = sub.copy()
        sub[feature_cols] = scaler.transform(
            sub[feature_cols].values.astype(np.float32)
        )
        return sub

    df_train_s = _scale_df(df_train)
    df_val_s   = _scale_df(df_val)
    df_test_s  = _scale_df(df_test)

    # Append GBDT probabilities to scaled dfs (proba stay unscaled, in [0,1])
    if args.gbdt_stack:
        # Train: OOF predictions (no leakage — each song predicted out-of-fold)
        for j, col in enumerate(gbdt_prob_cols):
            df_train_s[col] = gbdt_oof_proba[:, j]
        # Val / Test: final model trained on all training data
        for raw_df, scaled_df in [(df_val, df_val_s), (df_test, df_test_s)]:
            proba = gbdt_bst.predict(
                raw_df[feature_cols].values.astype(np.float32)
            )
            for j, col in enumerate(gbdt_prob_cols):
                scaled_df[col] = proba[:, j]

    train_seqs = build_sequences(df_train_s, aug_feature_cols, le)
    val_seqs   = build_sequences(df_val_s,   aug_feature_cols, le)
    test_seqs  = build_sequences(df_test_s,  aug_feature_cols, le)

    # ── Class weights (from train) ────────────────────────────────────────────
    counts = np.bincount(
        le.transform(df_train["label"].values), minlength=K
    ).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(counts.sum() / (K * counts), dtype=torch.float32,
                           device=device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SegmentSequenceModel(
        input_dim     = len(aug_feature_cols),
        hidden_dim    = args.hidden_dim,
        num_tags      = K,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        use_attention = args.use_attention,
        attn_heads    = args.attn_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-5,
    )

    best_val_f1   = -1.0
    best_state    = None
    patience_left = args.patience
    rng           = np.random.default_rng(seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_seqs := list(train_seqs))
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(train_seqs), args.batch_size):
            batch = train_seqs[i: i + args.batch_size]
            Xb, yb, mb = collate_sequences(batch, device)

            optimizer.zero_grad()
            loss = model(Xb, yb, mb, class_weights=weights,
                         lambda_ce=args.lambda_ce, crf_reg=args.crf_reg,
                         label_smoothing=args.label_smoothing,
                         noise_sigma=args.noise_sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        val_acc, val_f1, _ = _eval_split(model, val_seqs, device,
                                          args.batch_size)
        scheduler.step(val_f1)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  loss={total_loss/n_batches:.4f}"
                  f"  val_acc={val_acc:.4f}  val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                if verbose:
                    print(f"  Early stop at epoch {epoch}  best_val_F1={best_val_f1:.4f}")
                break

    # ── Test eval ─────────────────────────────────────────────────────────────
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    tr_acc, tr_f1, _   = _eval_split(model, train_seqs, device, args.batch_size)
    val_acc2, val_f12, _ = _eval_split(model, val_seqs,  device, args.batch_size)
    te_acc, te_f1, te_pc = _eval_split(model, test_seqs, device, args.batch_size)

    if verbose:
        y_true_all, y_pred_all = [], []
        model.eval()
        for seq in test_seqs:
            Xb, yb, mb = collate_sequences([seq], device)
            preds = model.predict(Xb, mb)
            y_true_all.extend(seq[1].tolist())
            y_pred_all.extend(preds[0])
        print(f"\n── Test ──")
        print(classification_report(
            y_true_all, y_pred_all,
            labels=list(range(K)), target_names=list(le.classes_),
            zero_division=0,
        ))
        print(f"  seed={seed:5d} | train_F1={tr_f1:.4f} | val_F1={val_f12:.4f} | test_F1={te_f1:.4f}")
        print(f"  train–val gap: {tr_f1 - val_f12:+.4f}  |  val–test gap: {val_f12 - te_f1:+.4f}")

    return {
        "seed":             seed,
        "best_val_f1":      best_val_f1,
        "train_acc":        tr_acc,
        "train_macro_f1":   tr_f1,
        "val_acc":          val_acc2,
        "val_macro_f1":     val_f12,
        "test_acc":         te_acc,
        "test_macro_f1":    te_f1,
        "per_class_f1":     {le.classes_[i]: v for i, v in te_pc.items()},
        "model_state":      best_state,
        "scaler":           scaler,
        "aug_feature_cols": aug_feature_cols,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="BiLSTM-CRF segment-label sequence model.")
    parser.add_argument("--input",        default=PARQUET_PATH)
    parser.add_argument("--extra-parquet", nargs="*", default=[])
    parser.add_argument("--merge-mode",   default="none", choices=list(_MERGE_MAPS))
    parser.add_argument("--feature-set",  default="full",
                        choices=["full", "acoustic", "acoustic_context",
                                 "no_context", "no_repetition"],
                        help="Feature subset to use (same sets as GBDT ablation).")
    parser.add_argument("--seeds",        nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--no-multi-seed", action="store_true")
    parser.add_argument("--val-size",     type=float, default=0.20)
    parser.add_argument("--test-size",    type=float, default=0.20)
    # Model hyper-parameters
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--num-layers",   type=int,   default=1)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--patience",     type=int,   default=18)
    parser.add_argument("--lambda-ce",    type=float, default=0.4,
                        help="Weight for class-weighted CE auxiliary loss.")
    parser.add_argument("--crf-reg",      type=float, default=0.01,
                        help="L2 penalty weight on CRF transition matrix.")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for auxiliary CE loss.")
    # Stacking
    parser.add_argument("--gbdt-stack",      action="store_true", default=False,
                        help="Append per-seed GBDT class probabilities as 7 extra features.")
    parser.add_argument("--gbdt-num-leaves", type=int,   default=31,
                        help="num_leaves for the stacking LightGBM.")
    parser.add_argument("--gbdt-n-rounds",   type=int,   default=500,
                        help="Max boosting rounds for stacking GBDT (early-stopped).")
    # Attention
    parser.add_argument("--use-attention",   action="store_true", default=False,
                        help="Add multi-head self-attention after BiLSTM.")
    parser.add_argument("--attn-heads",      type=int,   default=4,
                        help="Number of attention heads (must divide hidden_dim*2).")
    # Regularization
    parser.add_argument("--noise-sigma",     type=float, default=0.0,
                        help="Gaussian noise std on acoustic features during training. "
                             "0 = disabled. Recommended: 0.05.")
    args = parser.parse_args()

    if args.no_multi_seed:
        args.seeds = args.seeds[:1]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  |  torch {torch.__version__}")

    # ── Load & merge data ─────────────────────────────────────────────────────
    import pandas as pd
    df = load_dataset(args.input, extra_parquets=args.extra_parquet or [])
    df = apply_label_merge(df, args.merge_mode)

    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")

    _, _, _, le, all_feature_cols = build_features(df)
    feature_cols = select_feature_columns(all_feature_cols, args.feature_set)
    K  = len(le.classes_)
    print(f"\nFeature matrix: {len(df)} rows × {len(feature_cols)} features "
          f"(feature_set={args.feature_set}) | "
          f"Classes ({K}): {list(le.classes_)}")

    # ── Multi-seed run ────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  BiLSTM-CRF  merge_mode={args.merge_mode}  seeds={args.seeds}")
    print(f"{'#'*60}")
    t0 = time.perf_counter()

    all_results = []
    best_val_f1 = -1.0
    best_result = None

    for seed in args.seeds:
        print(f"\n── seed={seed} ──────────────────────────────────────────────")
        result = run_seed(seed, df, feature_cols, le, args.merge_mode, args,
                          device, verbose=(seed == args.seeds[0]))
        all_results.append(result)
        if result["best_val_f1"] > best_val_f1:
            best_val_f1 = result["best_val_f1"]
            best_result = result
        print(f"  seed={seed:5d} | train_F1={result['train_macro_f1']:.4f} "
              f"| val_F1={result['val_macro_f1']:.4f} "
              f"| test_F1={result['test_macro_f1']:.4f}")

    elapsed = time.perf_counter() - t0
    test_f1s  = [r["test_macro_f1"]  for r in all_results]
    val_f1s   = [r["val_macro_f1"]   for r in all_results]
    train_f1s = [r["train_macro_f1"] for r in all_results]

    print(f"\n  Train Macro-F1: {np.mean(train_f1s):.3f} ± {np.std(train_f1s):.3f}")
    print(f"  Val   Macro-F1: {np.mean(val_f1s):.3f} ± {np.std(val_f1s):.3f}")
    print(f"  Test  Macro-F1: {np.mean(test_f1s):.3f} ± {np.std(test_f1s):.3f}")
    print(f"  Train–Val gap : {np.mean(train_f1s)-np.mean(val_f1s):+.3f}")
    print(f"  Val–Test gap  : {np.mean(val_f1s)-np.mean(test_f1s):+.3f}")
    print(f"  Total time   : {elapsed:.1f}s")

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR,
                              f"segment_label_seq_{args.merge_mode}.pt")
    _aug_cols = best_result["aug_feature_cols"]
    torch.save({
        "model_state":   best_result["model_state"],
        "scaler_mean":   best_result["scaler"].mean_,
        "scaler_scale":  best_result["scaler"].scale_,
        "label_encoder": le,
        "feature_names": _aug_cols,
        "classes":       list(le.classes_),
        "merge_mode":    args.merge_mode,
        "config": {
            "input_dim":       len(_aug_cols),
            "hidden_dim":      args.hidden_dim,
            "num_layers":      args.num_layers,
            "dropout":         args.dropout,
            "num_tags":        K,
            "use_attention":   args.use_attention,
            "attn_heads":      args.attn_heads,
            "gbdt_stack":      args.gbdt_stack,
            "noise_sigma":     args.noise_sigma,
            "n_acoustic_feat": 60,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_macro_f1":  float(np.mean(val_f1s)),
        "test_macro_f1": float(np.mean(test_f1s)),
    }, model_path)
    print(f"\nBest model saved → {model_path}")

    # ── Save evaluation JSON ──────────────────────────────────────────────────
    os.makedirs(EVAL_DIR, exist_ok=True)
    out = {
        "merge_mode":    args.merge_mode,
        "classes":       list(le.classes_),
        "n_features":    len(_aug_cols),
        "seeds":         args.seeds,
        "train_macro_f1_mean": float(np.mean(train_f1s)),
        "train_macro_f1_std":  float(np.std(train_f1s)),
        "val_macro_f1_mean":  float(np.mean(val_f1s)),
        "val_macro_f1_std":   float(np.std(val_f1s)),
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std":  float(np.std(test_f1s)),
        "train_val_gap":      float(np.mean(train_f1s) - np.mean(val_f1s)),
        "val_test_gap":       float(np.mean(val_f1s) - np.mean(test_f1s)),
        "per_seed": [
            {
                "seed":           r["seed"],
                "train_macro_f1": r["train_macro_f1"],
                "val_macro_f1":   r["val_macro_f1"],
                "test_macro_f1":  r["test_macro_f1"],
                "per_class_f1":   r["per_class_f1"],
            }
            for r in all_results
        ],
        "best_seed_per_class_f1": best_result["per_class_f1"],
    }
    eval_path = os.path.join(EVAL_DIR, f"sequence_model_{args.merge_mode}.json")
    with open(eval_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Evaluation saved → {eval_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  merge_mode={args.merge_mode}  BiLSTM-CRF")
    print(f"{'='*60}")
    print(f"  Classes  : {list(le.classes_)}")
    print(f"  Features : {len(_aug_cols)}")
    print(f"  Epochs   : up to {args.epochs}  patience={args.patience}")
    print(f"\n  Split       Val F1    Test F1")
    print(f"  ----------------------------------")
    print(f"  Mean        {np.mean(val_f1s):.4f}    {np.mean(test_f1s):.4f}")
    print(f"  Std         {np.std(val_f1s):.4f}    {np.std(test_f1s):.4f}")
    print(f"\n  Best-seed per-class F1 (test):")
    for cls, v in sorted(best_result["per_class_f1"].items(), key=lambda x: -x[1]):
        print(f"    {cls:<15} {v:.4f}")
    print(f"\n  ► Compare with GBDT 9-class: 0.556 ± 0.013 (SALAMI-only)")


if __name__ == "__main__":
    main()

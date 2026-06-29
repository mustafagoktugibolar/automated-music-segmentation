#!/usr/bin/env python
"""
BiLSTM-CRF sequence model for music segment label classification.
==================================================================
Treats each song as an ordered sequence of segments and learns both
per-segment acoustics (BiLSTM) and structural grammar (CRF transitions).

Architecture:
  Input(87) → Linear(128) + LayerNorm + ReLU + Dropout
             → BiLSTM(hidden=128, layers=2, dropout=0.3)
             → Linear(K) [emissions]
             → CRF (learned K×K transition matrix, from-scratch impl)

Loss = CRF negative log-likelihood
     + λ · class-weighted cross-entropy on emissions
     (λ=0.4 keeps rare class recall while CRF learns grammar)

Split: same GroupShuffleSplit(raw_track_id) as GBDT → honest comparison.
Seeds: same DEFAULT_SEEDS → multi-seed mean±std reported.

Usage:
    python scripts/label_training/train_sequence_model.py
    python scripts/label_training/train_sequence_model.py --merge-mode none
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
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Re-use everything from GBDT trainer for identical split/data logic
from train_label_classifier import (  # noqa: E402
    load_dataset, apply_label_merge, build_features,
    make_grouped_split, _MERGE_MAPS, DEFAULT_SEEDS, META_COLS,
)

MODELS_DIR   = os.path.join(_app_root, "models")
EVAL_DIR     = os.path.join(MODELS_DIR, "evaluation")
PARQUET_PATH = os.path.join(_app_root, "data", "label_training", "segments.parquet")

# ── CRF (from scratch, no torchcrf dependency) ────────────────────────────────

class CRF(nn.Module):
    """Linear-chain CRF with log-space forward algorithm and Viterbi decode.

    Parameters
    ----------
    num_tags : int
        Number of label classes K.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] = score of transitioning FROM tag i TO tag j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_scores = nn.Parameter(torch.empty(num_tags))
        self.end_scores   = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_scores, -0.1, 0.1)
        nn.init.uniform_(self.end_scores,   -0.1, 0.1)

    # ── log-partition (forward algorithm) ────────────────────────────────────
    def _forward_alg(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return log Z (scalar) for a batch using the forward algorithm.

        emissions : (B, L, K)
        mask      : (B, L) bool
        """
        B, L, K = emissions.shape
        # alpha: (B, K)
        alpha = self.start_scores.unsqueeze(0) + emissions[:, 0]  # (B, K)

        for t in range(1, L):
            # (B, K, 1) + (K, K) → (B, K, K) → logsumexp → (B, K)
            score = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B,K,K)
            alpha_next = torch.logsumexp(score, dim=1) + emissions[:, t]  # (B,K)
            mask_t = mask[:, t].unsqueeze(1)  # (B,1)
            alpha = torch.where(mask_t, alpha_next, alpha)

        alpha = alpha + self.end_scores.unsqueeze(0)  # (B, K)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    # ── gold-sequence score ───────────────────────────────────────────────────
    def _score_sequence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return score of gold sequence for each item in batch.

        emissions : (B, L, K)
        tags      : (B, L)  int64
        mask      : (B, L)  bool
        """
        B, L, K = emissions.shape
        score = self.start_scores[tags[:, 0]] + emissions[:, 0].gather(
            1, tags[:, 0:1]
        ).squeeze(1)

        for t in range(1, L):
            m    = mask[:, t]
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_t  = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            score   = score + (trans_t + emit_t) * m.float()

        # end score for the last real tag
        last_tag_idx = mask.sum(1).long() - 1  # (B,)
        last_tags    = tags.gather(1, last_tag_idx.unsqueeze(1)).squeeze(1)
        score        = score + self.end_scores[last_tags]
        return score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean NLL over batch."""
        gold  = self._score_sequence(emissions, tags, mask)
        logZ  = self._forward_alg(emissions, mask)
        return (logZ - gold).mean()

    @torch.no_grad()
    def viterbi(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> list[list[int]]:
        """Viterbi decode; returns list of tag sequences (variable length)."""
        B, L, K = emissions.shape
        vit     = self.start_scores.unsqueeze(0) + emissions[:, 0]  # (B,K)
        backptr = []

        for t in range(1, L):
            score = vit.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B,K,K)
            best_scores, best_tags = score.max(dim=1)  # (B,K) each
            vit_next = best_scores + emissions[:, t]
            mask_t   = mask[:, t].unsqueeze(1)
            vit      = torch.where(mask_t, vit_next, vit)
            backptr.append(best_tags)                                  # list of (B,K)

        vit   = vit + self.end_scores.unsqueeze(0)
        lens  = mask.sum(1).long()  # (B,)
        best_last = vit.argmax(dim=1)  # (B,)

        seqs: list[list[int]] = []
        for b in range(B):
            seq  = [best_last[b].item()]
            L_b  = lens[b].item()
            for t in range(L_b - 2, -1, -1):
                seq.append(backptr[t][b, seq[-1]].item())
            seq.reverse()
            seqs.append(seq[:L_b])
        return seqs


# ── Model ─────────────────────────────────────────────────────────────────────

class SegmentSequenceModel(nn.Module):
    """Input projection → BiLSTM → emission head → CRF."""

    def __init__(self, input_dim: int, hidden_dim: int, num_tags: int,
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.emission_head = nn.Linear(hidden_dim * 2, num_tags)
        self.crf           = CRF(num_tags)
        self.num_tags      = num_tags

    def _emit(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) padded → emissions (B, L, K)."""
        h, _ = self.bilstm(self.proj(x))
        return self.emission_head(h)

    def forward(
        self, x: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
        class_weights: torch.Tensor | None = None, lambda_ce: float = 0.4,
    ) -> torch.Tensor:
        """Training forward. Returns combined loss."""
        emissions = self._emit(x)                                    # (B,L,K)
        crf_loss  = self.crf.neg_log_likelihood(emissions, tags, mask)

        if lambda_ce > 0 and class_weights is not None:
            # Flatten valid (unpadded) positions only
            flat_emit = emissions[mask]                              # (N_valid, K)
            flat_tags  = tags[mask]                                  # (N_valid,)
            ce_loss = F.cross_entropy(flat_emit, flat_tags,
                                      weight=class_weights)
            return crf_loss + lambda_ce * ce_loss
        return crf_loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Inference; returns per-song decoded tag lists."""
        emissions = self._emit(x)
        return self.crf.viterbi(emissions, mask)


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
    scaler = StandardScaler()
    X_tr_raw = df_train[feature_cols].values.astype(np.float32)
    scaler.fit(X_tr_raw)

    def _scale_df(sub: "pd.DataFrame") -> "pd.DataFrame":
        sub = sub.copy()
        sub[feature_cols] = scaler.transform(
            sub[feature_cols].values.astype(np.float32)
        )
        return sub

    df_train_s = _scale_df(df_train)
    df_val_s   = _scale_df(df_val)
    df_test_s  = _scale_df(df_test)

    train_seqs = build_sequences(df_train_s, feature_cols, le)
    val_seqs   = build_sequences(df_val_s,   feature_cols, le)
    test_seqs  = build_sequences(df_test_s,  feature_cols, le)

    # ── Class weights (from train) ────────────────────────────────────────────
    K = len(le.classes_)
    counts = np.bincount(
        le.transform(df_train["label"].values), minlength=K
    ).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(counts.sum() / (K * counts), dtype=torch.float32,
                           device=device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SegmentSequenceModel(
        input_dim  = len(feature_cols),
        hidden_dim = args.hidden_dim,
        num_tags   = K,
        num_layers = args.num_layers,
        dropout    = args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5, min_lr=1e-5,
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
                         lambda_ce=args.lambda_ce)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
        print(f"  seed={seed:5d} | val_F1={val_f12:.4f} | test_F1={te_f1:.4f}")

    return {
        "seed":          seed,
        "best_val_f1":   best_val_f1,
        "val_acc":       val_acc2,
        "val_macro_f1":  val_f12,
        "test_acc":      te_acc,
        "test_macro_f1": te_f1,
        "per_class_f1":  {le.classes_[i]: v for i, v in te_pc.items()},
        "model_state":   best_state,
        "scaler":        scaler,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="BiLSTM-CRF segment-label sequence model.")
    parser.add_argument("--input",        default=PARQUET_PATH)
    parser.add_argument("--extra-parquet", nargs="*", default=[])
    parser.add_argument("--merge-mode",   default="none", choices=list(_MERGE_MAPS))
    parser.add_argument("--seeds",        nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--no-multi-seed", action="store_true")
    parser.add_argument("--val-size",     type=float, default=0.20)
    parser.add_argument("--test-size",    type=float, default=0.20)
    # Model hyper-parameters
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--num-layers",   type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=120)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--lambda-ce",    type=float, default=0.4,
                        help="Weight for class-weighted CE auxiliary loss.")
    args = parser.parse_args()

    if args.no_multi_seed:
        args.seeds = args.seeds[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  torch {torch.__version__}")

    # ── Load & merge data ─────────────────────────────────────────────────────
    import pandas as pd
    df = load_dataset(args.input, extra_parquets=args.extra_parquet or [])
    df = apply_label_merge(df, args.merge_mode)

    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(df["label"])
    K  = len(le.classes_)
    print(f"\nFeature matrix: {len(df)} rows × {len(feature_cols)} features | "
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
        print(f"  seed={seed:5d} | val_F1={result['val_macro_f1']:.4f} "
              f"| test_F1={result['test_macro_f1']:.4f}")

    elapsed = time.perf_counter() - t0
    test_f1s = [r["test_macro_f1"] for r in all_results]
    val_f1s  = [r["val_macro_f1"]  for r in all_results]

    print(f"\n  Val  Macro-F1: {np.mean(val_f1s):.3f} ± {np.std(val_f1s):.3f}")
    print(f"  Test Macro-F1: {np.mean(test_f1s):.3f} ± {np.std(test_f1s):.3f}")
    print(f"  Val–Test gap : {np.mean(val_f1s)-np.mean(test_f1s):+.3f}")
    print(f"  Total time   : {elapsed:.1f}s")

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR,
                              f"segment_label_seq_{args.merge_mode}.pt")
    torch.save({
        "model_state":   best_result["model_state"],
        "scaler_mean":   best_result["scaler"].mean_,
        "scaler_scale":  best_result["scaler"].scale_,
        "label_encoder": le,
        "feature_names": feature_cols,
        "classes":       list(le.classes_),
        "merge_mode":    args.merge_mode,
        "config": {
            "input_dim":  len(feature_cols),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout":    args.dropout,
            "num_tags":   K,
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
        "n_features":    len(feature_cols),
        "seeds":         args.seeds,
        "val_macro_f1_mean":  float(np.mean(val_f1s)),
        "val_macro_f1_std":   float(np.std(val_f1s)),
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std":  float(np.std(test_f1s)),
        "val_test_gap":       float(np.mean(val_f1s) - np.mean(test_f1s)),
        "per_seed": [
            {
                "seed":          r["seed"],
                "val_macro_f1":  r["val_macro_f1"],
                "test_macro_f1": r["test_macro_f1"],
                "per_class_f1":  r["per_class_f1"],
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
    print(f"  Features : {len(feature_cols)}")
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

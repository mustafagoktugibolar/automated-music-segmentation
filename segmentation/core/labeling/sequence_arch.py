"""
BiLSTM-CRF architecture for sequence-based segment label classification.

Single source of truth for the model definition — shared by
``scripts/label_training/train_sequence_model.py`` (training) and
``workers/core/labeling/ml_sequence.py`` (inference) so the two never
drift apart.

Only depends on torch; safe to import from serving code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        crf_reg: float = 0.0,
    ) -> torch.Tensor:
        """Mean NLL over batch, with optional L2 penalty on transition matrix."""
        gold  = self._score_sequence(emissions, tags, mask)
        logZ  = self._forward_alg(emissions, mask)
        nll   = (logZ - gold).mean()
        if crf_reg > 0.0:
            nll = nll + crf_reg * (self.transitions ** 2).sum()
        return nll

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


class SegmentSequenceModel(nn.Module):
    """Input projection → BiLSTM → emission head → CRF."""

    def __init__(self, input_dim: int, hidden_dim: int, num_tags: int,
                 num_layers: int = 2, dropout: float = 0.3,
                 use_attention: bool = False, attn_heads: int = 4) -> None:
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
        self.use_attention = use_attention
        if use_attention:
            embed_dim = hidden_dim * 2
            self.attn      = nn.MultiheadAttention(embed_dim, num_heads=attn_heads,
                                                   dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(embed_dim)
        self.emission_head = nn.Linear(hidden_dim * 2, num_tags)
        self.crf           = CRF(num_tags)
        self.num_tags      = num_tags

    def _emit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, D) padded → emissions (B, L, K)."""
        h, _ = self.bilstm(self.proj(x))
        if self.use_attention and mask is not None:
            # key_padding_mask: True = ignore (PyTorch convention) → invert our mask
            attn_out, _ = self.attn(h, h, h, key_padding_mask=~mask)
            h = self.attn_norm(h + attn_out)
        return self.emission_head(h)

    def forward(
        self, x: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
        class_weights: torch.Tensor | None = None, lambda_ce: float = 0.4,
        crf_reg: float = 0.0, label_smoothing: float = 0.0,
        noise_sigma: float = 0.0,
    ) -> torch.Tensor:
        """Training forward. Returns combined loss."""
        if self.training and noise_sigma > 0.0:
            noise = torch.randn(x.shape[0], x.shape[1], 60,
                                device=x.device, dtype=x.dtype) * noise_sigma
            x = x.clone()
            x[:, :, :60] = x[:, :, :60] + noise * mask.unsqueeze(-1)
        emissions = self._emit(x, mask)                              # (B,L,K)
        crf_loss  = self.crf.neg_log_likelihood(emissions, tags, mask,
                                                crf_reg=crf_reg)

        if lambda_ce > 0 and class_weights is not None:
            # Flatten valid (unpadded) positions only
            flat_emit = emissions[mask]                              # (N_valid, K)
            flat_tags  = tags[mask]                                  # (N_valid,)
            ce_loss = F.cross_entropy(flat_emit, flat_tags,
                                      weight=class_weights,
                                      label_smoothing=label_smoothing)
            return crf_loss + lambda_ce * ce_loss
        return crf_loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Inference; returns per-song decoded tag lists."""
        emissions = self._emit(x, mask)
        return self.crf.viterbi(emissions, mask)

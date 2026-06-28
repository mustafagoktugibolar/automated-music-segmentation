"""
Weighted boundary fusion.

Groups candidate boundaries that fall within a merge window, applies a weighted
scoring scheme, and returns a deduplicated list sorted by time. Candidates from
multiple sources that agree within the merge window reinforce each other.
"""

from __future__ import annotations

import numpy as np

from .models import CandidateBoundary
from shared.logger import get_logger

logger = get_logger("fusion")

_DEFAULT_WEIGHTS: dict[str, float] = {
    "ssm": 0.42,
    "beat_grid": 0.24,
    "chord_proxy": 0.18,
    "beat_phrase": 0.18,
    "onset_flux": 0.06,
    "rms": 0.06,
    "lyrics": 0.10,
    "beat": 0.02,
}


class BoundaryFusion:
    """
    Fuse multiple lists of CandidateBoundary into a single consensus list.

    Candidates are grouped by proximity (merge_window_sec). Within each group,
    a weighted score is computed based on source-specific weights and per-candidate
    confidence. Groups above *threshold* become final boundaries.
    """

    def fuse(
        self,
        candidate_lists: list[list[CandidateBoundary]],
        weights: dict[str, float] | None = None,
        merge_window_sec: float = 1.75,
        threshold: float = 0.30,
    ) -> list[CandidateBoundary]:
        """
        Fuse multiple candidate lists into a single merged list.

        Parameters
        ----------
        candidate_lists   : One list per feature source.
        weights           : Mapping from source name to weight (0–1).
                            Defaults to the system-wide feature weights.
        merge_window_sec  : Candidates within this window are merged into one.
        threshold         : Minimum weighted score to keep a group.

        Returns
        -------
        Merged, sorted list of CandidateBoundary.
        """
        if weights is None:
            weights = _DEFAULT_WEIGHTS

        # Flatten all candidates.
        all_candidates: list[CandidateBoundary] = []
        for clist in candidate_lists:
            all_candidates.extend(clist)

        if not all_candidates:
            return []

        all_candidates.sort(key=lambda c: c.time_seconds)

        # Group candidates within merge_window_sec of each other.
        groups: list[list[CandidateBoundary]] = []
        for cand in all_candidates:
            if not groups:
                groups.append([cand])
                continue
            group_centre = float(
                np.mean([c.time_seconds for c in groups[-1]])
            )
            if abs(cand.time_seconds - group_centre) <= merge_window_sec:
                groups[-1].append(cand)
            else:
                groups.append([cand])

        fused: list[CandidateBoundary] = []
        for group in groups:
            # Per source, keep only the best-confidence candidate.
            best_by_source: dict[str, CandidateBoundary] = {}
            for cand in group:
                for src in cand.source:
                    if src not in best_by_source or cand.confidence > best_by_source[src].confidence:
                        best_by_source[src] = cand

            # Compute weighted score.
            weighted_sum = 0.0
            for src, cand in best_by_source.items():
                w = weights.get(src, 0.01)
                weighted_sum += w * cand.confidence

            # Multi-source bonus (capped at 0.15).
            n_sources = len(best_by_source)
            score = min(1.0, weighted_sum + min(0.15, 0.035 * max(0, n_sources - 1)))

            if score < threshold:
                continue

            # Anchor time: prefer structural sources.
            anchor = self._pick_anchor(best_by_source, weights)
            merged_sources = sorted(best_by_source.keys())

            fused.append(
                CandidateBoundary(
                    time_seconds=round(float(anchor.time_seconds), 3),
                    source=merged_sources,
                    confidence=round(score, 3),
                )
            )

        # Final dedup: if two fused boundaries are still within merge_window_sec,
        # keep the higher-confidence one.
        deduped: list[CandidateBoundary] = []
        for item in sorted(fused, key=lambda c: c.time_seconds):
            if not deduped or (item.time_seconds - deduped[-1].time_seconds) > merge_window_sec:
                deduped.append(item)
            elif item.confidence > deduped[-1].confidence:
                deduped[-1] = item

        logger.debug(
            "BoundaryFusion: %d raw → %d groups → %d fused (threshold=%.2f)",
            len(all_candidates),
            len(groups),
            len(deduped),
            threshold,
        )
        return deduped

    @staticmethod
    def _pick_anchor(
        best_by_source: dict[str, CandidateBoundary],
        weights: dict[str, float],
    ) -> CandidateBoundary:
        """
        Choose which candidate's timestamp to use as the group anchor.

        Prefer structural sources over fine-grained onset/beat sources.
        """
        for preferred in ("beat_grid", "beat_phrase", "ssm", "chord_proxy", "lyrics"):
            if preferred in best_by_source:
                return best_by_source[preferred]
        return max(
            best_by_source.values(),
            key=lambda c: weights.get(c.source[0] if c.source else "", 0.01) * c.confidence,
        )

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class BaseSeedingStrategy(BaseLabelingStrategy):
    """
    Base class for all cell seeding strategies.

    Encapsulates common configuration parameters such as marker gene dictionaries,
    and implements centralized cell label assignment logic, supporting both standard
    winner-takes-all assignment and quota-based budget allocation (`target_frac`).

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    target_frac : float | None, default None
        Fraction of total cells in `adata` to assign labels across qualifying cell types.
        When provided, activates exact budget allocation mode where candidates are ranked by score
        and selected per cluster up to their allocated quota.
    min_cells_per_type : int | None, default None
        Minimum guaranteed number of candidate cells required and allocated for a cell type to be seeded
        when `target_frac` is specified. If a cell type has fewer than `min_cells_per_type` eligible
        candidates, it is excluded from seeding (`unknown`).
    min_score : float | None, default None
        Absolute minimum score threshold required for a cell to be considered eligible during
        both winner-takes-all and quota-based label assignment.
    use_raw : bool, default True
        Whether to calculate scores or thresholds on `adata.raw` if present.
    unknown_label : str, default "unknown"
        Label assigned to cells that receive no seed.
    """

    _repr_exclude: Set[str] = {"markers"}

    def __init__(
        self,
        markers: Dict[str, List[str]],
        target_frac: float | None = None,
        min_cells_per_type: int | None = None,
        min_score: float | None = None,
        use_raw: bool = True,
        unknown_label: str = "unknown",
        **kwargs: Any,
    ) -> None:
        self.markers = markers
        self.target_frac = target_frac
        self.min_cells_per_type = min_cells_per_type
        self.min_score = min_score
        self.use_raw = use_raw
        self.unknown_label = unknown_label

    @property
    @abstractmethod
    def name(self) -> str:
        """Internal short-name identifying the strategy. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def execute_on(self, adata: AnnData) -> LabelingResult:
        """Execute the seeding strategy on the provided AnnData object."""
        pass

    def _assign_labels_from_scores(
        self,
        adata: AnnData,
        scores_df: pd.DataFrame,
        thresholds: Dict[str, float] | None = None,
        extra_uns: Dict[str, Any] | None = None,
    ) -> LabelingResult:
        """
        Assign labels from computed cell type scores using either winner-takes-all or quota selection.

        Parameters
        ----------
        adata : AnnData
            The input AnnData object being annotated.
        scores_df : pd.DataFrame
            DataFrame of dimensions (n_cells, n_cell_types) containing cell signature/marker scores.
        thresholds : Dict[str, float] | None, default None
            Optional dictionary of per-cell-type minimum threshold floors (e.g., from Otsu or QCQ minimum confidence).
        extra_uns : Dict[str, Any] | None, default None
            Additional metadata items to include in the returned `LabelingResult.uns`.

        Returns
        -------
        LabelingResult
            The complete labeling result ready for assignment into `adata`.
        """
        thresholds = thresholds or {}

        # 1. Identify pass mask for each cell across all types based on thresholds and min_score
        pass_mask = pd.DataFrame(False, index=scores_df.index, columns=scores_df.columns)
        for col in scores_df.columns:
            thresh = thresholds.get(col, -np.inf)
            if self.min_score is not None:
                thresh = max(thresh, self.min_score)
            pass_mask[col] = scores_df[col] >= thresh

        has_match = pass_mask.any(axis=1)
        best_match = scores_df.idxmax(axis=1)

        final_labels = pd.Series(self.unknown_label, index=scores_df.index, dtype=str)

        if self.target_frac is None:
            # Winner takes all among cells passing thresholds
            final_labels[has_match] = best_match[has_match]
        else:
            # Quota allocation selection
            final_labels = self._apply_quota_selection(
                scores_df=scores_df,
                pass_mask=pass_mask,
                best_match=best_match,
                has_match=has_match,
            )

        is_confident = final_labels != self.unknown_label
        uns_payload: Dict[str, Any] = {
            "thresholds": thresholds,
            "fraction_assigned": float(is_confident.mean()),
        }
        if extra_uns:
            uns_payload.update(extra_uns)

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={"max_score": scores_df.max(axis=1), "is_confident": is_confident},
            obsm={"scores": scores_df},
            uns=uns_payload,
        )

    def _apply_quota_selection(
        self,
        scores_df: pd.DataFrame,
        pass_mask: pd.DataFrame,
        best_match: pd.Series,
        has_match: pd.Series,
    ) -> pd.Series:
        """
        Execute exact budget allocation across cell types based on `target_frac` and `min_cells_per_type`.

        This is the notebook quota algorithm (``weak_label_quota_with_min_cells``):

        1. A cell is eligible when it clears the threshold for its best-scoring type.
        2. Types with fewer than `min_cells_per_type` eligible cells are dropped; if more
           types remain than ``budget // min_cells_per_type`` can accommodate, the most
           abundant types are kept.
        3. Every kept type gets `min_cells_per_type`; the remaining budget is shared in
           proportion to each type's surplus of eligible cells (floored, leftovers handed
           out one at a time from the largest share), capped by availability, then any
           deficit is refilled greedily from types with spare eligible cells.
        4. Within each type the highest-scoring eligible cells fill the quota.

        Parameters
        ----------
        scores_df : pd.DataFrame
            DataFrame of cell signature scores.
        pass_mask : pd.DataFrame
            Boolean DataFrame indicating which cells clear per-type thresholds and `min_score`.
        best_match : pd.Series
            Series mapping each cell to its highest-scoring cell type across columns.
        has_match : pd.Series
            Boolean Series indicating if a cell clears the threshold for at least one cell type.

        Returns
        -------
        pd.Series
            Series of assigned cell labels (either cell type name or 'unknown').
        """
        final_labels = pd.Series(self.unknown_label, index=scores_df.index, dtype=str)
        n_total = len(scores_df)
        target_total = int(round(n_total * self.target_frac))
        min_cells = self.min_cells_per_type if self.min_cells_per_type is not None else 0

        best_type = best_match.to_numpy().astype(str)
        best_score = scores_df.max(axis=1).to_numpy(dtype=float)
        eligible = (
            has_match.to_numpy()
            & (pass_mask.to_numpy()[np.arange(n_total), scores_df.columns.get_indexer(best_type)])
        )

        # Types with enough eligible cells, most abundant first.
        eligible_counts = pd.Series(best_type[eligible]).value_counts()
        keep_types = eligible_counts[eligible_counts >= max(min_cells, 1)].index.tolist()
        if not keep_types:
            return final_labels

        if min_cells > 0:
            max_types = target_total // min_cells
            if max_types == 0:
                return final_labels
            if len(keep_types) > max_types:
                keep_types = (
                    eligible_counts.loc[keep_types]
                    .sort_values(ascending=False)
                    .index[:max_types]
                    .tolist()
                )

        # Base allocation plus proportional share of the remaining budget.
        base = dict.fromkeys(keep_types, min_cells)
        remaining = max(0, target_total - sum(base.values()))
        avail = eligible_counts.loc[keep_types].to_dict()
        extra = dict.fromkeys(keep_types, 0)
        if remaining > 0:
            weights = np.array([max(0, avail[t] - base[t]) for t in keep_types], dtype=float)
            if weights.sum() > 0:
                props = weights / weights.sum()
                raw_extra = np.floor(props * remaining).astype(int)
                for t, e in zip(keep_types, raw_extra, strict=True):
                    extra[t] = int(e)
                leftover = remaining - sum(extra.values())
                order = np.argsort(-props)
                k = 0
                while leftover > 0:
                    extra[keep_types[order[k % len(order)]]] += 1
                    leftover -= 1
                    k += 1

        quota = {t: int(min(base[t] + extra[t], avail[t])) for t in keep_types}

        # Refill any deficit caused by capping from types with spare eligible cells.
        deficit = target_total - sum(quota.values())
        if deficit > 0:
            spare = {t: avail[t] - quota[t] for t in keep_types}
            for t in sorted(keep_types, key=lambda t: spare[t], reverse=True):
                if deficit <= 0:
                    break
                take = min(deficit, spare[t])
                if take > 0:
                    quota[t] += int(take)
                    deficit -= int(take)

        # Top-scoring eligible cells fill each type's quota.
        eligible_idx = np.where(eligible)[0]
        labels = final_labels.to_numpy().astype(object)
        for t in keep_types:
            idx_t = eligible_idx[best_type[eligible_idx] == t]
            if idx_t.size == 0:
                continue
            idx_sorted = idx_t[np.argsort(-best_score[idx_t])]
            labels[idx_sorted[: quota[t]]] = t

        return pd.Series(labels, index=scores_df.index, dtype=str)

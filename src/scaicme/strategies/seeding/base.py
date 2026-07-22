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
    """

    _repr_exclude: Set[str] = {"markers"}

    def __init__(
        self,
        markers: Dict[str, List[str]],
        target_frac: float | None = None,
        min_cells_per_type: int | None = None,
        min_score: float | None = None,
        use_raw: bool = True,
        **kwargs: Any,
    ) -> None:
        self.markers = markers
        self.target_frac = target_frac
        self.min_cells_per_type = min_cells_per_type
        self.min_score = min_score
        self.use_raw = use_raw

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

        final_labels = pd.Series("unknown", index=scores_df.index, dtype=str)

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

        is_confident = final_labels != "unknown"
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
        final_labels = pd.Series("unknown", index=scores_df.index, dtype=str)
        n_total = len(scores_df)
        total_budget = int(round(n_total * self.target_frac))
        min_cells = self.min_cells_per_type if self.min_cells_per_type is not None else 0

        # Collect candidate cells per cluster (cells where best_match is cluster and threshold cleared)
        candidates_per_type: Dict[str, List[str]] = {}
        for col in scores_df.columns:
            mask = has_match & (best_match == col) & pass_mask[col]
            candidates_per_type[col] = scores_df.index[mask].tolist()

        # Filter out types with fewer than min_cells eligible candidates
        eligible_types = [
            col for col, cands in candidates_per_type.items() if len(cands) >= min_cells and len(cands) > 0
        ]

        if not eligible_types:
            return final_labels

        # Calculate quotas per eligible cell type
        quotas: Dict[str, int] = {col: min_cells for col in eligible_types}
        base_total = sum(quotas.values())

        if total_budget > base_total:
            remaining_budget = total_budget - base_total
            total_avail = sum(len(candidates_per_type[col]) - min_cells for col in eligible_types)
            if total_avail > 0:
                # Proportional distribution of remaining budget based on available candidates above min_cells
                for col in eligible_types:
                    avail = len(candidates_per_type[col]) - min_cells
                    if avail > 0:
                        add = int(round(remaining_budget * (avail / total_avail)))
                        quotas[col] = min(min_cells + add, len(candidates_per_type[col]))

                # Greedily refill deficit or trim excess to match total_budget as closely as possible
                current_total = sum(quotas.values())
                while current_total < total_budget:
                    added = False
                    # Sort eligible types by highest available spare capacity
                    for col in sorted(
                        eligible_types,
                        key=lambda c: len(candidates_per_type[c]) - quotas[c],
                        reverse=True,
                    ):
                        if quotas[col] < len(candidates_per_type[col]):
                            quotas[col] += 1
                            current_total += 1
                            added = True
                            if current_total == total_budget:
                                break
                    if not added:
                        break
        else:
            # If base quota exceeds total budget, scale down across available
            for col in eligible_types:
                quotas[col] = min(quotas[col], len(candidates_per_type[col]))

        # Assign top q scoring cells for each eligible cell type
        for col, q in quotas.items():
            if q <= 0:
                continue
            cands = candidates_per_type[col]
            # Sort candidate indices by score in descending order
            sorted_cands = sorted(cands, key=lambda idx: scores_df.loc[idx, col], reverse=True)
            top_cands = sorted_cands[:q]
            final_labels.loc[top_cands] = col

        return final_labels

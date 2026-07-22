from typing import Any, Dict, List

import numpy as np
import pandas as pd
from anndata import AnnData

from ..base import LabelingResult
from .base import BaseSeedingStrategy


class QCQAdaptiveSeeding(BaseSeedingStrategy):
    """
    Quality-Checked Quantile (QCQ) Per-Gene Adaptive Thresholding for Seed Generation.

    Thresholds each marker gene independently using per-gene positive-expression quantiles,
    then assigns cell labels based on the fraction of active markers per cell.

    A cell is assigned a label if:
    1. Its active marker fraction for a cell type exceeds the hard minimum confidence value.
    2. It has the highest score among all qualifying types (winner-takes-all), or falls within
       the allocated quota when `target_frac` is provided.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    quantile : float, default 0.95
        The percentile used to compute each gene's positive-expression threshold.
    min_confidence : float, default 0.2
        The minimum fraction of active markers required to assign a label. Filters out weak matches
        even if the per-gene thresholds are satisfied.
    use_raw : bool, default True
        Whether to calculate thresholds and active marker fractions on `adata.raw` if present.
    target_frac : float | None, default None
        Fraction of total cells in `adata` to assign labels across qualifying cell types.
        When provided, activates quota-based budget allocation mode.
    min_cells_per_type : int | None, default None
        Minimum guaranteed number of seed candidates allocated per qualifying cluster when `target_frac`
        is specified.
    min_score : float | None, default None
        Absolute minimum score threshold required for eligibility during label selection.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        quantile: float = 0.95,
        min_confidence: float = 0.2,
        use_raw: bool = True,
        target_frac: float | None = None,
        min_cells_per_type: int | None = None,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            markers=markers,
            target_frac=target_frac,
            min_cells_per_type=min_cells_per_type,
            min_score=min_score,
            use_raw=use_raw,
            **kwargs,
        )
        self.quantile = quantile
        self.min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "qcq_adaptive_seeding"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        # 1. Calculate active marker fractions per cell type.
        scores_df = pd.DataFrame(index=adata.obs_names)
        gene_thresholds = {}

        for cell_type, genes in self.markers.items():
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names or (self.use_raw and adata.raw and g in adata.raw.var_names)
            ]

            if not valid_genes:
                scores_df[cell_type] = 0.0
                gene_thresholds[cell_type] = {}
                continue

            if self.use_raw and adata.raw is not None:
                X = adata.raw[:, valid_genes].X
            else:
                X = adata[:, valid_genes].X

            if hasattr(X, "toarray"):
                X = X.toarray()

            X = np.asarray(X)

            per_gene_thresholds = np.full(len(valid_genes), np.inf, dtype=float)
            active_mask = np.zeros_like(X, dtype=bool)
            for gene_idx in range(X.shape[1]):
                gene_values = X[:, gene_idx]
                positive_values = gene_values[gene_values > 0]
                if positive_values.size > 0:
                    threshold = float(np.quantile(positive_values, self.quantile))
                    per_gene_thresholds[gene_idx] = threshold
                    active_mask[:, gene_idx] = gene_values > threshold

            scores_df[cell_type] = active_mask.mean(axis=1)
            gene_thresholds[cell_type] = dict(
                zip(valid_genes, per_gene_thresholds.tolist(), strict=True)
            )

        # 2. Determine Thresholds (QC floor on the active-marker fraction).
        thresholds = dict.fromkeys(scores_df.columns, self.min_confidence)

        # 3. Assign Labels via centralized BaseSeedingStrategy method
        return self._assign_labels_from_scores(
            adata=adata,
            scores_df=scores_df,
            thresholds=thresholds,
            extra_uns={"gene_thresholds": gene_thresholds},
        )

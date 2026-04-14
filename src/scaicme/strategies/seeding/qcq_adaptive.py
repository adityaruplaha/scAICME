from typing import Dict, List

import numpy as np
import pandas as pd
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class QCQAdaptiveSeeding(BaseLabelingStrategy):
    """
    Quality-Checked Quantile (QCQ) Per-Gene Adaptive Thresholding for Seed Generation.

    Thresholds each marker gene independently using per-gene positive-expression quantiles,
    then assigns cell labels based on the fraction of active markers per cell.

    A cell is assigned a label if:
    1. Its active marker fraction for a cell type exceeds the hard minimum confidence value.
    2. It has the highest score among all qualifying types (winner-takes-all).

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
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        quantile: float = 0.95,
        min_confidence: float = 0.2,
        use_raw: bool = True,
        **kwargs,
    ):
        self.markers = markers
        self.quantile = quantile
        self.min_confidence = min_confidence
        self.use_raw = use_raw

    @property
    def name(self) -> str:
        return "qcq_adaptive_seeding"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        # 1. Calculate active marker fractions per cell type.
        # We store scores in a DataFrame: index=cells, columns=cell_types.
        scores_df = pd.DataFrame(index=adata.obs_names)
        gene_thresholds = {}

        for cell_type, genes in self.markers.items():
            # Filter genes that exist in the dataset.
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names or (self.use_raw and adata.raw and g in adata.raw.var_names)
            ]

            if not valid_genes:
                # If no markers found, score is 0.
                scores_df[cell_type] = 0.0
                gene_thresholds[cell_type] = {}
                continue

            if self.use_raw and adata.raw is not None:
                X = adata.raw[:, valid_genes].X
            else:
                X = adata[:, valid_genes].X

            # Handle sparse matrices.
            if hasattr(X, "toarray"):
                X = X.toarray()

            X = np.asarray(X)

            # Compute a per-gene threshold from positive expression values.
            per_gene_thresholds = np.full(len(valid_genes), np.inf, dtype=float)
            active_mask = np.zeros_like(X, dtype=bool)
            for gene_idx in range(X.shape[1]):
                gene_values = X[:, gene_idx]
                positive_values = gene_values[gene_values > 0]
                if positive_values.size > 0:
                    threshold = float(np.quantile(positive_values, self.quantile))
                    per_gene_thresholds[gene_idx] = threshold
                    active_mask[:, gene_idx] = gene_values > threshold

            # Score each cell by the fraction of markers that are active.
            scores_df[cell_type] = active_mask.mean(axis=1)
            gene_thresholds[cell_type] = dict(
                zip(valid_genes, per_gene_thresholds.tolist(), strict=True)
            )

        # 2. Determine Thresholds (QC floor on the active-marker fraction).
        thresholds = dict.fromkeys(scores_df.columns, self.min_confidence)

        # 3. Assign Labels
        final_labels = pd.Series("unknown", index=adata.obs_names)

        # Identify candidate cells (True if active-marker fraction clears the floor).
        pass_mask = pd.DataFrame(False, index=scores_df.index, columns=scores_df.columns)
        for col, thresh in thresholds.items():
            pass_mask[col] = scores_df[col] >= thresh

        # For cells passing at least one threshold, pick the max score
        # idxmax returns the column name (cell type) with the highest value
        # We only apply this to rows where at least one value is True
        has_match = pass_mask.any(axis=1)

        # "Winner Takes All" among passing types
        # Note: We look at the original scores_df to find the max, but only for valid rows
        best_match = scores_df.idxmax(axis=1)

        final_labels[has_match] = best_match[has_match]

        # 4. Return Rich Result
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={"max_score": scores_df.max(axis=1), "is_confident": has_match},
            obsm={"scores": scores_df},
            uns={
                "thresholds": thresholds,
                "gene_thresholds": gene_thresholds,
                "fraction_assigned": has_match.mean(),
            },
        )

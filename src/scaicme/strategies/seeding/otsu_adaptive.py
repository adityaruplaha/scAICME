from typing import Dict, List

import numpy as np
import pandas as pd
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class OtsuAdaptiveSeeding(BaseLabelingStrategy):
    """
    Otsu's Per-Gene Adaptive Thresholding for Seed Generation.

    Thresholds each marker gene independently using Otsu's method, then assigns cell
    labels based on the fraction of active markers per cell.

    A cell is assigned a label if:
    1. Its active marker fraction for a cell type exceeds a hard minimum confidence value.
    2. It has the highest score among all qualifying types (winner-takes-all).

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    bins : int, default 256
        Number of histogram bins to use for Otsu's threshold calculation.
        Higher values give more precise thresholds but are slightly slower.
    min_confidence : float, default 0.05
        Absolute minimum active marker fraction required to be considered.
    use_raw : bool, default True
        Whether to calculate thresholds and active marker fractions on `adata.raw` if present.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        bins: int = 256,
        min_confidence: float = 0.05,
        use_raw: bool = True,
        **kwargs,
    ):
        self.markers = markers
        self.bins = bins
        self.min_confidence = min_confidence
        self.use_raw = use_raw

    @property
    def name(self) -> str:
        return "otsu_adaptive_seeding"

    def _calculate_otsu_threshold(self, vals: np.ndarray) -> float:
        """Pure numpy implementation of Otsu's thresholding."""
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            return 0.0
        if float(vals.max()) == float(vals.min()):
            return float(vals.max())

        hist, bin_edges = np.histogram(vals, bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        weight1 = np.cumsum(hist).astype(float)
        weight2 = np.cumsum(hist[::-1]).astype(float)[::-1]

        sum1 = np.cumsum(hist * bin_centers)
        sum2 = np.cumsum((hist * bin_centers)[::-1])[::-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            mean1 = sum1 / weight1
            mean2 = sum2 / weight2

        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        variance12[np.isnan(variance12)] = 0

        return float(bin_centers[np.argmax(variance12)])

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
                threshold = self._calculate_otsu_threshold(gene_values)
                per_gene_thresholds[gene_idx] = threshold
                active_mask[:, gene_idx] = gene_values > threshold

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

        # For cells passing at least one threshold, pick the max score.
        has_match = pass_mask.any(axis=1)
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
                "fraction_assigned": float(has_match.mean()),
            },
        )

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class OtsuAdaptiveSeeding(BaseLabelingStrategy):
    """
    Otsu's Adaptive Thresholding for Seed Generation.

    Uses Otsu's method (traditionally used in computer vision for image binarization)
    to automatically find the optimal threshold that separates the marker score
    distribution into two classes: "Background" and "Signal".

    It calculates the threshold that maximizes inter-class variance, making it a
    parameter-free alternative to quantile-based thresholding.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    bins : int, default 256
        Number of histogram bins to use for Otsu's threshold calculation.
        Higher values give more precise thresholds but are slightly slower.
    min_score : float, default 0.05
        Absolute minimum score required. Acts as a Quality Check (QC) floor
        in case Otsu's method splits a unimodal noise distribution too low.
    use_raw : bool, default True
        Whether to calculate scores on `adata.raw` if present.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        bins: int = 256,
        min_score: float = 0.05,
        use_raw: bool = True,
        **kwargs,
    ):
        self.markers = markers
        self.bins = bins
        self.min_score = min_score
        self.use_raw = use_raw

    @property
    def name(self) -> str:
        return "otsu_adaptive_seeding"

    def _calculate_otsu_threshold(self, vals: np.ndarray) -> float:
        """Pure numpy implementation of Otsu's thresholding."""
        # Remove NaNs and handle edge cases
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return 0.0
        if vals.max() == vals.min():
            return float(vals.max())

        # 1. Compute histogram and probabilities
        hist, bin_edges = np.histogram(vals, bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. Cumulative weights (probabilities of being in class 1 or 2)
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # Avoid division by zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # 3. Cumulative means
            mean1 = np.cumsum(hist * bin_centers) / weight1
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # 4. Calculate between-class variance
        # $\sigma_b^2 = w_1 w_2 (\mu_1 - \mu_2)^2$
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        # Ignore NaNs that resulted from division by zero
        variance12[np.isnan(variance12)] = 0

        # 5. The threshold is the bin center that maximizes the variance
        idx = np.argmax(variance12)
        return float(bin_centers[idx])

    def execute_on(self, adata: AnnData) -> LabelingResult:
        # 1. Calculate Scores
        scores_df = pd.DataFrame(index=adata.obs_names)

        for cell_type, genes in self.markers.items():
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names or (self.use_raw and adata.raw and g in adata.raw.var_names)
            ]

            if not valid_genes:
                scores_df[cell_type] = 0.0
                continue

            temp_key = f"_temp_score_{cell_type}"
            try:
                sc.tl.score_genes(
                    adata, gene_list=valid_genes, score_name=temp_key, use_raw=self.use_raw
                )
                scores_df[cell_type] = adata.obs[temp_key].values
                del adata.obs[temp_key]
            except Exception:
                if self.use_raw and adata.raw is not None:
                    X = adata.raw[:, valid_genes].X
                else:
                    X = adata[:, valid_genes].X

                if hasattr(X, "toarray"):
                    X = X.toarray()
                scores_df[cell_type] = np.mean(X, axis=1)

        # 2. Determine Thresholds (Otsu + QC Floor)
        thresholds = {}
        for col in scores_df.columns:
            # Otsu's mathematically optimal split
            otsu_val = self._calculate_otsu_threshold(scores_df[col].values)
            # QC: Must be at least min_score
            thresholds[col] = max(otsu_val, self.min_score)

        # 3. Assign Labels
        final_labels = pd.Series("unknown", index=adata.obs_names)

        # Identify candidate cells (True if score > threshold)
        pass_mask = pd.DataFrame(False, index=scores_df.index, columns=scores_df.columns)
        for col, thresh in thresholds.items():
            pass_mask[col] = scores_df[col] > thresh

        # For cells passing at least one threshold, pick the max score
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
            uns={"thresholds": thresholds, "fraction_assigned": float(has_match.mean())},
        )

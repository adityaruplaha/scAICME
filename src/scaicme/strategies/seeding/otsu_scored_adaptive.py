from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..base import LabelingResult
from .base import BaseSeedingStrategy


class OtsuScoredAdaptiveSeeding(BaseSeedingStrategy):
    """
    Otsu's Adaptive Thresholding on Scored Markers for Seed Generation.

    Scores each marker gene set for every cell, thresholds them using Otsu's method, then assigns labels to the top scoring cells.

    A cell is assigned a label if:
    1. Its score for a cell type exceeds the Otsu threshold and the hard minimum score value.
    2. It has the highest score among all qualifying types (winner-takes-all), or falls within
       the allocated quota when `target_frac` is provided.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    bins : int, default 256
        Number of histogram bins to use for Otsu's threshold calculation.
        Higher values give more precise thresholds but are slightly slower.
    min_score : float, default 0.05
        Absolute minimum score required to be considered.
    use_raw : bool, default True
        Whether to calculate scores on `adata.raw` if present.
    target_frac : float | None, default None
        Fraction of total cells in `adata` to assign labels across qualifying cell types.
    min_cells_per_type : int | None, default None
        Minimum guaranteed number of seed candidates allocated per qualifying cluster when `target_frac`
        is specified.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        bins: int = 256,
        min_score: float = 0.05,
        use_raw: bool = True,
        target_frac: float | None = None,
        min_cells_per_type: int | None = None,
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
        self.bins = bins

    @property
    def name(self) -> str:
        return "otsu_scored_adaptive_seeding"

    def _calculate_otsu_threshold(self, vals: np.ndarray) -> float:
        """Pure numpy implementation of Otsu's thresholding."""
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return 0.0
        if vals.max() == vals.min():
            return float(vals.max())

        hist, bin_edges = np.histogram(vals, bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        variance12[np.isnan(variance12)] = 0

        idx = int(np.argmax(variance12))
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
                    adata,
                    gene_list=valid_genes,
                    score_name=temp_key,
                    use_raw=self.use_raw,
                    ctrl_size=50,
                    n_bins=25,
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
            otsu_val = self._calculate_otsu_threshold(scores_df[col].values)
            if self.min_score is not None:
                thresholds[col] = max(otsu_val, self.min_score)
            else:
                thresholds[col] = otsu_val

        # 3. Assign Labels via centralized BaseSeedingStrategy method
        return self._assign_labels_from_scores(
            adata=adata,
            scores_df=scores_df,
            thresholds=thresholds,
        )

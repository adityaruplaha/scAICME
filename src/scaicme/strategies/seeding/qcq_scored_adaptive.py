from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..base import LabelingResult
from .base import BaseSeedingStrategy


class QCQScoredAdaptiveSeeding(BaseSeedingStrategy):
    """
    Quality-Checked Quantile (QCQ) Adaptive Thresholding on Scored Markers for Seed Generation.

    Scores each marker gene set for every cell, then assigns cell labels based on those
    gene-set scores.

    A cell is assigned a label if:
    1. Its score for a cell type exceeds the population-based quantile threshold and the
       hard minimum score value.
    2. It has the highest score among all qualifying types (winner-takes-all), or falls within
       the allocated quota when `target_frac` is provided.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    quantile : float, default 0.95
        The percentile of the score distribution to use as the threshold.
        For example, 0.95 means only the top 5% of cells for a given signature are candidates.
    min_score : float, default 0.05
        Absolute minimum score required to be considered. Filters out weak matches even if
        they are in the top quantile.
    use_raw : bool, default True
        Whether to calculate scores on `adata.raw` if present.
    target_frac : float | None, default None
        Fraction of total cells in `adata` to assign labels across qualifying cell types.
        When provided, activates quota-based budget allocation mode.
    min_cells_per_type : int | None, default None
        Minimum guaranteed number of seed candidates allocated per qualifying cluster when `target_frac`
        is specified.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        quantile: float = 0.95,
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
        self.quantile = quantile

    @property
    def name(self) -> str:
        return "qcq_adaptive_seeding"

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
                if self.use_raw and adata.raw:
                    X = adata.raw[:, valid_genes].X
                else:
                    X = adata[:, valid_genes].X

                if hasattr(X, "toarray"):
                    X = X.toarray()

                scores_df[cell_type] = np.mean(X, axis=1)

        # 2. Determine Thresholds (Adaptive + QC)
        thresholds = {}
        for col in scores_df.columns:
            q_val = float(scores_df[col].quantile(self.quantile))
            if self.min_score is not None:
                thresholds[col] = max(q_val, self.min_score)
            else:
                thresholds[col] = q_val

        # 3. Assign Labels via centralized BaseSeedingStrategy method
        return self._assign_labels_from_scores(
            adata=adata,
            scores_df=scores_df,
            thresholds=thresholds,
        )

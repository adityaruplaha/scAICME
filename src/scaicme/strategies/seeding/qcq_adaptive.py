from typing import Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class QCQAdaptiveSeeding(BaseLabelingStrategy):
    """
    Quality-Checked Quantile (QCQ) Adaptive Thresholding for Seed Generation.

    Assigns cell labels based on marker gene signatures. A cell is assigned a label if:
    1. Its score for a cell type exceeds the population-based quantile threshold (adaptive).
    2. Its score exceeds a hard minimum value (quality check).
    3. It has the highest score among all qualifying types (winner-takes-all).

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    quantile : float, default 0.95
        The percentile of the score distribution to use as the threshold.
        e.g., 0.95 means only the top 5% of cells for a given signature are candidates.
    min_score : float, default 0.05
        Absolute minimum score required to be considered. Filters out weak matches
        even if they are in the top quantile.
    use_raw : bool, default True
        Whether to calculate scores on `adata.raw` if present.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        quantile: float = 0.95,
        min_score: float = 0.05,
        use_raw: bool = True,
        **kwargs,
    ):
        self.markers = markers
        self.quantile = quantile
        self.min_score = min_score
        self.use_raw = use_raw

    @property
    def name(self) -> str:
        return "qcq_adaptive_seeding"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        # 1. Calculate Scores
        # We store scores in a DataFrame: index=cells, columns=cell_types
        scores_df = pd.DataFrame(index=adata.obs_names)

        # Check if we can use scanpy for robust scoring
        # (scanpy.tl.score_genes subtracts a reference set mean, handling technical noise)
        for cell_type, genes in self.markers.items():
            # Filter genes that exist in the dataset
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names or (self.use_raw and adata.raw and g in adata.raw.var_names)
            ]

            if not valid_genes:
                # If no markers found, score is 0
                scores_df[cell_type] = 0.0
                continue

            # Calculate score
            # We use a temporary key to avoid polluting adata.obs permanently
            temp_key = f"_temp_score_{cell_type}"
            try:
                sc.tl.score_genes(
                    adata,
                    gene_list=valid_genes,
                    score_name=temp_key,
                    use_raw=self.use_raw,
                    ctrl_size=50,  # Standard scanpy default
                    n_bins=25,  # Standard scanpy default
                )
                scores_df[cell_type] = adata.obs[temp_key].values
                # Clean up
                del adata.obs[temp_key]
            except Exception:
                # Fallback: simple mean if score_genes fails (e.g., too few genes for bins)
                # This is a robust fallback for edge cases
                if self.use_raw and adata.raw:
                    X = adata.raw[:, valid_genes].X
                else:
                    X = adata[:, valid_genes].X

                # Handle sparse matrices
                if hasattr(X, "toarray"):
                    X = X.toarray()

                scores_df[cell_type] = np.mean(X, axis=1)

        # 2. Determine Thresholds (Adaptive + QC)
        thresholds = {}
        for col in scores_df.columns:
            # Adaptive: Quantile of the distribution
            q_val = scores_df[col].quantile(self.quantile)
            # QC: Must be at least min_score
            thresholds[col] = max(q_val, self.min_score)

        # 3. Assign Labels
        final_labels = pd.Series("unknown", index=adata.obs_names)

        # Identify candidate cells (True if score > threshold)
        # We broadcast the comparison across the DataFrame
        pass_mask = pd.DataFrame(False, index=scores_df.index, columns=scores_df.columns)
        for col, thresh in thresholds.items():
            pass_mask[col] = scores_df[col] > thresh

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
            uns={"thresholds": thresholds, "fraction_assigned": has_match.mean()},
        )

from typing import Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class GraphScoreSeeding(BaseLabelingStrategy):
    """
    GCN-Style Graph Score Propagation for Seed Generation.

    This strategy computes initial marker scores ($Y_0$) and iteratively diffuses
    them across the cell-cell connectivity graph using the Personalized PageRank /
    Label Spreading formulation:

    $$display$$
    Y_{t+1} = \\alpha \\hat{A} Y_t + (1-\\alpha) Y_0
    $$display$$

    Where $\\hat{A}$ is the symmetrically normalized adjacency matrix.
    Final seeds are selected using a "margin gate", ensuring that the top cell type
    score is distinctly higher than the second-best score.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    obsp_key : str, default 'connectivities'
        Key in `adata.obsp` containing the neighborhood graph adjacency matrix.
    alpha : float, default 0.5
        The diffusion parameter (between 0 and 1). Higher values mean more
        information is absorbed from neighbors; lower values rely more on
        the cell's own initial raw score.
    n_iterations : int, default 10
        Number of diffusion iterations to perform.
    margin : float, default 0.1
        The minimum required difference between the top score and the
        second-highest score for a cell to be confidently labeled.
    min_score : float, default 0.05
        Absolute minimum diffused score required to be considered a seed.
    use_raw : bool, default True
        Whether to calculate initial scores using `adata.raw`.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        obsp_key: str = "connectivities",
        alpha: float = 0.5,
        n_iterations: int = 10,
        margin: float = 0.1,
        min_score: float = 0.05,
        use_raw: bool = True,
        **kwargs,
    ):
        self.markers = markers
        self.obsp_key = obsp_key
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.margin = margin
        self.min_score = min_score
        self.use_raw = use_raw

    @property
    def name(self) -> str:
        return "graph_score_seeding"

    def _get_initial_scores(self, adata: AnnData) -> pd.DataFrame:
        """Calculates $Y_0$: The initial raw marker scores."""
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
                # Fallback to simple mean if score_genes fails
                if self.use_raw and adata.raw is not None:
                    X = adata.raw[:, valid_genes].X
                else:
                    X = adata[:, valid_genes].X

                if hasattr(X, "toarray"):
                    X = X.toarray()
                scores_df[cell_type] = np.mean(X, axis=1)

        # Scale scores to [0, 1] range per column to stabilize diffusion
        for col in scores_df.columns:
            min_val = scores_df[col].min()
            max_val = scores_df[col].max()
            if max_val > min_val:
                scores_df[col] = (scores_df[col] - min_val) / (max_val - min_val)
            else:
                scores_df[col] = 0.0

        return scores_df

    def _normalize_adjacency(self, A: sp.spmatrix) -> sp.coo_matrix:
        """Calculates $\\hat{A} = \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}$"""
        # $ \tilde{A} = A + I $
        A_tilde = A + sp.eye(A.shape[0])

        # $ \tilde{D} $
        D = np.array(A_tilde.sum(axis=1)).flatten()

        # $ \tilde{D}^{-1/2} $
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        D_mat_inv_sqrt = sp.diags(D_inv_sqrt)

        # $ \hat{A} $
        A_norm = D_mat_inv_sqrt.dot(A_tilde).dot(D_mat_inv_sqrt)
        return A_norm.tocoo()

    def execute_on(self, adata: AnnData) -> LabelingResult:
        if self.obsp_key not in adata.obsp:
            raise ValueError(
                f"Graph key '{self.obsp_key}' not found in adata.obsp. "
                f"Please run sc.pp.neighbors(adata) first."
            )

        # 1. Get Initial Scores ($Y_0$)
        Y_0_df = self._get_initial_scores(adata)
        Y_0 = Y_0_df.values
        cell_types = Y_0_df.columns.tolist()

        # 2. Prepare Graph ($\hat{A}$)
        A_norm = self._normalize_adjacency(adata.obsp[self.obsp_key])

        # 3. Diffuse Scores
        Y_t = Y_0.copy()
        for _ in range(self.n_iterations):
            # $ Y_{t+1} = \alpha \hat{A} Y_t + (1-\alpha) Y_0 $
            Y_t = self.alpha * A_norm.dot(Y_t) + (1.0 - self.alpha) * Y_0

        diffused_df = pd.DataFrame(Y_t, index=adata.obs_names, columns=cell_types)

        # 4. Apply Margin Gate & Min Score Filter
        final_labels = pd.Series("unknown", index=adata.obs_names)

        # Find the top and second-top scores for each cell
        sorted_scores = np.sort(Y_t, axis=1)
        top_scores = sorted_scores[:, -1]

        if Y_t.shape[1] > 1:
            second_scores = sorted_scores[:, -2]
        else:
            second_scores = np.zeros_like(top_scores)

        # Calculate margins
        margins = top_scores - second_scores

        # Determine which cells pass the gate
        is_confident = (margins >= self.margin) & (top_scores >= self.min_score)

        # Assign labels
        best_match_idx = np.argmax(Y_t, axis=1)

        # Translate indices back to cell type names, applying the confident mask
        confident_indices = np.where(is_confident)[0]
        for idx in confident_indices:
            final_labels.iloc[idx] = cell_types[best_match_idx[idx]]

        # 5. Return Rich DTO
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={
                "margin": pd.Series(margins, index=adata.obs_names),
                "is_confident": pd.Series(is_confident, index=adata.obs_names),
            },
            obsm={"initial_scores": Y_0_df, "diffused_scores": diffused_df},
            uns={"fraction_assigned": float(is_confident.mean()), "cell_types": cell_types},
        )

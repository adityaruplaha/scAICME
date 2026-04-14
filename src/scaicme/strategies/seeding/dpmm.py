from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from ..base import BaseLabelingStrategy, LabelingResult


class DPMMClusteredAdaptiveSeeding(BaseLabelingStrategy):
    r"""
    Dirichlet Process Mixture Model (DPMM) for Robust Cell Type Labeling.

    This strategy fits a Bayesian Gaussian Mixture Model with a Dirichlet Process
    prior to marker gene expression for each cell type. It uses per-gene expression
    thresholds to identify signal clusters and automatically discovers the optimal
    number of expression clusters without requiring manual specification.

    Algorithm:
    1. Compute per-gene thresholds from positive expression quantiles (configurable)
    2. Build marker activation score per cell (fraction of markers above threshold)
    3. Fit a BGM with DP prior to standardized marker expression (n_cells × n_markers)
    4. For each cell, compute posterior probabilities across components
    5. Filter components: keep only those with mean marker score >= cluster_score_min AND size >= min_cells_cluster
    6. For each cell, prob_signal = sum of posterior probabilities for signal components
    7. A cell is assigned to a cell type if its prob_signal >= min_confidence
    8. If multiple cell types are confident, assign the one with highest prob_signal

    The Dirichlet Process prior automatically collapses inactive components at
    convergence, effectively performing model selection to find the true number
    of expression clusters present in the data.

    Note: This strategy assumes that the input data (`adata.X` or `adata.raw.X`)
    is already log-normalized.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    min_confidence : float, default 0.8
        Minimum prob_signal threshold for assigning a cell to a cell type.
    high_expr_quantile : float or Dict[str, float], default 0.70
        Quantile used to set per-gene thresholds from positive expression values,
        or a mapping from marker gene to a gene-specific quantile. If a mapping is
        provided, genes not present in the mapping fall back to 0.70.
    min_cells_per_gene : int, default 20
        Minimum number of positive cells required to compute a per-gene threshold.
    min_expressed_markers : int, default 2
        Minimum number of markers that must be expressed above `min_cells_per_gene`.
    min_cell_enrichment : float, default 0.10
        Minimum fraction of cells with any marker expression to proceed.
    cluster_score_min : float, default 0.20
        Minimum mean marker score for a cluster to be considered signal.
    min_cells_cluster : int, default 5
        Minimum cluster size to be considered signal.
    weight_concentration_prior : float, default 0.05
        Dirichlet Process concentration parameter ($\gamma$). Controls component
        collapse behavior; lower values favor fewer active components at convergence.
    max_iter : int, default 1000
        Maximum iterations for DPMM EM algorithm fitting.
    use_raw : bool, default True
        Whether to extract marker expression from `adata.raw` (recommended for
        log-normalized data).
    random_state : int | None, default None
        Random seed for reproducible convergence.
    n_jobs : int, default 4
        Number of threads for parallel per-cell-type fitting.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        min_confidence: float = 0.8,
        high_expr_quantile: float | Dict[str, float] = 0.70,
        min_cells_per_gene: int = 20,
        min_expressed_markers: int = 2,
        min_cell_enrichment: float = 0.10,
        cluster_score_min: float = 0.20,
        min_cells_cluster: int = 5,
        weight_concentration_prior: float = 0.05,
        max_iter: int = 1000,
        use_raw: bool = True,
        random_state: int | None = None,
        n_jobs: int = 4,
        **kwargs,
    ):
        self.markers = markers
        self.min_confidence = min_confidence
        if isinstance(high_expr_quantile, defaultdict):
            self.per_gene_high_expr_quantile = high_expr_quantile
        elif isinstance(high_expr_quantile, dict):
            self.per_gene_high_expr_quantile = defaultdict(lambda: 0.70)
            self.per_gene_high_expr_quantile.update(high_expr_quantile)
        else:
            self.per_gene_high_expr_quantile = defaultdict(lambda: float(high_expr_quantile))
        self.min_cells_per_gene = min_cells_per_gene
        self.min_expressed_markers = min_expressed_markers
        self.min_cell_enrichment = min_cell_enrichment
        self.cluster_score_min = cluster_score_min
        self.min_cells_cluster = min_cells_cluster
        self.weight_concentration_prior = weight_concentration_prior
        self.max_iter = max_iter
        self.use_raw = use_raw
        self.random_state = random_state
        self.n_jobs = n_jobs

    @property
    def name(self) -> str:
        return "dpmm_clustered_adaptive_seeding"

    def _fit_single_ctype_adaptive(
        self, X: np.ndarray, gene_quantiles: List[float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit a marker-score DPMM for a single cell type's marker genes using the per-gene thresholding approach.

        This method:
        1. Computes per-gene thresholds from positive values.
        2. Builds a marker activation score per cell.
        3. Fits a DPMM on standardized marker expression (no PCA).
        4. Uses high marker-score clusters as signal components.

        Parameters
        ----------
        X : np.ndarray
            Expression matrix of shape (n_cells, n_markers) for this cell type.
        gene_quantiles : List[float]
            Per-gene quantile thresholds aligned to the columns of `X`.

        Returns
        -------
        prob_signal : np.ndarray
            Probability of belonging to a high marker-score component, shape (n_cells,).
        stats : Dict[str, Any]
            Convergence and component analysis diagnostics.
        """
        # Edge case: No data or completely zeroed data
        if X.shape[1] == 0 or np.all(X == 0):
            return np.zeros(X.shape[0]), {
                "converged": True,
                "n_total_components": 0,
                "n_signal_components": 0,
                "background_component_idx": None,
                "signal_component_indices": [],
                "component_sizes": [],
                "component_mean_sums": [],
                "marker_score_thresholds": [],
                "marker_score_cluster_means": [],
            }

        # Marker-score thresholds (Method 2, without PCA)
        expressed_cells_per_gene = (X > 0).sum(axis=0)
        expressed_marker_count = int(np.sum(expressed_cells_per_gene >= self.min_cells_per_gene))
        if expressed_marker_count < self.min_expressed_markers:
            return np.zeros(X.shape[0]), {
                "converged": True,
                "n_total_components": 0,
                "n_signal_components": 0,
                "background_component_idx": None,
                "signal_component_indices": [],
                "component_sizes": [],
                "component_mean_sums": [],
                "marker_score_thresholds": [],
                "marker_score_cluster_means": [],
                "reason": "insufficient_expressed_markers",
            }

        has_any_marker = (X > 0).sum(axis=1) > 0
        cell_enrichment = float(np.mean(has_any_marker))
        if cell_enrichment < self.min_cell_enrichment:
            return np.zeros(X.shape[0]), {
                "converged": True,
                "n_total_components": 0,
                "n_signal_components": 0,
                "background_component_idx": None,
                "signal_component_indices": [],
                "component_sizes": [],
                "component_mean_sums": [],
                "marker_score_thresholds": [],
                "marker_score_cluster_means": [],
                "reason": "low_cell_enrichment",
            }

        gene_thresholds = np.full(X.shape[1], np.inf, dtype=float)
        for gene_idx, gene_quantile in enumerate(gene_quantiles):
            gene_values = X[:, gene_idx]
            positive_values = gene_values[gene_values > 0]
            if positive_values.size >= self.min_cells_per_gene:
                gene_thresholds[gene_idx] = float(np.quantile(positive_values, gene_quantile))
            elif positive_values.size > 0:
                gene_thresholds[gene_idx] = float(np.median(positive_values))

        marker_activation_mask = X > gene_thresholds
        marker_activation_score = marker_activation_mask.sum(axis=1) / max(1, X.shape[1])

        scaler = StandardScaler(with_mean=True, with_std=True)
        standardized_expression = scaler.fit_transform(X)

        n_components = max(2, X.shape[0] // 10)
        bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=self.weight_concentration_prior,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        bgm.fit(standardized_expression)
        posterior_probabilities = bgm.predict_proba(standardized_expression)
        cluster_assignment = posterior_probabilities.argmax(axis=1)

        cluster_sizes = np.bincount(cluster_assignment, minlength=posterior_probabilities.shape[1])
        cluster_marker_score_means = np.zeros(posterior_probabilities.shape[1])
        for cluster_idx in range(posterior_probabilities.shape[1]):
            in_cluster = cluster_assignment == cluster_idx
            if in_cluster.any():
                cluster_marker_score_means[cluster_idx] = float(
                    marker_activation_score[in_cluster].mean()
                )

        signal_cluster_mask = (cluster_marker_score_means >= self.cluster_score_min) & (
            cluster_sizes >= self.min_cells_cluster
        )
        prob_signal = posterior_probabilities[:, signal_cluster_mask].sum(axis=1)

        stats = {
            "converged": bool(bgm.converged_),
            "n_total_components": int(posterior_probabilities.shape[1]),
            "n_signal_components": int(signal_cluster_mask.sum()),
            "background_component_idx": None,
            "signal_component_indices": np.where(signal_cluster_mask)[0].tolist(),
            "component_sizes": cluster_sizes.tolist(),
            "component_mean_sums": bgm.means_.sum(axis=1).tolist(),
            "marker_score_thresholds": gene_thresholds.tolist(),
            "marker_score_cluster_means": cluster_marker_score_means.tolist(),
        }

        return prob_signal, stats

    def execute_on(self, adata: AnnData) -> LabelingResult:
        """
        Execute DPMM labeling on AnnData object.

        Returns
        -------
        LabelingResult with:
        - labels: Assigned cell type per cell, or "unknown"
        - obs["max_confidence"]: Max prob_signal across cell types
        - obs["is_confident"]: Boolean indicator of assignment success
        - obsm["posterior_probabilities"]: prob_signal for each cell type (n_cells x n_types)
        - uns["dpmm_convergence_stats"]: Per-cell-type component diagnostics
        """
        cell_types = list(self.markers.keys())

        # Data structures to hold our DPMM outputs
        posteriors_df = pd.DataFrame(0.0, index=adata.obs_names, columns=cell_types)
        dpmm_stats = {}

        # 1. Prepare expression matrices for each cell type
        # We do this sequentially to safely interact with AnnData, then pass numpy arrays to threads
        ctype_matrices = {}
        ctype_quantiles = {}
        for ctype, genes in self.markers.items():
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names
                or (self.use_raw and adata.raw is not None and g in adata.raw.var_names)
            ]

            if not valid_genes:
                ctype_matrices[ctype] = np.zeros((adata.n_obs, 0))
                ctype_quantiles[ctype] = []
                continue

            if self.use_raw and adata.raw is not None:
                X_slice = adata.raw[:, valid_genes].X
            else:
                X_slice = adata[:, valid_genes].X

            # BGM requires dense arrays
            if sp.issparse(X_slice):
                X_slice = X_slice.toarray()

            ctype_matrices[ctype] = X_slice
            ctype_quantiles[ctype] = [self.per_gene_high_expr_quantile[g] for g in valid_genes]

        # 2. Fit BGM with DP prior for each cell type (IN PARALLEL)
        # Each fit returns prob_signal = P(high-expression component | cell)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Future -> Cell Type mapping
            future_to_ctype = {
                executor.submit(
                    self._fit_single_ctype_adaptive,
                    ctype_matrices[ctype],
                    ctype_quantiles[ctype],
                ): ctype
                for ctype in cell_types
            }

            for future in future_to_ctype:
                ctype = future_to_ctype[future]
                try:
                    posterior_probs, stats = future.result()
                    posteriors_df[ctype] = posterior_probs
                    dpmm_stats[ctype] = stats

                except Exception as e:
                    print(f"DPMM fitting failed for '{ctype}': {e}")
                    posteriors_df[ctype] = 0.0
                    dpmm_stats[ctype] = {"converged": False, "error": str(e)}

        # 3. Label Assignment
        final_labels = pd.Series("unknown", index=adata.obs_names)

        # Find cells that are confident in at least one cell type
        is_confident_mask = (posteriors_df >= self.min_confidence).any(axis=1)

        # For confident cells, assign the cell type with highest prob_signal
        best_matches = posteriors_df.idxmax(axis=1)
        final_labels[is_confident_mask] = best_matches[is_confident_mask]

        # Track max confidence across cell types and whether cell was assigned
        max_confidence = posteriors_df.max(axis=1)
        is_assigned = final_labels != "unknown"

        # 4. Return Rich DTO
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={"max_confidence": max_confidence, "is_confident": is_assigned},
            obsm={"posterior_probabilities": posteriors_df},
            uns={
                "fraction_assigned": float(is_assigned.mean()),
                "dpmm_convergence_stats": dpmm_stats,
            },
        )

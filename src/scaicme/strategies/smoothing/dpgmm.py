from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from ..base import BaseLabelingStrategy, LabelingResult


class DPGMMClusteredSmoothing(BaseLabelingStrategy):
    r"""
    Dirichlet Process Gaussian Mixture Model (DPGMM) for Robust Cell Type Labeling.

    This strategy fits a Bayesian Gaussian Mixture Model with a Dirichlet Process
    prior to marker gene expression for each cell type. Cluster scoring is driven
    by a precomputed seed-score matrix supplied by a prior seeding strategy, and
    the model automatically discovers the optimal number of expression clusters
    without requiring manual specification.

    Algorithm:
    1. Load the precomputed per-cell, per-cell-type seed scores from `adata.obsm`
    2. Compute per-gene positive quantile activation thresholds using `per_gene_pos_quantile`
    3. Fit a BGM with DP prior to standardized marker expression (n_cells × n_markers)
    4. For each cell, compute posterior probabilities across components
    5. Filter components: keep only those with mean seed score >= cluster_score_min AND size >= min_cells_cluster
    6. For each cell, prob_signal = sum of posterior probabilities for signal components
    7. A cell is assigned to a cell type if its prob_signal >= min_confidence
    8. If multiple cell types are confident, assign the one with highest prob_signal
    9. Enforce post-hoc consensus minimum cluster size using `min_cluster_size_post_hoc`

    The Dirichlet Process prior automatically collapses inactive components at
    convergence, effectively performing model selection to find the true number
    of expression clusters present in the data.

    Note: This strategy assumes that the input data (`adata.X` or `adata.raw.X`)
    is already log-normalized.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    initial_scores_key : str
        Key in `adata.obsm` pointing to a precomputed cell-by-cell-type score matrix
        from a prior seeding strategy. The matrix must have one column per marker
        set, in the same order as `markers`.
    min_confidence : float, default 0.8
        Minimum prob_signal threshold for assigning a cell to a cell type.
    min_cells_per_gene : int, default 20
        Minimum number of positive cells required to compute a per-gene threshold.
    min_expressed_markers : int, default 2
        Minimum number of markers that must be expressed above `min_cells_per_gene`.
    min_cell_enrichment : float, default 0.10
        Minimum fraction of cells with any marker expression to proceed.
    cluster_score_min : float, default 0.20
        Minimum mean marker score for a cluster to be considered signal.
    min_cells_cluster : int, default 5
        Minimum cluster size to be considered signal during component selection.
    weight_concentration_prior : float, default 0.05
        Dirichlet Process concentration parameter ($\gamma$). Controls component
        collapse behavior; lower values favor fewer active components at convergence.
    max_iter : int, default 1000
        Maximum iterations for DPGMM EM algorithm fitting.
    use_raw : bool, default True
        Whether to extract marker expression from `adata.raw` (recommended for
        log-normalized data).
    random_state : int | None, default None
        Random seed for reproducible convergence.
    n_jobs : int, default 4
        Number of threads for parallel per-cell-type fitting.
    per_gene_pos_quantile : float, default 0.90
        Quantile threshold applied across positive expression values to calculate per-gene
        activation thresholds before DPGMM clustering.
    min_cluster_size_post_hoc : int, default 15
        Post-hoc consensus size floor. Any assigned cell type cluster falling below this count
        post-smoothing is reassigned/dropped to "unknown" to prevent spurious small clusters.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        initial_scores_key: str,
        min_confidence: float = 0.8,
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
        per_gene_pos_quantile: float = 0.90,
        min_cluster_size_post_hoc: int = 15,
        **kwargs: Any,
    ) -> None:
        self.markers = markers
        self.min_confidence = min_confidence
        self.initial_scores_key = initial_scores_key
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
        self.per_gene_pos_quantile = per_gene_pos_quantile
        self.min_cluster_size_post_hoc = min_cluster_size_post_hoc

    @property
    def name(self) -> str:
        return "dpgmm_clustered_smoothing"

    def _get_initial_scores(self, adata: AnnData, cell_types: List[str]) -> pd.DataFrame:
        """Resolve a precomputed score matrix from `adata.obsm`."""
        if self.initial_scores_key not in adata.obsm:
            raise ValueError(
                f"Initial scores key '{self.initial_scores_key}' not found in adata.obsm"
            )

        initial_scores = adata.obsm[self.initial_scores_key]

        if isinstance(initial_scores, pd.DataFrame):
            scores_df = initial_scores.copy()
            if scores_df.shape[1] != len(cell_types):
                raise ValueError(
                    f"Initial score matrix '{self.initial_scores_key}' must have {len(cell_types)} columns."
                )
            if list(scores_df.columns) != cell_types:
                scores_df = scores_df.reindex(columns=cell_types)
        else:
            scores_array = np.asarray(initial_scores)
            if scores_array.ndim != 2:
                raise ValueError(
                    f"Initial score matrix '{self.initial_scores_key}' must be 2-dimensional."
                )
            if scores_array.shape[1] != len(cell_types):
                raise ValueError(
                    f"Initial score matrix '{self.initial_scores_key}' must have {len(cell_types)} columns."
                )
            scores_df = pd.DataFrame(scores_array, index=adata.obs_names, columns=cell_types)

        if scores_df.shape[0] != adata.n_obs:
            raise ValueError(
                f"Initial score matrix '{self.initial_scores_key}' must have {adata.n_obs} rows."
            )

        return scores_df.astype(float)

    def _fit_single_ctype_adaptive(
        self,
        X: np.ndarray,
        initial_scores: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit a seed-score-driven DPGMM for a single cell type's marker genes using the
        precomputed seeding score vector for that cell type.
        """
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

        if initial_scores is None:
            raise ValueError(
                "DPGMM requires a precomputed initial score vector for each cell type."
            )

        marker_activation_score = np.asarray(initial_scores, dtype=float)
        if marker_activation_score.shape[0] != X.shape[0]:
            raise ValueError("Initial score vector must have the same number of rows as X.")
        marker_activation_score = np.nan_to_num(marker_activation_score, nan=0.0)

        # Compute per-gene positive thresholds across marker genes using per_gene_pos_quantile
        per_gene_thresholds = []
        for gene_idx in range(X.shape[1]):
            gene_values = X[:, gene_idx]
            positive_values = gene_values[gene_values > 0]
            if positive_values.size > 0:
                threshold = float(np.quantile(positive_values, self.per_gene_pos_quantile))
            else:
                threshold = np.inf
            per_gene_thresholds.append(threshold)

        scaler = StandardScaler(with_mean=True, with_std=True)
        standardized_expression = scaler.fit_transform(X)

        n_components = max(2, int(np.sqrt(X.shape[0])))
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
            "marker_score_thresholds": per_gene_thresholds,
            "marker_score_cluster_means": cluster_marker_score_means.tolist(),
            "marker_score_source": "initial_scores",
        }

        return prob_signal, stats

    def execute_on(self, adata: AnnData) -> LabelingResult:
        """
        Execute DPGMM labeling on an AnnData object using precomputed seed scores.
        """
        cell_types = list(self.markers.keys())
        initial_scores_df = self._get_initial_scores(adata, cell_types)

        posteriors_df = pd.DataFrame(0.0, index=adata.obs_names, columns=cell_types)
        dpgmm_stats = {}

        ctype_matrices = {}
        for ctype, genes in self.markers.items():
            valid_genes = [
                g
                for g in genes
                if g in adata.var_names
                or (self.use_raw and adata.raw is not None and g in adata.raw.var_names)
            ]

            if not valid_genes:
                ctype_matrices[ctype] = np.zeros((adata.n_obs, 0))
                continue

            if self.use_raw and adata.raw is not None:
                X_slice = adata.raw[:, valid_genes].X
            else:
                X_slice = adata[:, valid_genes].X

            if sp.issparse(X_slice):
                X_slice = X_slice.toarray()

            ctype_matrices[ctype] = X_slice

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_ctype = {
                executor.submit(
                    self._fit_single_ctype_adaptive,
                    ctype_matrices[ctype],
                    None if initial_scores_df is None else initial_scores_df[ctype].to_numpy(),
                ): ctype
                for ctype in cell_types
            }

            for future in future_to_ctype:
                ctype = future_to_ctype[future]
                try:
                    posterior_probs, stats = future.result()
                    posteriors_df[ctype] = posterior_probs
                    dpgmm_stats[ctype] = stats

                except Exception as e:
                    print(f"DPGMM fitting failed for '{ctype}': {e}")
                    posteriors_df[ctype] = 0.0
                    dpgmm_stats[ctype] = {"converged": False, "error": str(e)}

        final_labels = pd.Series("unknown", index=adata.obs_names)

        is_confident_mask = (posteriors_df >= self.min_confidence).any(axis=1)
        best_matches = posteriors_df.idxmax(axis=1)
        final_labels[is_confident_mask] = best_matches[is_confident_mask]

        # Enforce post-hoc consensus minimum cluster size
        for ctype in cell_types:
            if (final_labels == ctype).sum() < self.min_cluster_size_post_hoc:
                final_labels[final_labels == ctype] = "unknown"

        max_confidence = posteriors_df.max(axis=1)
        is_assigned = final_labels != "unknown"

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={"max_confidence": max_confidence, "is_confident": is_assigned},
            obsm={"posterior_probabilities": posteriors_df},
            uns={
                "fraction_assigned": float(is_assigned.mean()),
                "dpgmm_convergence_stats": dpgmm_stats,
            },
        )

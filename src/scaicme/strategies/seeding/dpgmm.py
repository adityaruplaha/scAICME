from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from ..base import LabelingResult
from .base import BaseSeedingStrategy


class DPGMMSeeding(BaseSeedingStrategy):
    r"""
    Marker-set Dirichlet Process GMM seeding (self-contained, no prior scores needed).

    For each cell type, a Bayesian Gaussian Mixture with a Dirichlet Process prior is
    fitted to the standardized expression of that type's marker genes. Mixture
    components whose members are, on average, enriched for the type's markers are
    treated as signal, and every cell in a signal component receives the type's label
    with a confidence derived from its own marker activation. Per-type labels are then
    reconciled by confidence into a single seed label per cell.

    Algorithm (per cell type):

    1. Keep markers present in the data; skip the type if fewer than `min_genes_present`.
    2. Skip the type if fewer than `min_expressed_markers` markers are detected in at
       least `min_cells_per_gene` cells, or if fewer than `min_cell_enrichment` of cells
       express any marker.
    3. Per gene, an activation threshold is the `per_gene_pos_quantile` quantile of its
       positive values (the median when fewer than `min_cells_per_gene` cells are
       positive). A cell's *marker score* is the fraction of markers above threshold.
    4. Fit a DP-GMM (`n_components`, `weight_concentration_prior`, full covariance) to
       the standardized marker expression and take the hard component assignment.
    5. Components with mean marker score >= `cluster_score_min` and at least
       `min_cells_cluster` members are signal components.
    6. Cells in signal components are labeled with the type; their confidence is
       `marker_score / max(marker_score)` over all cells.

    Across types, each cell takes the type with the highest confidence (ties resolve
    in marker-dictionary order). Types that end up with fewer than
    `max(min_type_size, int(min_type_frac * n_cells))` cells are dropped to
    `unknown_label`.

    Note: this strategy assumes the input expression (`adata.X` or `adata.raw.X`) is
    already log-normalized.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    n_components : int | None, default None
        Number of mixture components for the DP-GMM. If None, `max(2, int(sqrt(n_cells)))`.
    weight_concentration_prior : float, default 0.05
        Dirichlet Process concentration parameter; lower values favor fewer active components.
    per_gene_pos_quantile : float, default 0.70
        Quantile of positive expression values used as each gene's activation threshold.
    cluster_score_min : float, default 0.20
        Minimum mean marker score for a component to count as signal.
    min_cells_cluster : int, default 100
        Minimum component size for it to count as signal.
    min_genes_present : int, default 3
        Minimum number of markers present in the data for a type to be fitted.
    min_expressed_markers : int, default 2
        Minimum number of markers detected in at least `min_cells_per_gene` cells.
    min_cells_per_gene : int, default 20
        Positive-cell count above which a marker counts as expressed and its threshold is
        taken from the quantile rather than the median.
    min_cell_enrichment : float, default 0.05
        Minimum fraction of cells expressing any marker for a type to be fitted.
    min_type_size : int, default 20
        Absolute floor on the final number of cells per assigned type.
    min_type_frac : float, default 0.001
        Fractional floor on the final number of cells per assigned type.
    covariance_type : str, default "full"
        Covariance type passed to `BayesianGaussianMixture`.
    max_iter : int, default 1000
        Maximum EM iterations.
    n_init : int, default 1
        Number of initializations for the mixture fit.
    random_state : int | None, default None
        Random seed for the mixture fit.
    use_raw : bool, default True
        Whether to read marker expression from `adata.raw` when present.
    unknown_label : str, default "unknown"
        Label for cells that receive no seed.
    n_jobs : int, default 1
        Number of threads used to fit cell types concurrently.
    verbose : bool, default False
        Print per-type diagnostics as they are produced.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        n_components: int | None = None,
        weight_concentration_prior: float = 0.05,
        per_gene_pos_quantile: float = 0.70,
        cluster_score_min: float = 0.20,
        min_cells_cluster: int = 100,
        min_genes_present: int = 3,
        min_expressed_markers: int = 2,
        min_cells_per_gene: int = 20,
        min_cell_enrichment: float = 0.05,
        min_type_size: int = 20,
        min_type_frac: float = 0.001,
        covariance_type: str = "full",
        max_iter: int = 1000,
        n_init: int = 1,
        random_state: int | None = None,
        use_raw: bool = True,
        unknown_label: str = "unknown",
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(markers=markers, use_raw=use_raw, unknown_label=unknown_label, **kwargs)
        self.n_components = n_components
        self.weight_concentration_prior = weight_concentration_prior
        self.per_gene_pos_quantile = per_gene_pos_quantile
        self.cluster_score_min = cluster_score_min
        self.min_cells_cluster = min_cells_cluster
        self.min_genes_present = min_genes_present
        self.min_expressed_markers = min_expressed_markers
        self.min_cells_per_gene = min_cells_per_gene
        self.min_cell_enrichment = min_cell_enrichment
        self.min_type_size = min_type_size
        self.min_type_frac = min_type_frac
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "dpgmm_seeding"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _marker_expression(self, adata: AnnData, genes: List[str]) -> tuple[np.ndarray, List[str]]:
        """Return dense (n_cells, n_present_markers) expression and the markers kept."""
        if self.use_raw and adata.raw is not None:
            source = adata.raw
        else:
            source = adata
        present = set(source.var_names.astype(str))
        genes = [g for g in genes if g in present]
        if not genes:
            return np.zeros((adata.n_obs, 0), dtype=float), genes
        X = source[:, genes].X
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=float), genes

    def _fit_single_type(
        self, ctype: str, X: np.ndarray, genes: List[str]
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fit one cell type. Returns (confidence, marker_score, diagnostics)."""
        n_cells = X.shape[0]
        conf = np.zeros(n_cells, dtype=float)
        marker_score = np.zeros(n_cells, dtype=float)
        diag: Dict[str, Any] = {"n_markers_used": len(genes), "n_labelled": 0, "pct_labelled": 0.0}

        if len(genes) < self.min_genes_present:
            diag["skipped"] = "insufficient_markers_present"
            return conf, marker_score, diag

        # Pre-filter 1: enough markers detected in enough cells.
        expr_per_gene = (X > 0).sum(axis=0)
        expressed_markers = int(np.sum(expr_per_gene >= self.min_cells_per_gene))
        diag["expressed_markers"] = expressed_markers
        if expressed_markers < self.min_expressed_markers:
            diag["skipped"] = "insufficient_expressed_markers"
            return conf, marker_score, diag

        # Pre-filter 2: enough cells express any marker.
        cell_enrichment = float(np.mean((X > 0).sum(axis=1) > 0))
        diag["cell_enrichment"] = cell_enrichment
        if cell_enrichment < self.min_cell_enrichment:
            diag["skipped"] = "low_cell_enrichment"
            return conf, marker_score, diag

        # Per-gene activation thresholds and per-cell marker score.
        gene_thr: Dict[str, float] = {}
        high_mask = np.zeros_like(X, dtype=bool)
        for j, g in enumerate(genes):
            pos = X[:, j][X[:, j] > 0]
            if pos.size >= self.min_cells_per_gene:
                thr = float(np.quantile(pos, self.per_gene_pos_quantile))
            elif pos.size > 0:
                thr = float(np.median(pos))
            else:
                thr = np.inf
            gene_thr[g] = thr
            if np.isfinite(thr):
                high_mask[:, j] = X[:, j] > thr
        marker_score = high_mask.sum(axis=1) / max(1, len(genes))
        diag["gene_thresholds"] = gene_thr

        # Standardize and fit the DP-GMM on raw marker expression (no PCA).
        X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        n_components = self.n_components
        if n_components is None:
            n_components = max(2, int(np.sqrt(n_cells)))
        bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=self.weight_concentration_prior,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        bgm.fit(X_scaled)
        components = bgm.predict(X_scaled)

        # Component-level gating on mean marker score and size.
        summary = (
            pd.DataFrame({"cluster": components, "score": marker_score})
            .groupby("cluster")["score"]
            .agg(["mean", "count"])
        )
        good = summary[
            (summary["mean"] >= self.cluster_score_min)
            & (summary["count"] >= self.min_cells_cluster)
        ].index.tolist()

        in_good = np.isin(components, good)
        if good:
            max_score = marker_score.max() if marker_score.max() > 0 else 1.0
            conf[in_good] = marker_score[in_good] / max_score

        n_lab = int(in_good.sum())
        diag.update(
            {
                "converged": bool(bgm.converged_),
                "n_components": n_components,
                "n_clusters": int(len(summary)),
                "good_clusters": [int(c) for c in good],
                "cluster_summary": {
                    str(int(c)): {"mean": float(r["mean"]), "count": int(r["count"])}
                    for c, r in summary.iterrows()
                },
                "n_labelled": n_lab,
                "pct_labelled": float(n_lab / n_cells * 100.0),
            }
        )
        return conf, marker_score, diag

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_on(self, adata: AnnData) -> LabelingResult:
        cell_types = list(self.markers.keys())
        n_cells = adata.n_obs

        inputs = {
            ctype: self._marker_expression(adata, genes) for ctype, genes in self.markers.items()
        }

        scores_df = pd.DataFrame(0.0, index=adata.obs_names, columns=cell_types)
        marker_scores_df = pd.DataFrame(0.0, index=adata.obs_names, columns=cell_types)
        diagnostics: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=max(1, self.n_jobs)) as executor:
            futures = {
                ctype: executor.submit(self._fit_single_type, ctype, X, genes)
                for ctype, (X, genes) in inputs.items()
            }
            for ctype in cell_types:
                conf, marker_score, diag = futures[ctype].result()
                scores_df[ctype] = conf
                marker_scores_df[ctype] = marker_score
                diagnostics[ctype] = diag
                if self.verbose:
                    if "skipped" in diag:
                        print(f"[{self.name}] {ctype}: skipped ({diag['skipped']})")
                    else:
                        print(
                            f"[{self.name}] {ctype}: labelled {diag['n_labelled']}/{n_cells} "
                            f"({diag['pct_labelled']:.2f}%), good_clusters={len(diag['good_clusters'])}, "
                            f"n_features={diag['n_markers_used']}"
                        )

        # Reconcile per-type labels: highest confidence wins, ties go to dictionary order.
        conf_values = scores_df.to_numpy()
        best_idx = conf_values.argmax(axis=1)
        best_conf = conf_values[np.arange(n_cells), best_idx]
        labels = np.where(
            best_conf > 0, np.asarray(cell_types, dtype=object)[best_idx], self.unknown_label
        )
        final_labels = pd.Series(labels, index=adata.obs_names, dtype=object)
        final_conf = pd.Series(best_conf, index=adata.obs_names, dtype=float)

        # Final size floor per assigned type.
        size_floor = max(self.min_type_size, int(self.min_type_frac * n_cells))
        counts = final_labels.value_counts()
        dropped = [t for t in cell_types if 0 < counts.get(t, 0) < size_floor]
        if dropped:
            drop_mask = final_labels.isin(dropped)
            final_labels[drop_mask] = self.unknown_label
            final_conf[drop_mask] = 0.0

        is_confident = final_labels != self.unknown_label

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels.astype(str),
            obs={"max_score": final_conf, "is_confident": is_confident},
            obsm={"scores": scores_df, "marker_scores": marker_scores_df},
            uns={
                "fraction_assigned": float(is_confident.mean()),
                "size_floor": int(size_floor),
                "dropped_types": dropped,
                "diagnostics": diagnostics,
            },
        )

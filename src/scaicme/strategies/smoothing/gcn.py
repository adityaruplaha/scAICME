import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

from ..base import BaseLabelingStrategy, LabelingResult


class GCNSmoothing(BaseLabelingStrategy):
    r"""
    GCN Smoothing for Seed Generation.

    This strategy computes initial marker scores ($Y_0$) and iteratively diffuses
    them across the cell-cell connectivity graph using the Personalized PageRank /
    Label Spreading formulation. When `initial_scores_key` is provided, the
    diffusion starts from the precomputed score matrix emitted by a prior seed
    strategy; otherwise, it falls back to on-the-fly scoring, which is deprecated.

    $$display$$
    Y_{t+1} = \alpha \hat{A} Y_t + (1-\alpha) Y_0
    $$display$$

    Where $\hat{A}$ is the symmetrically normalized adjacency matrix.
    Final seeds are selected using a "margin gate" or quantile-based adaptive gating,
    supporting exact quota allocation and rare cluster retention floors.

    Parameters
    ----------
    markers : Dict[str, List[str]]
        Dictionary mapping cell type names to lists of marker genes.
    obsp_key : str, default 'connectivities'
        Key in `adata.obsp` containing the neighborhood graph adjacency matrix.
    initial_scores_key : str | None, default None
        Key in `adata.obsm` containing a precomputed cell-by-cell-type score matrix
        from a prior seeding strategy. When provided, these scores are used as the
        initial signal for diffusion.
    alpha : float, default 0.5
        The diffusion parameter (between 0 and 1). Higher values mean more
        information is absorbed from neighbors; lower values rely more on
        the cell's own initial raw score.
    n_iterations : int, default 10
        Number of diffusion iterations to perform.
    margin : float, default 0.1
        The minimum required difference between the top score and the
        second-highest score for a cell to be confidently labeled when `adaptive_gate=False`.
    min_score : float, default 0.05
        Absolute minimum diffused score required to be considered a seed when `adaptive_gate=False`.
    use_raw : bool, default True
        Whether to calculate initial scores using `adata.raw` when
        `initial_scores_key` is not provided.
    temperature : float, default 1.0
        Softmax temperature scaling factor applied to initial scores ($Y_0 = \text{softmax}(Z / T)$)
        to sharpen probability vectors when `temperature != 1.0`.
    adaptive_gate : bool, default False
        Whether to apply quantile-based gating on top-1 prediction probability and margin before
        assigning labels.
    q_score : float, default 0.40
        Quantile threshold for top-1 probability when `adaptive_gate=True`.
    q_margin : float, default 0.35
        Quantile threshold for top-1 minus top-2 probability margin when `adaptive_gate=True`.
    tol : float | None, default None
        Convergence tolerance for early stopping in GCN diffusion ($\text{mean}(|Y_{t+1} - Y_t|) < \text{tol}$).
    target_frac : float | None, default None
        Fraction of total cells to assign labels post-gating across qualifying clusters.
    min_cells_per_type : int, default 100
        Base minimum cells budget allocated per cluster when `target_frac` is specified.
    min_cells_per_type_pct : float, default 0.001
        Population percentage threshold used alongside `min_cells_floor` to determine cluster retention floors.
    min_cells_floor : int, default 30
        Absolute minimum size floor for retaining rare clusters during smoothing.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]],
        obsp_key: str = "connectivities",
        initial_scores_key: str | None = None,
        alpha: float = 0.5,
        n_iterations: int = 10,
        margin: float = 0.1,
        min_score: float = 0.05,
        use_raw: bool = True,
        temperature: float = 1.0,
        adaptive_gate: bool = False,
        q_score: float = 0.40,
        q_margin: float = 0.35,
        tol: float | None = None,
        target_frac: float | None = None,
        min_cells_per_type: int = 100,
        min_cells_per_type_pct: float = 0.001,
        min_cells_floor: int = 30,
        **kwargs: Any,
    ) -> None:
        self.markers = markers
        self.obsp_key = obsp_key
        self.initial_scores_key = initial_scores_key
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.margin = margin
        self.min_score = min_score
        self.use_raw = use_raw
        self.temperature = temperature
        self.adaptive_gate = adaptive_gate
        self.q_score = q_score
        self.q_margin = q_margin
        self.tol = tol
        self.target_frac = target_frac
        self.min_cells_per_type = min_cells_per_type
        self.min_cells_per_type_pct = min_cells_per_type_pct
        self.min_cells_floor = min_cells_floor

    @property
    def name(self) -> str:
        return "gcn"

    def _get_initial_scores(self, adata: AnnData) -> pd.DataFrame:
        """Calculates $Y_0$ from precomputed seed scores or, deprecatedly, raw marker scores."""
        if self.initial_scores_key is not None:
            if self.initial_scores_key not in adata.obsm:
                raise ValueError(
                    f"Initial scores key '{self.initial_scores_key}' not found in adata.obsm"
                )

            initial_scores = adata.obsm[self.initial_scores_key]
            cell_types = list(self.markers.keys())

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

        warnings.warn(
            "GCNSmoothing is falling back to on-the-fly initial scoring. "
            "Pass initial_scores_key to reuse scores from a prior strategy.",
            DeprecationWarning,
            stacklevel=2,
        )

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

        for col in scores_df.columns:
            min_val = scores_df[col].min()
            max_val = scores_df[col].max()
            if max_val > min_val:
                scores_df[col] = (scores_df[col] - min_val) / (max_val - min_val)
            else:
                scores_df[col] = 0.0

        return scores_df

    def _normalize_adjacency(self, A: sp.spmatrix) -> sp.coo_matrix:
        r"""Calculates $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$"""
        A_tilde = A + sp.eye(A.shape[0])
        D = np.array(A_tilde.sum(axis=1)).flatten()
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        D_mat_inv_sqrt = sp.diags(D_inv_sqrt)
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
        if self.temperature != 1.0:
            Z = Y_0_df.values / self.temperature
            Z = Z - np.max(Z, axis=1, keepdims=True)
            exp_Z = np.exp(Z)
            sharpened = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
            Y_0_df = pd.DataFrame(sharpened, index=Y_0_df.index, columns=Y_0_df.columns)

        Y_0 = Y_0_df.values
        cell_types = Y_0_df.columns.tolist()

        # 2. Prepare Graph ($\hat{A}$)
        A_norm = self._normalize_adjacency(adata.obsp[self.obsp_key])

        # 3. Diffuse Scores
        Y_t = Y_0.copy()
        for _ in range(self.n_iterations):
            Y_next = self.alpha * A_norm.dot(Y_t) + (1.0 - self.alpha) * Y_0
            if self.tol is not None:
                delta = np.mean(np.abs(Y_next - Y_t))
                Y_t = Y_next
                if delta < self.tol:
                    break
            else:
                Y_t = Y_next

        diffused_df = pd.DataFrame(Y_t, index=adata.obs_names, columns=cell_types)

        # 4. Apply Margin Gate / Adaptive Gate & Quota Selection
        sorted_scores = np.sort(Y_t, axis=1)
        top_scores = sorted_scores[:, -1]
        second_scores = sorted_scores[:, -2] if Y_t.shape[1] > 1 else np.zeros_like(top_scores)
        margins = top_scores - second_scores

        uns_meta: Dict[str, Any] = {"cell_types": cell_types}

        if self.adaptive_gate:
            score_thresh = float(np.quantile(top_scores, self.q_score))
            margin_thresh = float(np.quantile(margins, self.q_margin))
            is_confident = (margins >= margin_thresh) & (top_scores >= score_thresh)
            uns_meta.update({"score_thresh": score_thresh, "margin_thresh": margin_thresh})
        else:
            is_confident = (margins >= self.margin) & (top_scores >= self.min_score)

        best_match_idx = np.argmax(Y_t, axis=1)
        best_match_names = pd.Series(
            [cell_types[idx] for idx in best_match_idx], index=adata.obs_names
        )
        final_labels = pd.Series("unknown", index=adata.obs_names, dtype=str)

        if self.target_frac is not None:
            n_total = len(adata)
            total_budget = int(round(n_total * self.target_frac))
            min_cells = self.min_cells_per_type

            candidates_per_type: Dict[str, List[str]] = {}
            for col in cell_types:
                mask = is_confident & (best_match_names == col)
                candidates_per_type[col] = adata.obs_names[mask].tolist()

            eligible_types = [
                col for col, cands in candidates_per_type.items() if len(cands) >= min_cells
            ]

            if eligible_types:
                quotas: Dict[str, int] = dict.fromkeys(eligible_types, min_cells)
                base_total = sum(quotas.values())

                if total_budget > base_total:
                    rem_budget = total_budget - base_total
                    total_avail = sum(
                        len(candidates_per_type[col]) - min_cells for col in eligible_types
                    )
                    if total_avail > 0:
                        for col in eligible_types:
                            avail = len(candidates_per_type[col]) - min_cells
                            if avail > 0:
                                add = int(round(rem_budget * (avail / total_avail)))
                                quotas[col] = min(min_cells + add, len(candidates_per_type[col]))

                        current_total = sum(quotas.values())
                        while current_total < total_budget:
                            added = False
                            for col in sorted(
                                eligible_types,
                                key=lambda c: len(candidates_per_type[c]) - quotas[c],
                                reverse=True,
                            ):
                                if quotas[col] < len(candidates_per_type[col]):
                                    quotas[col] += 1
                                    current_total += 1
                                    added = True
                                    if current_total == total_budget:
                                        break
                            if not added:
                                break
                else:
                    for col in eligible_types:
                        quotas[col] = min(quotas[col], len(candidates_per_type[col]))

                for col, q in quotas.items():
                    if q <= 0:
                        continue
                    cands = candidates_per_type[col]
                    sorted_cands = sorted(
                        cands, key=lambda idx: diffused_df.loc[idx, col], reverse=True
                    )
                    final_labels.loc[sorted_cands[:q]] = col
        else:
            confident_mask = pd.Series(is_confident, index=adata.obs_names)
            final_labels[confident_mask] = best_match_names[confident_mask]

        # Apply rare cluster retention floor when adaptive gating or quota selection is active
        if self.adaptive_gate or self.target_frac is not None:
            cluster_floor = max(
                self.min_cells_floor,
                int(round(adata.n_obs * self.min_cells_per_type_pct)),
                5,
            )
            for col in cell_types:
                if (final_labels == col).sum() < cluster_floor:
                    final_labels[final_labels == col] = "unknown"

        final_is_confident = pd.Series(final_labels != "unknown", index=adata.obs_names)
        uns_meta["fraction_assigned"] = float(final_is_confident.mean())

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={
                "margin": pd.Series(margins, index=adata.obs_names),
                "is_confident": final_is_confident,
            },
            obsm={"initial_scores": Y_0_df, "diffused_scores": diffused_df},
            uns=uns_meta,
        )


class GCNSeeding(BaseLabelingStrategy):
    r"""
    GCN-style label propagation of a prior score matrix with fixed gates.

    Starting from a prior strategy's cell-by-type score matrix (row-normalized to
    probabilities), scores are diffused over the kNN graph,

    $$Y_{t+1} = \alpha \hat{A} Y_t + (1-\alpha) Y_0, \qquad
    \hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2},$$

    for a fixed number of iterations. A cell becomes a seed when its top diffused
    score is at least `min_score` and its top-1/top-2 margin is at least `delta`.
    Optionally only the `target_frac` most confident seeds are kept, and types with
    fewer than ``max(min_cells_floor, floor(min_cells_per_type_pct * n_cells), 5)``
    seeds are dropped.

    Parameters
    ----------
    markers : Dict[str, List[str]] | List[str]
        Cell types, in the column order of the score matrix (a marker dictionary or
        a list of names).
    initial_scores_key : str
        Key in `adata.obsm` holding the prior cell-by-type score matrix.
    obsp_key : str, default "connectivities"
        Key in `adata.obsp` containing the neighborhood graph.
    target_frac : float | None, default 0.25
        Keep at most this fraction of cells (the most confident gated ones); ``None``
        disables the cap.
    min_score : float, default 0.55
        Gate on the top diffused score.
    delta : float, default 0.15
        Gate on the margin between the top two diffused scores.
    alpha : float, default 0.85
        Propagation strength.
    n_iter : int, default 100
        Number of propagation iterations.
    min_cells_per_type_pct : float, default 0.005
        Fractional floor on seeds per type.
    min_cells_floor : int, default 200
        Absolute floor on seeds per type.
    unknown_label : str, default "unknown"
        Label for cells that receive no seed.
    """

    def __init__(
        self,
        markers: Dict[str, List[str]] | List[str],
        initial_scores_key: str,
        obsp_key: str = "connectivities",
        target_frac: float | None = 0.25,
        min_score: float = 0.55,
        delta: float = 0.15,
        alpha: float = 0.85,
        n_iter: int = 100,
        min_cells_per_type_pct: float = 0.005,
        min_cells_floor: int = 200,
        unknown_label: str = "unknown",
        **kwargs: Any,
    ) -> None:
        self.markers = markers
        self.initial_scores_key = initial_scores_key
        self.obsp_key = obsp_key
        self.target_frac = target_frac
        self.min_score = min_score
        self.delta = delta
        self.alpha = alpha
        self.n_iter = n_iter
        self.min_cells_per_type_pct = min_cells_per_type_pct
        self.min_cells_floor = min_cells_floor
        self.unknown_label = unknown_label

    @property
    def name(self) -> str:
        return "gcn_seeding"

    def _cell_types(self) -> List[str]:
        return list(self.markers.keys()) if isinstance(self.markers, dict) else list(self.markers)

    def execute_on(self, adata: AnnData) -> LabelingResult:
        if self.obsp_key not in adata.obsp:
            raise ValueError(
                f"Graph key '{self.obsp_key}' not found in adata.obsp. "
                f"Please run sc.pp.neighbors(adata) first."
            )
        if self.initial_scores_key not in adata.obsm:
            raise ValueError(
                f"Initial scores key '{self.initial_scores_key}' not found in adata.obsm"
            )

        types = self._cell_types()
        scores = adata.obsm[self.initial_scores_key]
        if isinstance(scores, pd.DataFrame):
            scores = scores.reindex(columns=types).to_numpy(dtype=float)
        else:
            scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape != (adata.n_obs, len(types)):
            raise ValueError(
                f"Initial score matrix '{self.initial_scores_key}' must have shape "
                f"({adata.n_obs}, {len(types)})."
            )

        # Drop all-NaN types, then row-normalize to probabilities.
        usable = ~np.isnan(scores).all(axis=0)
        if not usable.any():
            raise ValueError("Score matrix has no usable type columns.")
        types = [t for t, u in zip(types, usable, strict=True) if u]
        S0 = np.nan_to_num(scores[:, usable], nan=0.0)
        Y0 = S0 / (S0.sum(axis=1, keepdims=True) + 1e-12)

        # Symmetrically normalized adjacency with self loops.
        N = adata.n_obs
        A = adata.obsp[self.obsp_key]
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        A_tilde = (A + sp.eye(N, format="csr")).tocsr()
        deg = np.asarray(A_tilde.sum(axis=1)).ravel()
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg + 1e-12), format="csr")
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        Y = Y0.copy()
        for _ in range(self.n_iter):
            Y = self.alpha * (A_hat @ Y) + (1.0 - self.alpha) * Y0
        Y = np.asarray(Y)

        # Gates on top-1 score and top-1/top-2 margin.
        top1_idx = Y.argmax(axis=1)
        top1 = Y[np.arange(N), top1_idx]
        Y2 = Y.copy()
        Y2[np.arange(N), top1_idx] = -np.inf
        top2 = Y2.max(axis=1)
        margin = top1 - top2
        gate = (top1 >= float(self.min_score)) & (margin >= float(self.delta))

        labels = np.array([self.unknown_label] * N, dtype=object)
        labels[gate] = np.array(types, dtype=object)[top1_idx[gate]]
        conf = np.zeros(N, dtype=float)
        conf[gate] = top1[gate]

        # Optional cap: keep the most confident target_frac of cells.
        if self.target_frac is not None:
            target_n = int(round(float(self.target_frac) * N))
            if target_n > 0:
                idx_labeled = np.where(labels != self.unknown_label)[0]
                if idx_labeled.size > target_n:
                    order = idx_labeled[np.argsort(-conf[idx_labeled])]
                    drop = np.setdiff1d(idx_labeled, order[:target_n], assume_unique=False)
                    labels[drop] = self.unknown_label
                    conf[drop] = 0.0

        # Drop types below the size floor.
        min_cells = max(
            int(self.min_cells_floor), int(np.floor(float(self.min_cells_per_type_pct) * N)), 5
        )
        counts = pd.Series(labels).value_counts()
        keep_types = set(counts[counts >= min_cells].index.tolist())
        keep_types.discard(self.unknown_label)
        small = (labels != self.unknown_label) & ~np.isin(labels, list(keep_types))
        labels[small] = self.unknown_label
        conf[small] = 0.0

        final_labels = pd.Series(labels, index=adata.obs_names, dtype=str)
        is_confident = final_labels != self.unknown_label
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={
                "max_score": pd.Series(conf, index=adata.obs_names),
                "top1": pd.Series(top1, index=adata.obs_names),
                "margin": pd.Series(margin, index=adata.obs_names),
                "is_confident": is_confident,
            },
            obsm={"scores": pd.DataFrame(Y, index=adata.obs_names, columns=types)},
            uns={
                "cell_types": types,
                "min_cells_used": int(min_cells),
                "gate_pass_fraction": float(gate.mean()),
                "fraction_assigned": float(is_confident.mean()),
            },
        )

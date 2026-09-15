"""Verbatim reference implementation from the PBMC 68k notebook.

Source SHA256 7dee319eda6c1b85e647338c4ac633d68667a6c8fdb920dc7abc42faad4448a7.

Cells 6 (quota seeding), 13 (GCN seed labeling), 16 (SVM), 24 (MLP) and 32
(compare_labels) are copied without modification, except that ``rbf_svm_labeling``
gains a ``random_state`` argument forwarded to ``SVC`` (the notebook sets none, so its
Platt scaling is not reproducible). The inline cells 18 (K-Means), 20 (KNN), 22 (RF)
and 27 (consensus) are wrapped as functions with their bodies unchanged. These are the
specification the package strategies are tested against in ``test_pbmc68k_notebook_parity.py``.
Do not "fix" them.
"""

# ruff: noqa
# fmt: off

from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    pairwise_distances,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC




# ===========================================================================
# Cell 6: Method 1, quota seeding
# ===========================================================================

def adaptive_marker_threshold(adata, gene, quantile=0.75):
    vals = adata[:, gene].X
    if hasattr(vals, "toarray"):
        vals = vals.toarray()
    vals = np.asarray(vals).ravel()
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.inf
    return float(np.quantile(vals, quantile))

def weak_label_quota_with_min_cells(
    adata,
    marker_sets,
    gene_quantile=0.75,
    target_frac=0.25,
    min_cells_per_type=200,
    min_score=0.2,
    score_mode="frac_high",   # "frac_high" as you used; keep for clarity
    verbose=True
):
    """
    Makes weak labels with:
      - total labeled fraction ~ target_frac
      - every labeled type has >= min_cells_per_type
      - optional min_score threshold
    """

    N = adata.n_obs
    target_total = int(round(target_frac * N))

    # --- 1) Compute per-type marker score (frac of markers above gene-specific threshold)
    scores = {}
    for cell_type, genes in marker_sets.items():
        genes = [g for g in genes if g in adata.var_names]
        if len(genes) == 0:
            continue
        thr = {g: adaptive_marker_threshold(adata, g, quantile=gene_quantile) for g in genes}

        M = sc.get.obs_df(adata, keys=list(thr.keys()))
        expr_high = M.gt(pd.Series(thr))
        frac_high = expr_high.sum(axis=1) / len(thr)

        scores[cell_type] = frac_high.astype(float)

    if len(scores) == 0:
        raise ValueError("No marker sets had genes present in adata.var_names")

    score_df = pd.DataFrame(scores, index=adata.obs_names)

    # --- 2) Best type + best score per cell
    best_score = score_df.max(axis=1).astype(float)
    best_type  = score_df.idxmax(axis=1).astype(str)

    # --- 3) Eligible cells by min_score
    eligible_mask = (best_score >= float(min_score))
    if verbose:
        print(f"[quota] Eligible (score >= {min_score}): {eligible_mask.sum()}/{N} ({eligible_mask.mean():.2%})")

    # --- 4) Keep only types that have >= min_cells_per_type eligible cells
    eligible_types_counts = best_type[eligible_mask].value_counts()
    keep_types = eligible_types_counts[eligible_types_counts >= min_cells_per_type].index.tolist()

    if verbose:
        print(f"[quota] Types with >= {min_cells_per_type} eligible cells: {len(keep_types)}/{score_df.shape[1]}")

    if len(keep_types) == 0:
        # nobody qualifies; everything unlabeled
        adata.obs["weak_label"] = pd.Categorical(["unlabeled"] * N)
        adata.obs["weak_conf"]  = best_score.values
        return adata, score_df

    # If too many types * min_cells exceeds target_total, we must drop some types
    # Keep the most abundant eligible types first.
    max_types_possible = target_total // min_cells_per_type
    if max_types_possible == 0:
        if verbose:
            print("[quota][warn] target_frac too small for min_cells_per_type. Labeling none.")
        adata.obs["weak_label"] = pd.Categorical(["unlabeled"] * N)
        adata.obs["weak_conf"]  = best_score.values
        return adata, score_df

    if len(keep_types) > max_types_possible:
        keep_types = eligible_types_counts.loc[keep_types].sort_values(ascending=False).index[:max_types_possible].tolist()
        if verbose:
            print(f"[quota][warn] Too many types for target budget; keeping top {len(keep_types)} types by eligible size.")

    # --- 5) For kept types, decide how many cells to label per type
    # Base allocation: min_cells_per_type each
    base = {t: min_cells_per_type for t in keep_types}
    base_total = sum(base.values())

    # Remaining budget to reach target_total
    remaining = max(0, target_total - base_total)

    # Available eligible cells per kept type
    avail = eligible_types_counts.loc[keep_types].to_dict()

    # Extra allocation proportional to availability beyond base
    extra = {t: 0 for t in keep_types}
    if remaining > 0:
        weights = np.array([max(0, avail[t] - base[t]) for t in keep_types], dtype=float)
        if weights.sum() > 0:
            props = weights / weights.sum()
            raw_extra = np.floor(props * remaining).astype(int)
            for t, e in zip(keep_types, raw_extra):
                extra[t] = int(e)

            # distribute leftover one-by-one
            leftover = remaining - sum(extra.values())
            if leftover > 0:
                order = np.argsort(-props)  # highest proportion first
                k = 0
                while leftover > 0:
                    t = keep_types[order[k % len(order)]]
                    extra[t] += 1
                    leftover -= 1
                    k += 1

    # Final per-type quota, capped by available eligible cells
    quota = {}
    for t in keep_types:
        q = base[t] + extra[t]
        q = min(q, avail[t])  # cannot exceed eligible availability
        quota[t] = int(q)

    # If capping reduced total, we can optionally refill from types with spare capacity
    chosen_total = sum(quota.values())
    deficit = target_total - chosen_total
    if deficit > 0:
        # refill greedily from types with spare eligible cells
        spare = {t: avail[t] - quota[t] for t in keep_types}
        order = sorted(keep_types, key=lambda t: spare[t], reverse=True)
        i = 0
        while deficit > 0 and i < len(order):
            t = order[i]
            take = min(deficit, spare[t])
            if take > 0:
                quota[t] += int(take)
                deficit -= int(take)
            i += 1

    # --- 6) Select top-scoring cells within each kept type according to quota
    weak_label = np.array(["unlabeled"] * N, dtype=object)

    # Precompute indices per type among eligible cells
    best_type_arr = best_type.values
    best_score_arr = best_score.values
    eligible_idx = np.where(eligible_mask.values)[0]

    for t in keep_types:
        idx_t = eligible_idx[best_type_arr[eligible_idx] == t]
        if idx_t.size == 0:
            continue
        # sort by best_score descending
        idx_sorted = idx_t[np.argsort(-best_score_arr[idx_t])]
        pick = idx_sorted[:quota[t]]
        weak_label[pick] = t

    adata.obs["weak_label"] = pd.Categorical(weak_label)
    adata.obs["weak_conf"]  = best_score.values

    if verbose:
        vc = adata.obs["weak_label"].value_counts()
        labeled = (adata.obs["weak_label"] != "unlabeled").sum()
        print(f"[quota] Target={target_frac:.0%} | Achieved={labeled/N:.2%} | labeled={labeled}/{N}")
        print(vc)
        small = vc[(vc.index != "unlabeled") & (vc < min_cells_per_type)]
        if len(small) > 0:
            print("[quota][ERROR] still found small labeled types:", small.to_dict())
        else:
            print(f"[quota] OK: all labeled types have >= {min_cells_per_type} cells")

    return adata, score_df


# ===========================================================================
# Cell 13: Method 3, GCN seed labeling
# ===========================================================================

def gcn_seed_labeling(
    adata,
    score_df,                       # pd.DataFrame: index=cells, columns=types
    target_frac=0.25,               # cap after gating (None to disable)
    min_score=0.55,                 # gate 1: top1 >= min_score
    delta=0.15,                     # gate 2: (top1 - top2) >= delta
    alpha=0.85,                     # propagation strength
    n_iter=100,                      # propagation iterations
    min_cells_per_type_pct=0.005,   # e.g., 0.005 = 0.5% of total cells
    min_cells_floor=200,            # absolute minimum
    label_col="weak_label_gcn",
    conf_col="seed_conf_gcn",
    neighbors_k=15,
    pca_comps=30,
    use_rep="X_pca",
    verbose=True
):
    """
    GCN-style label propagation on Scanpy kNN graph, then hard-label with gates:
      - top1 >= min_score
      - (top1 - top2) >= delta
      - drop types with < max(min_cells_floor, min_cells_per_type_pct*N)
      - optional cap: keep only top target_frac by confidence among gated cells

    Compatible with older AnnData versions (no .obsp_keys()).
    """

    # -------------------------
    # 0) Align score_df to adata
    # -------------------------
    if not isinstance(score_df, pd.DataFrame):
        raise ValueError("score_df must be a pandas DataFrame (cells x types).")

    score_df = score_df.copy()
    score_df.index = score_df.index.astype(str)
    score_df = score_df.reindex(adata.obs_names.astype(str))

    # drop columns that are all NaN
    score_df = score_df.loc[:, score_df.notna().any(axis=0)]
    if score_df.shape[1] == 0:
        raise ValueError("score_df has no usable type columns after filtering.")

    types = score_df.columns.astype(str).tolist()

    # fill NaN with 0
    S0 = score_df.fillna(0.0).to_numpy(dtype=float)

    # row-normalize to probabilities
    row_sum = S0.sum(axis=1, keepdims=True) + 1e-12
    Y0 = S0 / row_sum

    # -------------------------
    # 1) Ensure PCA exists
    # -------------------------
    if (use_rep not in getattr(adata, "obsm", {})) or (adata.obsm[use_rep] is None):
        sc.pp.pca(adata, n_comps=pca_comps, svd_solver="arpack")
        use_rep = "X_pca"

    # -------------------------
    # 2) Ensure neighbors graph exists
    #    Old/new Scanpy store connectivities in adata.obsp['connectivities']
    # -------------------------
    has_obsp = hasattr(adata, "obsp") and isinstance(adata.obsp, dict)
    has_conn = has_obsp and ("connectivities" in adata.obsp)

    if not has_conn:
        # build neighbors (writes into adata.obsp / adata.uns["neighbors"])
        n_pcs_use = min(pca_comps, adata.obsm[use_rep].shape[1])
        sc.pp.neighbors(adata, n_neighbors=neighbors_k, n_pcs=n_pcs_use)
        has_conn = ("connectivities" in adata.obsp)

    if not has_conn:
        # last fallback: some very old objects might store it differently
        # but in Scanpy this should normally exist after sc.pp.neighbors
        raise ValueError("Could not find neighbors connectivities in adata.obsp['connectivities'].")

    A = adata.obsp["connectivities"]
    if not sparse.isspmatrix(A):
        A = sparse.csr_matrix(A)

    # -------------------------
    # 3) Build normalized adjacency (GCN-style)
    #    A_hat = D^{-1/2} (A + I) D^{-1/2}
    # -------------------------
    N = adata.n_obs
    I = sparse.eye(N, format="csr")
    A_tilde = (A + I).tocsr()

    deg = np.asarray(A_tilde.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt, format="csr")
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # -------------------------
    # 4) Propagate
    #    Y_{t+1} = alpha A_hat Y_t + (1-alpha) Y0
    # -------------------------
    Y = Y0.copy()
    for _ in range(n_iter):
        Y = alpha * (A_hat @ Y) + (1.0 - alpha) * Y0

    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)

    # -------------------------
    # 5) Gates: top1>=min_score and margin>=delta
    # -------------------------
    top1_idx = Y.argmax(axis=1)
    top1 = Y[np.arange(N), top1_idx]

    Y2 = Y.copy()
    Y2[np.arange(N), top1_idx] = -np.inf
    top2 = Y2.max(axis=1)

    margin = top1 - top2
    gate = (top1 >= float(min_score)) & (margin >= float(delta))

    labels = np.array(["unlabeled"] * N, dtype=object)
    labels[gate] = np.array(types, dtype=object)[top1_idx[gate]]

    conf = np.zeros(N, dtype=float)
    conf[gate] = top1[gate]

    # -------------------------
    # 6) Optional cap to target_frac
    # -------------------------
    if target_frac is not None:
        target_n = int(round(float(target_frac) * N))
        if target_n > 0:
            idx_labeled = np.where(labels != "unlabeled")[0]
            if idx_labeled.size > target_n:
                order = idx_labeled[np.argsort(-conf[idx_labeled])]
                keep = order[:target_n]
                drop = np.setdiff1d(idx_labeled, keep, assume_unique=False)
                labels[drop] = "unlabeled"
                conf[drop] = 0.0

    # -------------------------
    # 7) Drop small types (pct rule + floor)
    # -------------------------
    min_cells_pct = int(np.floor(float(min_cells_per_type_pct) * N))
    min_cells = max(int(min_cells_floor), int(min_cells_pct), 5)

    vc = pd.Series(labels).value_counts()
    keep_types = set(vc[vc >= min_cells].index.tolist())
    keep_types.discard("unlabeled")

    small_mask = (labels != "unlabeled") & (~pd.Series(labels).isin(keep_types).values)
    labels[small_mask] = "unlabeled"
    conf[small_mask] = 0.0

    # -------------------------
    # 8) Write back
    # -------------------------
    adata.obs[label_col] = pd.Categorical(labels)
    if "unlabeled" not in adata.obs[label_col].cat.categories:
        adata.obs[label_col] = adata.obs[label_col].cat.add_categories(["unlabeled"])
    adata.obs[label_col] = adata.obs[label_col].fillna("unlabeled")

    adata.obs[conf_col] = conf

    if verbose:
        print(f"[GCN] min_cells_per_type = {min_cells} (max(floor={min_cells_floor}, pct={min_cells_pct}, 5))")
        print("[GCN] final label distribution:")
        print(adata.obs[label_col].value_counts())
        print(f"[GCN] labeled fraction: {(adata.obs[label_col] != 'unlabeled').mean():.2%}")
        print(f"[GCN] gate pass (before size filter/cap): {gate.mean():.2%}")
        print(f"[GCN] median top1={np.median(top1):.3f}, median margin={np.median(margin):.3f}")

    return {
        "Y_propagated": Y,
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "types": types,
        "min_cells_used": min_cells,
    }


# ===========================================================================
# Cell 16: SVM
# ===========================================================================

def rbf_svm_labeling(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    out_col="label_svm_rbf",
    out_conf_col="svm_rbf_conf",
    C=5.0,
    gamma="scale",
    min_conf=0.0,
    random_state=None,  # added: the notebook sets none
):
    X = adata.obsm[X_key]
    y = adata.obs[seed_col].astype(str).values
    known_mask = y != "unlabeled"
    unknown_mask = ~known_mask

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(C=C, gamma=gamma, kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
    )
    clf.fit(X[known_mask], y[known_mask])

    proba = clf.predict_proba(X[unknown_mask])
    pred = clf.classes_[np.argmax(proba, axis=1)]
    conf = np.max(proba, axis=1)

    y_out = y.copy()
    y_out[unknown_mask] = pred

    if min_conf > 0:
        idx_unknown = np.where(unknown_mask)[0]
        y_out[idx_unknown[conf < min_conf]] = "unlabeled"

    adata.obs[out_col] = pd.Categorical(y_out)
    conf_all = np.zeros(adata.n_obs, dtype=float)
    conf_all[unknown_mask] = conf
    adata.obs[out_conf_col] = conf_all
    return clf


# ===========================================================================
# Cells 18, 20, 22: K-Means, KNN, Random Forest
# ===========================================================================

# Cell 18, wrapped: X, labels_seed, known_mask, known_types come from cell 15.
def kmeans_labeling(X, labels_seed, known_mask, known_types, C=None):
    # 3) K-means (k = number of known types), map clusters by seed-majority
    k = len(known_types)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km_labels = km.fit_predict(X)

    # Map kmeans cluster → type using majority of seeded cells in that cluster
    cluster_to_type = {}
    for c in range(k):
        seed_in_c = labels_seed[(km_labels == c) & known_mask]
        if len(seed_in_c) == 0:
            # If no seeds, map to nearest centroid
            center = km.cluster_centers_[c][None, :]
            cl_type = known_types[pairwise_distances(center, C).argmin()]
        else:
            cl_type = Counter(seed_in_c).most_common(1)[0][0]
        cluster_to_type[c] = cl_type

    label_kmeans = np.array([cluster_to_type[c] for c in km_labels], dtype=object)
    return label_kmeans


# Cell 20, wrapped.
def knn_labeling(X, labels_seed, known_mask):
    # 4) k-NN classifier (fit on seeds, predict all)
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance",p=1)
    knn.fit(X[known_mask], labels_seed[known_mask])
    label_knn = knn.predict(X)
    proba_knn = knn.predict_proba(X)  # for confidence, if needed
    return label_knn


# Cell 22, wrapped.
def rf_labeling(X, labels_seed, known_mask):
    # 5) Random Forest classifier (fit on seeds, predict all)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=42
    )
    rf.fit(X[known_mask], labels_seed[known_mask])
    label_rf = rf.predict(X)
    proba_rf = rf.predict_proba(X)
    return label_rf


# ===========================================================================
# Cell 24: MLP
# ===========================================================================

def mlp_labeling(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    out_col="label_mlp",
    out_conf_col="mlp_conf",
    hidden_layer_sizes=(256, 128),
    alpha=1e-4,
    max_iter=300,
    min_conf=0.0,
    random_state=0,
    early_stopping=True,        # works safely now because y is numeric
    validation_fraction=0.1
):
    X = np.asarray(adata.obsm[X_key], dtype=np.float32)
    y_str = adata.obs[seed_col].astype(str).values

    known_mask = y_str != "unlabeled"
    unknown_mask = ~known_mask

    if known_mask.sum() < 10:
        raise ValueError(f"Too few labeled seeds: {known_mask.sum()}")

    # ---- Encode labels to integers (prevents np.isnan on strings) ----
    le = LabelEncoder()
    y_known = le.fit_transform(y_str[known_mask])  # int labels: 0..C-1

    # if only 1 class present, classifier can't train
    if len(le.classes_) < 2:
        raise ValueError(f"Only one labeled class present in seeds: {le.classes_[0]}")

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=15,
            verbose=False
        )
    )

    clf.fit(X[known_mask], y_known)

    # Predict probabilities for unknown cells
    proba = clf.predict_proba(X[unknown_mask])              # shape: (n_unknown, n_classes)
    pred_int = np.argmax(proba, axis=1)
    pred_lbl = le.inverse_transform(pred_int)               # back to strings
    conf = np.max(proba, axis=1)

    # Fill output labels
    y_out = y_str.copy()
    y_out[unknown_mask] = pred_lbl

    # Optional confidence gate
    if min_conf > 0:
        idx_unknown = np.where(unknown_mask)[0]
        y_out[idx_unknown[conf < min_conf]] = "unlabeled"

    adata.obs[out_col] = pd.Categorical(y_out)

    conf_all = np.zeros(adata.n_obs, dtype=float)
    conf_all[unknown_mask] = conf
    adata.obs[out_conf_col] = conf_all

    return clf, le


# ===========================================================================
# Cell 27: consensus
# ===========================================================================

# Cell 27, wrapped.
def notebook_consensus(adata):
    # 6) Consensus label for previously unknown cells
    methods = np.vstack([
        adata.obs["label_svm_rbf"].astype(str).values,
        adata.obs["label_kmeans"].astype(str).values,
        adata.obs["label_knn"].astype(str).values,
        adata.obs["label_rf"].astype(str).values,
         adata.obs["label_mlp"].astype(str).values
    ])  # shape: 5 x n_cells

    consensus = np.array(["unlabeled"] * adata.n_obs, dtype=object)
    agree_frac = np.zeros(adata.n_obs)

    for i in range(adata.n_obs):
        votes = [m[i] for m in methods]
        # don’t count 'unlabeled' as a valid vote if other labels exist
        filtered = [v for v in votes if v != "unlabeled"]
        if len(filtered) == 0:
            consensus[i] = "unlabeled"
            agree_frac[i] = 0.0
        else:
            c = Counter(filtered)
            maj_label, maj_votes = c.most_common(1)[0]
            consensus[i] = maj_label
            agree_frac[i] = maj_votes / 5.0
    return consensus, agree_frac


# ===========================================================================
# Cell 32: compare_labels
# ===========================================================================

def compare_labels(
    adata,
    pred_col,
    ref_col,
    ignore_labels=("unlabeled", "Unknown"),
    evaluate_on="both"  # "all" or "labeled" or "both"
):
    """
    Compare adata.obs[pred_col] (prediction) vs adata.obs[ref_col] (reference)
    using ARI, NMI, macro-F1, accuracy, and coverage.

    ignore_labels are treated as 'unlabeled/unknown' and excluded in 'labeled' mode.
    """

    y_pred = adata.obs[pred_col].astype(str).values
    y_ref  = adata.obs[ref_col].astype(str).values

    ignore = set(map(str, ignore_labels))
    pred_labeled = ~np.isin(y_pred, list(ignore))
    ref_labeled  = ~np.isin(y_ref,  list(ignore))

    # For "labeled" evaluation, keep only cells where BOTH are labeled
    mask_labeled = pred_labeled & ref_labeled

    def _metrics(mask):
        yp = y_pred[mask]
        yr = y_ref[mask]
        if yp.size == 0:
            return {"ARI": np.nan, "NMI": np.nan, "macroF1": np.nan, "acc": np.nan, "n_eval": 0}

        return {
            "ARI": adjusted_rand_score(yr, yp),
            "NMI": normalized_mutual_info_score(yr, yp),
            "macroF1": f1_score(yr, yp, average="macro", zero_division=0),
            "acc": accuracy_score(yr, yp),
            "n_eval": int(yp.size),
        }

    out = {
        "pred_col": pred_col,
        "ref_col": ref_col,
        "coverage_pred": float(pred_labeled.mean()),
        "coverage_ref": float(ref_labeled.mean()),
        "coverage_both": float(mask_labeled.mean()),
        "n_total": int(len(y_pred)),
    }

    if evaluate_on in ("all", "both"):
        out_all = _metrics(np.ones(len(y_pred), dtype=bool))
        out.update({f"all_{k}": v for k, v in out_all.items()})

    if evaluate_on in ("labeled", "both"):
        out_lab = _metrics(mask_labeled)
        out.update({f"labeled_{k}": v for k, v in out_lab.items()})

    return out

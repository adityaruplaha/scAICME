"""Verbatim reference implementation from the final PBMC 68k notebook (Sept 2026).

Source SHA256 ed3e912d9a4e1bd99a4248661fe3bd6e9a55edeea1385a9d71e7c322108cc6ac.

Cells 3 (Method 2, DP-GMM seeding without PCA), 5 (training-data helper, SVM, KNN,
random forest, MLP), 9 (metrics) and 15 (cluster quality) are copied without
modification. The inline k-means and consensus blocks of cell 5 are wrapped as
functions with their bodies unchanged. This is the specification the package strategies
are tested against in ``test_pbmc68k_dp_parity.py``. Do not "fix" them.

Cells 1 and 2 (Method 1 quota seeding and Method 3 quota GCN seeding) are defined in the
notebook but were never executed in the saved run, and the pipeline below does not use
them; the earlier fixed-gate GCN seeder is covered by
``reference_pbmc68k_notebook.py``.
"""

# ruff: noqa
# fmt: off

from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC




# ============================================================================
# Cell 3: Method 2, DP-GMM marker-set seeding (no PCA)
# ============================================================================

def dp_seed_by_marker_sets_soft_no_pca(
    adata,
    marker_sets,
    n_components=30,
    weight_concentration_prior=0.05,
    random_state=0,
    per_gene_pos_quantile=0.70,
    cluster_score_min=0.20,
    min_cells_cluster=100,
    label_col_prefix="dp_seed",
    conf_col_prefix="dp_seed_conf",
    verbose=True
):
    diagnostics = {}
    all_vars = set(adata.var_names.astype(str))
    n_cells = adata.n_obs

    # --- Pre-filter thresholds ---
    min_genes_present     = 3
    min_expressed_markers = 2
    min_cells_per_gene    = 20
    min_cell_enrichment   = 0.05

    for setname, genes in marker_sets.items():
        # 0) Keep only markers present in adata
        genes = [g for g in genes if g in all_vars]
        if len(genes) < min_genes_present:
            if verbose:
                print(f"[DP-soft][skip] {setname}: only {len(genes)} markers present in adata.")
            continue

        # 1) Extract expression
        Xg = adata[:, genes].X
        if hasattr(Xg, "toarray"):
            Xg = Xg.toarray()
        Xg = np.asarray(Xg, dtype=float)

        # 2) Pre-filter 1: expressed markers
        expr_per_gene = (Xg > 0).sum(axis=0)
        expressed_markers = int(np.sum(expr_per_gene >= min_cells_per_gene))
        if expressed_markers < min_expressed_markers:
            if verbose:
                print(f"[DP-soft][skip] {setname}: only {expressed_markers} expressed markers.")
            continue

        # 3) Pre-filter 2: cell enrichment
        cells_with_any_marker = (Xg > 0).sum(axis=1) > 0
        cell_enrichment = float(np.mean(cells_with_any_marker))
        if cell_enrichment < min_cell_enrichment:
            if verbose:
                print(f"[DP-soft][skip] {setname}: cell enrichment {cell_enrichment:.4f} < {min_cell_enrichment}.")
            continue

        # 4) Per-gene thresholds
        gene_thr = {}
        for j, g in enumerate(genes):
            v = Xg[:, j]
            pos = v[v > 0]
            if pos.size >= 20:
                gene_thr[g] = float(np.quantile(pos, per_gene_pos_quantile))
            elif pos.size > 0:
                gene_thr[g] = float(np.median(pos))
            else:
                gene_thr[g] = np.inf

        # 5) Cell marker scores
        high_mask = np.zeros_like(Xg, dtype=bool)
        for j, g in enumerate(genes):
            thr = gene_thr[g]
            if np.isinf(thr):
                continue
            high_mask[:, j] = Xg[:, j] > thr

        marker_score = high_mask.sum(axis=1) / max(1, len(genes))

        # 6) Standardize + BGM (NO PCA)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xg_scaled = scaler.fit_transform(Xg)

        bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=weight_concentration_prior,
            covariance_type="full",
            max_iter=1000,
            n_init=1,
            random_state=random_state,
        )
        bgm.fit(Xg_scaled)
        cl = bgm.predict(Xg_scaled)

        # 7) Cluster-level stats
        df_cluster = pd.DataFrame({"cluster": cl, "score": marker_score})
        cluster_summary = df_cluster.groupby("cluster")["score"].agg(["mean", "count"])

        good_clusters = cluster_summary[
            (cluster_summary["mean"] >= cluster_score_min) &
            (cluster_summary["count"] >= min_cells_cluster)
        ].index.tolist()

        # 8) Assign labels + confidence
        label_col = f"{label_col_prefix}_{setname}"
        conf_col  = f"{conf_col_prefix}_{setname}"

        labels = np.array(["unlabeled"] * n_cells, dtype=object)
        confs  = np.zeros(n_cells, dtype=float)

        if len(good_clusters) > 0:
            max_score = marker_score.max() if marker_score.max() > 0 else 1.0
            for cid in good_clusters:
                idx = np.where(cl == cid)[0]
                labels[idx] = setname
                confs[idx] = marker_score[idx] / max_score

        adata.obs[label_col] = pd.Categorical(labels)
        adata.obs[conf_col]  = confs

        # 9) Diagnostics
        n_lab = (labels != "unlabeled").sum()
        pct_lab = n_lab / n_cells * 100.0
        diagnostics[setname] = {
            "n_markers_used": len(genes),
            "n_clusters": int(len(cluster_summary)),
            "good_clusters": good_clusters,
            "n_labelled": int(n_lab),
            "pct_labelled": float(pct_lab),
            "cluster_summary": cluster_summary,
            "expressed_markers": expressed_markers,
            "cell_enrichment": cell_enrichment,
            "n_features_used": int(Xg_scaled.shape[1]),
        }

        if verbose:
            print(
                f"{setname}: labelled {n_lab}/{n_cells} ({pct_lab:.2f}%), "
                f"good_clusters={len(good_clusters)}, n_features={Xg_scaled.shape[1]}"
            )

    # --- Build consensus weak label across marker sets ---
    label_cols = [f"{label_col_prefix}_{s}" for s in marker_sets.keys() if f"{label_col_prefix}_{s}" in adata.obs]
    conf_cols  = [f"{conf_col_prefix}_{s}" for s in marker_sets.keys() if f"{conf_col_prefix}_{s}" in adata.obs]

    if len(label_cols) > 0:
        lab_df = adata.obs[label_cols].astype(str)
        conf_df = adata.obs[conf_cols].astype(float).fillna(0.0)

        dp_consensus_label = []
        dp_consensus_conf  = []

        for i in adata.obs_names:
            row_labels = lab_df.loc[i].values
            row_confs  = conf_df.loc[i].values
            row_confs = np.where(row_labels == "unlabeled", 0.0, row_confs)

            best_idx = int(np.argmax(row_confs))
            best_conf = float(row_confs[best_idx])
            best_label = row_labels[best_idx] if best_conf > 0 else "unlabeled"

            dp_consensus_label.append(best_label)
            dp_consensus_conf.append(best_conf)

        adata.obs["weak_label"] = pd.Categorical(dp_consensus_label)
        adata.obs["weak_conf"]  = np.array(dp_consensus_conf, dtype=float)

        # --- Final size filter ---
        threshold = max(100, int(0.002 * adata.n_obs))
        counts = adata.obs["weak_label"].value_counts()

        keep_types = set(counts[counts >= threshold].index.tolist())

        mask_keep = adata.obs["weak_label"].isin(keep_types)
        adata.obs.loc[~mask_keep, "weak_label"] = "unlabeled"
        adata.obs.loc[~mask_keep, "weak_conf"]  = 0.0

        if verbose:
            print("\n[DP-soft no PCA] Final weak_label distribution:")
            print(adata.obs["weak_label"].value_counts(normalize=True) * 100)

    return diagnostics


# ============================================================================
# Cell 5: training data and classifiers
# ============================================================================

def get_training_data(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    conf_col="weak_conf",
    min_seed_conf=0.3,
    max_pcs=30
):
    X = np.asarray(adata.obsm[X_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{X_key} must be 2D.")

    # use only first max_pcs PCs
    X = X[:, :min(max_pcs, X.shape[1])]

    y = adata.obs[seed_col].astype(str).values

    if conf_col in adata.obs.columns:
        conf = np.asarray(adata.obs[conf_col], dtype=float)
        known_mask = (y != "unlabeled") & (conf >= min_seed_conf)
    else:
        known_mask = (y != "unlabeled")

    if known_mask.sum() < 20:
        raise ValueError(f"Too few confident labeled seeds: {known_mask.sum()}")

    return X, y, known_mask


def rbf_svm_labeling(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    conf_col="weak_conf",
    out_col="label_svm_rbf",
    out_conf_col="svm_rbf_conf",
    C=2.0,
    gamma="scale",
    min_seed_conf=0.3,
    min_conf=0.6,
    max_pcs=30
):
    X, y, known_mask = get_training_data(
        adata,
        X_key=X_key,
        seed_col=seed_col,
        conf_col=conf_col,
        min_seed_conf=min_seed_conf,
        max_pcs=max_pcs
    )

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )

    clf.fit(X[known_mask], y[known_mask])

    proba = clf.predict_proba(X)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    conf = np.max(proba, axis=1)

    y_out = pred.copy()
    y_out[conf < min_conf] = "unlabeled"

    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

    return clf


# Inline k-means block of cell 5, wrapped; body unchanged.
def kmeans_labeling(adata):
    # ----- Feature space -----
    X = np.asarray(adata.obsm["X_pca"][:, :30], dtype=float)

    # ----- Seed labels -----
    labels_seed = adata.obs["weak_label"].astype(str).values
    conf = adata.obs.get("weak_conf", pd.Series(np.ones(len(labels_seed))))

    # Use only confident seeds
    seed_mask = (labels_seed != "unlabeled") & (conf > 0.3)

    known_types = np.unique(labels_seed[seed_mask])
    print("Known types:", known_types)

    # ----- Fit KMeans -----
    k = min(30, max(10, int(np.sqrt(adata.n_obs/2))))   # adaptive cluster number
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
        max_iter=500
    )
    km.fit(X)

    # ----- Compute centroids of seed types (fallback) -----
    centroids = []
    for t in known_types:
        centroids.append(X[seed_mask & (labels_seed == t)].mean(axis=0))
    C = np.vstack(centroids)

    # ----- Map clusters to labels -----
    cluster_to_type = {}
    cluster_conf = {}

    for c in np.unique(km.labels_):

        idx = np.where(km.labels_ == c)[0]

        seed_in_c = labels_seed[idx]
        seed_in_c = seed_in_c[seed_in_c != "unlabeled"]

        # Case 1: cluster has seeds
        if len(seed_in_c) > 0:

            counts = Counter(seed_in_c)
            maj_label, maj_count = counts.most_common(1)[0]

            cluster_to_type[c] = maj_label
            cluster_conf[c] = maj_count / len(idx)

        # Case 2: no seeds → assign by nearest seed centroid
        else:

            center = km.cluster_centers_[c].reshape(1, -1)
            nearest = pairwise_distances(center, C).argmin()

            cluster_to_type[c] = known_types[nearest]
            cluster_conf[c] = 0.0


    # ----- Assign labels to cells -----
    labels_out = np.array([cluster_to_type[c] for c in km.labels_])
    conf_out = np.array([cluster_conf[c] for c in km.labels_])
    return labels_out, conf_out


def knn_labeling(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    conf_col="weak_conf",
    out_col="label_knn",
    out_conf_col="knn_conf",
    n_neighbors=9,
    min_seed_conf=0.3,
    min_conf=0.55,
    max_pcs=30
):
    X, y, known_mask = get_training_data(
        adata,
        X_key=X_key,
        seed_col=seed_col,
        conf_col=conf_col,
        min_seed_conf=min_seed_conf,
        max_pcs=max_pcs
    )

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        p=2
    )
    knn.fit(X[known_mask], y[known_mask])

    pred = knn.predict(X)
    proba = knn.predict_proba(X)
    conf = np.max(proba, axis=1)

    y_out = pred.copy()
    y_out[conf < min_conf] = "unlabeled"

    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

    return knn


def rf_labeling(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    conf_col="weak_conf",
    out_col="label_rf",
    out_conf_col="rf_conf",
    n_estimators=500,
    max_depth=18,
    min_samples_leaf=5,
    min_seed_conf=0.3,
    min_conf=0.55,
    max_pcs=30,
    random_state=42
):
    X, y, known_mask = get_training_data(
        adata,
        X_key=X_key,
        seed_col=seed_col,
        conf_col=conf_col,
        min_seed_conf=min_seed_conf,
        max_pcs=max_pcs
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state
    )

    rf.fit(X[known_mask], y[known_mask])

    pred = rf.predict(X)
    proba = rf.predict_proba(X)
    conf = np.max(proba, axis=1)

    y_out = pred.copy()
    y_out[conf < min_conf] = "unlabeled"

    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

    return rf


def mlp_labeling(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    conf_col="weak_conf",
    out_col="label_mlp",
    out_conf_col="mlp_conf",
    hidden_layer_sizes=(128, 64),
    alpha=1e-3,
    max_iter=400,
    min_seed_conf=0.3,
    min_conf=0.60,
    max_pcs=30,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
):
    X, y_str, known_mask = get_training_data(
        adata,
        X_key=X_key,
        seed_col=seed_col,
        conf_col=conf_col,
        min_seed_conf=min_seed_conf,
        max_pcs=max_pcs
    )

    le = LabelEncoder()
    y_known = le.fit_transform(y_str[known_mask])

    if len(le.classes_) < 2:
        raise ValueError(f"Only one labeled class present in seeds: {le.classes_[0]}")

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False
        )
    )

    clf.fit(X[known_mask], y_known)

    proba = clf.predict_proba(X)
    pred_int = np.argmax(proba, axis=1)
    pred_lbl = le.inverse_transform(pred_int)
    conf = np.max(proba, axis=1)

    y_out = pred_lbl.copy()
    y_out[conf < min_conf] = "unlabeled"

    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

    return clf, le


# Inline consensus block of cell 5, wrapped; body unchanged.
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


# ============================================================================
# Cell 9: metrics
# ============================================================================

def compare_labels_ari_acc_spec_sens(
    adata,
    pred_col,
    ref_col,
    ignore_labels=("unlabeled", "Unknown"),
    evaluate_on="both"   # "all" or "labeled" or "both"
):
    """
    Compare adata.obs[pred_col] (prediction) vs adata.obs[ref_col] (reference)
    using:
      - ARI
      - Accuracy
      - Specificity (macro, one-vs-rest)
      - Sensitivity/Recall (macro, one-vs-rest)
    """

    y_pred_all = adata.obs[pred_col].astype(str).values
    y_ref_all  = adata.obs[ref_col].astype(str).values

    ignore = set(map(str, ignore_labels))
    pred_labeled = ~np.isin(y_pred_all, list(ignore))
    ref_labeled  = ~np.isin(y_ref_all,  list(ignore))
    mask_labeled = pred_labeled & ref_labeled

    def _spec_sens_macro(yr, yp):
        """
        Returns (specificity_macro, sensitivity_macro) for multiclass via one-vs-rest.
        """
        classes = np.unique(np.concatenate([yr, yp]))
        specs, sens = [], []

        for cls in classes:
            yr_bin = (yr == cls).astype(int)
            yp_bin = (yp == cls).astype(int)

            cm = confusion_matrix(yr_bin, yp_bin, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan     # TNR
            sen  = tp / (tp + fn) if (tp + fn) > 0 else np.nan     # TPR (Recall)

            specs.append(spec)
            sens.append(sen)

        return float(np.nanmean(specs)), float(np.nanmean(sens))

    def _metrics(mask):
        yr = y_ref_all[mask]
        yp = y_pred_all[mask]
        if yp.size == 0:
            return {
                "ARI": np.nan,
                "Accuracy": np.nan,
                "Specificity_macro": np.nan,
                "Sensitivity_macro": np.nan,
                "n_eval": 0
            }

        spec_m, sens_m = _spec_sens_macro(yr, yp)

        return {
            "ARI": float(adjusted_rand_score(yr, yp)),
            "Accuracy": float(accuracy_score(yr, yp)),
            "Specificity_macro": float(spec_m),
            "Sensitivity_macro": float(sens_m),
            "n_eval": int(yp.size),
        }

    out = {
        "pred_col": pred_col,
        "ref_col": ref_col,
        "coverage_pred": float(pred_labeled.mean()),
        "coverage_ref": float(ref_labeled.mean()),
        "coverage_both": float(mask_labeled.mean()),
        "n_total": int(len(y_pred_all)),
    }

    if evaluate_on in ("all", "both"):
        out_all = _metrics(np.ones(len(y_pred_all), dtype=bool))
        out.update({f"all_{k}": v for k, v in out_all.items()})

    if evaluate_on in ("labeled", "both"):
        out_lab = _metrics(mask_labeled)
        out.update({f"labeled_{k}": v for k, v in out_lab.items()})

    return out


# ============================================================================
# Cell 15: cluster quality
# ============================================================================

def clustering_metrics(adata, cluster_col, X_key="X_pca"):
    X = np.asarray(adata.obsm[X_key])
    labels = adata.obs[cluster_col].astype(str).values

    if len(np.unique(labels)) < 2:
        return {
            "method": cluster_col,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        }

    return {
        "method": cluster_col,
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels)
    }

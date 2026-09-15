"""Verbatim reference implementation from the GSE225475 spatial notebook.

Source: the GSE225475 spatial notebook (cells 4 and 6),
SHA256 ef1c0683a45d9d81bd8025c90d7b6b035d80b7eae4e87c57a542a30c9060b342.

The functions below are copied without modification (only the module-level
PCA/neighbors block and the method run loop were dropped, and the consensus block
was wrapped as ``notebook_consensus``). They are the specification that the package
strategies are tested against in ``test_gse225475_notebook_parity.py``. Do not "fix" them.
"""

# ruff: noqa
# fmt: off

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# ===========================================================================
# Cell 4: dp_seed_by_marker_sets_soft
# ===========================================================================

def dp_seed_by_marker_sets_soft(
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
        #threshold = max(100, int(0.002 * adata.n_obs))
        threshold = max(20, int(0.001 * adata.n_obs))
        counts = adata.obs["weak_label"].value_counts()

        keep_types = set(counts[counts >= threshold].index.tolist())

        mask_keep = adata.obs["weak_label"].isin(keep_types)
        adata.obs.loc[~mask_keep, "weak_label"] = "unlabeled"
        adata.obs.loc[~mask_keep, "weak_conf"]  = 0.0

        if verbose:
            print("\n[DP-soft no PCA] Final weak_label distribution:")
            print(adata.obs["weak_label"].value_counts(normalize=True) * 100)

    return diagnostics


# ===========================================================================
# Cell 6: get_training_data, classifiers, consensus
# ===========================================================================

# -----------------------------
# Helper
# -----------------------------
def get_training_data(
    adata,
    X_key="X_pca",
    seed_col="weak_label",
    conf_col="weak_conf",
    min_seed_conf=0.05,
    max_pcs=15
):
    X = np.asarray(adata.obsm[X_key], dtype=np.float32)
    X = X[:, :min(max_pcs, X.shape[1])]
    y = adata.obs[seed_col].astype(str).values

    if conf_col in adata.obs.columns:
        conf = np.asarray(adata.obs[conf_col], dtype=float)
        known_mask = (y != "unlabeled") & (conf >= min_seed_conf)
    else:
        known_mask = (y != "unlabeled")

    y_known = y[known_mask]
    classes = np.unique(y_known)

    print("Known labeled spots:", known_mask.sum())
    print("Known classes:", classes.tolist())

    if known_mask.sum() < 20:
        raise ValueError(f"Too few confident labeled seeds: {known_mask.sum()}")
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 labeled classes, got {len(classes)}: {classes.tolist()}")

    return X, y, known_mask

# -----------------------------
# Methods
# -----------------------------
def rbf_svm_labeling(adata, X_key="X_pca", seed_col="weak_label", conf_col="weak_conf",
                     out_col="label_svm_rbf", out_conf_col="svm_rbf_conf",
                     C=2.0, gamma="scale", min_seed_conf=0.05, min_conf=0.3, max_pcs=15):
    X, y, known_mask = get_training_data(adata, X_key, seed_col, conf_col, min_seed_conf, max_pcs)
    clf = make_pipeline(
        StandardScaler(),
        SVC(C=C, gamma=gamma, kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    )
    clf.fit(X[known_mask], y[known_mask])
    proba = clf.predict_proba(X)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    conf = np.max(proba, axis=1)
    y_out = pred.copy()
    y_out[conf < min_conf] = "unlabeled"
    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

def kmeans_seed_transfer(adata, X_key="X_pca", seed_col="weak_label", conf_col="weak_conf",
                         out_col="label_kmeans", out_conf_col="kmeans_conf", min_seed_conf=0.05):
    X, y, known_mask = get_training_data(adata, X_key, seed_col, conf_col, min_seed_conf, 30)
    k = min(10, max(8, int(np.sqrt(adata.n_obs / 2))))
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    km.fit(X)
    known_types = np.unique(y[known_mask])
    centroids = np.vstack([X[known_mask & (y == t)].mean(axis=0) for t in known_types])

    cluster_to_type, cluster_conf = {}, {}
    for c in np.unique(km.labels_):
        idx = np.where(km.labels_ == c)[0]
        seed_in_c = y[idx]
        seed_in_c = seed_in_c[seed_in_c != "unlabeled"]
        if len(seed_in_c) > 0:
            counts = Counter(seed_in_c)
            maj_label, maj_count = counts.most_common(1)[0]
            cluster_to_type[c] = maj_label
            cluster_conf[c] = maj_count / len(idx)
        else:
            nearest = pairwise_distances(km.cluster_centers_[c].reshape(1, -1), centroids).argmin()
            cluster_to_type[c] = known_types[nearest]
            cluster_conf[c] = 0.0

    labels_out = np.array([cluster_to_type[c] for c in km.labels_], dtype=object)
    conf_out = np.array([cluster_conf[c] for c in km.labels_], dtype=float)
    adata.obs[out_col] = pd.Categorical(labels_out)
    adata.obs[out_conf_col] = conf_out

def knn_labeling(adata, X_key="X_pca", seed_col="weak_label", conf_col="weak_conf",
                 out_col="label_knn", out_conf_col="knn_conf",
                 n_neighbors=9, min_seed_conf=0.05, min_conf=0.3, max_pcs=15):
    X, y, known_mask = get_training_data(adata, X_key, seed_col, conf_col, min_seed_conf, max_pcs)
    k = min(n_neighbors, max(1, known_mask.sum() - 1))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", p=2)
    knn.fit(X[known_mask], y[known_mask])
    pred = knn.predict(X)
    proba = knn.predict_proba(X)
    conf = np.max(proba, axis=1)
    y_out = pred.copy()
    y_out[conf < min_conf] = "unlabeled"
    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

def rf_labeling(adata, X_key="X_pca", seed_col="weak_label", conf_col="weak_conf",
                out_col="label_rf", out_conf_col="rf_conf",
                n_estimators=300, max_depth=18, min_samples_leaf=3,
                min_seed_conf=0.05, min_conf=0.3, max_pcs=15):
    X, y, known_mask = get_training_data(adata, X_key, seed_col, conf_col, min_seed_conf, max_pcs)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )
    rf.fit(X[known_mask], y[known_mask])
    pred = rf.predict(X)
    proba = rf.predict_proba(X)
    conf = np.max(proba, axis=1)
    y_out = pred.copy()
    y_out[conf < min_conf] = "unlabeled"
    adata.obs[out_col] = pd.Categorical(y_out)
    adata.obs[out_conf_col] = conf

def mlp_labeling(adata, X_key="X_pca", seed_col="weak_label", conf_col="weak_conf",
                 out_col="label_mlp", out_conf_col="mlp_conf",
                 hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300,
                 min_seed_conf=0.05, min_conf=0.3, max_pcs=15):
    X, y_str, known_mask = get_training_data(adata, X_key, seed_col, conf_col, min_seed_conf, max_pcs)
    le = LabelEncoder()
    y_known = le.fit_transform(y_str[known_mask])

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
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


def notebook_consensus(adata, method_outputs):
    """Cell 6 consensus block, verbatim, wrapped as a function."""
    if len(method_outputs) == 0:
        raise ValueError("No classifier output was generated. Check weak_label distribution.")

    methods = np.vstack([
        adata.obs[c].astype(str).values
        for c in method_outputs
    ])

    consensus = np.array(["unlabeled"] * adata.n_obs, dtype=object)
    agree_frac = np.zeros(adata.n_obs)

    for i in range(adata.n_obs):
        votes = [m[i] for m in methods]
        filtered = [v for v in votes if v != "unlabeled"]
        if len(filtered) == 0:
            consensus[i] = "unlabeled"
            agree_frac[i] = 0.0
        else:
            c = Counter(filtered)
            maj_label, maj_votes = c.most_common(1)[0]
            consensus[i] = maj_label
            agree_frac[i] = maj_votes / len(method_outputs)
    return consensus, agree_frac

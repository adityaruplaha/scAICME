"""EXAMPLE: PBMC 68k annotation reproducing the final PBMC notebook (Sept 2026).

Pipeline, as executed in that notebook's saved run:

1. Load the 10x PBMC 68k filtered matrix and attach the reference annotation
   (one column, attached by row order, renamed to ``pseudo_cell_type``).
2. Standard QC and log-normalization (``icme.pp``).
3. Method 2 — DP-GMM seeding per marker set, fitted on standardised marker expression
   with no PCA (``DPGMMSeeding``) -> ``weak_label``.
4. PCA (30 components) and a 15-neighbour graph, then SVM, k-means, KNN, random forest
   and MLP propagation, and a plurality consensus over the five.
5. Leiden clustering, agreement metrics against the reference annotation
   (``classwise_metrics.csv``) and cluster-quality scores
   (``pbmc_scACIME_cluster_metrics.csv``).

Methods 1 and 3 of that notebook (quota seeding, and quota GCN seeding) are defined
there but were never executed in the saved run and are not part of this pipeline; the
package equivalents are ``QCQAdaptiveSeeding(target_frac=...)`` and ``GCNSeeding``.

Input layout (``SCAICME_PBMC68K_DIR``, default ``data/pbmc68k``)::

    <dir>/filtered_matrices_mex/hg19/{matrix.mtx,genes.tsv,barcodes.tsv}
    <dir>/pbmc_annot.csv
    <dir>/cell_type_true.csv    # optional; the manual ground truth, preferred reference
    <dir>/label_scanvi.csv      # optional, attached when present
"""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import scAICME as icme

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("SCAICME_PBMC68K_DIR", "data/pbmc68k"))
MTX_DIR = DATA_DIR / "filtered_matrices_mex" / "hg19"
ANNOT_CSV = DATA_DIR / "pbmc_annot.csv"
SCANVI_CSV = DATA_DIR / "label_scanvi.csv"
TRUTH_CSV = DATA_DIR / "cell_type_true.csv"
OUTPUT_DIR = Path("examples/pbmc68k/outputs")
SAVE_H5AD = True
# Silhouette is O(n^2) in the number of cells; set False to skip the cluster-quality table.
CLUSTER_QUALITY = True

UNLABELED = "unlabeled"
SEED_KEY = "weak_label"
CONSENSUS_KEY = "label_consensus"
# The notebook evaluates against "cell_type_true", the manually curated ground-truth
# annotation, which is supplied to it from outside rather than built in the notebook.
# It is used when available (see TRUTH_CSV); otherwise the loader's own annotation
# column stands in and the metrics are against that instead.
TRUTH_KEY = "cell_type_true"
FALLBACK_REF_KEY = "pseudo_cell_type"
RANDOM_STATE = 42

# Marker panels (the notebook's SKIN_MARKERS, which hold PBMC panels).
PBMC_MARKERS = {
    "CD8+/CD45RA+ Naive Cytotoxic": [
        "CD3D", "CD3E", "TRAC", "CD8A", "CD8B", "CCR7", "LEF1", "TCF7", "LTB", "IL7R", "MAL", "LST1",
    ],
    "CD4+/CD25 T Reg": [
        "CD3D", "CD3E", "TRAC", "CD4", "IL2RA", "FOXP3", "IKZF2", "CTLA4", "TIGIT", "TNFRSF18",
        "CCR7", "LTB",
    ],
    "CD8+ Cytotoxic T": [
        "CD3D", "CD3E", "TRAC", "CD8A", "CD8B", "NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "CTSW",
        "KLRD1", "CCL5",
    ],
    "CD56+ NK": [
        "NKG7", "GNLY", "PRF1", "GZMB", "GZMH", "CTSW", "KLRD1", "FCGR3A", "TRDC", "XCL1", "XCL2",
    ],
    "CD19+ B": [
        "MS4A1", "CD79A", "CD79B", "CD74", "HLA-DRA", "HLA-DRB1", "CD37", "CD19", "BANK1", "CD22",
        "CD83",
    ],
    "CD14+ Monocyte": [
        "LYZ", "S100A8", "S100A9", "CTSS", "FCN1", "LGALS3", "LST1", "TYROBP", "FCER1G", "CTSD",
        "MNDA", "IL1B",
    ],
    "CD4+/CD45RO+ Memory": [
        "CD3D", "CD3E", "TRAC", "CD4", "IL7R", "LTB", "CCR7", "MAL", "NOSIP", "TCF7", "LEF1", "CXCR4",
    ],
    "CD4+/CD45RA+/CD25- Naive T": [
        "CD3D", "CD3E", "TRAC", "CD4", "CCR7", "LEF1", "TCF7", "IL7R", "LTB", "MAL", "NOSIP", "LST1",
    ],
    "Dendritic": [
        "FCER1A", "CD1C", "CLEC10A", "ITGAX", "LILRA4", "GZMB", "HLA-DRA", "HLA-DRB1", "IRF7",
    ],
    "CD34+": ["CD34", "SPINK2", "GATA2", "MPO", "HBB", "TYMP", "MEIS1", "AVP"],
    "CD4+ T Helper2": [
        "CD3D", "CD3E", "TRAC", "CD4", "IL7R", "GATA3", "IL4", "IL5", "IL13", "CCR4", "CCR6", "ICOS",
    ],
}  # fmt: skip

# Method 2 seeding, from the notebook's call. n_components is min(100, sqrt(N)) and the
# post-hoc size floor is max(100, 0.002 N).
SEEDING = {
    "per_gene_pos_quantile": 0.4,
    "cluster_score_min": 0.1,
    "min_cells_cluster": 100,
    "weight_concentration_prior": 0.05,
    "min_cell_enrichment": 0.05,
    "min_type_size": 100,
    "min_type_frac": 0.002,
    "random_state": RANDOM_STATE,
}

# Feature space and propagation.
N_COMPS = 30
N_NEIGHBORS = 15
MAX_PCS = 30
MIN_SEED_CONF = 0.30
LEIDEN_RESOLUTION = 1.6

METHOD_KEYS = ["label_svm_rbf", "label_kmeans", "label_knn", "label_rf", "label_mlp"]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    adata = load_pbmc68k()
    adata = preprocess(adata)
    attach_ground_truth(adata)
    run_seeding(adata)
    prepare_features(adata)
    run_propagation(adata)
    run_consensus(adata)
    run_leiden(adata)
    attach_scanvi(adata)
    evaluate(adata)
    export(adata)


def load_pbmc68k() -> ad.AnnData:
    """Read the 10x matrix (gene symbols) and attach the reference annotation by row order."""
    if not MTX_DIR.exists():
        raise FileNotFoundError(
            f"PBMC 68k matrix directory not found: {MTX_DIR}. Set SCAICME_PBMC68K_DIR or "
            "extract the 10x filtered matrices into data/pbmc68k/."
        )
    # cache=True writes an h5ad next to the matrix; newer anndata needs the opt-in below.
    with ad.settings.override(allow_write_nullable_strings=True):
        adata = sc.read_10x_mtx(MTX_DIR, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if ANNOT_CSV.exists():
        df = pd.read_csv(ANNOT_CSV)
        if df.shape[1] == 1 and (
            not df.columns[0] or str(df.columns[0]).lower().startswith("unnamed")
        ):
            df.columns = ["org_annot"]
        if len(df) != adata.n_obs:
            raise ValueError(f"{ANNOT_CSV} has {len(df)} rows; expected {adata.n_obs} cells.")
        for col in df.columns:
            safe_col = str(col).strip() or "org_annot"
            if safe_col in adata.obs.columns:
                safe_col = f"{safe_col}_csv"
            adata.obs[safe_col] = pd.Categorical(df[col].astype(str).values)
        print("Attached columns to adata.obs:", list(df.columns))
        first = str(df.columns[0]).strip() or "org_annot"
        adata.obs = adata.obs.rename(columns={first: FALLBACK_REF_KEY})
    else:
        print(f"[warn] {ANNOT_CSV} not found; reference metrics will be skipped.")
    return adata


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    adata = icme.pp.qc_filter(adata)
    icme.pp.normalize_log1p(adata)
    print(f"READY | cells={adata.n_obs:,} genes={adata.n_vars:,}")
    if FALLBACK_REF_KEY in adata.obs:
        adata.obs[FALLBACK_REF_KEY] = adata.obs[FALLBACK_REF_KEY].astype(str).astype("category")
        print(adata.obs[FALLBACK_REF_KEY].value_counts())
    return adata


def attach_ground_truth(adata: ad.AnnData) -> None:
    """Attach the manual ground-truth annotation when it has been supplied."""
    if not TRUTH_CSV.exists():
        print(f"\n[info] {TRUTH_CSV} not found; metrics will use {FALLBACK_REF_KEY}.")
        return
    truth = pd.read_csv(TRUTH_CSV)
    column = TRUTH_KEY if TRUTH_KEY in truth.columns else truth.columns[0]
    if len(truth) != adata.n_obs:
        raise ValueError(
            f"{TRUTH_CSV} has {len(truth)} rows but {adata.n_obs} cells remain after QC. "
            "Attach it by row order over all cells before QC, or supply it with a "
            "barcode column so it can be joined."
        )
    adata.obs[TRUTH_KEY] = pd.Categorical(truth[column].astype(str).values)
    print(f"\nAttached {TRUTH_KEY} from {TRUTH_CSV}")


def reference_key(adata: ad.AnnData) -> str | None:
    """The ground truth when present, else the loader's annotation, else nothing."""
    for key in (TRUTH_KEY, FALLBACK_REF_KEY):
        if key in adata.obs:
            return key
    return None


def run_seeding(adata: ad.AnnData) -> None:
    """Method 2: per-marker-set DP-GMM on standardised marker expression, no PCA."""
    seeder = icme.strategies.DPGMMSeeding(
        markers=PBMC_MARKERS,
        n_components=min(100, int(np.sqrt(adata.n_obs))),
        use_raw=False,
        unknown_label=UNLABELED,
        verbose=True,
        **SEEDING,
    )
    icme.tl.label(adata, seeder, key_added=SEED_KEY)
    uns = adata.uns[f"{SEED_KEY}_uns"]
    print(f"\n[DP-soft no PCA] size floor={uns['size_floor']}, dropped={uns['dropped_types']}")
    print(f"[DP-soft no PCA] Final {SEED_KEY} distribution:")
    print(adata.obs[SEED_KEY].value_counts(normalize=True) * 100)


def prepare_features(adata: ad.AnnData) -> None:
    sc.pp.pca(adata, n_comps=N_COMPS, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_COMPS)


def run_propagation(adata: ad.AnnData) -> None:
    """The notebook's five classifiers, each predicting every cell from the seeds."""
    common = {
        "seed_key": SEED_KEY,
        "unknown_label": UNLABELED,
        "keep_seeds": False,
        "max_pcs": MAX_PCS,
    }
    methods = {
        "label_svm_rbf": icme.strategies.SVMPropagation(
            kernel="rbf",
            c=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            scale_features=True,
            min_seed_conf=MIN_SEED_CONF,
            min_conf=0.60,
            random_state=RANDOM_STATE,
            **common,
        ),
        # k = min(30, max(10, sqrt(N/2))). The per-cluster vote uses every labelled seed
        # while the seedless-cluster fallback uses only seeds above 0.3, as in the notebook.
        "label_kmeans": icme.strategies.KMeansPropagation(
            n_clusters=min(30, max(10, int(np.sqrt(adata.n_obs / 2)))),
            n_init=20,
            max_iter=500,
            scale_features=False,
            min_seed_conf=0.0,
            fallback_min_seed_conf=0.3,
            min_conf=0.0,
            random_state=RANDOM_STATE,
            **common,
        ),
        "label_knn": icme.strategies.KNNPropagation(
            n_neighbors=9,
            weights="distance",
            p=2,
            min_seed_conf=MIN_SEED_CONF,
            min_conf=0.55,
            **common,
        ),
        "label_rf": icme.strategies.RandomForestPropagation(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_seed_conf=MIN_SEED_CONF,
            min_conf=0.55,
            random_state=RANDOM_STATE,
            **common,
        ),
        "label_mlp": icme.strategies.NeuralNetworkPropagation(
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            scale_features=True,
            min_seed_conf=MIN_SEED_CONF,
            min_conf=0.60,
            random_state=RANDOM_STATE,
            **common,
        ),
    }
    results = icme.tl.label(adata, strategies=methods, n_jobs=1)
    missing = [k for k in methods if k not in results]
    if missing:
        raise ValueError(f"Propagation did not produce: {missing}")
    print("\nClassifier outputs:", list(methods))


def run_consensus(adata: ad.AnnData) -> None:
    """Plurality vote; agreement is votes divided by the number of classifiers."""
    consensus = icme.strategies.ConsensusVoting(
        keys=METHOD_KEYS, majority_fraction=None, fraction_of="all", unknown_label=UNLABELED
    )
    icme.tl.label(adata, consensus, key_added=CONSENSUS_KEY)
    print("\nConsensus:")
    print(adata.obs[CONSENSUS_KEY].value_counts())


def run_leiden(adata: ad.AnnData) -> None:
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added="label_leiden")
    n = adata.obs["label_leiden"].nunique()
    print(f"\nLeiden clusters (resolution {LEIDEN_RESOLUTION}): {n}")


def attach_scanvi(adata: ad.AnnData) -> None:
    """Attach saved scANVI predictions when the file is present (the notebook reads them)."""
    if not SCANVI_CSV.exists():
        print(f"\n[info] {SCANVI_CSV} not found; skipping the scANVI column.")
        return
    pred = pd.read_csv(SCANVI_CSV)
    if len(pred) != adata.n_obs:
        print(
            f"[warn] {SCANVI_CSV} has {len(pred)} rows but {adata.n_obs} cells remain after QC; "
            "skipping the scANVI column."
        )
        return
    labels = pred["label_scanvi"].astype(str)
    # The notebook relabels this class after prediction.
    adata.obs["label_scanvi"] = labels.replace("CD4+ T Helper2", "CD34+").values
    print(f"\nAttached label_scanvi from {SCANVI_CSV}")


def evaluate(adata: ad.AnnData) -> None:
    """Agreement with the reference annotation, and cluster quality of the consensus."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_key = reference_key(adata)
    if ref_key is not None:
        pred_keys = [CONSENSUS_KEY, "label_knn", "label_rf", "label_kmeans", "label_svm_rbf",
                     "label_mlp"]  # fmt: skip
        metrics = icme.evaluation.compare_many(
            adata, pred_keys, ref_key, ignore_labels=(UNLABELED, "Unknown")
        )
        cols = ["pred_col", "coverage_pred", "coverage_both", "labeled_ARI", "labeled_acc",
                "labeled_specificity_macro", "labeled_sensitivity_macro", "labeled_n_eval"]  # fmt: skip
        print(f"\nAgreement with the reference annotation ({ref_key}):")
        print(metrics[cols].sort_values("labeled_ARI", ascending=False).to_string(index=False))
        metrics.to_csv(OUTPUT_DIR / "classwise_metrics.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'classwise_metrics.csv'}")

    if CLUSTER_QUALITY:
        print("\nCluster quality (silhouette is O(n^2); this takes a while):")
        table = pd.DataFrame([icme.evaluation.cluster_quality(adata, CONSENSUS_KEY)])
        print(table.to_string(index=False))
        table.to_csv(OUTPUT_DIR / "pbmc_scACIME_cluster_metrics.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'pbmc_scACIME_cluster_metrics.csv'}")


def export(adata: ad.AnnData) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        c
        for c in [
            TRUTH_KEY,
            FALLBACK_REF_KEY,
            SEED_KEY,
            f"{SEED_KEY}_max_score",
            *METHOD_KEYS,
            CONSENSUS_KEY,
            f"{CONSENSUS_KEY}_agreement_fraction",
            "label_leiden",
            "label_scanvi",
        ]
        if c in adata.obs
    ]
    labels_path = OUTPUT_DIR / "pbmc68k_labels.csv"
    adata.obs[cols].to_csv(labels_path, index=True)
    print(f"\nLabels written to {labels_path}")
    if SAVE_H5AD:
        h5ad_path = OUTPUT_DIR / "pbmc68k_annotated.h5ad"
        with ad.settings.override(allow_write_nullable_strings=True):
            adata.write_h5ad(h5ad_path)
        print(f"Annotated AnnData written to {h5ad_path}")


if __name__ == "__main__":
    main()

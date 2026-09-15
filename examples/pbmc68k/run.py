"""EXAMPLE: PBMC 68k annotation reproducing the PBMC 68k notebook.

Pipeline, as executed in the notebook:

1. Load the 10x PBMC 68k filtered matrix and attach the reference annotation
   (``pbmc_annot.csv``, one ``pbmcannot`` column, attached by row order).
2. Standard QC and log-normalization (``icme.pp``).
3. Method 1 — per-gene quantile marker scores with quota selection
   (``QCQAdaptiveSeeding``); its score matrix feeds Method 3.
4. Method 3 — GCN propagation of those scores over the kNN graph with fixed gates
   (``GCNSeeding``). Its output is the ``weak_label`` used downstream.
   (Method 2, DP-GMM seeding, is defined in the notebook but was not executed and
   its output is overwritten by Method 3; run it with ``RUN_DPGMM = True``.)
5. SVM / K-Means / KNN / Random Forest / MLP on 15 PCs, then plurality consensus.
6. Rare/novel flag and agreement metrics against the reference annotation
   (``ablation_metrics.csv``).

Input layout (``SCAICME_PBMC68K_DIR``, default ``data/pbmc68k``)::

    <dir>/filtered_matrices_mex/hg19/{matrix.mtx,genes.tsv,barcodes.tsv}
    <dir>/pbmc_annot.csv
"""

import math
import os
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

import scAICME as icme

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("SCAICME_PBMC68K_DIR", "data/pbmc68k"))
MTX_DIR = DATA_DIR / "filtered_matrices_mex" / "hg19"
ANNOT_CSV = DATA_DIR / "pbmc_annot.csv"
OUTPUT_DIR = Path("examples/pbmc68k/outputs")
SAVE_H5AD = True
RUN_DPGMM = False

UNLABELED = "unlabeled"
REF_KEY = "pseudo_cell_type"
SCORE_SEED_KEY = "weak_label_quota"  # Method 1 (its scores feed Method 3)
SEED_KEY = "weak_label"  # Method 3, used for propagation
CONSENSUS_KEY = "label_consensus"
RANDOM_STATE = 42

# Notebook cell 5 (named SKIN_MARKERS there; these are PBMC panels).
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

# Method 1 (cell 7).
QUOTA = {"quantile": 0.6, "target_frac": 0.55, "min_cells_per_type": 50, "min_score": 0.2}
# Method 3 (cell 14).
GCN = {
    "target_frac": 0.55,
    "min_score": 0.2,
    "delta": 0.15,
    "min_cells_per_type_pct": 0.005,
    "min_cells_floor": 200,
}
# Method 2 (cell 11); not executed in the notebook.
DPGMM = {
    "per_gene_pos_quantile": 0.8,
    "cluster_score_min": 0.20,
    "min_cells_cluster": 100,
    "n_components": 30,
    "weight_concentration_prior": 0.05,
    "min_cell_enrichment": 0.10,
    "random_state": RANDOM_STATE,
}

# Feature space (cell 15): 15 PCs, kNN graph on them.
N_COMPS = 15
N_NEIGHBORS = 15
# Reference counts printed by the notebook for the Method 3 seeds (cell 14).
NOTEBOOK_SEEDS = {"CD19+ B": 3940, "CD14+ Monocyte": 3281, "CD8+ Cytotoxic T": 660, "CD4+/CD45RO+ Memory": 354}  # fmt: skip


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    adata = load_pbmc68k()
    adata = preprocess(adata)
    run_seeding(adata)
    prepare_features(adata)
    method_keys = run_propagation(adata)
    run_consensus(adata, method_keys)
    evaluate(adata, method_keys)
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
        adata.obs = adata.obs.rename(columns={first: REF_KEY})
    else:
        print(f"[warn] {ANNOT_CSV} not found; reference metrics will be skipped.")
    return adata


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    adata = icme.pp.qc_filter(adata)
    icme.pp.normalize_log1p(adata)
    print(f"READY | cells={adata.n_obs:,} genes={adata.n_vars:,}")
    if REF_KEY in adata.obs:
        adata.obs[REF_KEY] = adata.obs[REF_KEY].astype(str).astype("category")
        print(adata.obs[REF_KEY].value_counts())
    return adata


def run_seeding(adata: ad.AnnData) -> None:
    """Method 1 (quota) -> its score matrix -> Method 3 (GCN) -> weak_label."""
    quota = icme.strategies.QCQAdaptiveSeeding(
        markers=PBMC_MARKERS,
        min_confidence=QUOTA["min_score"],
        unknown_label=UNLABELED,
        **QUOTA,
    )
    icme.tl.label(adata, quota, key_added=SCORE_SEED_KEY)
    labeled = (adata.obs[SCORE_SEED_KEY] != UNLABELED).sum()
    print(
        f"[quota] Target={QUOTA['target_frac']:.0%} | Achieved={labeled / adata.n_obs:.2%} "
        f"| labeled={labeled}/{adata.n_obs}"
    )
    print(adata.obs[SCORE_SEED_KEY].value_counts())

    if RUN_DPGMM:
        n = adata.n_obs
        dpgmm = icme.strategies.DPGMMSeeding(
            markers=PBMC_MARKERS,
            unknown_label=UNLABELED,
            min_type_size=math.ceil(0.02 * n),  # the notebook's 2% keep rule
            min_type_frac=0.0,
            verbose=True,
            **DPGMM,
        )
        icme.tl.label(adata, dpgmm, key_added="weak_label_dpgmm")

    # Method 3 needs the kNN graph; the notebook builds it inside gcn_seed_labeling
    # (PCA 30 comps, 15 neighbors) before cell 15 recomputes PCA with 15 comps.
    sc.pp.pca(adata, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    gcn = icme.strategies.GCNSeeding(
        markers=PBMC_MARKERS,
        initial_scores_key=f"{SCORE_SEED_KEY}_scores",
        unknown_label=UNLABELED,
        **GCN,
    )
    icme.tl.label(adata, gcn, key_added=SEED_KEY)
    counts = adata.obs[SEED_KEY].value_counts()
    print("[GCN] final label distribution:")
    print(counts)
    print(f"[GCN] labeled fraction: {(adata.obs[SEED_KEY] != UNLABELED).mean():.2%}")
    comparison = (
        pd.DataFrame(
            {
                "this_run": counts.drop(UNLABELED, errors="ignore"),
                "notebook": pd.Series(NOTEBOOK_SEEDS),
            }
        )
        .fillna(0)
        .astype(int)
    )
    comparison["diff"] = comparison["this_run"] - comparison["notebook"]
    print("Parity check of Method 3 seeds against the notebook output:")
    print(comparison.to_string())


def prepare_features(adata: ad.AnnData) -> None:
    """Cell 15: PCA(15, arpack) and a 15-neighbor graph on those PCs (overwrites Method 3's)."""
    sc.pp.pca(adata, n_comps=N_COMPS, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
    seeds = adata.obs[SEED_KEY].astype(str)
    known = seeds != UNLABELED
    print("Known types (seeds):", sorted(seeds[known].unique()), "| n_seeds =", int(known.sum()))


def run_propagation(adata: ad.AnnData) -> list[str]:
    n_types = adata.obs[SEED_KEY].astype(str).pipe(lambda s: s[s != UNLABELED].nunique())
    common = {"seed_key": SEED_KEY, "unknown_label": UNLABELED}
    methods = {
        # Cell 16: seeds kept, unknown cells predicted, min_conf on predictions only.
        "label_svm_rbf": icme.strategies.SVMPropagation(
            kernel="rbf",
            c=5.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            scale_features=True,
            keep_seeds=True,
            min_conf=0.55,
            random_state=RANDOM_STATE,  # the notebook sets none; fixed here for reproducibility
            **common,
        ),
        # Cell 18: k = number of seeded types, default init, clusters mapped by seed majority.
        "label_kmeans": icme.strategies.KMeansPropagation(
            n_clusters=n_types,
            n_init="auto",
            max_iter=300,
            scale_features=False,
            keep_seeds=False,
            min_conf=0.0,
            random_state=RANDOM_STATE,
            **common,
        ),
        # Cell 20: 3 nearest seeds, distance-weighted, Manhattan metric, all cells predicted.
        "label_knn": icme.strategies.KNNPropagation(
            n_neighbors=3, weights="distance", p=1, keep_seeds=False, min_conf=0.0, **common
        ),
        # Cell 22.
        "label_rf": icme.strategies.RandomForestPropagation(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            keep_seeds=False,
            min_conf=0.0,
            random_state=RANDOM_STATE,
            **common,
        ),
        # Cells 24-25: seeds kept, unknown cells predicted, min_conf on predictions only.
        "label_mlp": icme.strategies.NeuralNetworkPropagation(
            hidden_layer_sizes=(256, 128),
            alpha=1e-4,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            scale_features=True,
            keep_seeds=True,
            min_conf=0.55,
            random_state=RANDOM_STATE,
            **common,
        ),
    }
    results = icme.tl.label(adata, strategies=methods, n_jobs=1)
    completed = [k for k in methods if k in results]
    print("\nAvailable classifier outputs:", completed)
    if not completed:
        raise ValueError("No classifier output was generated. Check the seed distribution.")
    return completed


def run_consensus(adata: ad.AnnData, method_keys: list[str]) -> None:
    consensus = icme.strategies.ConsensusVoting(
        keys=method_keys, majority_fraction=None, fraction_of="all", unknown_label=UNLABELED
    )
    icme.tl.label(adata, consensus, key_added=CONSENSUS_KEY)
    print("\nConsensus:")
    print(adata.obs[CONSENSUS_KEY].value_counts())

    # Cell 30: rare/novel flag.
    rare = icme.evaluation.flag_rare(
        adata, CONSENSUS_KEY, f"{CONSENSUS_KEY}_agreement_fraction", max_agreement=0.4
    )
    adata.obs["novel_flag"] = rare
    print(f"\nRare/novel (flagged) cells: {int(rare.sum())} ({rare.mean():.2%})")


def evaluate(adata: ad.AnnData, method_keys: list[str]) -> None:
    """Cells 32-34: agreement with the reference annotation."""
    if REF_KEY not in adata.obs:
        return
    pred_keys = [
        CONSENSUS_KEY,
        "label_knn",
        "label_rf",
        "label_kmeans",
        "label_svm_rbf",
        "label_mlp",
    ]
    metrics = icme.evaluation.compare_many(
        adata, [k for k in pred_keys if k in method_keys or k == CONSENSUS_KEY], REF_KEY,
        ignore_labels=(UNLABELED, "Unknown"),
    )  # fmt: skip
    cols = [
        "pred_col", "coverage_pred", "coverage_both",
        "labeled_ARI", "labeled_NMI", "labeled_macroF1", "labeled_acc", "labeled_n_eval",
    ]  # fmt: skip
    print("\nAgreement with the reference annotation:")
    print(metrics[cols].sort_values("labeled_ARI", ascending=False).to_string())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'ablation_metrics.csv'}")


def export(adata: ad.AnnData) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [c for c in [REF_KEY, SEED_KEY, CONSENSUS_KEY, f"{CONSENSUS_KEY}_agreement_fraction", "novel_flag"] if c in adata.obs]  # fmt: skip
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

"""EXAMPLE: Annotating GSE225475 psoriasis skin Visium sections (six pooled samples).

Package-level reproduction of the GSE225475 spatial notebook:
load six Space Ranger outputs, pool them, QC + normalize, seed spots with a marker-set
DP-GMM, propagate the seeds with five classifiers on PCA space, and take a plurality
consensus. The final labels are written to ``scAICME_spatial_labels.csv``.

Input layout (``SCAICME_GSE225475_DIR``, default ``data/gse225475``)::

    <dir>/NS1/filtered_feature_bc_matrix.h5
    <dir>/NS1/spatial/tissue_positions_list.csv
    ...

Each ``GSM70491xx_<sample>.tar.gz`` from GEO extracts to exactly this layout.
"""

import os
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

import scAICME as icme

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("SCAICME_GSE225475_DIR", "data/gse225475"))
OUTPUT_DIR = Path("examples/gse225475/outputs")
SAVE_H5AD = True

# Sample name -> condition (NS: non-lesional, PP: psoriasis plaque).
SAMPLES = {
    "NS1": "NS",
    "NS2": "NS",
    "PP1": "PP",
    "PP2": "PP",
    "PP3": "PP",
    "PP4": "PP",
}

UNLABELED = "unlabeled"
SEED_KEY = "weak_label"
CONSENSUS_KEY = "label_consensus"
RANDOM_STATE = 42

# Marker sets for DP-GMM seeding (notebook: SPATIAL_SKIN_MARKERS).
SPATIAL_SKIN_MARKERS = {
    "Keratinocyte": [
        "KRT14",
        "KRT1",
        "KRT10",
        "KRT6A",
        "KRT16",
        "DMKN",
        "S100A7",
        "S100A8",
        "S100A9",
    ],
    "Fibroblast": ["DCN", "COL1A1", "COL1A2", "COL3A1", "LUM", "SFRP2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "CLDN5"],
    "Myeloid": ["LST1", "TYROBP", "FCER1G", "HLA-DRA", "CD74", "IL1B", "CLEC10A"],
    "T_cell": ["CD3D", "CD3E", "TRAC", "IL7R", "LTB"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "KIT"],
    "Smooth_muscle": ["ACTA2", "TAGLN", "MYL9", "RGS5"],
    "Eccrine_gland": ["DCD", "PIP", "MUCL1", "KRT19"],
}

# Seeding settings used in the notebook run (cell 5).
SEEDING = {
    "n_components": 15,
    "weight_concentration_prior": 0.1,
    "per_gene_pos_quantile": 0.3,
    "cluster_score_min": 0.08,
    "min_cells_cluster": 30,
    "random_state": RANDOM_STATE,
}

# Feature space for propagation (cell 6).
N_COMPS = 20
N_PCS_USE = 15
N_NEIGHBORS = 20

# Shared propagation settings (cell 6, get_training_data + per-method min_conf).
MIN_SEED_CONF = 0.05
MIN_CONF = 0.3

# Consensus label counts reported in the notebook output, for a quick parity check.
NOTEBOOK_CONSENSUS = {
    "Keratinocyte": 3534,
    "Fibroblast": 2622,
    "Smooth_muscle": 561,
    "Eccrine_gland": 295,
    "Endothelial": 65,
    "T_cell": 49,
    "Myeloid": 45,
    "Mast": 13,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    adata = load_samples()
    adata = preprocess(adata)
    run_seeding(adata)
    prepare_features(adata)
    method_keys = run_propagation(adata)
    run_consensus(adata, method_keys)
    export(adata)


def load_samples() -> ad.AnnData:
    """Read every Space Ranger sample and pool them (outer join on genes)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"GSE225475 directory not found: {DATA_DIR}. Set SCAICME_GSE225475_DIR or "
            "extract the GEO sample archives into data/gse225475/."
        )

    adatas = []
    for sample, condition in SAMPLES.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read_visium deprecation, missing images
            sample_adata = sc.read_visium(
                DATA_DIR / sample, count_file="filtered_feature_bc_matrix.h5"
            )
        sample_adata.var_names_make_unique()
        sample_adata.obs["sample"] = sample
        sample_adata.obs["condition"] = condition
        sample_adata.obs["barcode"] = sample_adata.obs_names
        sample_adata.obs_names = [f"{sample}_{bc}" for bc in sample_adata.obs_names]
        adatas.append(sample_adata)
        print(sample, sample_adata.shape)

    adata = ad.concat(
        adatas, join="outer", label="batch", keys=list(SAMPLES.keys()), index_unique=None
    )
    print(adata)
    return adata


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    """Keep in-tissue spots, then standard QC, normalization, and log1p."""
    if "in_tissue" in adata.obs.columns:
        adata = adata[adata.obs["in_tissue"].astype(int) == 1].copy()

    adata = icme.pp.qc_filter(adata)
    icme.pp.normalize_log1p(adata)
    print(f"READY | cells={adata.n_obs:,} genes={adata.n_vars:,}")
    return adata


def run_seeding(adata: ad.AnnData) -> None:
    """Marker-set DP-GMM seeding -> obs['weak_label'], obs['weak_label_max_score']."""
    seeder = icme.strategies.DPGMMSeeding(
        markers=SPATIAL_SKIN_MARKERS, unknown_label=UNLABELED, verbose=True, **SEEDING
    )
    icme.tl.label(adata, seeder, key_added=SEED_KEY)

    print(f"\n[{seeder.name}] Final {SEED_KEY} distribution:")
    print(adata.obs[SEED_KEY].value_counts(normalize=True) * 100)


def prepare_features(adata: ad.AnnData) -> None:
    """PCA on the full log-normalized matrix plus a kNN graph (as in the notebook)."""
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=N_COMPS, svd_solver="arpack")
    if "connectivities" not in adata.obsp:
        n_pcs = min(N_PCS_USE, adata.obsm["X_pca"].shape[1])
        sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs)


def run_propagation(adata: ad.AnnData) -> list[str]:
    """Train five classifiers on the seeds; return the keys that completed."""
    common = {
        "seed_key": SEED_KEY,
        "unknown_label": UNLABELED,
        "keep_seeds": False,
        "min_seed_conf": MIN_SEED_CONF,
    }
    methods = {
        "label_svm_rbf": icme.strategies.SVMPropagation(
            kernel="rbf",
            c=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            scale_features=True,
            min_conf=MIN_CONF,
            max_pcs=N_PCS_USE,
            random_state=RANDOM_STATE,
            **common,
        ),
        "label_kmeans": icme.strategies.KMeansPropagation(
            n_clusters=None,  # min(10, max(8, sqrt(N/2)))
            n_init=20,
            max_iter=500,
            scale_features=False,
            min_conf=0.0,
            max_pcs=30,
            random_state=RANDOM_STATE,
            **common,
        ),
        "label_knn": icme.strategies.KNNPropagation(
            n_neighbors=9,
            weights="distance",
            min_conf=MIN_CONF,
            max_pcs=N_PCS_USE,
            **common,
        ),
        "label_rf": icme.strategies.RandomForestPropagation(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_conf=MIN_CONF,
            max_pcs=N_PCS_USE,
            random_state=RANDOM_STATE,
            **common,
        ),
        "label_mlp": icme.strategies.NeuralNetworkPropagation(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            scale_features=True,
            min_conf=MIN_CONF,
            max_pcs=N_PCS_USE,
            random_state=RANDOM_STATE,
            **common,
        ),
    }

    # Sequential execution keeps the run deterministic and mirrors the notebook loop.
    results = icme.tl.label(adata, strategies=methods, n_jobs=1)
    completed = [k for k in methods if k in results]
    print("\nAvailable classifier outputs:", completed)
    if not completed:
        raise ValueError("No classifier output was generated. Check the seed distribution.")
    return completed


def run_consensus(adata: ad.AnnData, method_keys: list[str]) -> None:
    """Plurality vote across classifiers; agreement is votes / number of classifiers."""
    consensus = icme.strategies.ConsensusVoting(
        keys=method_keys, majority_fraction=None, fraction_of="all", unknown_label=UNLABELED
    )
    icme.tl.label(adata, consensus, key_added=CONSENSUS_KEY)

    counts = adata.obs[CONSENSUS_KEY].value_counts(dropna=False)
    print("\nConsensus created successfully.")
    print(counts)

    comparison = (
        pd.DataFrame({"this_run": counts, "notebook": pd.Series(NOTEBOOK_CONSENSUS)})
        .fillna(0)
        .astype(int)
    )
    comparison["diff"] = comparison["this_run"] - comparison["notebook"]
    print("\nParity check against the notebook output:")
    print(comparison.to_string())


def export(adata: ad.AnnData) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = adata.obs[["sample", "barcode", CONSENSUS_KEY]].copy()
    labels_path = OUTPUT_DIR / "scAICME_spatial_labels.csv"
    labels.to_csv(labels_path, index=True)
    print(f"\nLabels written to {labels_path}")
    print(labels.head())

    if SAVE_H5AD:
        h5ad_path = OUTPUT_DIR / "gse225475_annotated.h5ad"
        # Pooled Visium var names arrive as nullable strings; opt in to writing them.
        with ad.settings.override(allow_write_nullable_strings=True):
            adata.write_h5ad(h5ad_path)
        print(f"Annotated AnnData written to {h5ad_path}")


if __name__ == "__main__":
    main()

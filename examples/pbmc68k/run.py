"""EXAMPLE: Annotating the PBMC 68k dataset using scAICME in exact parity with notebook."""

import os
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scAICME as icme

# Minimum genes detected per cell and minimum cells per gene
MIN_GENES_TO_RETAIN = 200
MIN_CELLS_PER_GENE = 3

# High-end cutoffs are set by percentiles to trim extreme outliers
QC_PERCENTILES = {
    "umi_hi": 99.5,
    "genes_hi": 99.5,
    "mito_hi": 95.0,
}
# Mitochondrial fraction ceiling; acts as a floor on the dynamic cutoff
MITO_FLOOR = 20.0

BASELINE_PLOT_KEYS = [
    "leiden_res0.2",
    "leiden_res0.4",
    "leiden_res0.6",
    "leiden_res0.8",
    "leiden_res1.0",
]
SAVE_OUTPUTS = True
OUTPUT_DIR = Path("examples/pbmc68k/outputs")
FIGURE_FORMAT = "png"

# Unified color palette across cell types and labels
UNIFIED_PALETTE = {
    "CD8+/CD45RA+ Naive Cytotoxic": "#1f77b4",
    "CD4+/CD25 T Reg": "#ff7f0e",
    "CD8+ Cytotoxic T": "#2ca02c",
    "CD56+ NK": "#d62728",
    "CD19+ B": "#9467bd",
    "CD14+ Monocyte": "#8c564b",
    "CD4+/CD45RO+ Memory": "#e377c2",
    "CD4+/CD45RA+/CD25- Naive T": "#7f7f7f",
    "Dendritic": "#bcbd22",
    "CD34+": "#17becf",
    "CD4+ T Helper2": "#aec7e8",
    "unknown": "#cccccc",
    "True": "#2ca02c",
    "False": "#d62728",
}

# Cell type marker genes for seeding PBMC 68k
PBMC_MARKERS = {
    "CD8+/CD45RA+ Naive Cytotoxic": [
        "CD3D",
        "CD3E",
        "TRAC",
        "CD8A",
        "CD8B",
        "CCR7",
        "LEF1",
        "TCF7",
        "LTB",
        "IL7R",
        "MAL",
        "LST1",
    ],
    "CD4+/CD25 T Reg": [
        "CD3D",
        "CD3E",
        "TRAC",
        "CD4",
        "IL2RA",
        "FOXP3",
        "IKZF2",
        "CTLA4",
        "TIGIT",
        "TNFRSF18",
        "CCR7",
        "LTB",
    ],
    "CD8+ Cytotoxic T": [
        "CD3D",
        "CD3E",
        "CD8A",
        "CD8B",
        "NKG7",
        "GNLY",
        "GZMB",
        "GZMH",
        "PRF1",
        "CTSW",
        "KLRD1",
        "CCL5",
    ],
    "CD56+ NK": [
        "NKG7",
        "GNLY",
        "PRF1",
        "GZMB",
        "GZMH",
        "CTSW",
        "KLRD1",
        "FCGR3A",
        "XCL1",
        "XCL2",
    ],
    "CD19+ B": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "CD74",
        "HLA-DRA",
        "HLA-DRB1",
        "CD37",
        "CD19",
        "BANK1",
        "CD22",
        "CD83",
    ],
    "CD14+ Monocyte": [
        "LYZ",
        "S100A8",
        "S100A9",
        "CTSS",
        "FCN1",
        "LGALS3",
        "LST1",
        "TYROBP",
        "FCER1G",
        "CTSD",
        "MNDA",
        "IL1B",
    ],
    "CD4+/CD45RO+ Memory": [
        "CD3D",
        "CD3E",
        "TRAC",
        "CD4",
        "IL7R",
        "LTB",
        "CCR7",
        "MAL",
        "NOSIP",
        "TCF7",
        "LEF1",
        "CXCR4",
    ],
    "CD4+/CD45RA+/CD25- Naive T": [
        "CD3D",
        "CD3E",
        "TRAC",
        "CD4",
        "CCR7",
        "LEF1",
        "TCF7",
        "IL7R",
        "LTB",
        "MAL",
        "NOSIP",
        "LST1",
    ],
    "Dendritic": [
        "FCER1A",
        "CD1C",
        "CLEC10A",
        "ITGAX",
        "LILRA4",
        "GZMB",
        "HLA-DRA",
        "HLA-DRB1",
        "IRF7",
    ],
    "CD34+": [
        "CD34",
        "SPINK2",
        "GATA2",
        "MPO",
        "HBB",
        "TYMP",
        "MEIS1",
        "AVP",
    ],
    "CD4+ T Helper2": [
        "CD3D",
        "CD3E",
        "CD4",
        "IL7R",
        "GATA3",
        "IL4",
        "CCR4",
        "CCR6",
        "ICOS",
    ],
}


def main() -> None:
    """Run the PBMC 68k annotation and evaluation pipeline."""
    adata = preprocess_pbmc68k()
    run_icme_pipelines(adata)


def add_qc_gene_sets(adata: sc.AnnData) -> None:
    """Annotate mitochondrial, ribosomal, and hemoglobin gene flags."""
    vn_up = adata.var_names.str.upper()
    adata.var["mt"] = vn_up.str.startswith("MT-")
    adata.var["ribo"] = vn_up.str.startswith(("RPS", "RPL", "MRPS", "MRPL"))
    adata.var["hb"] = adata.var_names.str.match(r"^(HB[ABEDM][A-Z0-9]*)", case=False)


def compute_qc_metrics(adata: sc.AnnData) -> None:
    """Compute standard QC metrics using the annotated gene sets."""
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )


def derive_thresholds(adata: sc.AnnData) -> tuple[float, float, float]:
    """Return data-driven thresholds for UMI, gene counts, and mito fraction."""
    umi_hi = np.percentile(adata.obs["total_counts"], QC_PERCENTILES["umi_hi"])
    genes_hi = np.percentile(adata.obs["n_genes_by_counts"], QC_PERCENTILES["genes_hi"])
    mito_hi = max(
        MITO_FLOOR,
        np.percentile(adata.obs["pct_counts_mt"], QC_PERCENTILES["mito_hi"]),
    )
    return umi_hi, genes_hi, mito_hi


def filter_cells(adata: sc.AnnData, umi_hi: float, genes_hi: float, mito_hi: float) -> sc.AnnData:
    """Filter cells by gene counts, UMI counts, and mitochondrial fraction."""
    keep_cells = (
        (adata.obs["n_genes_by_counts"] >= MIN_GENES_TO_RETAIN)
        & (adata.obs["n_genes_by_counts"] <= genes_hi)
        & (adata.obs["total_counts"] <= umi_hi)
        & (adata.obs["pct_counts_mt"] <= mito_hi)
    )
    return adata[keep_cells].copy()


def preprocess_pbmc68k() -> sc.AnnData:
    """Load PBMC 68k and run QC filtering plus normalization."""
    candidates = [
        Path("/bulk/PBMC Dataset/pbmc68k_10x/filtered_matrices_mex/hg19"),
        Path("pbmc68k_10x/filtered_matrices_mex/hg19"),
        Path.home() / "pbmc68k_10x/filtered_matrices_mex/hg19",
    ]
    mtx_dir = next((p for p in candidates if p.exists()), None)
    if not mtx_dir:
        raise FileNotFoundError(
            f"PBMC 68k matrix directory not found in candidate paths: {candidates}"
        )

    print(f"Loading PBMC 68k from {mtx_dir}...")
    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Attach annotations if available
    annot_candidates = [
        mtx_dir / "pbmc_annot.csv",
        Path("pbmc68k_10x/filtered_matrices_mex/hg19/pbmc_annot.csv"),
    ]
    csv_path = next((p for p in annot_candidates if p.exists()), None)
    if csv_path:
        df = pd.read_csv(csv_path)
        if df.shape[1] == 1:
            if not df.columns[0] or str(df.columns[0]).lower().startswith("unnamed"):
                df.columns = ["org_annot"]
        for col in df.columns:
            safe_col = str(col).strip() or "org_annot"
            if safe_col in adata.obs.columns:
                safe_col = f"{safe_col}_csv"
            adata.obs[safe_col] = pd.Categorical(df[col].astype(str).values)
        print(f"Attached annotations from {csv_path}: {list(df.columns)}")

        # Rename the annotation column to pseudo_cell_type in parity with notebook
        if "pbmcannot" in adata.obs.columns:
            adata.obs = adata.obs.rename(columns={"pbmcannot": "pseudo_cell_type"})
        elif "pbmcannot_csv" in adata.obs.columns:
            adata.obs = adata.obs.rename(columns={"pbmcannot_csv": "pseudo_cell_type"})

    print(f"START  | cells={adata.n_obs:,}  genes={adata.n_vars:,}")
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    add_qc_gene_sets(adata)
    compute_qc_metrics(adata)

    print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())

    umi_hi, genes_hi, mito_hi = derive_thresholds(adata)
    print(
        f"Thresholds -> min_genes={MIN_GENES_TO_RETAIN}, umi_hi~{umi_hi:.0f}, "
        f"genes_hi~{genes_hi:.0f}, mito_hi~{mito_hi:.1f}%"
    )

    before = adata.n_obs
    adata = filter_cells(adata, umi_hi, genes_hi, mito_hi)
    print(f"CELL FILTER | kept {adata.n_obs:,}/{before:,} ({adata.n_obs / before:.1%})")

    before_g = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"GENE FILTER | kept {adata.n_vars:,}/{before_g:,} (>= {MIN_CELLS_PER_GENE} cells)")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    print(f"READY | cells={adata.n_obs:,} genes={adata.n_vars:,}")

    return adata


def prepare_features(adata: sc.AnnData, n_pcs: int = 50) -> sc.AnnData:
    """Compute highly variable genes, PCA, and neighborhood graph for downstream SSA strategies."""
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=50, n_pcs=n_pcs)
    return adata


def run_icme_pipelines(adata: sc.AnnData) -> sc.AnnData:
    """Run seeding, smoothing, propagation, and consensus voting on PBMC 68k."""
    adata = prepare_features(adata, n_pcs=50)

    # For PBMC 68k (~68k cells), 0.1% min_cells_per_type is ~68 cells
    min_cells_per_type = max(10, int(round(0.001 * adata.n_obs)))
    print(f"\nUsing min_cells_per_type = {min_cells_per_type} for PBMC 68k seeding strategies.")

    # ========== Seed Generation ==========
    base_seed_strategies = {
        "qcq_adaptive": icme.strategies.QCQAdaptiveSeeding(
            markers=PBMC_MARKERS,
            target_frac=0.6,
            min_cells_per_type=min_cells_per_type,
            min_score=0.2,
        ),
        "otsu_scored_adaptive": icme.strategies.OtsuScoredAdaptiveSeeding(
            markers=PBMC_MARKERS,
            target_frac=0.6,
            min_cells_per_type=min_cells_per_type,
            min_score=0.2,
        ),
        "otsu_adaptive": icme.strategies.OtsuAdaptiveSeeding(
            markers=PBMC_MARKERS,
            target_frac=0.6,
            min_cells_per_type=min_cells_per_type,
        ),
    }

    smoothened_seed_strategies = {}
    for base_name, _base_strategy in base_seed_strategies.items():
        smoothened_seed_strategies[f"{base_name}_dpgmm"] = icme.strategies.DPGMMClusteredSmoothing(
            markers=PBMC_MARKERS,
            initial_scores_key=f"{base_name}_scores",
            min_confidence=0.6,
            min_cells_cluster=30,
            weight_concentration_prior=0.01,
            per_gene_pos_quantile=0.70,
            min_cluster_size_post_hoc=min_cells_per_type,
        )
        smoothened_seed_strategies[f"{base_name}_gcn"] = icme.strategies.GCNSmoothing(
            markers=PBMC_MARKERS,
            initial_scores_key=f"{base_name}_scores",
            temperature=0.1,
            alpha=0.5,
            tol=1e-4,
            min_cells_floor=30,
        )

    print("\nRunning base seeding strategies...")
    seed_results = icme.tl.label(adata, strategies=base_seed_strategies, n_jobs=4)
    print("Running smoothened seeding strategies...")
    seed_results.update(icme.tl.label(adata, strategies=smoothened_seed_strategies, n_jobs=4))

    compute_labeling_counts_matrix(adata, seed_results)

    # ========== Propagation (Exact Parity with Notebook) ==========
    seed_names = list(seed_results.keys())
    propagation_factories = [
        lambda seed_key: icme.strategies.KNNPropagation(
            seed_key=seed_key,
            n_neighbors=9,
            weights="distance",
            min_seed_conf=0.30,
            min_conf=0.55,
            max_pcs=30,
        ),
        lambda seed_key: icme.strategies.RandomForestPropagation(
            seed_key=seed_key,
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=5,
            min_seed_conf=0.30,
            min_conf=0.55,
            max_pcs=30,
            random_state=42,
        ),
        lambda seed_key: icme.strategies.SVMPropagation(
            seed_key=seed_key,
            c=2.0,
            gamma="scale",
            min_seed_conf=0.30,
            min_conf=0.60,
            max_pcs=30,
            scale_features=True,
        ),
        lambda seed_key: icme.strategies.NeuralNetworkPropagation(
            seed_key=seed_key,
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            min_seed_conf=0.30,
            min_conf=0.60,
            max_pcs=30,
            random_state=42,
        ),
        lambda seed_key: icme.strategies.KMeansPropagation(
            seed_key=seed_key,
            n_clusters=30,
            scale_features=False,
            min_seed_conf=0.30,
            max_pcs=30,
            random_state=42,
        ),
    ]
    all_propagation_strategies = {}
    seed_abbr_to_names = {}
    seed_prop_keys = {}

    for seed_name in seed_names:
        seed_abbr = seed_name.removeprefix("seeds_")
        seed_abbr_to_names[seed_abbr] = seed_name
        seed_prop_keys[seed_abbr] = []
        for factory in propagation_factories:
            strategy = factory(seed_name)
            key = f"prop_{strategy.name}_{seed_abbr}"
            all_propagation_strategies[key] = strategy
            seed_prop_keys[seed_abbr].append(key)

    print(f"\nExecuting {len(all_propagation_strategies)} propagation strategies across seeds...")
    max_jobs = os.cpu_count() or 1
    icme.tl.label(
        adata,
        strategies=all_propagation_strategies,
        n_jobs=min(len(all_propagation_strategies), max_jobs),
    )

    existing_propagation_keys = {
        key for key in all_propagation_strategies if key in adata.obs.columns
    }
    missing_propagation_keys = [
        key for key in all_propagation_strategies if key not in existing_propagation_keys
    ]
    if missing_propagation_keys:
        warnings.warn(
            "Skipping missing propagation outputs before consensus: "
            + ", ".join(sorted(missing_propagation_keys)),
            stacklevel=2,
        )

    # ========== Per-Seed Consensus ==========
    consensus_tasks = {}
    for seed_abbr in seed_abbr_to_names.keys():
        prop_keys = [key for key in seed_prop_keys[seed_abbr] if key in existing_propagation_keys]
        if not prop_keys:
            warnings.warn(
                f"Skipping consensus for seed '{seed_abbr}' because no propagation labels were created.",
                stacklevel=2,
            )
            continue
        consensus_key = f"consensus_{seed_abbr}"
        consensus_tasks[consensus_key] = icme.strategies.ConsensusVoting(
            keys=prop_keys, majority_fraction=0.1
        )

    seed_consensus_keys = list(consensus_tasks.keys())
    if consensus_tasks:
        print("\nComputing per-seed consensus labels...")
        icme.tl.label(adata, strategies=consensus_tasks, n_jobs=4)
    else:
        warnings.warn(
            "No per-seed consensus labels were created, so final consensus will be skipped.",
            stacklevel=2,
        )

    # ========== Final Consensus ==========
    if seed_consensus_keys:
        print("Computing final consensus across all seeds...")
        final_consensus = icme.strategies.ConsensusVoting(
            keys=seed_consensus_keys, majority_fraction=0.1
        )
        icme.tl.label(adata, strategies=final_consensus, key_added="labels_final")
    else:
        warnings.warn(
            "Skipping final consensus because no per-seed consensus labels exist.", stacklevel=2
        )

    # ========== Plot Keys and Visualization ==========
    seed_plot_keys = list(seed_results.keys())
    dpgmm_plot_keys = [k for k in seed_plot_keys if k.endswith("_dpgmm")]
    gcn_plot_keys = [k for k in seed_plot_keys if k.endswith("_gcn")]
    prop_plot_keys = list(consensus_tasks.keys())
    if "labels_final" in adata.obs:
        prop_plot_keys.append("labels_final")

    print("\nRunning clustering baselines...")
    run_baselines(adata)
    ablation_cols = [
        col
        for col in adata.obs.columns
        if col == "labels_final" or col.startswith("leiden_") or col in seed_consensus_keys
    ]
    compute_ablation_metrics(adata, sorted(ablation_cols))
    plot_icme_umaps(
        adata,
        seed_plot_keys=seed_plot_keys,
        dpgmm_plot_keys=dpgmm_plot_keys,
        gcn_plot_keys=gcn_plot_keys,
        prop_plot_keys=prop_plot_keys,
        baseline_plot_keys=BASELINE_PLOT_KEYS,
        save_plots=SAVE_OUTPUTS,
        output_dir=OUTPUT_DIR,
    )
    return adata


def compute_labeling_counts_matrix(adata: sc.AnnData, seed_results: dict) -> None:
    """Print seed labeling statistics."""
    print("\n" + "=" * 70)
    print("SEED LABELING STATISTICS")
    print("=" * 70)

    n_cells = adata.n_obs
    print("\nSeed Counts by Strategy:")
    seed_data = []
    for seed_col in seed_results.keys():
        seed_count = int((adata.obs[seed_col] != "unknown").sum())
        pct = 100.0 * seed_count / n_cells
        seed_data.append(
            {"strategy": seed_col, "seed_count": seed_count, "seed_pct": f"{pct:.1f}%"}
        )

    seed_df = pd.DataFrame(seed_data).set_index("strategy").sort_index()
    print(seed_df.to_string())

    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(seed_data).to_csv(OUTPUT_DIR / "seed_counts.csv", index=False)
        print(f"\nStatistics saved to {OUTPUT_DIR}/")
    print("=" * 70)


def run_baselines(adata: sc.AnnData) -> None:
    """Compute canonical clustering baselines for comparison."""
    for res in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sc.tl.leiden(
            adata, key_added=f"leiden_res{res}", resolution=res, flavor="igraph", n_iterations=2
        )
    sc.tl.tsne(adata, n_pcs=50)


def compute_ablation_metrics(adata: sc.AnnData, label_columns: list[str]) -> None:
    """Compute pairwise ARI and NMI metrics between all label predictions."""
    if "labels_final" not in adata.obs:
        print("No final consensus labels found; skipping ablation metrics.")
        return

    print("\n" + "=" * 70)
    print("ABLATION METRICS: Pairwise Agreement")
    print("=" * 70)

    all_label_cols = [col for col in label_columns if col in adata.obs.columns]
    if not all_label_cols:
        print("No label columns found for comparison.")
        return

    n_methods = len(all_label_cols)
    ari_matrix = np.zeros((n_methods, n_methods))
    nmi_matrix = np.zeros((n_methods, n_methods))

    for i, col1 in enumerate(all_label_cols):
        for j, col2 in enumerate(all_label_cols):
            labels1 = adata.obs[col1]
            labels2 = adata.obs[col2]
            valid = (labels1 != "unknown") & (labels2 != "unknown")

            if valid.sum() == 0:
                ari_matrix[i, j] = np.nan
                nmi_matrix[i, j] = np.nan
            else:
                labels1_valid = labels1[valid]
                labels2_valid = labels2[valid]
                ari_matrix[i, j] = adjusted_rand_score(labels1_valid, labels2_valid)
                nmi_matrix[i, j] = normalized_mutual_info_score(labels1_valid, labels2_valid)

    ari_df = pd.DataFrame(ari_matrix, index=all_label_cols, columns=all_label_cols)
    nmi_df = pd.DataFrame(nmi_matrix, index=all_label_cols, columns=all_label_cols)

    print("\n--- Adjusted Rand Index (ARI) Matrix ---")
    print(ari_df.to_string(float_format=lambda x: f"{x:.3f}" if not np.isnan(x) else "nan"))
    print("\n--- Normalized Mutual Information (NMI) Matrix ---")
    print(nmi_df.to_string(float_format=lambda x: f"{x:.3f}" if not np.isnan(x) else "nan"))

    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ari_df.to_csv(OUTPUT_DIR / "ari_matrix.csv")
        nmi_df.to_csv(OUTPUT_DIR / "nmi_matrix.csv")
        print(f"\nARI matrix saved to: {OUTPUT_DIR / 'ari_matrix.csv'}")
        print(f"NMI matrix saved to: {OUTPUT_DIR / 'nmi_matrix.csv'}")
    print("=" * 70)


def _build_legend_handles(color_keys: list[str], adata: sc.AnnData, palette: dict | None) -> dict:
    """Build legend patches from data categories and colors."""
    handles_dict = {}
    for color_key in color_keys:
        if color_key not in adata.obs:
            continue
        col_data = adata.obs[color_key]
        if col_data.dtype in ("float64", "float32", "int64", "int32"):
            continue
        if hasattr(col_data, "cat"):
            categories = col_data.cat.categories
        else:
            categories = sorted(col_data.unique())

        for cat in categories:
            cat_str = str(cat)
            if cat_str not in handles_dict:
                if palette is not None:
                    color = palette.get(cat_str, "#cccccc")
                else:
                    cmap = mpl.colormaps.get_cmap("tab20")
                    idx = list(categories).index(cat)
                    color = cmap(idx % 20)
                handles_dict[cat_str] = mpl.patches.Patch(facecolor=color)
    return handles_dict


def plot_icme_umaps(
    adata: sc.AnnData,
    seed_plot_keys: list[str],
    dpgmm_plot_keys: list[str],
    gcn_plot_keys: list[str],
    prop_plot_keys: list[str],
    baseline_plot_keys: list[str],
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot UMAPs and t-SNEs for seeds, propagation outputs, and final consensus labels."""
    if save_plots:
        plot_dir = output_dir or OUTPUT_DIR
        plot_dir.mkdir(parents=True, exist_ok=True)

    adata_plot = adata[:, :]
    adata_plot.obs = adata.obs.copy()
    for col in adata_plot.obs.columns:
        if adata_plot.obs[col].dtype == bool:
            adata_plot.obs[col] = adata_plot.obs[col].astype(str)

    sc.tl.umap(adata_plot)

    def plot_and_save(color_keys, title_suffix, palette=UNIFIED_PALETTE, consolidate_legend=True):
        fig = sc.pl.umap(
            adata_plot,
            color=color_keys,
            ncols=2,
            palette=palette,
            legend_loc=None if consolidate_legend else "right",
            return_fig=True,
        )
        if consolidate_legend:
            for ax in fig.axes:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            handles_dict = _build_legend_handles(color_keys, adata_plot, palette)
            if handles_dict:
                fig.legend(
                    handles_dict.values(),
                    handles_dict.keys(),
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )
        if save_plots:
            fig.savefig(
                OUTPUT_DIR / f"umap{title_suffix}.{FIGURE_FORMAT}",
                bbox_inches="tight",
                dpi=100,
            )

    plot_and_save(seed_plot_keys, "_seeds", consolidate_legend=True)
    plot_and_save(dpgmm_plot_keys, "_dpgmm", consolidate_legend=False)
    plot_and_save(gcn_plot_keys, "_gcn", consolidate_legend=False)
    plot_and_save(prop_plot_keys, "_prop", consolidate_legend=True)
    plot_and_save(baseline_plot_keys, "_baselines", palette=None, consolidate_legend=False)

    fig = sc.pl.tsne(
        adata_plot,
        color=baseline_plot_keys,
        ncols=2,
        palette=None,
        legend_loc="right",
        return_fig=True,
    )
    if save_plots:
        fig.savefig(
            OUTPUT_DIR / f"tsne_baselines.{FIGURE_FORMAT}",
            bbox_inches="tight",
            dpi=100,
        )


if __name__ == "__main__":
    main()

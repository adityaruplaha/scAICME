"""EXAMPLE: Annotating the PBMC3k dataset"""

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scAICME as icme

# Minimum genes detected per cell to keep obvious low-quality droplets.
MIN_GENES_TO_RETAIN = 200
# Minimum number of cells a gene must appear in to retain it.
MIN_CELLS_PER_GENE = 3
# High-end cutoffs are set by percentiles to trim extreme outliers.
QC_PERCENTILES = {
    "umi_hi": 99.5,
    "genes_hi": 99.5,
    "mito_hi": 95.0,
}
# Mitochondrial fraction ceiling; acts as a floor on the dynamic cutoff.
MITO_FLOOR = 20.0  # = 20% mitochondrial reads, a common conservative threshold for human PBMCs.

BASELINE_PLOT_KEYS = [
    "leiden_res0.2",
    "leiden_res0.4",
    "leiden_res0.6",
    "leiden_res0.8",
    "leiden_res1.0",
]
SAVE_OUTPUTS = True
OUTPUT_DIR = Path("examples/pbmc3k/outputs")
FIGURE_FORMAT = "png"

# Unified color palette for all cell types and labels.
# "unknown" is assigned a neutral gray (appears hollow when edge-colored).
UNIFIED_PALETTE = {
    # PBMC cell types
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
    # Unknown/unlabeled
    "unknown": "#cccccc",
    # Boolean fields (from is_confident, etc.)
    "True": "#2ca02c",
    "False": "#d62728",
}

# Cell type marker genes for seeding
PBMC_MARKERS = {
    # -------------------------------------------------
    # CD8+/CD45RA+ Naive Cytotoxic (naive CD8 T)
    # -------------------------------------------------
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
        "LST1",  # (optional; remove LST1 if you see myeloid leakage)
    ],
    # -------------------------------------------------
    # CD4+/CD25 T Reg (Treg)
    # -------------------------------------------------
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
    # -------------------------------------------------
    # CD8+ Cytotoxic T
    # -------------------------------------------------
    "CD8+ Cytotoxic T": [
        "CD3D",
        "CD3E",
        "TRAC",
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
    # -------------------------------------------------
    # CD56+ NK
    # -------------------------------------------------
    "CD56+ NK": [
        "NKG7",
        "GNLY",
        "PRF1",
        "GZMB",
        "GZMH",
        "CTSW",
        "KLRD1",
        "FCGR3A",
        "TRDC",  # optional (rare)
        "XCL1",
        "XCL2",
    ],
    # -------------------------------------------------
    # CD19+ B
    # -------------------------------------------------
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
        "CD83",  # activation sometimes
    ],
    # -------------------------------------------------
    # CD14+ Monocyte (classical monocytes)
    # -------------------------------------------------
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
    # -------------------------------------------------
    # CD4+/CD45RO+ Memory
    # -------------------------------------------------
    "CD4+/CD45RO+ Memory": [
        "CD3D",
        "CD3E",
        "TRAC",
        "CD4",
        "IL7R",
        "LTB",
        "CCR7",  # central memory
        "MAL",
        "NOSIP",
        "TCF7",
        "LEF1",  # may be lower than naive
        "CXCR4",
    ],
    # -------------------------------------------------
    # CD4+/CD45RA+/CD25- Naive T (naive CD4)
    # -------------------------------------------------
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
        "LST1",  # optional; remove if contaminating monocytes
    ],
    # -------------------------------------------------
    # Dendritic (cDC / pDC mixed depending on dataset)
    # -------------------------------------------------
    "Dendritic": [
        "FCER1A",
        "CD1C",
        "CLEC10A",  # cDC2
        "ITGAX",
        "LILRA4",  # ITGAX general DC; LILRA4 pDC
        "GZMB",  # pDC hallmark (often)
        "HLA-DRA",
        "HLA-DRB1",
        "IRF7",
    ],
    # -------------------------------------------------
    # CD34+ (HSPC / progenitors)
    # -------------------------------------------------
    "CD34+": [
        "CD34",
        "SPINK2",
        "GATA2",
        "MPO",  # may indicate myeloid progenitors
        "HBB",  # remove if RBC contamination
        "TYMP",
        "MEIS1",
        "AVP",  # optional depending on platform
    ],
    # -------------------------------------------------
    # CD4+ T Helper2 (rare; dataset has 19 cells)
    # -------------------------------------------------
    "CD4+ T Helper2": [
        "CD3D",
        "CD3E",
        "TRAC",
        "CD4",
        "IL7R",
        "GATA3",
        "IL4",
        "IL5",
        "IL13",
        "CCR4",
        "CCR6",
        "ICOS",
    ],
}


def main() -> None:
    """Run the PBMC3k preprocessing example."""
    adata = preprocess_pbmc3k()
    run_icme_pipelines(adata)


def add_qc_gene_sets(adata: sc.AnnData) -> None:
    """Annotate mitochondrial, ribosomal, and hemoglobin gene flags."""
    vn_up = adata.var_names.str.upper()
    # Human and mouse 'mt-' both become 'MT-' after uppercasing.
    adata.var["mt"] = vn_up.str.startswith("MT-")
    adata.var["ribo"] = vn_up.str.startswith(("RPS", "RPL", "MRPS", "MRPL"))
    # Hemoglobin genes mark ambient RNA or erythrocyte contamination.
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
    # Keep the bulk while removing extreme high-count and high-gene outliers.
    umi_hi = np.percentile(adata.obs["total_counts"], QC_PERCENTILES["umi_hi"])
    genes_hi = np.percentile(adata.obs["n_genes_by_counts"], QC_PERCENTILES["genes_hi"])
    # Use a percentile-based mito cutoff, but never below a conservative floor.
    mito_hi = max(
        MITO_FLOOR,
        np.percentile(adata.obs["pct_counts_mt"], QC_PERCENTILES["mito_hi"]),
    )
    return umi_hi, genes_hi, mito_hi


def filter_cells(adata: sc.AnnData, umi_hi: float, genes_hi: float, mito_hi: float) -> sc.AnnData:
    """Filter cells by gene counts, UMI counts, and mitochondrial fraction."""
    # Combine gene/UMI/mito filters to remove low-quality and outlier cells.
    keep_cells = (
        (adata.obs["n_genes_by_counts"] >= MIN_GENES_TO_RETAIN)
        & (adata.obs["n_genes_by_counts"] <= genes_hi)
        & (adata.obs["total_counts"] <= umi_hi)
        & (adata.obs["pct_counts_mt"] <= mito_hi)
    )
    return adata[keep_cells].copy()


def preprocess_pbmc3k() -> sc.AnnData:
    """Load PBMC3k and run QC filtering plus normalization."""
    adata = sc.datasets.pbmc3k()

    print(f"START  | cells={adata.n_obs:,}  genes={adata.n_vars:,}")
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()  # avoids duplicate symbol problems in downstream tools
    adata.obs_names_make_unique()

    add_qc_gene_sets(adata)
    compute_qc_metrics(adata)

    print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())

    umi_hi, genes_hi, mito_hi = derive_thresholds(adata)
    print(
        "Thresholds -> min_genes="
        f"{MIN_GENES_TO_RETAIN}, umi_hi~{umi_hi:.0f}, genes_hi~{genes_hi:.0f}, mito_hi~{mito_hi:.1f}%"
    )

    before = adata.n_obs
    adata = filter_cells(adata, umi_hi, genes_hi, mito_hi)
    print(f"CELL FILTER | kept {adata.n_obs:,}/{before:,} ({adata.n_obs / before:.1%})")

    before_g = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"GENE FILTER | kept {adata.n_vars:,}/{before_g:,} (>= {MIN_CELLS_PER_GENE} cells)")

    # Library size normalization, then log-transform for downstream analysis.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    print(f"READY | cells={adata.n_obs:,} genes={adata.n_vars:,}")

    return adata


def prepare_features(adata: sc.AnnData, n_pcs: int = 50) -> sc.AnnData:
    """Compute PCA and neighbors for downstream SSA strategies."""
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=n_pcs)
    return adata


def run_icme_pipelines(adata: sc.AnnData) -> sc.AnnData:
    """Run seeding, propagation from each seed, per-seed consensus, and final consensus."""
    adata = prepare_features(adata)

    # ========== Seed Generation ==========
    seed_strategy_instances = [
        icme.strategies.QCQAdaptiveSeeding(markers=PBMC_MARKERS),
        icme.strategies.OtsuAdaptiveSeeding(markers=PBMC_MARKERS),
        icme.strategies.GraphScoreSeeding(markers=PBMC_MARKERS),
        icme.strategies.DPMMClusteredAdaptiveSeeding(
            markers=PBMC_MARKERS,
            min_confidence=0.6,
            min_cells_cluster=3,
            weight_concentration_prior=0.01,
        ),
    ]
    seed_strategies = {f"seeds_{s.name}": s for s in seed_strategy_instances}
    seed_results = icme.tl.label(adata, strategies=seed_strategies, n_jobs=4)

    # Print seed counts before propagation
    compute_labeling_counts_matrix(adata, seed_results)

    # ========== Propagation ==========
    seed_names = list(seed_strategies.keys())
    propagation_factories = [
        lambda seed_key: icme.strategies.KNNPropagation(seed_key=seed_key),
        lambda seed_key: icme.strategies.RandomForestPropagation(seed_key=seed_key, random_state=0),
        lambda seed_key: icme.strategies.NearestCentroidPropagation(seed_key=seed_key),
        lambda seed_key: icme.strategies.SVMPropagation(seed_key=seed_key),
    ]
    all_propagation_strategies = {}
    seed_abbr_to_names = {}
    seed_prop_keys = {}

    for seed_name in seed_names:
        seed_abbr = seed_name.split("_")[1]
        seed_abbr_to_names[seed_abbr] = seed_name

        seed_prop_keys[seed_abbr] = []
        for factory in propagation_factories:
            strategy = factory(seed_name)
            key = f"prop_{strategy.name}_{seed_abbr}"
            all_propagation_strategies[key] = strategy
            seed_prop_keys[seed_abbr].append(key)

    # Execute all propagations in parallel
    max_jobs = os.cpu_count() or 1
    icme.tl.label(
        adata,
        strategies=all_propagation_strategies,
        n_jobs=min(len(all_propagation_strategies), max_jobs),
    )

    # ========== Per-Seed Consensus ==========
    consensus_tasks = {}
    for seed_abbr in seed_abbr_to_names.keys():
        prop_keys = seed_prop_keys[seed_abbr]
        consensus_key = f"consensus_{seed_abbr}"
        consensus_tasks[consensus_key] = icme.strategies.ConsensusVoting(
            keys=prop_keys, majority_fraction=0.66
        )

    # Execute all per-seed consensus votes in parallel
    seed_consensus_keys = list(consensus_tasks.keys())
    icme.tl.label(adata, strategies=consensus_tasks, n_jobs=4)

    # ========== Final Consensus ==========
    final_consensus = icme.strategies.ConsensusVoting(
        keys=seed_consensus_keys, majority_fraction=0.66
    )
    icme.tl.label(adata, strategies=final_consensus, key_added="labels_final")

    # ========== Plot Keys and Visualization ==========
    seed_plot_keys = list(seed_strategies.keys())
    dpmm_key = next((k for k in seed_plot_keys if "dpmm" in k), None)
    dpmm_plot_keys = (
        [dpmm_key, f"{dpmm_key}_max_confidence", f"{dpmm_key}_is_confident"] if dpmm_key else []
    )
    prop_plot_keys = list(consensus_tasks.keys()) + ["labels_final"]

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
        dpmm_plot_keys=dpmm_plot_keys,
        prop_plot_keys=prop_plot_keys,
        baseline_plot_keys=BASELINE_PLOT_KEYS,
        save_plots=SAVE_OUTPUTS,
        output_dir=OUTPUT_DIR,
    )
    return adata


def compute_labeling_counts_matrix(adata: sc.AnnData, seed_results: dict) -> None:
    """Print seed labeling statistics.

    Args:
        adata: AnnData object with labeled cells in obs
        seed_results: Dict returned from icme.tl.label() with column keys as keys
    """
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

    seed_df = pd.DataFrame(seed_data).set_index("strategy")
    print(seed_df.to_string())

    # ========== Save ==========
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(seed_data).to_csv(OUTPUT_DIR / "seed_counts.csv", index=False)
        print(f"\nStatistics saved to {OUTPUT_DIR}/")

    print("=" * 70)


def run_baselines(adata: sc.AnnData) -> None:
    """Compute canonical clustering baselines for comparison."""
    # Leiden with multiple resolutions for sensitivity analysis.
    for res in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sc.tl.leiden(
            adata, key_added=f"leiden_res{res}", resolution=res, flavor="igraph", n_iterations=2
        )

    sc.tl.tsne(adata, n_pcs=50)


def compute_ablation_metrics(adata: sc.AnnData, label_columns: list[str]) -> None:
    """Compute pairwise ARI and NMI metrics between all label predictions.

    Creates symmetric matrices showing agreement between:
    - Per-seed consensus (consensus_*)
    - Final consensus (labels_final)
    - Leiden baselines (leiden_res*)
    """
    if "labels_final" not in adata.obs:
        print("No final consensus labels found; skipping ablation metrics.")
        return

    print("\n" + "=" * 70)
    print("ABLATION METRICS: Pairwise Agreement")
    print("=" * 70)

    # ========== Collect Label Columns ==========
    # Group labels: per-seed consensus, final, and baselines
    all_label_cols = [col for col in label_columns if col in adata.obs.columns]

    if not all_label_cols:
        print("No label columns found for comparison.")
        return

    n_methods = len(all_label_cols)
    ari_matrix = np.zeros((n_methods, n_methods))
    nmi_matrix = np.zeros((n_methods, n_methods))

    # ========== Compute Pairwise Metrics ==========
    # Only compute valid pairs (cells labeled in both methods)
    for i, col1 in enumerate(all_label_cols):
        for j, col2 in enumerate(all_label_cols):
            labels1 = adata.obs[col1]
            labels2 = adata.obs[col2]

            # Find cells labeled in both methods
            valid = (labels1 != "unknown") & (labels2 != "unknown")

            if valid.sum() == 0:
                ari_matrix[i, j] = np.nan
                nmi_matrix[i, j] = np.nan
            else:
                labels1_valid = labels1[valid]
                labels2_valid = labels2[valid]

                ari_matrix[i, j] = adjusted_rand_score(labels1_valid, labels2_valid)
                nmi_matrix[i, j] = normalized_mutual_info_score(labels1_valid, labels2_valid)

    # Create DataFrames with method names as indices
    ari_df = pd.DataFrame(ari_matrix, index=all_label_cols, columns=all_label_cols)
    nmi_df = pd.DataFrame(nmi_matrix, index=all_label_cols, columns=all_label_cols)

    # Print results
    print("\n--- Adjusted Rand Index (ARI) Matrix ---")
    print("(Higher = more similar labeling)")
    print(ari_df.to_string(float_format=lambda x: f"{x:.3f}" if not np.isnan(x) else "nan"))

    print("\n--- Normalized Mutual Information (NMI) Matrix ---")
    print("(Higher = more similar labeling)")
    print(nmi_df.to_string(float_format=lambda x: f"{x:.3f}" if not np.isnan(x) else "nan"))

    # ========== Save Metrics ==========
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        ari_df.to_csv(OUTPUT_DIR / "ari_matrix.csv")
        nmi_df.to_csv(OUTPUT_DIR / "nmi_matrix.csv")

        print(f"\nARI matrix saved to: {OUTPUT_DIR / 'ari_matrix.csv'}")
        print(f"NMI matrix saved to: {OUTPUT_DIR / 'nmi_matrix.csv'}")

    print("=" * 70)


def _build_legend_handles(color_keys: list[str], adata: sc.AnnData, palette: dict | None) -> dict:
    """Build legend patches from data categories and colors.

    Args:
        color_keys: List of obs column names to extract categories from.
        adata: AnnData object containing the observations.
        palette: Optional dict mapping category names to colors. If None, uses tab20 colormap.

    Returns:
        Dict mapping category strings to matplotlib Patch handles.

    Note:
        Skips numerical columns to preserve their continuous visualization as colormaps.
    """
    handles_dict = {}
    for color_key in color_keys:
        if color_key not in adata.obs:
            continue

        col_data = adata.obs[color_key]

        # Skip numerical columns; let Scanpy handle their continuous colormaps
        if col_data.dtype in ("float64", "float32", "int64", "int32"):
            continue

        # Only process categorical or string columns
        if hasattr(col_data, "cat"):
            categories = col_data.cat.categories
        else:
            # For string/object columns, only create legend entries
            categories = sorted(col_data.unique())

        for cat in categories:
            cat_str = str(cat)
            if cat_str not in handles_dict:
                if palette is not None:
                    color = palette.get(cat_str, "#cccccc")
                else:
                    # Use categorical cmap if no palette specified
                    cmap = mpl.colormaps.get_cmap("tab20")
                    idx = list(categories).index(cat)
                    color = cmap(idx % 20)

                handles_dict[cat_str] = mpl.patches.Patch(facecolor=color)

    return handles_dict


def plot_icme_umaps(
    adata: sc.AnnData,
    seed_plot_keys: list[str],
    dpmm_plot_keys: list[str],
    prop_plot_keys: list[str],
    baseline_plot_keys: list[str],
    save_plots: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Plot UMAPs for seeds, propagation outputs, and final consensus labels.

    Uses a unified palette across all plots for consistent coloring.
    Removes individual subplot legends and adds a single shared legend per figure.
    """
    if save_plots:
        plot_dir = output_dir or OUTPUT_DIR
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Use a copy of adata for plotting to avoid modifying the original .obs
    adata_plot = adata[:, :]
    adata_plot.obs = adata.obs.copy()

    # Convert boolean columns to categorical strings for plotting
    for col in adata_plot.obs.columns:
        if adata_plot.obs[col].dtype == bool:
            adata_plot.obs[col] = adata_plot.obs[col].astype(str)

    sc.tl.umap(adata_plot)

    # Helper function to save figure with optional consolidated legend
    def plot_and_save(color_keys, title_suffix, palette=UNIFIED_PALETTE, consolidate_legend=True):
        # When consolidating, suppress individual legends; otherwise use default legend rendering
        fig = sc.pl.umap(
            adata_plot,
            color=color_keys,
            ncols=2,
            palette=palette,
            legend_loc=None if consolidate_legend else "right",
            return_fig=True,
        )

        # Only consolidate legend when comparing subplots with same cell types
        if consolidate_legend:
            # Remove all individual legends from subplots
            for ax in fig.axes:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            # Build legend from categories and colors
            handles_dict = _build_legend_handles(color_keys, adata_plot, palette)

            # Add single shared legend to the figure
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
    plot_and_save(dpmm_plot_keys, "_dpmm", consolidate_legend=False)
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

    # Default legends are rendered for each subplot

    if save_plots:
        fig.savefig(
            OUTPUT_DIR / f"tsne_baselines.{FIGURE_FORMAT}",
            bbox_inches="tight",
            dpi=100,
        )


if __name__ == "__main__":
    main()

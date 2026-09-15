"""Preprocessing helpers shared by the examples.

These wrap the standard scanpy QC recipe used throughout the project: flag QC gene
sets, compute metrics, derive data-driven upper cutoffs from percentiles, filter, then
library-size normalize and log1p.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import scanpy as sc
from anndata import AnnData


def flag_qc_genes(adata: AnnData) -> None:
    """Annotate mitochondrial, ribosomal, and hemoglobin genes in ``adata.var``.

    Gene symbols are uppercased before matching so human ``MT-`` and mouse ``mt-``
    prefixes are both recognized.
    """
    adata.var_names = adata.var_names.astype(str)
    vn_up = adata.var_names.str.upper()
    adata.var["mt"] = vn_up.str.startswith("MT-")
    adata.var["ribo"] = vn_up.str.startswith(("RPS", "RPL", "MRPS", "MRPL"))
    adata.var["hb"] = adata.var_names.str.match(r"^(HB[ABEDM][A-Z0-9]*)", case=False)


def qc_thresholds(
    adata: AnnData,
    umi_hi_pct: float = 99.5,
    genes_hi_pct: float = 99.5,
    mito_pct: float = 95.0,
    mito_floor: float = 20.0,
) -> Dict[str, float]:
    """Derive upper cutoffs for total counts, detected genes, and mito fraction.

    Requires ``sc.pp.calculate_qc_metrics`` outputs in ``adata.obs``. The mitochondrial
    cutoff is the given percentile but never below ``mito_floor`` percent.
    """
    return {
        "umi_hi": float(np.percentile(adata.obs["total_counts"], umi_hi_pct)),
        "genes_hi": float(np.percentile(adata.obs["n_genes_by_counts"], genes_hi_pct)),
        "mito_hi": float(max(mito_floor, np.percentile(adata.obs["pct_counts_mt"], mito_pct))),
    }


def qc_filter(
    adata: AnnData,
    min_genes: int = 200,
    min_cells_per_gene: int = 3,
    umi_hi_pct: float = 99.5,
    genes_hi_pct: float = 99.5,
    mito_pct: float = 95.0,
    mito_floor: float = 20.0,
    verbose: bool = True,
) -> AnnData:
    """Standard QC: flag gene sets, compute metrics, filter cells and genes.

    Cells are kept when they have at least ``min_genes`` detected genes and fall
    at or below the percentile-derived cutoffs for total counts, detected genes, and
    mitochondrial fraction (see :func:`qc_thresholds`). Genes detected in fewer than
    ``min_cells_per_gene`` cells are then removed. Thresholds are stored in
    ``adata.uns["qc_thresholds"]``.

    Returns a filtered copy; the input is annotated in place with QC metrics.
    """
    if verbose:
        print(f"START  | cells={adata.n_obs:,}  genes={adata.n_vars:,}")

    adata.var_names_make_unique()
    flag_qc_genes(adata)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=False, inplace=True
    )
    if verbose:
        print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())

    thr = qc_thresholds(adata, umi_hi_pct, genes_hi_pct, mito_pct, mito_floor)
    if verbose:
        print(
            f"Thresholds -> min_genes={min_genes}, umi_hi~{thr['umi_hi']:.0f}, "
            f"genes_hi~{thr['genes_hi']:.0f}, mito_hi~{thr['mito_hi']:.1f}%"
        )

    keep = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= thr["genes_hi"])
        & (adata.obs["total_counts"] <= thr["umi_hi"])
        & (adata.obs["pct_counts_mt"] <= thr["mito_hi"])
    )
    before = adata.n_obs
    adata = adata[keep].copy()
    if verbose:
        print(f"CELL FILTER | kept {adata.n_obs:,}/{before:,} ({adata.n_obs / before:.1%})")

    before_g = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    if verbose:
        print(f"GENE FILTER | kept {adata.n_vars:,}/{before_g:,} (>= {min_cells_per_gene} cells)")

    adata.uns["qc_thresholds"] = {
        "min_genes": min_genes,
        "min_cells_per_gene": min_cells_per_gene,
        **thr,
    }
    return adata


def normalize_log1p(adata: AnnData, target_sum: float = 1e4, set_raw: bool = True) -> AnnData:
    """Library-size normalize to ``target_sum`` and log1p in place; optionally freeze ``.raw``."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if set_raw:
        adata.raw = adata
    return adata

# PBMC 68k: notebook reproduction

Package-level reproduction of the PBMC 68k notebook (SHA256
`7dee319eda6c1b85e647338c4ac633d68667a6c8fdb920dc7abc42faad4448a7`), the PBMC 68k
notebook. It replaces the earlier `examples/pbmc68k/run.py`, which was an unfinished
attempt at a later PBMC notebook.

## Data

```
data/pbmc68k/filtered_matrices_mex/hg19/{matrix.mtx,genes.tsv,barcodes.tsv}
data/pbmc68k/pbmc_annot.csv        # one column, attached by row order
```

The matrix is the public 10x "Fresh 68k PBMCs (Donor A)" filtered gene/barcode
matrix:

```bash
mkdir -p data/pbmc68k && cd data/pbmc68k
curl -sSLO https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
tar -xzf fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
```

`pbmc_annot.csv` is the notebook's reference annotation (`pbmcannot` column, renamed
`pseudo_cell_type`). It is **not** the public 10x `68k_pbmc_barcodes_annotation.tsv`:
the notebook's class counts (e.g. CD8+/CD45RA+ Naive Cytotoxic 21,950, CD4+/CD25 T Reg
14,058 after QC) differ from the public labels. If the notebook's file is unavailable,
a stand-in can be derived from the public TSV (barcode order matches the matrix):

```python
import pandas as pd
bc = pd.read_csv("data/pbmc68k/filtered_matrices_mex/hg19/barcodes.tsv", header=None)[0]
ann = pd.read_csv("data/pbmc68k/68k_pbmc_barcodes_annotation.tsv", sep="\t").set_index("barcodes")
pd.DataFrame({"pbmcannot": ann.loc[bc, "celltype"].values}).to_csv("data/pbmc68k/pbmc_annot.csv", index=False)
```

The annotation only feeds the evaluation step; labels do not depend on it.
`SCAICME_PBMC68K_DIR` overrides the data location.

## Run

```bash
PYTHONPATH=src uv run python examples/pbmc68k/run.py
```

Outputs in `examples/pbmc68k/outputs/`: `pbmc68k_labels.csv` (reference, seeds,
consensus, agreement, rare flag per cell), `ablation_metrics.csv` (the notebook's
metrics table) and `pbmc68k_annotated.h5ad`.

## Pipeline ↔ notebook mapping

| Notebook | Package |
| --- | --- |
| cell 0: `read_10x_mtx`, annotation by row order, QC, normalize + log1p | `load_pbmc68k()`, `icme.pp.qc_filter`, `icme.pp.normalize_log1p` |
| cells 6–7: Method 1 `weak_label_quota_with_min_cells` (quantile 0.6, target 55 %, ≥ 50 cells/type, min score 0.2) | `QCQAdaptiveSeeding(quantile=0.6, target_frac=0.55, min_cells_per_type=50, min_score=0.2)`; its score matrix is `obsm["weak_label_quota_scores"]` |
| cells 10–11: Method 2 `dp_seed_by_marker_sets_soft` — defined, **not executed**, and overwritten by Method 3 | `DPGMMSeeding(...)` with `RUN_DPGMM = True` (off by default) |
| cells 13–14: Method 3 `gcn_seed_labeling` on Method 1's scores (PCA 30 / 15 neighbors built inside; gates 0.2 / 0.15; cap 55 %; floor max(200, 0.5 %)) → `weak_label` | `GCNSeeding(initial_scores_key="weak_label_quota_scores", ...)` |
| cell 15: PCA(15, arpack), neighbors(15), UMAP | `prepare_features()` (UMAP skipped: it feeds no downstream step) |
| cells 16–25: SVM (C=5, balanced, seeds kept, min_conf 0.55), K-Means (k = #seed types), KNN (k=3, distance, Manhattan), RF (300 trees, balanced_subsample), MLP ((256,128), seeds kept, min_conf 0.55) | the five propagation strategies with those settings |
| cell 27: plurality consensus, agreement = votes / 5 | `ConsensusVoting(majority_fraction=None, fraction_of="all")` |
| cell 30: rare/novel flag (agreement ≤ 0.4 or type < 0.5 %) | `icme.evaluation.flag_rare` |
| cells 32–34: `compare_labels` per method → `ablation_metrics.csv` | `icme.evaluation.compare_many` |
| cells 35–39: UMAP figures of `adata_syn` (a synthetic object not defined in this notebook) | not reproduced |

Deviations worth knowing:

- The notebook's SVM sets no `random_state`, so its Platt scaling is not reproducible;
  the example fixes `random_state=42`.
- The notebook's K-Means fallback for a seedless cluster references an undefined
  variable (`C`); the package uses the nearest seed-class centroid. With four seeded
  types and ~8k seeds no cluster is seedless, so this path is not exercised.
- PCA/neighbors for Method 3 use the notebook's defaults (30 PCs, 15 neighbors) and are
  then recomputed with 15 PCs for propagation, exactly as the notebook does.

## Parity record (2026-09-15)

Run in this repository's environment (Python 3.12, scanpy 1.12, scikit-learn 1.8,
anndata 0.12.6, numpy 2.3.5), about 5 minutes:

- QC: 68,154 cells × 17,676 genes; thresholds `umi_hi≈3874`, `genes_hi≈1265`,
  `mito_hi=20.0%` — identical to the notebook output.
- Method 1 (quota seeds): **identical** to the notebook — 34,601 labeled and all nine
  per-type counts (7203 / 7183 / 5476 / 4133 / 4085 / 3392 / 2239 / 593 / 297).
- Method 3 (GCN seeds), against the notebook's printed counts:

  | type | this run | notebook |
  | --- | ---: | ---: |
  | CD19+ B | 3940 | 3940 |
  | CD14+ Monocyte | 3283 | 3281 |
  | CD8+ Cytotoxic T | 642 | 660 |
  | CD4+/CD45RO+ Memory | 353 | 354 |
  | total seeds | 8218 | 8235 |

- Running the notebook's own functions verbatim (cells 6, 13, 16, 18, 20, 22, 24, 27,
  32; `tests/reference_pbmc68k_notebook.py`) on the same in-memory data and the same kNN
  graph gives **identical** Method 1 scores and labels, Method 3 labels and
  confidences, all five classifier label sets, the consensus and agreement fractions,
  and the `compare_labels` metrics. The 17-seed Method 3 difference above therefore
  comes from the inputs to that stage — `sc.pp.neighbors` (approximate kNN) and
  ARPACK PCA — differing between this environment and the one the notebook
  was run in, not from the pipeline. The same equality is asserted on synthetic
  data in `tests/test_pbmc68k_notebook_parity.py`.
- Consensus (this run): CD4+/CD45RO+ Memory 42,237; CD8+ Cytotoxic T 16,370; CD19+ B
  5,042; CD14+ Monocyte 4,505; 1,534 cells (2.25 %) flagged rare/novel (notebook:
  1,683, 2.47 %). The notebook does not print its consensus counts.
- Metrics table: computed against the public-annotation stand-in, so the values are
  not comparable to the notebook's `ablation_metrics.csv` until the original
  `pbmc_annot.csv` is supplied.

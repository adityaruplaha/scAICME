# PBMC 68k: notebook reproduction

Package-level reproduction of the final PBMC 68k notebook (SHA256
`ed3e912d9a4e1bd99a4248661fe3bd6e9a55edeea1385a9d71e7c322108cc6ac`, Sept 2026).
It replaces the earlier reproduction of the previous PBMC notebook, whose lineage
remains covered by `tests/test_pbmc68k_notebook_parity.py`.

The saved run of that notebook seeds with **Method 2** (DP-GMM per marker set, no PCA):
its Method 1 and Method 3 cells are defined but were never executed
(`execution_count: null`), and the cells that were executed run in the order
load → Method 2 → classifiers → Leiden → metrics → scANVI.

## Data

```
data/pbmc68k/filtered_matrices_mex/hg19/{matrix.mtx,genes.tsv,barcodes.tsv}
data/pbmc68k/pbmc_annot.csv        # one column, attached by row order
data/pbmc68k/cell_type_true.csv    # optional; the manual ground truth, preferred reference
data/pbmc68k/label_scanvi.csv      # optional; attached when present
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

Outputs in `examples/pbmc68k/outputs/`: `pbmc68k_labels.csv` (reference, seeds, the five
classifier labels, consensus and agreement, Leiden, scANVI when available),
`classwise_metrics.csv`, `pbmc_scACIME_cluster_metrics.csv` and
`pbmc68k_annotated.h5ad`.

The DP-GMM stage fits a 100-component full-covariance mixture per marker set and the
silhouette score is quadratic in the number of cells, so the run is long; set
`CLUSTER_QUALITY = False` in `run.py` to skip the cluster-quality table.

## Pipeline ↔ notebook mapping

| Notebook | Package |
| --- | --- |
| cell 0: `read_10x_mtx`, annotation by row order, QC, normalize + log1p, marker panels | `load_pbmc68k()`, `icme.pp.qc_filter`, `icme.pp.normalize_log1p` |
| cells 1–2: Method 1 quota seeding and Method 3 quota GCN seeding — defined, **never executed** | not part of this pipeline; `QCQAdaptiveSeeding(target_frac=...)` and `GCNSeeding` are the package equivalents |
| cells 3–4: Method 2 `dp_seed_by_marker_sets_soft_no_pca` (per-gene quantile 0.4, cluster score ≥ 0.1, ≥ 100 cells/cluster, `n_components = min(100, √N)`, prior 0.05, floor max(100, 0.002 N)) → `weak_label` | `DPGMMSeeding(...)` with those settings and `use_raw=False` |
| cell 5: PCA(30, arpack), neighbors(15, 30 PCs) | `prepare_features()` |
| cell 5: SVM (C=2, balanced, min_conf 0.60), k-means (k = min(30, max(10, √(N/2))), n_init 20), KNN (k=9, distance, Euclidean, min_conf 0.55), RF (500 trees, depth 18, leaf 5, balanced_subsample, min_conf 0.55), MLP ((128,64), α=1e-3, 400 iters, early stopping, min_conf 0.60); all with `min_seed_conf=0.30`, 30 PCs, and every cell predicted | the five propagation strategies with those settings and `keep_seeds=False` |
| cell 5: plurality consensus, agreement = votes / 5 | `ConsensusVoting(majority_fraction=None, fraction_of="all")` |
| cell 6: `sc.tl.leiden(resolution=1.6)` | `run_leiden()` |
| cells 9–10: `compare_labels_ari_acc_spec_sens` per method → `classwise_metrics.csv` | `icme.evaluation.compare_many` (ARI, accuracy, macro specificity and sensitivity, plus NMI and macro-F1) |
| cells 11–13: scANVI predictions, read from `label_scanvi.csv` and one class relabelled | `attach_scanvi()` when the file is present; the notebook's own training cell is not reproduced |
| cell 15: `clustering_metrics` on the consensus → `pbmc_scACIME_cluster_metrics.csv` | `icme.evaluation.cluster_quality` |
| cells 7–8, 14: marker dot plot and heat map, rare-cell UMAP panels | not reproduced (figure code) |

Deviations worth knowing:

- **DP seeding uses no PCA.** Cell 4 calls `dp_seed_by_marker_sets_soft`, the variant
  that reduces each marker block with PCA, and its stored output reports `n_pcs=10`; but
  the notebook defines only `dp_seed_by_marker_sets_soft_no_pca`, so that call resolved
  to a definition left in the kernel from an earlier session. The no-PCA version is the
  one reproduced here, by decision, which means the seed counts below are **not**
  expected to match the stored cell 4 output.
- The notebook evaluates against `cell_type_true`, the manually curated ground-truth
  annotation, which is supplied to the notebook from outside rather than built inside
  it. Drop those labels in as `data/pbmc68k/cell_type_true.csv` and the example uses
  them as the reference; until then it falls back to `pseudo_cell_type` from the
  loader, and the metrics are against that instead.
- The notebook's k-means takes its per-cluster majority vote over every labelled seed
  but builds the seedless-cluster fallback centroids only from seeds above 0.3
  confidence. `KMeansPropagation(min_seed_conf=0.0, fallback_min_seed_conf=0.3)`
  reproduces both masks.
- scANVI is attached from the saved CSV rather than retrained, as the notebook does in
  cell 12.
- **No rare-type carve-out.** The May 2026 PBMC notebook exempted
  `{"CD4+ T Helper2", "CD34+"}` from the DP size filter, and the July and September
  versions dropped that exemption; this example follows the September version, so
  nothing is exempt. It made no difference to the run below (the floor came out at 136,
  CD4+ T Helper2 cleared it with about 2,357 cells and CD34+ received no seeds at all),
  but `DPGMMSeeding(always_keep=("CD4+ T Helper2", "CD34+"))` restores the older
  behaviour if a future run puts either type just under the floor.

## Parity record (2026-09-28)

Run in this repository's environment (Python 3.12, scanpy 1.12, scikit-learn 1.8,
anndata 0.12.6, numpy 2.3.5). About 45 minutes wall clock, most of it the eleven
DP-GMM fits, plus the silhouette.

- QC: 68,154 cells × 17,676 genes; thresholds `umi_hi≈3874`, `genes_hi≈1265`,
  `mito_hi=20.0%` — identical to the notebook output.
- DP seeding, per marker set, against the notebook's stored cell 4 output. The size
  floor resolves to 136 in both, and no type falls below it.

  | marker set | markers present | this run | notebook | notebook `n_pcs` |
  | --- | ---: | ---: | ---: | ---: |
  | CD56+ NK | 10 | **12998** | **12998** | 10 |
  | Dendritic | 9 | **8223** | **8223** | 9 |
  | CD34+ | 7 | **0** | **0** | 7 |
  | CD4+ T Helper2 | 9 | **37753** | **37753** | 9 |
  | CD8+/CD45RA+ Naive Cytotoxic | 11 | 54927 | 54345 | 10 |
  | CD4+/CD25 T Reg | 11 | 48795 | 49827 | 10 |
  | CD8+ Cytotoxic T | 12 | 29221 | 30690 | 10 |
  | CD19+ B | 11 | 12364 | 10534 | 10 |
  | CD14+ Monocyte | 12 | 5371 | 7304 | 10 |
  | CD4+/CD45RO+ Memory | 11 | 54063 | 55304 | 10 |
  | CD4+/CD45RA+/CD25- Naive T | 11 | 54906 | 54548 | 10 |

  The split is exact and it explains itself. Every set with at most 10 markers present
  reproduces the notebook's labelled count and its number of signal components exactly;
  every set with 11 or more differs. A
  full-covariance Gaussian mixture is equivariant under invertible linear maps, so when
  the notebook's PCA keeps all `n_features` dimensions it is a pure rotation and changes
  nothing, while at 11 or 12 markers `pca_n=10` genuinely discards a dimension and the
  fit changes. So the stored output did come from the PCA variant, the port itself is
  confirmed correct wherever PCA was lossless, and the remaining differences are the
  intended consequence of dropping PCA rather than an implementation discrepancy.

- Consensus (this run): CD4+/CD25 T Reg 25,226; CD8+/CD45RA+ Naive Cytotoxic 19,885;
  CD8+ Cytotoxic T 9,314; CD56+ NK 4,531; CD19+ B 4,303; CD14+ Monocyte 3,864;
  Dendritic 564; CD4+ T Helper2 317; CD4+/CD45RA+/CD25- Naive T 102;
  CD4+/CD45RO+ Memory 48. The notebook does not print its consensus counts.
- Leiden at resolution 1.6 gives 18 clusters. Cluster quality of the consensus:
  silhouette 0.053, Calinski-Harabasz 5579, Davies-Bouldin 2.80.
- The metrics table was computed against the public-annotation stand-in, so its values
  are **not** comparable to the notebook's `classwise_metrics.csv` until the original
  `pbmc_annot.csv` is supplied. Ranking in this run, by ARI on cells labelled in both:
  random forest 0.538, SVM 0.498, MLP 0.450, k-means 0.203, consensus 0.202, KNN 0.178.
  The three rejecting classifiers cover 28-41 % of cells, k-means and the consensus 100 %.

The notebook's own functions (cells 3, 5, 9 and 15) are kept verbatim in
`tests/reference_pbmc68k_dp_notebook.py`, and `tests/test_pbmc68k_dp_parity.py` asserts
that the package reproduces them label for label on synthetic data: DP seed labels and
confidences, all five classifier label sets, the k-means confidences, the consensus and
its agreement fractions, and both metric functions.

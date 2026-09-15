# GSE225475: psoriasis skin Visium sections

Package-level reproduction of the GSE225475 spatial notebook
(SHA256 `ef1c0683a45d9d81bd8025c90d7b6b035d80b7eae4e87c57a542a30c9060b342`).
The notebook pools six Visium sections (NS1, NS2, PP1–PP4; GSM7049132–GSM7049137),
seeds spots with a per-marker-set Dirichlet-process GMM, propagates the seeds with
five classifiers on PCA space, and takes a plurality consensus.

## Data

Download the six sample archives from
[GSE225475](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225475)
(about 120 MB in total) and extract each into `data/gse225475/`:

```bash
mkdir -p data/gse225475 && cd data/gse225475
for s in GSM7049132_NS1 GSM7049133_NS2 GSM7049134_PP1 GSM7049135_PP2 GSM7049136_PP3 GSM7049137_PP4; do
  curl -sSLO "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7049nnn/${s%%_*}/suppl/$s.tar.gz"
  tar -xzf "$s.tar.gz"
done
```

This yields `data/gse225475/<sample>/filtered_feature_bc_matrix.h5` and
`data/gse225475/<sample>/spatial/tissue_positions_list.csv`, which is what
`sc.read_visium` needs (no images are shipped; the missing-image warning is harmless).
Set `SCAICME_GSE225475_DIR` to use a different location.

## Run

```bash
PYTHONPATH=src uv run python examples/gse225475/run.py
```

Outputs go to `examples/gse225475/outputs/`:

- `scAICME_spatial_labels.csv` — `sample`, `barcode`, `label_consensus` per spot
  (the notebook's export)
- `gse225475_annotated.h5ad` — the full annotated AnnData (seeds, per-method labels,
  probabilities, consensus, PCA, kNN graph, QC thresholds)

Runtime is about a minute on a laptop.

## Pipeline ↔ notebook mapping

| Notebook | Package |
| --- | --- |
| cells 0–1: `sc.read_visium` per sample, `ad.concat(join="outer")` | `load_samples()` |
| cell 2: `in_tissue` filter, QC metrics, percentile cutoffs, `normalize_total` + `log1p`, `adata.raw = adata` | `icme.pp.qc_filter`, `icme.pp.normalize_log1p` |
| cells 3–5: `SPATIAL_SKIN_MARKERS`, `dp_seed_by_marker_sets_soft(...)` → `weak_label`, `weak_conf` | `DPGMMSeeding(...)` → `weak_label`, `weak_label_max_score` |
| cell 6: PCA(20, arpack), neighbors(20, 15 PCs) | `prepare_features()` |
| cell 6: `rbf_svm_labeling`, `kmeans_seed_transfer`, `knn_labeling`, `rf_labeling`, `mlp_labeling` | `SVMPropagation`, `KMeansPropagation`, `KNNPropagation`, `RandomForestPropagation`, `NeuralNetworkPropagation` with the notebook's settings and `keep_seeds=False` |
| cell 6: plurality consensus, `agree_frac = votes / n_methods` | `ConsensusVoting(majority_fraction=None, fraction_of="all")` |
| cell 7: CSV export | `export()` |

## Parity record (2026-09-15)

Run in this repository's environment (Python 3.12, scanpy 1.12, scikit-learn 1.8,
anndata 0.12.6, numpy 2.3.5):

- QC: 7,184 spots × 21,216 genes retained; thresholds `umi_hi≈134528`,
  `genes_hi≈10497`, `mito_hi=20.0%` — identical to the notebook output.
- Seeding: per-type labelled counts and signal-component counts are identical to the
  notebook output for all eight types (e.g. Keratinocyte 7171 / 13 components,
  T_cell 1696 / 12), as is the `weak_label` distribution to six decimals.
- Running the notebook's own functions (cells 4 and 6, verbatim) on the same in-memory
  data in this environment gives **identical labels** for the seeds, all five
  classifiers, and the consensus. The same check runs on synthetic data in
  `tests/test_gse225475_notebook_parity.py`.
- Consensus counts against the numbers printed in the notebook:

  | type | this run | notebook |
  | --- | ---: | ---: |
  | Keratinocyte | 3534 | 3534 |
  | Fibroblast | 2619 | 2622 |
  | Smooth_muscle | 564 | 561 |
  | Eccrine_gland | 294 | 295 |
  | Endothelial | 66 | 65 |
  | T_cell | 49 | 49 |
  | Myeloid | 44 | 45 |
  | Mast | 14 | 13 |

  The 10-spot difference (0.14%) is environment drift, not a pipeline difference:
  the notebook ran under a different scikit-learn/scanpy build, and SVM Platt
  scaling, the MLP validation split, and ARPACK PCA are
  all sensitive to that. The seeding stage, which has no such dependence, matches
  exactly.

# scAICME: scRNA-seq Annotation by Identifying Canonical Marker Expressions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

`scAICME` is a Python package for marker-driven semi-supervised annotation in scRNA-seq data. The pipeline combines seed-labelling strategies (adaptive thresholding, GCNs, DPGMMs), identity extension models (ensemble supervised learning methods), and consensus voting to infer likely cell types from canonical marker genes.

### Intended Use

This package is designed for annotating scRNA-seq datasets when you have prior knowledge of marker genes for target cell types. It is not intended for unsupervised clustering or novel cell type discovery (yet!). Instead, it provides a robust framework for leveraging known biology to generate high-confidence annotations.

**Key characteristics:**
- Combines multiple seeding strategies for robust identity evidence
- Extends identity assignments from high-confidence seeds to unlabeled cells
- Evaluates method agreement using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
- Runs efficiently on standard hardware, taking advantage of parallel processing where possible

**Requirements and Limitations:**
- Requires prior knowledge of marker genes for target cell types
- Identification accuracy depends directly on marker gene quality
- Limited to cell types with defined markers (no automated discovery)
- Works best with well-separated cell types in the data

## Installation

The package requires Python ≥ 3.10. Core dependencies (scanpy, scikit-learn, pandas, numpy) are automatically installed.

## Quick Start

### Basic Usage

```python
import scanpy as sc
import scAICME as icme

# Load your scRNA-seq data
adata = sc.read_h5ad("data/pbmc.h5ad")

# Define marker genes for cell types of interest
markers = {
    "CD8_T": ["CD8A", "CD8B", "GZMA"],
    "CD4_T": ["CD4", "IL7R", "TCF7"],
    "B_cells": ["CD19", "MS4A1", "CD79A"],
    "Dendritic": ["ITGAX", "CD1C", "FCER1A"],
    # ... more cell types
}

# Phase 1: Generate seed labels from multiple strategies (no consensus)
seed_strategies = {
    "seeds_qcq_scored": icme.strategies.QCQScoredAdaptiveSeeding(markers=markers),
    "seeds_qcq_per_gene": icme.strategies.QCQAdaptiveSeeding(markers=markers),
    "seeds_otsu_scored": icme.strategies.OtsuScoredAdaptiveSeeding(markers=markers),
    "seeds_otsu_per_gene": icme.strategies.OtsuAdaptiveSeeding(markers=markers),
    "seeds_dpgmm": icme.strategies.DPGMMSeeding(markers=markers, random_state=0),
}
icme.tl.label(adata, strategies=seed_strategies, n_jobs=4)

# Phase 2: Propagate labels independently from each seed source
# This allows different propagation methods to learn from different seed qualities
propagation_methods = ["knn", "rf"]  # KNN and Random Forest
all_propagated_labels = []

for seed_name in seed_strategies.keys():
    propagators = {
        f"prop_{method}_{seed_name.split('_')[1]}": (
            icme.strategies.KNNPropagation(seed_key=seed_name)
            if method == "knn"
            else icme.strategies.RandomForestPropagation(seed_key=seed_name)
        )
        for method in propagation_methods
    }
    icme.tl.label(adata, strategies=propagators, n_jobs=2)
    all_propagated_labels.extend(propagators.keys())

# Phase 3: Final consensus across all propagated predictions
final_consensus = icme.strategies.ConsensusVoting(
    keys=all_propagated_labels,
    majority_fraction=0.66
)
icme.tl.label(adata, strategies=final_consensus, key_added="labels_final")

# Results are stored in adata.obs
print(adata.obs["labels_final"].value_counts())
```

### Working with Individual Strategies

```python
# Apply a single scored strategy
strategy = icme.strategies.QCQScoredAdaptiveSeeding(markers=markers, quantile=0.95)
result = icme.tl.label(adata, strategies=strategy, key_added="my_labels")

# Access the result
print(f"Assigned labels: {result['my_labels'].labels.value_counts()}")

# Check confidence scores if available
if "my_labels_max_confidence" in adata.obs:
    print(f"Mean confidence: {adata.obs['my_labels_max_confidence'].mean():.2f}")
```

### Batch Processing with Async

```python
import asyncio

async def label_multiple():
    strategies = [
        icme.strategies.QCQScoredAdaptiveSeeding(markers),
        icme.strategies.QCQAdaptiveSeeding(markers),
        icme.strategies.OtsuScoredAdaptiveSeeding(markers),
        icme.strategies.OtsuAdaptiveSeeding(markers),
    ]
    
    results = await asyncio.gather(
        *[icme.tl.label_async(adata, s) for s in strategies]
    )
    return results

# results = asyncio.run(label_multiple())
```


## Example: GSE225475 psoriasis skin (Visium)

`examples/gse225475/run.py` is a package-level reproduction of the GSE225475
spatial notebook: six pooled Visium sections → QC (`icme.pp.qc_filter`) →
`DPGMMSeeding` → PCA → SVM / K-Means / KNN / Random Forest / MLP propagation →
plurality consensus → `scAICME_spatial_labels.csv`.

```bash
# Download the six GEO sample archives into data/gse225475/<sample>/ (~120 MB total)
# (or point SCAICME_GSE225475_DIR at an existing extraction), then:
PYTHONPATH=src uv run python examples/gse225475/run.py
```

See `examples/gse225475/README.md` for the data layout and the parity record.

## Example: PBMC 68k

`examples/pbmc68k/run.py` reproduces the PBMC 68k notebook: QC →
quota seeding (`QCQAdaptiveSeeding` with `target_frac`) → `GCNSeeding` on its score
matrix → 15 PCs → SVM / K-Means / KNN / Random Forest / MLP → plurality consensus →
rare-cell flag and agreement metrics against a reference annotation
(`icme.evaluation.compare_many`). See `examples/pbmc68k/README.md`.

```bash
PYTHONPATH=src uv run python examples/pbmc68k/run.py
```

### Evaluation helpers

`icme.evaluation.compare_labels(adata, pred_key, ref_key)` returns ARI, NMI, macro-F1,
accuracy and coverage (over all cells and over cells labeled in both columns);
`compare_many` tabulates several prediction columns; `flag_rare` marks cells with weak
consensus agreement or tiny consensus types.

## Example: PBMC3k Dataset

A complete end-to-end pipeline is provided in `examples/pbmc3k/run.py`:

```bash
# Run the PBMC3k example
uv run icme-examples pbmc3k
```

This demonstrates:
- QC filtering with data-driven thresholds
- Feature preprocessing (HVGs, PCA, neighbors)
- Phase 1 seeding with 4 independent strategies (no early consensus)
- Phase 2 propagation from each seed independently (3 × 4 combinations)
- Baseline clustering (Leiden at multiple resolutions)
- Visualization (UMAP, t-SNE)
- Quantitative ablation metrics (ARI, NMI)


## Architecture

### Strategy Pattern

All labeling methods inherit from `BaseLabelingStrategy` and implement:

```python
class MyStrategy(BaseLabelingStrategy):
    def __init__(self, markers, **kwargs):
        self.markers = markers
    
    @property
    def name(self) -> str:
        return "my_strategy"
    
    def execute_on(self, adata: AnnData) -> LabelingResult:
        # Implement labeling logic
        labels = self.predict(adata)
        
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=labels,
            obs={"confidence": confidence_scores},  # Optional
            uns={"parameters": self.__dict__},      # Optional
        )
```

### Data Flow

```
Raw scRNA-seq data
    ↓
QC Filtering (genes, UMI, mitochondrial content)
    ↓
Normalization (log1p)
    ↓
Preprocessing (as needed: HVG selection, PCA, neighbors, etc.)
    ↓
┌──────────────────────────────────────────┐
│   PHASE 1: Weak Labeling Strategies      │
│  (QCQ, Otsu, Graph, DPMM, etc.)          │
│  → Generate independent seed labels      │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│   PHASE 2: Independent Propagation       │
│  Each propagator (KNN, RF, Centroid)     │
│  learns from EACH seed independently     │
│  → Generates: prop_<method>_<seed>       │
└──────────────────────────────────────────┘
    ↓
Final Consensus Voting (across all Phase 2 outputs)
    ↓
Quantitative Evaluation (ARI, NMI)
```

**Design rationale:**
- Phase 1 generates independent seeds from diverse methods
- Phase 2 propagates from each seed independently, allowing diverse approaches to coexist
- Final consensus aggregates all Phase 2 predictions for robust classification


## Implementation

### Phase 0: Marker Gene Specification

Define markers as a dictionary mapping cell type names to lists of genes:

```python
markers = {
    "T_cells": ["CD3D", "CD3E", "CD3G"],
    "B_cells": ["CD19", "MS4A1", "CD79A"],
    "Monocytes": ["LYZ", "S100A8", "S100A9"],
    "NK_cells": ["GNLY", "NKG7", "GZMB"],
}
```

**Guidelines:**
- Use 3-5 robust marker genes per cell type for reliability
- Prefer genes with high expression in target cell type
- Prefer genes with low expression in other cell types
- Test markers on reference data before large-scale analysis

### Phase 1: Initial Labeling

Use any (or many!) of the following strategies to generate independent seed labels. Each is propagated independently in Phase 2:

| Strategy | Class | Description |
|----------|-------|-------------|
| QCQ Adaptive Thresholding on Scored Markers | `QCQScoredAdaptiveSeeding` | `score_genes`-based marker set scoring with quantile + minimum score gates |
| QCQ Per-Gene Adaptive Thresholding | `QCQAdaptiveSeeding` | Per-gene positive-expression thresholds with active-marker-fraction (`min_confidence`) gating |
| Otsu Adaptive Thresholding on Scored Markers | `OtsuScoredAdaptiveSeeding` | `score_genes`-based marker set scoring with Otsu + minimum score gates |
| Otsu Per-Gene Adaptive Thresholding | `OtsuAdaptiveSeeding` | Per-gene Otsu thresholds with active-marker-fraction (`min_confidence`) gating |
| Marker-set DP-GMM Seeding | `DPGMMSeeding` | Self-contained: per cell type, a Dirichlet-process GMM on standardized marker expression; signal components are those enriched for the type's markers; confidence is the cell's marker activation |
| GCN Seeding | `GCNSeeding` | Row-normalizes a prior strategy's score matrix, diffuses it over the kNN graph for a fixed number of iterations, then gates on top score and top-1/top-2 margin, optionally keeps only the most confident `target_frac`, and drops small types |
| GCN Smoothing | `GCNSmoothing` | Graph-convolutional smoothing of a prior strategy's score matrix with adaptive quantile gates and quota selection |
| DP-GMM Clustered Smoothing | `DPGMMClusteredSmoothing` | Bayesian mixture model gated by a prior strategy's seed scores |

**Common Parameters:**
- `markers` (dict): Cell type → marker gene list mapping
- `unknown_label` (str, default "unknown"): Label for unlabeled cells
- `use_raw` (bool, default True): Read marker expression from `adata.raw` when present

With `target_frac`, the QCQ/Otsu seeders allocate an exact labeling budget: every type with at least `min_cells_per_type` eligible cells gets that many seeds, the remaining budget is shared in proportion to each type's surplus of eligible cells, and the highest-scoring eligible cells fill each quota. Their per-type score matrix is stored in `obsm["<key>_scores"]` and can feed `GCNSeeding` (`initial_scores_key="<key>_scores"`).

`DPGMMSeeding` needs no prior scores. It skips a type whose markers are mostly absent or barely expressed, gates mixture components by mean marker score and size, reconciles types by confidence, and drops types that end up below `max(min_type_size, min_type_frac * n_cells)` cells. The per-type confidence matrix is stored in `obsm["<key>_scores"]`, the raw marker-activation fractions in `obsm["<key>_marker_scores"]`, and per-type fit diagnostics in `uns["<key>_uns"]["diagnostics"]`.

```python
seeder = icme.strategies.DPGMMSeeding(
    markers=markers,
    n_components=15,                # None -> max(2, int(sqrt(n_cells)))
    weight_concentration_prior=0.1,
    per_gene_pos_quantile=0.3,      # activation threshold per marker gene
    cluster_score_min=0.08,         # mean marker score for a component to count as signal
    min_cells_cluster=30,
    random_state=42,
)
icme.tl.label(adata, seeder, key_added="weak_label")
```

### Phase 2: Identity Extension Strategies

Extend seed identities to unlabeled cells using supervised learning:

| Strategy | Class | Description |
|----------|-------|-------------|
| KNN Propagation | `KNNPropagation` | k-Nearest neighbor classification |
| Random Forest Propagation | `RandomForestPropagation` | Ensemble-based classification (`max_depth`, `min_samples_leaf`, `max_features`, `class_weight`) |
| SVM Propagation | `SVMPropagation` | Kernel SVM with optional Platt probabilities (`class_weight="balanced"` supported) |
| Neural Network Propagation | `NeuralNetworkPropagation` | MLP classifier with early stopping (`validation_fraction`, `n_iter_no_change`, `scale_features`) |
| K-Means Propagation | `KMeansPropagation` | Cluster all cells, label each cluster by its majority seed; confidence is that majority's share of the cluster, seedless clusters take the nearest seed-class centroid |
| Nearest Centroid Propagation | `NearestCentroidPropagation` | Centroid-based assignment |

**Common Parameters:**
- `seed_key` (str): Column in `adata.obs` containing seed labels
- `obsm_key` (str, default "X_pca"): Feature representation for classification
- `max_pcs` (int | None): Use only the first `max_pcs` columns of the feature matrix
- `unknown_label` (str, default "unknown"): Label for unlabeled cells
- `keep_seeds` (bool, default True): Preserve original seed labels; set `False` to re-predict seeds
- `min_seed_conf` (float, default 0.0): Train only on seeds whose confidence (`<seed_key>_max_score`, `<seed_key>_max_confidence`, or `conf_key`) reaches this value
- `min_conf` (float, default 0.0): Predictions below this confidence become `unknown_label`

For the probabilistic classifiers the label is the `argmax` of the same probability vector that supplies the confidence, so a cell's label and its confidence always refer to the same class.

### Consensus Strategy

Obtain a final consensus label by combining multiple strategies with majority voting:

| Strategy | Class | Description |
|----------|-------|-------------|
| Consensus Voting | `ConsensusVoting` | Combine multiple predictions via majority voting |

**Parameters:**
- `keys` (list[str]): Column names to combine
- `majority_fraction` (float | None, default 0.66): Fraction of votes required (0.51 to 1.0); `None` means plurality (the most common valid vote always wins)
- `fraction_of` ("valid" | "all", default "valid"): Whether the agreement fraction is taken over the cell's valid (non-unknown) votes or over all voters in `keys`
- `unknown_label` (str, default "unknown"): Label for unlabeled cells

**Example Usage:**
```python
# Combine propagation outputs from all seeds at the final stage
consensus = icme.strategies.ConsensusVoting(
    keys=["prop_knn_qcq", "prop_knn_otsu", "prop_rf_qcq", "prop_rf_otsu"],
    majority_fraction=0.66  # Supermajority
)
icme.tl.label(adata, strategies=consensus, key_added="labels_final")
```

#### Using `majority_fraction` for Consensus Voting

Controls how many independent strategies must agree to assign a label:

- `0.51`: Simple majority (loose, more cells labeled)
- `0.66`: Supermajority (balanced, recommended)
- `1.00`: Unanimous (strict, fewer cells labeled, higher confidence)

**Example:**
```python
# Strict consensus (all methods must agree)
consensus = icme.strategies.ConsensusVoting(
    keys=["seeds_qcq", "seeds_otsu", "seeds_graph"],
    majority_fraction=1.0
)

# Loose consensus (2 out of 3)
consensus = icme.strategies.ConsensusVoting(
    keys=["seeds_qcq", "seeds_otsu", "seeds_graph"],
    majority_fraction=0.51
)
```


## Output Format

Labeling results are stored in `adata.obs` with the following convention:

```
├── {key}                         # Main labels ("Unknown" = unlabeled)
├── {key}_max_confidence          # Maximum voting score (if available)
├── {key}_is_confident            # Boolean confidence flag (optional)
└── {key}_params                  # Strategy parameters in adata.uns
```

Example output structure for Phase 1 seeding:
```
adata.obs columns:
├── seeds_qcq              # QCQ strategy labels
├── seeds_otsu             # Otsu strategy labels
├── seeds_graph            # Graph strategy labels
├── seeds_dpmm             # DPMM strategy labels
├── seeds_dpmm_max_confidence    # DPMM confidence score
├── seeds_dpmm_is_confident      # DPMM confidence boolean
# Note: seeds_consensus is no longer generated; seeds are propagated independently
```

## Performance Considerations

- QCQ and Otsu strategies run in seconds to minutes depending on data size
- Graph-based methods may take longer due to network construction
- DPMM is computationally intensive; consider subsampling for large datasets. Parallel processing is available for DPMM using the `n_jobs` parameter.
- Multiple strategies can be run in parallel using `label_async` and `asyncio.gather` for efficient batch processing

## Troubleshooting

### "No labeled cells found in the seed column"
- Phase 2 propagation requires at least some seeds from Phase 1
- Check that Phase 1 resulted in labeled cells (not all "unknown")
- Try looser marker gene definitions or `majority_fraction=0.51`

### "Key 'X_pca' not found in adata.obsm"
- Ensure PCA was computed before running propagation strategies
- Run: `sc.tl.pca(adata)` and `sc.pp.neighbors(adata)`

### Few cells labeled in Phase 1
- Markers might be missing from your dataset
- Check gene names match annotation (case-sensitive)
- Markers might be too specific; try more permissive thresholds

### Inconsistent labels between methods
- This is expected! Different methods have different biases
- Use consensus voting to increase confidence
- Check `majority_fraction` parameter

## FAQ

**Q: Can I use gene expression data from different platforms (10x, Smart-seq)?**  
A: Yes! Ensure proper normalization before running (`sc.pp.normalize_total` + `sc.pp.log1p`)

**Q: What if I have confidence scores for true cell types?**  
A: Filter your labeled data to high-confidence cells before using as seed labels

**Q: Can I use this for novel cell type discovery?**  
A: Not at the moment. This package is designed for annotation with known markers, not unsupervised clustering. Future versions may include treating "unknown" cells as a separate cluster for discovery, using a hierarchical approach.

**Q: How many marker genes do I need?**  
A: 3-5 robust markers per cell type is a good starting point. More markers reduce false positives.

**Q: Can I combine predictions with a reference atlas?**  
A: Not built-in, but is possible using a custom strategy.


## Authorship and Acknowledgments

This package was developed as part of the Master of Statistics (M.Stat.) project at the Indian Statistical Institute in partial fulfillment of curriculum requirements. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed author and contributor information, including ORCIDs.

I am immensely grateful to my advisors, Prof. Raghunath Chatterjee and Dr. Jayant Jha, for their guidance and support throughout this project. I also thank Dr. Snehalika Lall for her valuable insights, experience and infrastructure support, without which this work would not have been possible. Finally, I acknowledge the open-source community and the developers of the libraries used in this project for their contributions to scientific software.

## Licensing and Citation

BSD 3-Clause License - See [LICENSE](LICENSE) file for details.

Please cite this package appropriately if you use it in your research. A citation file will be provided upon publication.

---

**For questions, issues, or feature requests, please open an issue on GitHub.**

Last Updated: 14 February 2026

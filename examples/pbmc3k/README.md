# PBMC3k: generic pipeline demo

A first-pass demonstration of the generic scAICME pipeline on the 10x PBMC3k dataset,
kept as a quick smoke test: it needs no download (`sc.datasets.pbmc3k()`, with a copy
in `data/pbmc3k_raw.h5ad`) and exercises most strategies. It is **not** a notebook
reproduction and makes no parity claims; the dataset examples that do are
`examples/gse225475/` and `examples/pbmc68k/`.

```bash
PYTHONPATH=src uv run --group examples python examples/pbmc3k/run.py
```

Pipeline: QC (percentile thresholds) → HVG/PCA/neighbors → three base seeders
(`QCQAdaptiveSeeding`, `OtsuScoredAdaptiveSeeding`, `OtsuAdaptiveSeeding`), each also
smoothed by `DPGMMClusteredSmoothing` and `GCNSmoothing` → four propagators (KNN,
Random Forest, Nearest Centroid, SVM) per seed set → per-seed and final 2/3-majority
consensus → Leiden baselines, UMAP/t-SNE plots and pairwise ARI/NMI under
`examples/pbmc3k/outputs/`.

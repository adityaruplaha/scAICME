"""
Pytest configuration and fixtures for scAICME test suite.

Provides reproducible synthetic AnnData objects with injected biological signals
for testing all labeling strategies.
"""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture
def synthetic_adata():
    """
    Create a 1000 cells x 80 genes synthetic AnnData object with injected signals for 8 classes.

    Classes with partially overlapping marker genes (125 cells per class):
    - Class A (cells 0-124): genes 0-4
    - Class B (cells 125-249): genes 4-8 (overlaps with A at gene_4)
    - Class C (cells 250-374): genes 8-12
    - Class D (cells 375-499): genes 12-16 (overlaps with C at gene_12)
    - Class E (cells 500-624): genes 16-20
    - Class F (cells 625-749): genes 20-24 (overlaps with E at gene_20)
    - Class G (cells 750-874): genes 24-28
    - Class H (cells 875-999): genes 28-32 (overlaps with G at gene_28)

    The object includes standard preprocessing: normalization, log1p, PCA, and
    neighborhood graph computation.
    """
    np.random.seed(42)
    n_cells, n_genes = 1000, 80

    # Initialize with low baseline Poisson noise
    X = np.random.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Inject Class A signal in genes 0-4 (cells 0-124)
    X[0:125, 0:5] += np.random.poisson(lam=5.0, size=(125, 5)).astype(np.float32)

    # Inject Class B signal in genes 4-8 (cells 125-249, overlaps with A)
    X[125:250, 4:9] += np.random.poisson(lam=4.5, size=(125, 5)).astype(np.float32)

    # Inject Class C signal in genes 8-12 (cells 250-374)
    X[250:375, 8:13] += np.random.poisson(lam=5.0, size=(125, 5)).astype(np.float32)

    # Inject Class D signal in genes 12-16 (cells 375-499, overlaps with C)
    X[375:500, 12:17] += np.random.poisson(lam=4.5, size=(125, 5)).astype(np.float32)

    # Inject Class E signal in genes 16-20 (cells 500-624)
    X[500:625, 16:21] += np.random.poisson(lam=5.0, size=(125, 5)).astype(np.float32)

    # Inject Class F signal in genes 20-24 (cells 625-749, overlaps with E)
    X[625:750, 20:25] += np.random.poisson(lam=4.5, size=(125, 5)).astype(np.float32)

    # Inject Class G signal in genes 24-28 (cells 750-874)
    X[750:875, 24:29] += np.random.poisson(lam=5.0, size=(125, 5)).astype(np.float32)

    # Inject Class H signal in genes 28-32 (cells 875-999, overlaps with G)
    X[875:1000, 28:33] += np.random.poisson(lam=4.5, size=(125, 5)).astype(np.float32)

    # Create AnnData object
    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Preprocessing: normalization, log transform, PCA, and neighborhood graph
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=42)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=42)

    # Store raw for score calculation
    adata.raw = adata

    return adata


@pytest.fixture
def marker_dict():
    """
    Create a marker gene dictionary for 8 classes with partially overlapping genes.

    Classes A and B: overlapping on gene_2
    Classes C and D: overlapping on gene_6
    Classes E and F: overlapping on gene_10
    Classes G and H: overlapping on gene_14
    """
    return {
        "Class A": ["gene_0", "gene_1", "gene_2"],
        "Class B": ["gene_2", "gene_3", "gene_4"],
        "Class C": ["gene_4", "gene_5", "gene_6"],
        "Class D": ["gene_6", "gene_7", "gene_8"],
        "Class E": ["gene_8", "gene_9", "gene_10"],
        "Class F": ["gene_10", "gene_11", "gene_12"],
        "Class G": ["gene_12", "gene_13", "gene_14"],
        "Class H": ["gene_14", "gene_15", "gene_16"],
    }


@pytest.fixture
def synthetic_adata_with_seeds(synthetic_adata):
    """
    Create a synthetic AnnData with pre-labeled seed cells.

    Seeds (cells 0-5): labeled as "Class A"
    Seeds (cells 125-130): labeled as "Class B"
    All other cells: labeled as "unknown"
    """
    adata = synthetic_adata.copy()

    # Initialize labels
    labels = pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

    # Add Class A seeds (cells 0-5)
    labels.iloc[0:6] = "Class A"

    # Add Class B seeds (cells 125-130)
    labels.iloc[125:131] = "Class B"

    adata.obs["seed_labels"] = labels

    return adata


@pytest.fixture
def synthetic_adata_with_multi_class_seeds(synthetic_adata):
    """
    Create a synthetic AnnData with pre-labeled seed cells from multiple classes.

    Seeds (cells 0-5): labeled as "Class A"
    Seeds (cells 125-130): labeled as "Class B"
    Seeds (cells 250-255): labeled as "Class C"
    Seeds (cells 375-380): labeled as "Class D"
    All other cells: labeled as "unknown"

    Used for testing classifiers that require multiple classes (e.g., NearestCentroid).
    """
    adata = synthetic_adata.copy()

    # Initialize labels
    labels = pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)

    # Add Class A seeds (cells 0-5)
    labels.iloc[0:6] = "Class A"

    # Add Class B seeds (cells 125-130)
    labels.iloc[125:131] = "Class B"

    # Add Class C seeds (cells 250-255)
    labels.iloc[250:256] = "Class C"

    # Add Class D seeds (cells 375-380)
    labels.iloc[375:381] = "Class D"

    adata.obs["seed_labels_multi"] = labels

    return adata


@pytest.fixture
def adata_ubiquitous_shared():
    """
    Create AnnData with ubiquitous (house-keeping) genes shared across ALL cell types.

    Structure:
    - Genes 0-4: Ubiquitous housekeeping genes (present in all 8 classes)
    - Genes 5-7: Class A-specific
    - Genes 8-10: Class B-specific
    etc.
    """
    np.random.seed(43)
    n_cells, n_genes = 1000, 80

    X = np.random.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)
    X[:, 0:5] += np.random.poisson(lam=6.0, size=(n_cells, 5)).astype(np.float32)

    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 5),
            (125, 250, 8),
            (250, 375, 11),
            (375, 500, 14),
            (500, 625, 17),
            (625, 750, 20),
            (750, 875, 23),
            (875, 1000, 26),
        ]
    ):
        gene_end = gene_start + 3
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=5.0, size=(end_cell - start_cell, 3)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=43)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=43)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_ubiquitous():
    """Marker dict with ubiquitous genes shared by all + specific genes."""
    ubiquitous = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]
    return {
        "Class A": ubiquitous + ["gene_5", "gene_6", "gene_7"],
        "Class B": ubiquitous + ["gene_8", "gene_9", "gene_10"],
        "Class C": ubiquitous + ["gene_11", "gene_12", "gene_13"],
        "Class D": ubiquitous + ["gene_14", "gene_15", "gene_16"],
        "Class E": ubiquitous + ["gene_17", "gene_18", "gene_19"],
        "Class F": ubiquitous + ["gene_20", "gene_21", "gene_22"],
        "Class G": ubiquitous + ["gene_23", "gene_24", "gene_25"],
        "Class H": ubiquitous + ["gene_26", "gene_27", "gene_28"],
    }


@pytest.fixture
def adata_highly_specific():
    """Create AnnData with completely non-overlapping marker genes per class."""
    np.random.seed(44)
    n_cells, n_genes = 1000, 80
    X = np.random.poisson(lam=0.5, size=(n_cells, n_genes)).astype(np.float32)

    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 0),
            (125, 250, 4),
            (250, 375, 8),
            (375, 500, 12),
            (500, 625, 16),
            (625, 750, 20),
            (750, 875, 24),
            (875, 1000, 28),
        ]
    ):
        gene_end = gene_start + 4
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=6.0, size=(end_cell - start_cell, 4)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=44)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=44)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_highly_specific():
    """Marker dict with completely non-overlapping genes."""
    return {
        "Class A": ["gene_0", "gene_1", "gene_2", "gene_3"],
        "Class B": ["gene_4", "gene_5", "gene_6", "gene_7"],
        "Class C": ["gene_8", "gene_9", "gene_10", "gene_11"],
        "Class D": ["gene_12", "gene_13", "gene_14", "gene_15"],
        "Class E": ["gene_16", "gene_17", "gene_18", "gene_19"],
        "Class F": ["gene_20", "gene_21", "gene_22", "gene_23"],
        "Class G": ["gene_24", "gene_25", "gene_26", "gene_27"],
        "Class H": ["gene_28", "gene_29", "gene_30", "gene_31"],
    }


@pytest.fixture
def adata_hierarchical_overlap():
    """Create AnnData with hierarchical overlap structure."""
    np.random.seed(45)
    n_cells, n_genes = 1000, 80
    X = np.random.poisson(lam=0.3, size=(n_cells, n_genes)).astype(np.float32)

    X[:, 0:3] += np.random.poisson(lam=3.0, size=(n_cells, 3)).astype(np.float32)
    X[0:500, 3:6] += np.random.poisson(lam=3.0, size=(500, 3)).astype(np.float32)
    X[500:1000, 6:9] += np.random.poisson(lam=3.0, size=(500, 3)).astype(np.float32)
    X[0:250, 9:12] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)
    X[250:500, 12:15] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)
    X[500:750, 15:18] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)
    X[750:1000, 18:21] += np.random.poisson(lam=3.0, size=(250, 3)).astype(np.float32)

    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 21),
            (125, 250, 24),
            (250, 375, 27),
            (375, 500, 30),
            (500, 625, 33),
            (625, 750, 36),
            (750, 875, 39),
            (875, 1000, 42),
        ]
    ):
        gene_end = gene_start + 3
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=5.0, size=(end_cell - start_cell, 3)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=45)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=45)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_hierarchical():
    """Marker dict with hierarchical overlap structure."""
    return {
        "Class A": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_21",
            "gene_22",
            "gene_23",
        ],
        "Class B": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_24",
            "gene_25",
            "gene_26",
        ],
        "Class C": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_27",
            "gene_28",
            "gene_29",
        ],
        "Class D": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_30",
            "gene_31",
            "gene_32",
        ],
        "Class E": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_15",
            "gene_16",
            "gene_17",
            "gene_33",
            "gene_34",
            "gene_35",
        ],
        "Class F": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_15",
            "gene_16",
            "gene_17",
            "gene_36",
            "gene_37",
            "gene_38",
        ],
        "Class G": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_18",
            "gene_19",
            "gene_20",
            "gene_39",
            "gene_40",
            "gene_41",
        ],
        "Class H": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_18",
            "gene_19",
            "gene_20",
            "gene_42",
            "gene_43",
            "gene_44",
        ],
    }


@pytest.fixture
def adata_complex_mixed():
    """Create AnnData with complex mixed overlap patterns."""
    np.random.seed(46)
    n_cells, n_genes = 1000, 80
    X = np.random.poisson(lam=0.4, size=(n_cells, n_genes)).astype(np.float32)

    X[:, 0:3] += np.random.poisson(lam=4.0, size=(n_cells, 3)).astype(np.float32)
    background_mask = np.random.rand(n_cells) > 0.3
    X[background_mask, 3:6] += np.random.poisson(lam=2.0, size=(background_mask.sum(), 3)).astype(
        np.float32
    )
    X[np.concatenate([np.arange(0, 250), np.arange(500, 750)]), 6:9] += np.random.poisson(
        lam=3.0, size=(500, 3)
    ).astype(np.float32)

    for _, (start, end) in enumerate([(0, 250), (250, 500), (500, 750), (750, 1000)]):
        X[start:end, 9:12] += np.random.poisson(lam=3.0, size=(end - start, 3)).astype(np.float32)

    X[0:375, 12:15] += np.random.poisson(lam=2.0, size=(375, 3)).astype(np.float32)

    for _, (start_cell, end_cell, gene_start) in enumerate(
        [
            (0, 125, 15),
            (125, 250, 18),
            (250, 375, 21),
            (375, 500, 24),
            (500, 625, 27),
            (625, 750, 30),
            (750, 875, 33),
            (875, 1000, 36),
        ]
    ):
        gene_end = gene_start + 3
        X[start_cell:end_cell, gene_start:gene_end] += np.random.poisson(
            lam=5.5, size=(end_cell - start_cell, 3)
        ).astype(np.float32)

    adata = AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=10, random_state=46)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=46)
    adata.raw = adata
    return adata


@pytest.fixture
def marker_dict_complex_mixed():
    """Marker dict with complex mixed overlap patterns."""
    return {
        "Class A": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_15",
            "gene_16",
            "gene_17",
        ],
        "Class B": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_18",
            "gene_19",
            "gene_20",
        ],
        "Class C": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_12",
            "gene_13",
            "gene_14",
            "gene_21",
            "gene_22",
            "gene_23",
        ],
        "Class D": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_24",
            "gene_25",
            "gene_26",
        ],
        "Class E": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_27",
            "gene_28",
            "gene_29",
        ],
        "Class F": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_5",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_30",
            "gene_31",
            "gene_32",
        ],
        "Class G": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_7",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_33",
            "gene_34",
            "gene_35",
        ],
        "Class H": [
            "gene_0",
            "gene_1",
            "gene_2",
            "gene_3",
            "gene_4",
            "gene_5",
            "gene_6",
            "gene_8",
            "gene_9",
            "gene_10",
            "gene_11",
            "gene_36",
            "gene_37",
            "gene_38",
        ],
    }

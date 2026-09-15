"""
Parity tests against the verbatim PBMC 68k notebook implementation.

Each stage of the notebook is run with its settings via the package and compared,
label for label, with the notebook function from ``reference_pbmc68k_notebook`` on the
same synthetic data. Exact equality is intended.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scaicme import evaluation, strategies, tl

sys.path.insert(0, str(Path(__file__).parent))
import reference_pbmc68k_notebook as ref  # noqa: E402
from conftest import MARKER_DICT, build_synthetic_adata  # noqa: E402

UNLABELED = "unlabeled"
RANDOM_STATE = 42

# Notebook settings (cells 7 and 14); the size floor is lowered for the 1000-cell fixture.
QUOTA = {"gene_quantile": 0.6, "target_frac": 0.55, "min_cells_per_type": 50, "min_score": 0.2}
GCN = {
    "target_frac": 0.55,
    "min_score": 0.2,
    "delta": 0.15,
    "min_cells_per_type_pct": 0.005,
    "min_cells_floor": 20,
}


@pytest.fixture(scope="module")
def base():
    """Synthetic data with the graph the notebook's GCN would build (PCA 30 -> 10 here)."""
    adata = build_synthetic_adata()
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=adata.obsm["X_pca"].shape[1])
    return adata


@pytest.fixture(scope="module")
def seeded(base):
    """Method 1 + Method 3 via the package, and via the notebook functions."""
    markers = {k: list(v) for k, v in MARKER_DICT.items()}

    pkg = base.copy()
    quota = strategies.QCQAdaptiveSeeding(
        markers=markers,
        quantile=QUOTA["gene_quantile"],
        min_confidence=QUOTA["min_score"],
        target_frac=QUOTA["target_frac"],
        min_cells_per_type=QUOTA["min_cells_per_type"],
        min_score=QUOTA["min_score"],
        unknown_label=UNLABELED,
    )
    tl.label(pkg, quota, key_added="weak_label_quota")
    gcn = strategies.GCNSeeding(
        markers=markers,
        initial_scores_key="weak_label_quota_scores",
        unknown_label=UNLABELED,
        **GCN,
    )
    tl.label(pkg, gcn, key_added="weak_label")

    nb = base.copy()
    nb, score_df = ref.weak_label_quota_with_min_cells(nb, markers, verbose=False, **QUOTA)
    nb.obs["weak_label_quota"] = nb.obs["weak_label"].astype(str)
    ref.gcn_seed_labeling(
        nb, score_df=score_df, label_col="weak_label", conf_col="seed_conf", verbose=False, **GCN
    )
    return pkg, nb, score_df


class TestQuotaSeedingParity:
    def test_scores_identical(self, seeded):
        pkg, _, score_df = seeded
        np.testing.assert_array_equal(
            pkg.obsm["weak_label_quota_scores"], score_df.to_numpy(dtype=float)
        )

    def test_labels_identical(self, seeded):
        pkg, nb, _ = seeded
        a = pkg.obs["weak_label_quota"].astype(str).values
        b = nb.obs["weak_label_quota"].astype(str).values
        assert (a == b).all()
        assert (a != UNLABELED).sum() > 100  # non-vacuous

    def test_confidence_identical(self, seeded):
        pkg, nb, _ = seeded
        np.testing.assert_array_equal(
            pkg.obs["weak_label_quota_max_score"].values, nb.obs["weak_conf"].values
        )

    def test_quota_respects_budget_and_floor(self, seeded):
        pkg, _, _ = seeded
        counts = pkg.obs["weak_label_quota"].value_counts().drop(UNLABELED, errors="ignore")
        assert (counts >= QUOTA["min_cells_per_type"]).all()
        assert counts.sum() <= round(QUOTA["target_frac"] * pkg.n_obs)


class TestGCNSeedingParity:
    def test_labels_identical(self, seeded):
        pkg, nb, _ = seeded
        a = pkg.obs["weak_label"].astype(str).values
        b = nb.obs["weak_label"].astype(str).values
        assert (a == b).all()
        assert (a != UNLABELED).sum() > 50
        assert len(set(a) - {UNLABELED}) >= 2

    def test_confidence_identical(self, seeded):
        pkg, nb, _ = seeded
        np.testing.assert_array_equal(
            pkg.obs["weak_label_max_score"].values, nb.obs["seed_conf"].values
        )


class TestPropagationParity:
    @pytest.fixture(scope="class")
    def propagated(self, seeded):
        pkg, _, _ = seeded
        pkg = pkg.copy()
        nb = pkg.copy()

        # Cell 15 inputs for the notebook functions.
        X = nb.obsm["X_pca"]
        labels_seed = nb.obs["weak_label"].astype(str).values
        known_mask = labels_seed != UNLABELED
        known_types = np.unique(labels_seed[known_mask])
        n_types = len(known_types)

        common = {"seed_key": "weak_label", "unknown_label": UNLABELED}
        methods = {
            "label_svm_rbf": strategies.SVMPropagation(
                kernel="rbf",
                c=5.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                scale_features=True,
                keep_seeds=True,
                min_conf=0.55,
                random_state=RANDOM_STATE,
                **common,
            ),
            "label_kmeans": strategies.KMeansPropagation(
                n_clusters=n_types,
                n_init="auto",
                max_iter=300,
                scale_features=False,
                keep_seeds=False,
                min_conf=0.0,
                random_state=RANDOM_STATE,
                **common,
            ),
            "label_knn": strategies.KNNPropagation(
                n_neighbors=3, weights="distance", p=1, keep_seeds=False, min_conf=0.0, **common
            ),
            "label_rf": strategies.RandomForestPropagation(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced_subsample",
                n_jobs=-1,
                keep_seeds=False,
                min_conf=0.0,
                random_state=RANDOM_STATE,
                **common,
            ),
            "label_mlp": strategies.NeuralNetworkPropagation(
                hidden_layer_sizes=(256, 128),
                alpha=1e-4,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                scale_features=True,
                keep_seeds=True,
                min_conf=0.55,
                random_state=RANDOM_STATE,
                **common,
            ),
        }
        results = tl.label(pkg, strategies=methods, n_jobs=1)
        assert set(results) == set(methods)

        ref.rbf_svm_labeling(nb, seed_col="weak_label", min_conf=0.55, random_state=RANDOM_STATE)
        nb.obs["label_kmeans"] = pd.Categorical(
            ref.kmeans_labeling(X, labels_seed, known_mask, known_types)
        )
        nb.obs["label_knn"] = pd.Categorical(ref.knn_labeling(X, labels_seed, known_mask))
        nb.obs["label_rf"] = pd.Categorical(ref.rf_labeling(X, labels_seed, known_mask))
        ref.mlp_labeling(
            nb,
            seed_col="weak_label",
            out_col="label_mlp",
            out_conf_col="mlp_conf",
            hidden_layer_sizes=(256, 128),
            min_conf=0.55,
            random_state=RANDOM_STATE,
        )
        return pkg, nb

    @pytest.mark.parametrize(
        "key", ["label_svm_rbf", "label_kmeans", "label_knn", "label_rf", "label_mlp"]
    )
    def test_method_labels_identical(self, propagated, key):
        pkg, nb = propagated
        assert (pkg.obs[key].astype(str).values == nb.obs[key].astype(str).values).all()

    def test_consensus_and_metrics_identical(self, propagated):
        pkg, nb = propagated
        keys = ["label_svm_rbf", "label_kmeans", "label_knn", "label_rf", "label_mlp"]
        consensus = strategies.ConsensusVoting(
            keys=keys, majority_fraction=None, fraction_of="all", unknown_label=UNLABELED
        )
        tl.label(pkg, consensus, key_added="label_consensus")
        expected, agree = ref.notebook_consensus(nb)
        assert (pkg.obs["label_consensus"].values == expected).all()
        np.testing.assert_array_equal(pkg.obs["label_consensus_agreement_fraction"].values, agree)

        # Cell 32: metrics against a reference column (use the KNN labels as a stand-in).
        nb.obs["label_consensus"] = pd.Categorical(expected)
        got = evaluation.compare_labels(
            pkg, "label_consensus", "label_knn", ignore_labels=(UNLABELED, "Unknown")
        )
        want = ref.compare_labels(
            nb, "label_consensus", "label_knn", ignore_labels=(UNLABELED, "Unknown")
        )
        assert got.keys() == want.keys()
        for k in want:
            if isinstance(want[k], float):
                assert got[k] == pytest.approx(want[k], nan_ok=True), k
            else:
                assert got[k] == want[k], k

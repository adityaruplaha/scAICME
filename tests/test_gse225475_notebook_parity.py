"""
Parity tests against the verbatim GSE225475 spatial notebook implementation.

Each package strategy is run with the notebook's settings and compared, label for
label, with the corresponding notebook function from ``reference_gse225475_notebook``
on the same synthetic data. These are exact-equality tests by design: the package
is meant to reproduce the notebook, not approximate it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from scaicme import strategies, tl

sys.path.insert(0, str(Path(__file__).parent))
import reference_gse225475_notebook as ref  # noqa: E402
from conftest import MARKER_DICT, build_synthetic_adata  # noqa: E402

UNLABELED = "unlabeled"
SEED_KEY = "weak_label"
RANDOM_STATE = 42

# Notebook cell 5 settings.
SEEDING = {
    "n_components": 15,
    "weight_concentration_prior": 0.1,
    "per_gene_pos_quantile": 0.3,
    "cluster_score_min": 0.08,
    "min_cells_cluster": 30,
    "random_state": RANDOM_STATE,
}
# Notebook cell 6 settings shared by the classifiers.
COMMON = {
    "seed_key": SEED_KEY,
    "unknown_label": UNLABELED,
    "keep_seeds": False,
    "min_seed_conf": 0.05,
}
MIN_CONF = 0.3
MAX_PCS = 15


def _package_methods():
    """The five notebook classifiers as package strategies (cell 6 settings)."""
    return {
        "label_svm_rbf": strategies.SVMPropagation(
            kernel="rbf",
            c=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            scale_features=True,
            min_conf=MIN_CONF,
            max_pcs=MAX_PCS,
            random_state=RANDOM_STATE,
            **COMMON,
        ),
        "label_kmeans": strategies.KMeansPropagation(
            n_clusters=None,
            n_init=20,
            max_iter=500,
            scale_features=False,
            min_conf=0.0,
            max_pcs=30,
            random_state=RANDOM_STATE,
            **COMMON,
        ),
        "label_knn": strategies.KNNPropagation(
            n_neighbors=9, weights="distance", min_conf=MIN_CONF, max_pcs=MAX_PCS, **COMMON
        ),
        "label_rf": strategies.RandomForestPropagation(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_conf=MIN_CONF,
            max_pcs=MAX_PCS,
            random_state=RANDOM_STATE,
            **COMMON,
        ),
        "label_mlp": strategies.NeuralNetworkPropagation(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            scale_features=True,
            min_conf=MIN_CONF,
            max_pcs=MAX_PCS,
            random_state=RANDOM_STATE,
            **COMMON,
        ),
    }


REFERENCE_METHODS = {
    "label_svm_rbf": ref.rbf_svm_labeling,
    "label_kmeans": ref.kmeans_seed_transfer,
    "label_knn": ref.knn_labeling,
    "label_rf": ref.rf_labeling,
    "label_mlp": ref.mlp_labeling,
}


@pytest.fixture(scope="module")
def synthetic_adata():
    """Module-scoped synthetic data: parity fixtures are expensive, build once."""
    return build_synthetic_adata()


@pytest.fixture(scope="module")
def marker_dict():
    return {k: list(v) for k, v in MARKER_DICT.items()}


@pytest.fixture(scope="module")
def seeded(synthetic_adata, marker_dict):
    """Package-seeded AnnData plus a copy seeded by the notebook function."""
    pkg = synthetic_adata.copy()
    seeder = strategies.DPGMMSeeding(markers=marker_dict, unknown_label=UNLABELED, **SEEDING)
    tl.label(pkg, seeder, key_added=SEED_KEY)

    nb = synthetic_adata.copy()
    ref.dp_seed_by_marker_sets_soft(nb, marker_dict, verbose=False, **SEEDING)
    return pkg, nb


class TestSeedingParity:
    def test_labels_identical(self, seeded):
        pkg, nb = seeded
        assert (nb.obs["weak_label"].astype(str).values == pkg.obs[SEED_KEY].values).all()
        assert (pkg.obs[SEED_KEY] != UNLABELED).sum() > 100  # the comparison is not vacuous

    def test_confidence_identical(self, seeded):
        pkg, nb = seeded
        np.testing.assert_array_equal(
            nb.obs["weak_conf"].values, pkg.obs[f"{SEED_KEY}_max_score"].values
        )

    def test_per_type_confidence_identical(self, seeded, marker_dict):
        pkg, nb = seeded
        scores = pkg.obsm[f"{SEED_KEY}_scores"]
        for j, ctype in enumerate(marker_dict):
            col = f"dp_seed_conf_{ctype}"
            if col in nb.obs:
                np.testing.assert_array_equal(nb.obs[col].values, scores[:, j])
            else:  # type skipped by the notebook -> zero confidence in the package
                assert not scores[:, j].any()


class TestPropagationParity:
    @pytest.fixture(scope="class")
    def propagated(self, seeded):
        pkg, _ = seeded
        pkg = pkg.copy()
        nb = pkg.copy()
        nb.obs["weak_conf"] = nb.obs[f"{SEED_KEY}_max_score"]

        methods = _package_methods()
        results = tl.label(pkg, strategies=methods, n_jobs=1)
        assert set(results) == set(methods)
        for fn in REFERENCE_METHODS.values():
            fn(nb)
        return pkg, nb

    @pytest.mark.parametrize("key", list(REFERENCE_METHODS))
    def test_method_labels_identical(self, propagated, key):
        pkg, nb = propagated
        assert (pkg.obs[key].astype(str).values == nb.obs[key].astype(str).values).all()

    def test_consensus_identical(self, propagated):
        pkg, nb = propagated
        keys = list(REFERENCE_METHODS)
        consensus = strategies.ConsensusVoting(
            keys=keys, majority_fraction=None, fraction_of="all", unknown_label=UNLABELED
        )
        tl.label(pkg, consensus, key_added="label_consensus")
        expected_labels, expected_agree = ref.notebook_consensus(nb, keys)
        assert (pkg.obs["label_consensus"].values == expected_labels).all()
        np.testing.assert_array_equal(
            pkg.obs["label_consensus_agreement_fraction"].values, expected_agree
        )

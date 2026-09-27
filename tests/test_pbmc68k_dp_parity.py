"""
Parity tests against the verbatim final PBMC 68k notebook implementation.

That notebook's saved run seeds with Method 2 (DP-GMM on marker sets, no PCA) and
propagates with SVM, k-means, KNN, random forest and MLP before a plurality consensus.
Each stage is run through the package with the notebook's settings and compared, label
for label, with the notebook function from ``reference_pbmc68k_dp_notebook``.

The settings below are the notebook's except for `min_cells_cluster`, lowered from 100
to 30 so the 1000-cell synthetic fixture yields seeds at all; both sides receive the
same value, so the comparison is unaffected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from scaicme import evaluation, strategies, tl

sys.path.insert(0, str(Path(__file__).parent))
import reference_pbmc68k_dp_notebook as ref  # noqa: E402
from conftest import MARKER_DICT, build_synthetic_adata  # noqa: E402

UNLABELED = "unlabeled"
SEED_KEY = "weak_label"
RANDOM_STATE = 42
MAX_PCS = 30

# Cell 4's call, with the fixture-sized cluster floor noted in the module docstring.
DP = {
    "per_gene_pos_quantile": 0.4,
    "cluster_score_min": 0.1,
    "min_cells_cluster": 30,
    "weight_concentration_prior": 0.05,
    "random_state": RANDOM_STATE,
}
# The notebook's post-hoc size filter: max(100, int(0.002 * n_cells)).
SIZE_FLOOR = {"min_type_size": 100, "min_type_frac": 0.002}


def _n_components(n_obs: int) -> int:
    return min(100, int(np.sqrt(n_obs)))


def _n_clusters(n_obs: int) -> int:
    return min(30, max(10, int(np.sqrt(n_obs / 2))))


@pytest.fixture(scope="module")
def seeded():
    """DP-seeded AnnData via the package, and a copy seeded by the notebook function."""
    base = build_synthetic_adata()
    markers = {k: list(v) for k, v in MARKER_DICT.items()}
    n_components = _n_components(base.n_obs)

    pkg = base.copy()
    seeder = strategies.DPGMMSeeding(
        markers=markers,
        n_components=n_components,
        min_cell_enrichment=0.05,
        use_raw=False,
        unknown_label=UNLABELED,
        n_jobs=1,
        **DP,
        **SIZE_FLOOR,
    )
    tl.label(pkg, seeder, key_added=SEED_KEY)

    nb = base.copy()
    ref.dp_seed_by_marker_sets_soft_no_pca(
        nb, markers, n_components=n_components, verbose=False, **DP
    )
    return pkg, nb


class TestSeedingParity:
    def test_labels_identical(self, seeded):
        pkg, nb = seeded
        got = pkg.obs[SEED_KEY].astype(str).to_numpy()
        want = nb.obs["weak_label"].astype(str).to_numpy()
        assert (got == want).all()
        assert (got != UNLABELED).sum() > 200  # the comparison is not vacuous
        assert len(set(got) - {UNLABELED}) >= 2

    def test_confidence_identical(self, seeded):
        pkg, nb = seeded
        np.testing.assert_array_equal(
            pkg.obs[f"{SEED_KEY}_max_score"].to_numpy(), nb.obs["weak_conf"].to_numpy()
        )

    def test_size_floor_dropped_small_types(self, seeded):
        pkg, _ = seeded
        uns = pkg.uns[f"{SEED_KEY}_uns"]
        assert uns["size_floor"] == max(100, int(0.002 * pkg.n_obs))
        assert uns["dropped_types"], "expected the floor to drop at least one type"


class TestPropagationParity:
    @pytest.fixture(scope="class")
    def propagated(self, seeded):
        pkg, _ = seeded
        pkg = pkg.copy()
        nb = pkg.copy()
        # The notebook functions read the confidence from "weak_conf".
        nb.obs["weak_conf"] = nb.obs[f"{SEED_KEY}_max_score"]

        common = {"seed_key": SEED_KEY, "unknown_label": UNLABELED, "keep_seeds": False}
        methods = {
            "label_svm_rbf": strategies.SVMPropagation(
                kernel="rbf",
                c=2.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                scale_features=True,
                min_seed_conf=0.30,
                min_conf=0.60,
                max_pcs=MAX_PCS,
                random_state=RANDOM_STATE,
                **common,
            ),
            # The vote uses every labelled seed; only the seedless-cluster fallback is
            # restricted to confident seeds, matching the notebook's two masks.
            "label_kmeans": strategies.KMeansPropagation(
                n_clusters=_n_clusters(pkg.n_obs),
                n_init=20,
                max_iter=500,
                scale_features=False,
                min_seed_conf=0.0,
                fallback_min_seed_conf=0.3,
                min_conf=0.0,
                max_pcs=MAX_PCS,
                random_state=RANDOM_STATE,
                **common,
            ),
            "label_knn": strategies.KNNPropagation(
                n_neighbors=9,
                weights="distance",
                p=2,
                min_seed_conf=0.30,
                min_conf=0.55,
                max_pcs=MAX_PCS,
                **common,
            ),
            "label_rf": strategies.RandomForestPropagation(
                n_estimators=500,
                max_depth=18,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                min_seed_conf=0.30,
                min_conf=0.55,
                max_pcs=MAX_PCS,
                random_state=RANDOM_STATE,
                **common,
            ),
            "label_mlp": strategies.NeuralNetworkPropagation(
                hidden_layer_sizes=(128, 64),
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                scale_features=True,
                min_seed_conf=0.30,
                min_conf=0.60,
                max_pcs=MAX_PCS,
                random_state=RANDOM_STATE,
                **common,
            ),
        }
        results = tl.label(pkg, strategies=methods, n_jobs=1)
        assert set(results) == set(methods)

        ref.rbf_svm_labeling(nb, min_seed_conf=0.30, min_conf=0.60, max_pcs=MAX_PCS)
        labels_out, conf_out = ref.kmeans_labeling(nb)
        nb.obs["label_kmeans"] = labels_out
        nb.obs["kmeans_conf"] = conf_out
        ref.knn_labeling(nb, n_neighbors=9, min_seed_conf=0.30, min_conf=0.55, max_pcs=MAX_PCS)
        ref.rf_labeling(nb, min_seed_conf=0.30, min_conf=0.55, max_pcs=MAX_PCS)
        ref.mlp_labeling(
            nb,
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            min_seed_conf=0.30,
            min_conf=0.60,
            max_pcs=MAX_PCS,
            random_state=RANDOM_STATE,
        )
        return pkg, nb

    @pytest.mark.parametrize(
        "key", ["label_svm_rbf", "label_kmeans", "label_knn", "label_rf", "label_mlp"]
    )
    def test_method_labels_identical(self, propagated, key):
        pkg, nb = propagated
        assert (pkg.obs[key].astype(str).to_numpy() == nb.obs[key].astype(str).to_numpy()).all()

    def test_kmeans_confidence_identical(self, propagated):
        pkg, nb = propagated
        np.testing.assert_allclose(
            pkg.obs["label_kmeans_confidence"].to_numpy(), nb.obs["kmeans_conf"].to_numpy()
        )

    def test_consensus_identical(self, propagated):
        pkg, nb = propagated
        keys = ["label_svm_rbf", "label_kmeans", "label_knn", "label_rf", "label_mlp"]
        consensus = strategies.ConsensusVoting(
            keys=keys, majority_fraction=None, fraction_of="all", unknown_label=UNLABELED
        )
        tl.label(pkg, consensus, key_added="label_consensus")
        expected, agree = ref.notebook_consensus(nb)
        assert (pkg.obs["label_consensus"].to_numpy() == expected).all()
        np.testing.assert_array_equal(
            pkg.obs["label_consensus_agreement_fraction"].to_numpy(), agree
        )


class TestMetricsParity:
    """The package reports the notebook's metrics under its own key names."""

    KEY_MAP = {
        "ARI": "ARI",
        "Accuracy": "acc",
        "Specificity_macro": "specificity_macro",
        "Sensitivity_macro": "sensitivity_macro",
        "n_eval": "n_eval",
    }

    def test_compare_labels_identical(self, seeded):
        pkg, _ = seeded
        # Compare the seeds against the injected class structure of the fixture.
        truth = np.repeat([f"Class {c}" for c in "ABCDEFGH"], pkg.n_obs // 8)
        pkg.obs["cell_type_true"] = truth
        got = evaluation.compare_labels(
            pkg, SEED_KEY, "cell_type_true", ignore_labels=(UNLABELED, "Unknown")
        )
        want = ref.compare_labels_ari_acc_spec_sens(
            pkg, SEED_KEY, "cell_type_true", ignore_labels=(UNLABELED, "Unknown")
        )
        for shared in ("coverage_pred", "coverage_ref", "coverage_both", "n_total"):
            assert got[shared] == pytest.approx(want[shared]), shared
        for prefix in ("all", "labeled"):
            for nb_key, pkg_key in self.KEY_MAP.items():
                assert got[f"{prefix}_{pkg_key}"] == pytest.approx(
                    want[f"{prefix}_{nb_key}"], nan_ok=True
                ), f"{prefix}_{nb_key}"

    def test_cluster_quality_identical(self, seeded):
        pkg, _ = seeded
        got = evaluation.cluster_quality(pkg, SEED_KEY)
        want = ref.clustering_metrics(pkg, SEED_KEY)
        assert got["method"] == want["method"]
        for k in ("silhouette", "calinski_harabasz", "davies_bouldin"):
            assert got[k] == pytest.approx(want[k], nan_ok=True), k

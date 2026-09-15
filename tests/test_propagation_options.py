"""Tests for propagation, consensus, preprocessing and result-writing options added for
notebook parity (seed-purity K-Means, extra RF/MLP knobs, plurality consensus,
scaicme.pp, h5ad-safe parameters)."""

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from scaicme import pp, strategies, tl


class TestKMeansPropagation:
    def test_default_n_clusters_heuristic(self, synthetic_adata_with_multi_class_seeds):
        adata = synthetic_adata_with_multi_class_seeds
        strategy = strategies.KMeansPropagation(seed_key="seed_labels_multi", random_state=42)
        tl.label(adata, strategy, key_added="km")
        expected = min(10, max(8, int(np.sqrt(adata.n_obs / 2))))
        assert adata.uns["km_uns"]["n_clusters"] == expected
        assert adata.obs["km_kmeans_cluster"].nunique() <= expected

    def test_confidence_is_seed_purity(self, synthetic_adata_with_multi_class_seeds):
        adata = synthetic_adata_with_multi_class_seeds
        strategy = strategies.KMeansPropagation(
            seed_key="seed_labels_multi", n_clusters=8, keep_seeds=False, random_state=42
        )
        tl.label(adata, strategy, key_added="km")
        conf = adata.obs["km_confidence"]
        clusters = adata.obs["km_kmeans_cluster"]
        seeds = adata.obs["seed_labels_multi"]
        assert conf.between(0, 1).all()
        for k in clusters.unique():
            members = clusters == k
            seeds_in_k = seeds[members & (seeds != "unknown")]
            if len(seeds_in_k):
                top = seeds_in_k.value_counts().iloc[0]
                assert np.isclose(conf[members].iloc[0], top / members.sum())
                assert adata.obs.loc[members, "km"].iloc[0] == seeds_in_k.value_counts().index[0]
            else:
                assert (conf[members] == 0).all()

    def test_seedless_clusters_fall_back_to_nearest_class(
        self, synthetic_adata_with_multi_class_seeds
    ):
        adata = synthetic_adata_with_multi_class_seeds
        # Many clusters relative to 24 seeds guarantees seedless clusters.
        strategy = strategies.KMeansPropagation(
            seed_key="seed_labels_multi", n_clusters=40, keep_seeds=False, random_state=42
        )
        tl.label(adata, strategy, key_added="km")
        seedless = adata.obs["km_confidence"] == 0
        assert seedless.any()
        # Seedless clusters still get a known class (nearest seed centroid), never unknown.
        assert "unknown" not in adata.obs["km"].values
        assert set(adata.obs.loc[seedless, "km"]) <= {"Class A", "Class B", "Class C", "Class D"}

    def test_min_conf_rejects_low_purity(self, synthetic_adata_with_multi_class_seeds):
        adata = synthetic_adata_with_multi_class_seeds
        strategy = strategies.KMeansPropagation(
            seed_key="seed_labels_multi",
            n_clusters=40,
            keep_seeds=False,
            min_conf=0.01,
            random_state=42,
        )
        tl.label(adata, strategy, key_added="km")
        assert (adata.obs.loc[adata.obs["km_confidence"] < 0.01, "km"] == "unknown").all()


class TestClassifierOptions:
    def test_random_forest_accepts_tree_options(self, synthetic_adata_with_seeds):
        strategy = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=0,
        )
        tl.label(synthetic_adata_with_seeds, strategy, key_added="rf")
        params = synthetic_adata_with_seeds.uns["rf_params"]["params"]
        assert params["max_depth"] == 3 and params["class_weight"] == "balanced_subsample"
        assert "rf_probabilities" in synthetic_adata_with_seeds.obsm

    def test_mlp_scaling_and_early_stopping_options(self, synthetic_adata_with_seeds):
        strategy = strategies.NeuralNetworkPropagation(
            seed_key="seed_labels",
            hidden_layer_sizes=(16, 8),
            scale_features=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            max_iter=50,
            random_state=0,
        )
        tl.label(synthetic_adata_with_seeds, strategy, key_added="nn")
        params = synthetic_adata_with_seeds.uns["nn_params"]["params"]
        assert params["scale_features"] is True and params["n_iter_no_change"] == 5
        # Tuples are stored as lists so the AnnData can be written to h5ad.
        assert params["hidden_layer_sizes"] == [16, 8]

    def test_labels_follow_argmax_of_probabilities(self, synthetic_adata_with_multi_class_seeds):
        adata = synthetic_adata_with_multi_class_seeds
        for key, strategy in {
            "svm": strategies.SVMPropagation(
                seed_key="seed_labels_multi",
                keep_seeds=False,
                class_weight="balanced",
                random_state=0,
            ),
            "knn": strategies.KNNPropagation(seed_key="seed_labels_multi", keep_seeds=False),
            "rf": strategies.RandomForestPropagation(
                seed_key="seed_labels_multi", keep_seeds=False, n_estimators=30, random_state=0
            ),
            "nn": strategies.NeuralNetworkPropagation(
                seed_key="seed_labels_multi", keep_seeds=False, max_iter=50, random_state=0
            ),
        }.items():
            tl.label(adata, strategy, key_added=key)
            probs = adata.obsm[f"{key}_probabilities"]
            classes = np.array(sorted(set(adata.obs["seed_labels_multi"]) - {"unknown"}))
            expected = classes[probs.argmax(axis=1)]
            assert (adata.obs[key].values == expected).all(), key
            np.testing.assert_allclose(adata.obs[f"{key}_confidence"].values, probs.max(axis=1))


class TestConsensusOptions:
    @pytest.fixture
    def votes(self):
        adata = ad.AnnData(X=np.zeros((4, 1)))
        adata.obs_names = [f"c{i}" for i in range(4)]
        adata.obs["m1"] = ["A", "A", "unknown", "unknown"]
        adata.obs["m2"] = ["B", "A", "B", "unknown"]
        adata.obs["m3"] = ["C", "unknown", "unknown", "unknown"]
        return adata

    def test_plurality_always_assigns_a_valid_vote(self, votes):
        strategy = strategies.ConsensusVoting(keys=["m1", "m2", "m3"], majority_fraction=None)
        tl.label(votes, strategy, key_added="c")
        # c0: three-way tie -> first key wins (Counter order); c3: all abstain -> unknown.
        assert list(votes.obs["c"]) == ["A", "A", "B", "unknown"]

    def test_fraction_of_all_uses_total_voters(self, votes):
        strategy = strategies.ConsensusVoting(
            keys=["m1", "m2", "m3"], majority_fraction=None, fraction_of="all"
        )
        tl.label(votes, strategy, key_added="c")
        np.testing.assert_allclose(votes.obs["c_agreement_fraction"], [1 / 3, 2 / 3, 1 / 3, 0.0])
        assert votes.uns["c_uns"]["fraction_of"] == "all"

    def test_fraction_of_valid_is_default(self, votes):
        strategy = strategies.ConsensusVoting(keys=["m1", "m2", "m3"], majority_fraction=None)
        tl.label(votes, strategy, key_added="c")
        np.testing.assert_allclose(votes.obs["c_agreement_fraction"], [1 / 3, 1.0, 1.0, 0.0])

    def test_threshold_applies_to_chosen_denominator(self, votes):
        strategy = strategies.ConsensusVoting(
            keys=["m1", "m2", "m3"], majority_fraction=0.5, fraction_of="all"
        )
        tl.label(votes, strategy, key_added="c")
        # Only c1 reaches 2/3 of all voters; c2 is unanimous among valid votes but 1/3 overall.
        assert list(votes.obs["c"]) == ["unknown", "A", "unknown", "unknown"]

    def test_invalid_fraction_of(self):
        with pytest.raises(ValueError):
            strategies.ConsensusVoting(keys=["m1"], fraction_of="some")


class TestPreprocessing:
    @pytest.fixture
    def counts(self):
        rng = np.random.default_rng(0)
        X = rng.poisson(1.0, size=(300, 400)).astype(np.float32)
        X[:5] = 0  # empty droplets: fewer than min_genes
        X[5, :] = 200  # extreme library size
        adata = ad.AnnData(X=X)
        adata.var_names = [f"MT-{i}" if i < 10 else f"GENE{i}" for i in range(400)]
        adata.obs_names = [f"cell{i}" for i in range(300)]
        return adata

    def test_qc_filter_thresholds_and_filtering(self, counts):
        adata = pp.qc_filter(counts, min_genes=50, verbose=False)
        thr = adata.uns["qc_thresholds"]
        assert set(thr) >= {"umi_hi", "genes_hi", "mito_hi", "min_genes", "min_cells_per_gene"}
        assert thr["mito_hi"] >= 20.0
        assert (
            adata.n_obs < 300 and "cell0" not in adata.obs_names and "cell5" not in adata.obs_names
        )
        assert (adata.obs["n_genes_by_counts"] >= 50).all()
        assert adata.var["mt"].sum() == 10

    def test_qc_filter_rejects_dropped_genes(self, counts):
        counts.X[:, 20] = 0
        adata = pp.qc_filter(counts, min_genes=50, min_cells_per_gene=3, verbose=False)
        assert "GENE20" not in adata.var_names

    def test_normalize_log1p_sets_raw(self, counts):
        adata = pp.qc_filter(counts, min_genes=50, verbose=False)
        pp.normalize_log1p(adata)
        assert adata.raw is not None
        assert "log1p" in adata.uns
        totals = np.expm1(adata.X).sum(axis=1)
        np.testing.assert_allclose(np.asarray(totals).ravel(), 1e4, rtol=1e-3)

    def test_pp_exported_under_both_names(self):
        import scAICME

        assert scAICME.pp is pp


class TestResultWriting:
    def test_annotated_adata_round_trips_to_h5ad(self, synthetic_adata_with_seeds, tmp_path):
        adata = synthetic_adata_with_seeds
        strategy = strategies.NeuralNetworkPropagation(
            seed_key="seed_labels", hidden_layer_sizes=(8,), max_iter=20, random_state=0
        )
        tl.label(adata, strategy, key_added="nn")
        path = tmp_path / "out.h5ad"
        with ad.settings.override(allow_write_nullable_strings=True):
            adata.write_h5ad(path)
        back = sc.read_h5ad(path)
        assert list(back.uns["nn_params"]["params"]["hidden_layer_sizes"]) == [8]
        assert (back.obs["nn"].astype(str).values == adata.obs["nn"].astype(str).values).all()

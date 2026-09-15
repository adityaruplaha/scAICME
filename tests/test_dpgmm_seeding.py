"""Behavioral tests for DPGMMSeeding (marker-set DP-GMM seeding)."""

import numpy as np
import pytest

from scaicme import strategies, tl

SETTINGS = {
    "n_components": 15,
    "weight_concentration_prior": 0.1,
    "per_gene_pos_quantile": 0.3,
    "cluster_score_min": 0.08,
    "min_cells_cluster": 30,
    "random_state": 42,
}


class TestDPGMMSeeding:
    def test_basic_execution_and_outputs(self, synthetic_adata, marker_dict):
        strategy = strategies.DPGMMSeeding(markers=marker_dict, **SETTINGS)
        result = tl.label(synthetic_adata, strategy, key_added="seeds")

        assert "seeds" in result
        obs = synthetic_adata.obs
        assert set(obs["seeds"].unique()) <= set(marker_dict) | {"unknown"}
        assert (obs["seeds"] != "unknown").sum() > 0
        assert "seeds_max_score" in obs and "seeds_is_confident" in obs
        assert synthetic_adata.obsm["seeds_scores"].shape == (
            synthetic_adata.n_obs,
            len(marker_dict),
        )
        assert (
            synthetic_adata.obsm["seeds_marker_scores"].shape
            == synthetic_adata.obsm["seeds_scores"].shape
        )

        uns = synthetic_adata.uns["seeds_uns"]
        assert set(uns["diagnostics"]) == set(marker_dict)
        assert uns["size_floor"] == max(20, int(0.001 * synthetic_adata.n_obs))
        assert 0 < uns["fraction_assigned"] <= 1

    def test_confidence_is_max_over_types_and_bounded(self, synthetic_adata, marker_dict):
        strategy = strategies.DPGMMSeeding(markers=marker_dict, **SETTINGS)
        tl.label(synthetic_adata, strategy, key_added="seeds")
        scores = synthetic_adata.obsm["seeds_scores"]
        conf = synthetic_adata.obs["seeds_max_score"].to_numpy()
        assert scores.min() >= 0 and scores.max() <= 1
        assigned = synthetic_adata.obs["seeds"] != "unknown"
        np.testing.assert_allclose(conf[assigned], scores.max(axis=1)[assigned])
        assert (conf[~assigned] == 0).all()

    def test_seeds_land_on_the_right_cells(self, synthetic_adata, marker_dict):
        strategy = strategies.DPGMMSeeding(markers=marker_dict, **SETTINGS)
        tl.label(synthetic_adata, strategy, key_added="seeds")
        labels = synthetic_adata.obs["seeds"]
        # Class A signal was injected in cells 0-124 (genes 0-4).
        block = labels.iloc[0:125]
        assigned = block[block != "unknown"]
        assert len(assigned) > 0
        assert (assigned == "Class A").mean() > 0.8

    def test_unknown_label_is_configurable(self, synthetic_adata, marker_dict):
        strategy = strategies.DPGMMSeeding(
            markers=marker_dict, unknown_label="unlabeled", min_type_size=10_000, **SETTINGS
        )
        tl.label(synthetic_adata, strategy, key_added="seeds")
        assert (synthetic_adata.obs["seeds"] == "unlabeled").all()
        assert "unknown" not in synthetic_adata.obs["seeds"].values

    def test_size_floor_drops_small_types(self, synthetic_adata, marker_dict):
        strategy = strategies.DPGMMSeeding(markers=marker_dict, min_type_size=10_000, **SETTINGS)
        tl.label(synthetic_adata, strategy, key_added="seeds")
        uns = synthetic_adata.uns["seeds_uns"]
        assert uns["size_floor"] == 10_000
        assert set(uns["dropped_types"]) > set()
        assert (synthetic_adata.obs["seeds"] == "unknown").all()
        assert (synthetic_adata.obs["seeds_max_score"] == 0).all()

    def test_missing_markers_are_skipped(self, synthetic_adata, marker_dict):
        markers = dict(marker_dict)
        markers["Ghost"] = ["nope_1", "nope_2", "nope_3"]
        markers["Thin"] = ["gene_0", "nope_2"]  # only one present -> below min_genes_present
        strategy = strategies.DPGMMSeeding(markers=markers, **SETTINGS)
        tl.label(synthetic_adata, strategy, key_added="seeds")

        diag = synthetic_adata.uns["seeds_uns"]["diagnostics"]
        assert diag["Ghost"]["skipped"] == "insufficient_markers_present"
        assert diag["Thin"]["skipped"] == "insufficient_markers_present"
        assert "Ghost" not in synthetic_adata.obs["seeds"].values
        assert "Thin" not in synthetic_adata.obs["seeds"].values
        scores = synthetic_adata.obsm["seeds_scores"]
        assert not scores[:, list(markers).index("Ghost")].any()

    def test_low_enrichment_is_skipped(self, synthetic_adata, marker_dict):
        # gene_60..62 carry only Poisson(0.5) noise; demanding near-universal expression skips it.
        markers = {"Noise": ["gene_60", "gene_61", "gene_62"]}
        strategy = strategies.DPGMMSeeding(markers=markers, min_cell_enrichment=0.999, **SETTINGS)
        tl.label(synthetic_adata, strategy, key_added="seeds")
        assert synthetic_adata.uns["seeds_uns"]["diagnostics"]["Noise"]["skipped"] == (
            "low_cell_enrichment"
        )
        assert (synthetic_adata.obs["seeds"] == "unknown").all()

    def test_default_n_components_is_sqrt_n(self, synthetic_adata, marker_dict):
        strategy = strategies.DPGMMSeeding(
            markers={"Class A": marker_dict["Class A"]}, random_state=42, min_cells_cluster=30
        )
        tl.label(synthetic_adata, strategy, key_added="seeds")
        diag = synthetic_adata.uns["seeds_uns"]["diagnostics"]["Class A"]
        assert diag["n_components"] == max(2, int(np.sqrt(synthetic_adata.n_obs)))

    def test_deterministic_with_random_state(self, synthetic_adata, marker_dict):
        a = synthetic_adata.copy()
        b = synthetic_adata.copy()
        tl.label(a, strategies.DPGMMSeeding(markers=marker_dict, **SETTINGS), key_added="s")
        tl.label(b, strategies.DPGMMSeeding(markers=marker_dict, **SETTINGS), key_added="s")
        assert (a.obs["s"] == b.obs["s"]).all()

    def test_threaded_fit_matches_sequential(self, synthetic_adata, marker_dict):
        a = synthetic_adata.copy()
        b = synthetic_adata.copy()
        tl.label(a, strategies.DPGMMSeeding(markers=marker_dict, n_jobs=1, **SETTINGS), "s")
        tl.label(b, strategies.DPGMMSeeding(markers=marker_dict, n_jobs=4, **SETTINGS), "s")
        assert (a.obs["s"] == b.obs["s"]).all()
        np.testing.assert_array_equal(a.obsm["s_scores"], b.obsm["s_scores"])

    def test_works_without_raw(self, synthetic_adata, marker_dict):
        synthetic_adata.raw = None
        strategy = strategies.DPGMMSeeding(markers=marker_dict, use_raw=False, **SETTINGS)
        tl.label(synthetic_adata, strategy, key_added="seeds")
        assert (synthetic_adata.obs["seeds"] != "unknown").sum() > 0

    @pytest.mark.parametrize("n_jobs", [1, 3])
    def test_exported_under_both_package_names(self, n_jobs):
        import scAICME

        assert scAICME.strategies.DPGMMSeeding is strategies.DPGMMSeeding
        assert strategies.DPGMMSeeding(markers={}, n_jobs=n_jobs).name == "dpgmm_seeding"

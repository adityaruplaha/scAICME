"""
Tests for Tooling / Dispatcher (tl.label function).

Tests the main dispatcher function `icme.tl.label` which handles:
- Single strategy execution
- Batch execution with lists (auto-named)
- Batch execution with dicts (custom-named)
"""

from scAICME import strategies, tl


class TestLabelDispatcher:
    """Test suite for the tl.label dispatcher function."""

    def test_single_strategy_execution(self, synthetic_adata, marker_dict):
        """Test dispatching a single strategy with explicit key."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added="my_labels")

        # Result should be a dict with the explicit key
        assert isinstance(result, dict)
        assert "my_labels" in result
        assert len(result) == 1

        # Labels should be added to adata.obs
        assert "my_labels" in synthetic_adata.obs

    def test_single_strategy_auto_named(self, synthetic_adata, marker_dict):
        """Test that single strategy gets auto-named based on strategy.name if key_added is None."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added=None)

        # Result should have an auto-generated key based on strategy name
        assert isinstance(result, dict)
        # Should have exactly one result
        assert len(result) == 1

        # The key should contain the strategy name
        key = list(result.keys())[0]
        assert "qcq" in key

    def test_batch_list_auto_named(self, synthetic_adata, marker_dict):
        """Test batch execution with list generates auto-names for each strategy."""
        strategies_list = [
            strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01),
            strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.01),
        ]

        result = tl.label(synthetic_adata, strategies_list, n_jobs=2)

        # Result should have two keys
        assert isinstance(result, dict)
        assert len(result) == 2

        # Keys should be auto-generated based on strategy names
        keys = list(result.keys())
        assert any("qcq" in k for k in keys)
        assert any("otsu" in k for k in keys)

        # Labels should be added for both strategies
        for key in keys:
            assert key in synthetic_adata.obs

    def test_batch_dict_custom_named(self, synthetic_adata, marker_dict):
        """Test batch execution with dict uses custom names."""
        strategies_dict = {
            "seeds_strict": strategies.QCQAdaptiveSeeding(
                markers=marker_dict, quantile=0.95, min_score=0.02
            ),
            "seeds_loose": strategies.QCQAdaptiveSeeding(
                markers=marker_dict, quantile=0.8, min_score=0.001
            ),
        }

        result = tl.label(synthetic_adata, strategies_dict, n_jobs=2)

        # Result should have two keys with custom names
        assert isinstance(result, dict)
        assert "seeds_strict" in result
        assert "seeds_loose" in result
        assert len(result) == 2

        # Labels should be added with custom keys
        assert "seeds_strict" in synthetic_adata.obs
        assert "seeds_loose" in synthetic_adata.obs

    def test_multiple_strategies_different_results(self, synthetic_adata, marker_dict):
        """
        Test that different strategies can produce different labels for
        the same set of cells (expected behavior).
        """
        strategies_dict = {
            "qcq": strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01),
            "otsu": strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.01),
        }

        result = tl.label(synthetic_adata, strategies_dict, n_jobs=2)

        # Both should produce results
        assert len(result) == 2

        # Labels may differ between strategies
        qcq_labels = synthetic_adata.obs["qcq"]
        otsu_labels = synthetic_adata.obs["otsu"]

        # At least one cell should have valid labels
        unknown_count_qcq = (qcq_labels == "unknown").sum()
        unknown_count_otsu = (otsu_labels == "unknown").sum()

        assert unknown_count_qcq < len(qcq_labels) or unknown_count_otsu < len(otsu_labels)

    def test_batch_with_graph_score(self, synthetic_adata, marker_dict):
        """Test batch execution including GraphScorePropagation strategy."""
        strategies_list = [
            strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.01),
            strategies.GraphScoreSeeding(
                markers=marker_dict,
                alpha=0.8,
                n_iterations=10,
                margin=0.1,
                min_score=0.01,
            ),
        ]

        result = tl.label(synthetic_adata, strategies_list, n_jobs=2)

        # Should have two results
        assert len(result) == 2

        # Both should produce labels
        all_columns = synthetic_adata.obs.columns
        label_columns = [c for c in all_columns if "_icme_label_" in c or c.startswith("_")]

        # At least 2 label columns should be created
        assert len(label_columns) >= 2

    def test_n_jobs_parameter(self, synthetic_adata, marker_dict):
        """Test that n_jobs parameter is respected (batch mode only)."""
        strategies_list = [
            strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01),
            strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.01),
        ]

        # Should work with different n_jobs values
        result_1job = tl.label(synthetic_adata.copy(), strategies_list, n_jobs=1)
        result_2jobs = tl.label(synthetic_adata.copy(), strategies_list, n_jobs=2)

        # Both should produce the same number of results
        assert len(result_1job) == len(result_2jobs)

    def test_result_contains_labeling_result_objects(self, synthetic_adata, marker_dict):
        """Test that returned dict values are LabelingResult objects."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added="test")

        # Check that result values are LabelingResult objects
        for _key, labeling_result in result.items():
            assert hasattr(labeling_result, "adata")
            assert hasattr(labeling_result, "strategy")
            assert hasattr(labeling_result, "labels")

    def test_labeling_result_attributes(self, synthetic_adata, marker_dict):
        """Test that LabelingResult has expected attributes."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added="attrs_test")
        labeling_result = result["attrs_test"]

        # Check essential attributes
        assert labeling_result.adata is synthetic_adata
        assert labeling_result.strategy is strategy
        assert len(labeling_result.labels) == synthetic_adata.n_obs

        # Check optional attributes
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

    def test_auxillary_data_written_to_adata(self, synthetic_adata_with_seeds):
        """
        Test that auxiliary observation and unstructured data from LabelingResult
        are properly written to the AnnData object.
        """
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_neighbors=5
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="aux_test")

        # Check that auxiliary obs data is written
        # KNNPropagation stores confidence in obs
        assert "aux_test_confidence" in synthetic_adata_with_seeds.obs

        # Check that auxiliary obsm data is written
        # KNNPropagation stores probabilities in obsm
        assert "aux_test_probabilities" in synthetic_adata_with_seeds.obsm

    def test_batch_continues_on_error(self, synthetic_adata):
        """
        Test that batch mode continues processing when one strategy fails.
        (This depends on error handling in tl.label)
        """
        # Create strategies: one should work, one should fail (missing seed_key)
        good_strategy = strategies.QCQAdaptiveSeeding(
            markers={"Class A": ["gene_0", "gene_1"]}, quantile=0.9, min_score=0.01
        )

        # This will fail because there's no "fake_seeds" column
        bad_strategy = strategies.KNNPropagation(seed_key="fake_seeds", obsm_key="X_pca")

        # In batch mode, bad_strategy should fail but good_strategy should succeed
        result = tl.label(synthetic_adata, [good_strategy, bad_strategy], n_jobs=2)

        # At least the good strategy should have produced a result
        assert len(result) >= 1

    def test_unique_key_generation_for_duplicates(self, synthetic_adata, marker_dict):
        """
        Test that auto-generated keys are made unique if duplicates occur.
        """
        # Use the same strategy twice (same type, so same name)
        strategies_list = [
            strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01),
            strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.95, min_score=0.02),
        ]

        result = tl.label(synthetic_adata, strategies_list, n_jobs=2)

        # Should have two distinct keys
        keys = list(result.keys())
        assert len(keys) == 2
        assert keys[0] != keys[1]

        # Both keys should be in obs
        assert keys[0] in synthetic_adata.obs
        assert keys[1] in synthetic_adata.obs

        # One key should have a numeric suffix (from uniqueness check)
        assert any("_1" in k for k in keys) or any("_2" in k for k in keys)

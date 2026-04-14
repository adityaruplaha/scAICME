"""
Tests for Phase 2: ML Propagation Strategies.

Tests machine learning-based label propagation strategies:
- KNNPropagation: k-Nearest Neighbors classifier
- RandomForestPropagation: Random Forest classifier
- NeuralNetworkPropagation: MLP classifier
- SVMPropagation: Support Vector Machine classifier
- NearestCentroidPropagation: Nearest Centroid classifier
"""

import numpy as np
import pandas as pd
import pytest

from scAICME import strategies, tl


class TestKNNPropagation:
    """Test suite for KNNPropagation strategy."""

    def test_basic_execution(self, synthetic_adata_with_seeds):
        """Test that KNN propagation runs without error and produces valid DTO."""
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_neighbors=5
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="knn_prop")
        labeling_result = result["knn_prop"]

        # Validate result dict
        assert "knn_prop" in result
        assert labeling_result is not None

        # Validate DTO structure and attributes
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Verify references
        assert labeling_result.adata is synthetic_adata_with_seeds
        assert labeling_result.strategy is strategy

        # Validate labels written to obs
        assert "knn_prop" in synthetic_adata_with_seeds.obs

        # Verify payload contains ML-specific outputs
        assert "confidence" in labeling_result.obs
        assert "probabilities" in labeling_result.obsm

    def test_near_seed_cells_labeled(self, synthetic_adata_with_seeds):
        """Test that cells near labeled seeds are propagated the same label."""
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_neighbors=5
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="knn_test")

        # Cells 0-5 are seeds (Class A)
        # Cell 6 should be labeled as Class A if it's close in PCA space
        label_cell_6 = synthetic_adata_with_seeds.obs.loc["cell_6", "knn_test"]

        # Cell 6 is in high Class A signal region, should be Class A
        assert label_cell_6 == "Class A", (
            f"Cell 6 should be labeled Class A by KNN, got {label_cell_6}"
        )

    def test_confidence_scores_stored(self, synthetic_adata_with_seeds):
        """Test that confidence scores are computed and stored."""
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_neighbors=5
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="knn_conf")
        result["knn_conf"]

        # Check confidence scores in obs
        assert "knn_conf_confidence" in synthetic_adata_with_seeds.obs
        confidences = synthetic_adata_with_seeds.obs["knn_conf_confidence"]

        # Confidences should be between 0 and 1
        assert confidences.min() >= 0
        assert confidences.max() <= 1

    def test_probabilities_stored(self, synthetic_adata_with_seeds):
        """Test that class probabilities are computed and stored."""
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_neighbors=5
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="knn_probs")
        result["knn_probs"]

        # Check probabilities in obsm
        assert "knn_probs_probabilities" in synthetic_adata_with_seeds.obsm
        probs = synthetic_adata_with_seeds.obsm["knn_probs_probabilities"]

        # Each row should sum to ~1.0
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1.0 per cell"

        # Should have columns for both Class A and Class B (since both are in seeds)
        assert probs.shape[1] == 2

    def test_seeds_preserved(self, synthetic_adata_with_seeds):
        """Test that seed cells keep their original labels."""
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_neighbors=5,
            keep_seeds=True,
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="knn_seeds")

        # Seeds (cells 0-5) should remain Class A
        for i in range(6):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "knn_seeds"]
            assert label == "Class A", f"Seed cell_{i} should remain Class A"

        # Seeds (cells 125-130) should remain Class B
        for i in range(125, 131):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "knn_seeds"]
            assert label == "Class B", f"Seed cell_{i} should remain Class B"

    def test_propagation_fraction_computed(self, synthetic_adata_with_seeds):
        """Test that propagation fraction is computed."""
        strategy = strategies.KNNPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_neighbors=5
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="knn_frac")
        result["knn_frac"]

        # Get the proportions from uns (stored under the key_uns suffix)
        key_with_suffix = "knn_frac"
        assert f"{key_with_suffix}_uns" in synthetic_adata_with_seeds.uns
        fraction = synthetic_adata_with_seeds.uns[f"{key_with_suffix}_uns"].get(
            "fraction_propagated", None
        )

        # 6 labeled out of 100, so ~0.94 should be propagated
        if fraction is not None:
            assert 0.9 <= fraction <= 0.99, f"Expected ~0.94 propagated, got {fraction}"

    def test_missing_seed_column_error(self, synthetic_adata):
        """Test that missing seed column raises ValueError."""
        strategy = strategies.KNNPropagation(seed_key="nonexistent_seeds", obsm_key="X_pca")

        with pytest.raises(ValueError, match="not found"):
            tl.label(synthetic_adata, strategy, key_added="knn_error")


class TestRandomForestPropagation:
    """Test suite for RandomForestPropagation strategy."""

    def test_basic_execution(self, synthetic_adata_with_seeds):
        """Test that Random Forest propagation runs without error."""
        strategy = strategies.RandomForestPropagation(
            seed_key="seed_labels", obsm_key="X_pca", n_estimators=10, random_state=42
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="rf_prop")

        assert "rf_prop" in result
        assert "rf_prop" in synthetic_adata_with_seeds.obs

    def test_near_seed_cells_labeled(self, synthetic_adata_with_seeds):
        """Test that cells are propagated labels from seeds."""
        strategy = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_estimators=10,
            random_state=42,
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="rf_test")

        # Most cells should get a label (not all unknown)
        labels = synthetic_adata_with_seeds.obs["rf_test"]
        unknown_count = (labels == "unknown").sum()

        assert unknown_count < len(labels) / 2, "Most cells should be labeled, not unknown"

    def test_probabilities_sum_to_one(self, synthetic_adata_with_seeds):
        """Test that predicted probabilities sum to 1.0 per cell."""
        strategy = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_estimators=10,
            random_state=42,
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="rf_probs")
        result["rf_probs"]

        # Check probabilities in obsm
        assert "rf_probs_probabilities" in synthetic_adata_with_seeds.obsm
        probs = synthetic_adata_with_seeds.obsm["rf_probs_probabilities"]

        # Each row should sum to ~1.0
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1.0 per cell"

    def test_confidence_scores(self, synthetic_adata_with_seeds):
        """Test that confidence scores are reasonable."""
        strategy = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_estimators=10,
            random_state=42,
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="rf_conf")

        # Check confidence scores
        assert "rf_conf_confidence" in synthetic_adata_with_seeds.obs
        confidences = synthetic_adata_with_seeds.obs["rf_conf_confidence"]

        # All confidences should be valid probabilities
        assert confidences.min() >= 0
        assert confidences.max() <= 1

    def test_random_state_reproducibility(self, synthetic_adata_with_seeds):
        """Test that same random_state produces same results."""
        strategy1 = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_estimators=10,
            random_state=42,
        )

        strategy2 = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_estimators=10,
            random_state=42,
        )

        adata1 = synthetic_adata_with_seeds.copy()
        adata2 = synthetic_adata_with_seeds.copy()

        tl.label(adata1, strategy1, key_added="rf_rs1")
        tl.label(adata2, strategy2, key_added="rf_rs2")

        # Same seeds and random state should give identical results
        labels1 = adata1.obs["rf_rs1"]
        labels2 = adata2.obs["rf_rs2"]

        pd.testing.assert_series_equal(labels1, labels2, check_names=False)

    def test_seeds_preserved(self, synthetic_adata_with_seeds):
        """Test that seed cells keep their original labels."""
        strategy = strategies.RandomForestPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            n_estimators=10,
            random_state=42,
            keep_seeds=True,
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="rf_seeds")

        # Seeds (cells 0-5) should remain Class A
        for i in range(6):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "rf_seeds"]
            assert label == "Class A", f"Seed cell_{i} should remain Class A"

        # Seeds (cells 125-130) should remain Class B
        for i in range(125, 131):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "rf_seeds"]
            assert label == "Class B", f"Seed cell_{i} should remain Class B"


class TestSVMPropagation:
    """Test suite for SVMPropagation strategy."""

    def test_basic_execution(self, synthetic_adata_with_seeds):
        """Test that SVM propagation runs without error."""
        strategy = strategies.SVMPropagation(seed_key="seed_labels", obsm_key="X_pca")

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="svm_prop")

        assert "svm_prop" in result
        assert "svm_prop" in synthetic_adata_with_seeds.obs

    def test_probabilities_sum_to_one(self, synthetic_adata_with_seeds):
        """Test that predicted probabilities sum to 1.0 per cell when enabled."""
        strategy = strategies.SVMPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            probability=True,
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="svm_probs")
        result["svm_probs"]

        assert "svm_probs_probabilities" in synthetic_adata_with_seeds.obsm
        probs = synthetic_adata_with_seeds.obsm["svm_probs_probabilities"]

        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1.0 per cell"

    def test_seeds_preserved(self, synthetic_adata_with_seeds):
        """Test that seed cells keep their original labels."""
        strategy = strategies.SVMPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            keep_seeds=True,
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="svm_seeds")

        for i in range(6):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "svm_seeds"]
            assert label == "Class A", f"Seed cell_{i} should remain Class A"

        for i in range(125, 131):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "svm_seeds"]
            assert label == "Class B", f"Seed cell_{i} should remain Class B"


class TestNeuralNetworkPropagation:
    """Test suite for NeuralNetworkPropagation strategy."""

    def test_basic_execution(self, synthetic_adata_with_seeds):
        """Test that neural network propagation runs without error."""
        strategy = strategies.NeuralNetworkPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            max_iter=100,
            random_state=0,
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="nn_prop")

        assert "nn_prop" in result
        assert "nn_prop" in synthetic_adata_with_seeds.obs

    def test_probabilities_sum_to_one(self, synthetic_adata_with_seeds):
        """Test that predicted probabilities sum to 1.0 per cell."""
        strategy = strategies.NeuralNetworkPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            max_iter=100,
            random_state=0,
        )

        result = tl.label(synthetic_adata_with_seeds, strategy, key_added="nn_probs")
        result["nn_probs"]

        assert "nn_probs_probabilities" in synthetic_adata_with_seeds.obsm
        probs = synthetic_adata_with_seeds.obsm["nn_probs_probabilities"]

        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities should sum to 1.0 per cell"

    def test_seeds_preserved(self, synthetic_adata_with_seeds):
        """Test that seed cells keep their original labels."""
        strategy = strategies.NeuralNetworkPropagation(
            seed_key="seed_labels",
            obsm_key="X_pca",
            keep_seeds=True,
            max_iter=100,
            random_state=0,
        )

        tl.label(synthetic_adata_with_seeds, strategy, key_added="nn_seeds")

        for i in range(6):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "nn_seeds"]
            assert label == "Class A", f"Seed cell_{i} should remain Class A"

        for i in range(125, 131):
            label = synthetic_adata_with_seeds.obs.loc[f"cell_{i}", "nn_seeds"]
            assert label == "Class B", f"Seed cell_{i} should remain Class B"


class TestNearestCentroidPropagation:
    """Test suite for NearestCentroidPropagation strategy."""

    def test_basic_execution(self, synthetic_adata_with_multi_class_seeds):
        """Test that Nearest Centroid propagation runs without error."""
        strategy = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi", obsm_key="X_pca", metric="euclidean"
        )

        result = tl.label(
            synthetic_adata_with_multi_class_seeds, strategy, key_added="centroid_prop"
        )

        assert "centroid_prop" in result
        assert "centroid_prop" in synthetic_adata_with_multi_class_seeds.obs

    def test_cells_assigned_to_centroid(self, synthetic_adata_with_multi_class_seeds):
        """Test that all cells are assigned to nearest seed centroid."""
        strategy = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi", obsm_key="X_pca", metric="euclidean"
        )

        tl.label(synthetic_adata_with_multi_class_seeds, strategy, key_added="centroid_test")

        # All cells should have a label (not "unknown", as NearestCentroid always assigns)
        labels = synthetic_adata_with_multi_class_seeds.obs["centroid_test"]

        # Should have Class A, B, C, or D labels
        unique_labels = set(labels.unique())
        assert any(c in unique_labels for c in ["Class A", "Class B", "Class C", "Class D"])

    def test_near_seed_cells_labeled_correctly(self, synthetic_adata_with_multi_class_seeds):
        """Test that cells near seed centroid get the seed label."""
        strategy = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi", obsm_key="X_pca", metric="euclidean"
        )

        tl.label(synthetic_adata_with_multi_class_seeds, strategy, key_added="centroid_near")

        # Cells close to Class A seeds should be labeled Class A, B, C, or D based on proximity
        label_cell_6 = synthetic_adata_with_multi_class_seeds.obs.loc["cell_6", "centroid_near"]
        assert label_cell_6 in ["Class A", "Class B", "Class C", "Class D"]

    def test_seeds_preserved(self, synthetic_adata_with_multi_class_seeds):
        """Test that seed cells keep their original labels."""
        strategy = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi",
            obsm_key="X_pca",
            metric="euclidean",
            keep_seeds=True,
        )

        tl.label(synthetic_adata_with_multi_class_seeds, strategy, key_added="centroid_seeds")

        # Class A seeds (cells 0-5) should remain Class A
        for i in range(6):
            label = synthetic_adata_with_multi_class_seeds.obs.loc[f"cell_{i}", "centroid_seeds"]
            assert label == "Class A", f"Seed cell_{i} should remain Class A"

        # Class B seeds (cells 125-130) should remain Class B
        for i in range(125, 131):
            label = synthetic_adata_with_multi_class_seeds.obs.loc[f"cell_{i}", "centroid_seeds"]
            assert label == "Class B", f"Seed cell_{i} should remain Class B"

        # Class C seeds (cells 250-255) should remain Class C
        for i in range(250, 256):
            label = synthetic_adata_with_multi_class_seeds.obs.loc[f"cell_{i}", "centroid_seeds"]
            assert label == "Class C", f"Seed cell_{i} should remain Class C"

        # Class D seeds (cells 375-380) should remain Class D
        for i in range(375, 381):
            label = synthetic_adata_with_multi_class_seeds.obs.loc[f"cell_{i}", "centroid_seeds"]
            assert label == "Class D", f"Seed cell_{i} should remain Class D"

    def test_metric_parameter(self, synthetic_adata_with_multi_class_seeds):
        """Test that different metrics can be specified."""
        # Test with euclidean metric
        strategy1 = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi", obsm_key="X_pca", metric="euclidean"
        )

        # Test with manhattan metric
        strategy2 = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi", obsm_key="X_pca", metric="manhattan"
        )

        adata1 = synthetic_adata_with_multi_class_seeds.copy()
        adata2 = synthetic_adata_with_multi_class_seeds.copy()

        tl.label(adata1, strategy1, key_added="centroid_euc")
        tl.label(adata2, strategy2, key_added="centroid_man")

        # Both should have valid labels
        assert all(adata1.obs["centroid_euc"] != "")
        assert all(adata2.obs["centroid_man"] != "")

    def test_missing_seed_column_error(self, synthetic_adata):
        """Test that missing seed column raises ValueError."""
        strategy = strategies.NearestCentroidPropagation(
            seed_key="nonexistent_seeds", obsm_key="X_pca"
        )

        with pytest.raises(ValueError, match="not found"):
            tl.label(synthetic_adata, strategy, key_added="centroid_error")

    def test_missing_obsm_key_error(self, synthetic_adata_with_multi_class_seeds):
        """Test that missing obsm key raises ValueError."""
        strategy = strategies.NearestCentroidPropagation(
            seed_key="seed_labels_multi", obsm_key="nonexistent_pca"
        )

        with pytest.raises(ValueError, match="not found"):
            tl.label(
                synthetic_adata_with_multi_class_seeds, strategy, key_added="centroid_obsm_error"
            )

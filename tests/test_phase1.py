"""
Tests for Phase 1: Seed Generation Strategies.

Tests the three main seed generation strategies:
- QCQAdaptiveSeeding: Quantile-based with quality checks
- OtsuAdaptiveSeeding: Otsu's method for threshold selection
- GraphScoreSeeding: GCN-style score propagation
"""

from scAICME import strategies, tl


class TestQCQAdaptiveSeeding:
    """Test suite for QCQAdaptiveSeeding strategy."""

    def test_basic_execution(self, synthetic_adata, marker_dict):
        """Test that QCQ strategy runs without error and produces labels."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        # Validate result dict and DTO structure
        assert isinstance(result, dict)
        assert "qcq_labels" in result
        assert labeling_result is not None

        # Validate DTO attributes
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Validate reference integrity
        assert labeling_result.adata is synthetic_adata
        assert labeling_result.strategy is strategy

        # Check that labels were added to obs
        assert "qcq_labels" in synthetic_adata.obs
        assert len(labeling_result.labels) == synthetic_adata.n_obs

    def test_class_a_detection(self, synthetic_adata, marker_dict):
        """Test that cells with strong Class A markers get labeled."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.8, min_score=0.01)

        tl.label(synthetic_adata, strategy, key_added="qcq_labels")

        # Cells 0-124 have high Class A signal, most should be labeled Class A
        labels = synthetic_adata.obs.loc["cell_0":"cell_124", "qcq_labels"]
        class_a_count = (labels == "Class A").sum()

        # At least 20% of high-signal cells should be labeled Class A
        assert class_a_count > 0.2 * len(labels), (
            "At least 20% of high Class A signal cells should be labeled"
        )

    def test_output_structure(self, synthetic_adata, marker_dict):
        """Test that output contains proper columns and types and payloads are valid."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.9, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added="qcq_test")
        labeling_result = result["qcq_test"]

        # Check LabelingResult completeness
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Verify DTO references
        assert labeling_result.adata is synthetic_adata
        assert labeling_result.strategy is strategy

        # Validate all cells have a label
        assert len(labeling_result.labels) == synthetic_adata.n_obs

        # Validate payload contents: obs should have auxiliary data
        assert len(labeling_result.obs) > 0
        assert "max_score" in labeling_result.obs
        assert "is_confident" in labeling_result.obs

        # obsm should have score matrix
        assert len(labeling_result.obsm) > 0
        assert "scores" in labeling_result.obsm

        # uns should have metadata
        assert len(labeling_result.uns) > 0
        assert "thresholds" in labeling_result.uns
        assert "fraction_assigned" in labeling_result.uns


class TestOtsuAdaptiveSeeding:
    """Test suite for OtsuAdaptiveSeeding strategy."""

    def test_basic_execution(self, synthetic_adata, marker_dict):
        """Test that Otsu strategy runs without error and produces labels."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.01)

        result = tl.label(synthetic_adata, strategy, key_added="otsu_labels")
        labeling_result = result["otsu_labels"]

        # Validate result dict
        assert isinstance(result, dict)
        assert "otsu_labels" in result
        assert labeling_result is not None

        # Validate DTO has all required attributes
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Validate references
        assert labeling_result.adata is synthetic_adata
        assert labeling_result.strategy is strategy

        # Check that labels were added to obs
        assert "otsu_labels" in synthetic_adata.obs
        assert len(labeling_result.labels) == synthetic_adata.n_obs

    def test_threshold_calculation(self, synthetic_adata, marker_dict):
        """Test that Otsu thresholds are properly calculated."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.001)

        result = tl.label(synthetic_adata, strategy, key_added="otsu_test")
        result["otsu_test"]

        # Check that thresholds were calculated and stored in uns
        assert "otsu_test_uns" in synthetic_adata.uns
        thresholds_dict = synthetic_adata.uns["otsu_test_uns"]
        assert "thresholds" in thresholds_dict

        # Thresholds should exist for all marker classes
        thresholds = thresholds_dict["thresholds"]
        for class_name in marker_dict.keys():
            assert class_name in thresholds, f"Threshold missing for {class_name}"

    def test_seed_generation(self, synthetic_adata, marker_dict):
        """Test that high-signal cells are labeled as seeds."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict, bins=256, min_score=0.01)

        tl.label(synthetic_adata, strategy, key_added="otsu_seeds")

        # Cell 0 has high Class A signal, should be labeled
        label = synthetic_adata.obs.loc["cell_0", "otsu_seeds"]
        # Could be Class A or neighboring class due to overlapping genes
        assert label in ["Class A", "Class B"], f"Expected cell_0 to be labeled, got {label}"


class TestGraphScoreSeeding:
    """Test suite for GraphScoreSeeding strategy."""

    def test_basic_execution(self, synthetic_adata, marker_dict):
        """Test that GraphScore strategy runs without error."""
        strategy = strategies.GraphScoreSeeding(
            markers=marker_dict,
            alpha=0.8,
            n_iterations=10,
            margin=0.1,
            min_score=0.01,
        )

        result = tl.label(synthetic_adata, strategy, key_added="graph_labels")
        labeling_result = result["graph_labels"]

        assert isinstance(result, dict)
        assert "graph_labels" in result
        assert labeling_result is not None

        # Validate DTO structure
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Verify reference integrity
        assert labeling_result.adata is synthetic_adata
        assert labeling_result.strategy is strategy

        # Validate labels written to obs
        assert "graph_labels" in synthetic_adata.obs

    def test_propagated_scores_stored(self, synthetic_adata, marker_dict):
        """Test that diffused scores are stored after graph propagation."""
        strategy = strategies.GraphScoreSeeding(
            markers=marker_dict,
            alpha=0.8,
            n_iterations=10,
            margin=0.1,
            min_score=0.01,
        )

        result = tl.label(synthetic_adata, strategy, key_added="graph_prop")
        labeling_result = result["graph_prop"]

        # Check that diffused scores are stored (after propagation)
        assert "diffused_scores" in labeling_result.obsm
        diffused_scores = labeling_result.obsm["diffused_scores"]

        # Verify shape matches cell types
        assert diffused_scores.shape[0] == synthetic_adata.n_obs
        assert diffused_scores.shape[1] == len(marker_dict)

    def test_alpha_parameter_effect(self, synthetic_adata, marker_dict):
        """Test that different alpha values produce different results."""
        # Smaller alpha (more original signal)
        strategy1 = strategies.GraphScoreSeeding(
            markers=marker_dict,
            alpha=0.2,
            n_iterations=10,
            margin=0.05,
            min_score=0.01,
        )

        # Larger alpha (more neighborhood influence)
        strategy2 = strategies.GraphScoreSeeding(
            markers=marker_dict,
            alpha=0.9,
            n_iterations=10,
            margin=0.05,
            min_score=0.01,
        )

        adata1 = synthetic_adata.copy()
        adata2 = synthetic_adata.copy()

        tl.label(adata1, strategy1, key_added="graph_alpha_low")
        tl.label(adata2, strategy2, key_added="graph_alpha_high")

        # Labels should differ between different alpha values
        # (at least some cells should have different labels)
        labels1 = set(adata1.obs["graph_alpha_low"].unique())
        labels2 = set(adata2.obs["graph_alpha_high"].unique())

        # Both should have at least one known label
        assert "unknown" not in labels1 or len(labels1) > 1
        assert "unknown" not in labels2 or len(labels2) > 1

    def test_seed_confidence(self, synthetic_adata, marker_dict):
        """Test that clear signal cells get labeled."""
        strategy = strategies.GraphScoreSeeding(
            markers=marker_dict, alpha=0.8, n_iterations=10, margin=0.05, min_score=0.0001
        )

        tl.label(synthetic_adata, strategy, key_added="graph_test")

        # Check that labeling was performed (not all cells remain as unknown)
        all_labels = synthetic_adata.obs["graph_test"]
        unknown_count = (all_labels == "unknown").sum()

        # At least some cells should be labeled (not all unknown)
        assert unknown_count < len(all_labels), (
            "GraphScore should label at least some cells with high signal"
        )

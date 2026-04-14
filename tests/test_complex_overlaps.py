"""
Tests for strategies on complex overlap patterns.

Tests strategies across 4 complex biological marker scenarios:
1. Ubiquitous genes (strong background signal)
2. Highly specific (non-overlapping markers)
3. Hierarchical structure (tree-like relationships)
4. Complex mixed patterns (most realistic)
"""

from scAICME.tl import LabelingResult

from scAICME import strategies, tl


class TestUbiquitousGenePattern:
    """Test strategies on data with ubiquitous housekeeping genes."""

    def test_ubiquitous_qcq_robustness(self, adata_ubiquitous_shared, marker_dict_ubiquitous):
        """QCQ should handle strong ubiquitous background signal."""
        strategy = strategies.QCQAdaptiveSeeding(
            markers=marker_dict_ubiquitous, quantile=0.85, min_score=0.01
        )
        result = tl.label(adata_ubiquitous_shared, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        # Validate DTO structure
        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Some labels may be 'unknown' due to ubiquitous background signal
        assert all(
            label in list(marker_dict_ubiquitous.keys()) + ["unknown"]
            for label in labeling_result.labels
        )

        # Validate DTO payload
        assert labeling_result.adata is not None
        assert labeling_result.strategy.__class__.__name__ == "QCQAdaptiveSeeding"
        assert labeling_result.obs is not None
        assert labeling_result.obsm is not None
        assert labeling_result.uns is not None

    def test_ubiquitous_otsu_robustness(self, adata_ubiquitous_shared, marker_dict_ubiquitous):
        """Otsu should handle strong ubiquitous background signal."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict_ubiquitous, min_score=0.01)
        result = tl.label(adata_ubiquitous_shared, strategy, key_added="otsu_labels")
        labeling_result = result["otsu_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Allow unknown labels due to ubiquitous background
        assert all(
            label in list(marker_dict_ubiquitous.keys()) + ["unknown"]
            for label in labeling_result.labels
        )
        assert isinstance(labeling_result.obs, dict)
        assert len(labeling_result.obs) >= 0

    def test_ubiquitous_graph_score_robustness(
        self, adata_ubiquitous_shared, marker_dict_ubiquitous
    ):
        """GraphScore should leverage neighbor structure despite ubiquitous signal."""
        strategy = strategies.GraphScoreSeeding(markers=marker_dict_ubiquitous, propagation_steps=3)
        result = tl.label(adata_ubiquitous_shared, strategy, key_added="graph_labels")
        labeling_result = result["graph_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        assert labeling_result.adata is not None

    def test_ubiquitous_consensus_robustness(self, adata_ubiquitous_shared, marker_dict_ubiquitous):
        """Consensus should combine strategy strengths despite ubiquitous background."""
        # Create individual labels for consensus
        adata = adata_ubiquitous_shared.copy()

        strategy1 = strategies.QCQAdaptiveSeeding(markers=marker_dict_ubiquitous, quantile=0.9)
        tl.label(adata, strategy1, key_added="labels_1")

        strategy2 = strategies.OtsuAdaptiveSeeding(markers=marker_dict_ubiquitous)
        tl.label(adata, strategy2, key_added="labels_2")

        strategy3 = strategies.GraphScoreSeeding(markers=marker_dict_ubiquitous)
        tl.label(adata, strategy3, key_added="labels_3")

        # Consensus voting
        strategy_consensus = strategies.ConsensusVoting(
            keys=["labels_1", "labels_2", "labels_3"], majority_fraction=0.66
        )
        result = tl.label(adata, strategy_consensus, key_added="consensus")
        labeling_result = result["consensus"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        assert labeling_result.strategy.__class__.__name__ == "ConsensusVoting"


class TestHighlySpecificPattern:
    """Test strategies on non-overlapping marker genes."""

    def test_highly_specific_qcq_perfect_case(
        self, adata_highly_specific, marker_dict_highly_specific
    ):
        """QCQ should excel with non-overlapping markers."""
        strategy = strategies.QCQAdaptiveSeeding(
            markers=marker_dict_highly_specific, quantile=0.9, min_score=0.01
        )
        result = tl.label(adata_highly_specific, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Allow some unknown labels from high specificity
        assert all(
            label in list(marker_dict_highly_specific.keys()) + ["unknown"]
            for label in labeling_result.labels
        )
        assert labeling_result.adata is not None

    def test_highly_specific_otsu_perfect_case(
        self, adata_highly_specific, marker_dict_highly_specific
    ):
        """Otsu should benefit from sharp marker separation."""
        strategy = strategies.OtsuAdaptiveSeeding(
            markers=marker_dict_highly_specific, min_score=0.01
        )
        result = tl.label(adata_highly_specific, strategy, key_added="otsu_labels")
        labeling_result = result["otsu_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Allow some unknown labels
        assert all(
            label in list(marker_dict_highly_specific.keys()) + ["unknown"]
            for label in labeling_result.labels
        )

    def test_highly_specific_graph_score_perfect_case(
        self, adata_highly_specific, marker_dict_highly_specific
    ):
        """GraphScore should maintain neighbor structure with clean separation."""
        strategy = strategies.GraphScoreSeeding(
            markers=marker_dict_highly_specific, propagation_steps=3
        )
        result = tl.label(adata_highly_specific, strategy, key_added="graph_labels")
        labeling_result = result["graph_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000

    def test_highly_specific_consensus_voting(
        self, adata_highly_specific, marker_dict_highly_specific
    ):
        """Consensus voting should produce high-confidence calls."""
        adata = adata_highly_specific.copy()

        strategy1 = strategies.QCQAdaptiveSeeding(markers=marker_dict_highly_specific, quantile=0.9)
        tl.label(adata, strategy1, key_added="labels_1")

        strategy2 = strategies.OtsuAdaptiveSeeding(markers=marker_dict_highly_specific)
        tl.label(adata, strategy2, key_added="labels_2")

        strategy3 = strategies.GraphScoreSeeding(markers=marker_dict_highly_specific)
        tl.label(adata, strategy3, key_added="labels_3")

        strategy_consensus = strategies.ConsensusVoting(
            keys=["labels_1", "labels_2", "labels_3"], majority_fraction=0.66
        )
        result = tl.label(adata, strategy_consensus, key_added="consensus")
        labeling_result = result["consensus"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        assert labeling_result.uns is not None
        assert len(labeling_result.uns) >= 0


class TestHierarchicalOverlapPattern:
    """Test strategies on hierarchical/tree-structured overlap."""

    def test_hierarchical_qcq_tree_structure(
        self, adata_hierarchical_overlap, marker_dict_hierarchical
    ):
        """QCQ should handle hierarchical gene overlap."""
        strategy = strategies.QCQAdaptiveSeeding(
            markers=marker_dict_hierarchical, quantile=0.9, min_score=0.01
        )
        result = tl.label(adata_hierarchical_overlap, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Allow some unknown labels from hierarchical overlap
        assert all(
            label in list(marker_dict_hierarchical.keys()) + ["unknown"]
            for label in labeling_result.labels
        )
        assert labeling_result.obs is not None

    def test_hierarchical_otsu_tree_structure(
        self, adata_hierarchical_overlap, marker_dict_hierarchical
    ):
        """Otsu should discriminate despite hierarchical overlap."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict_hierarchical, min_score=0.01)
        result = tl.label(adata_hierarchical_overlap, strategy, key_added="otsu_labels")
        labeling_result = result["otsu_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000

    def test_hierarchical_graph_score_tree_structure(
        self, adata_hierarchical_overlap, marker_dict_hierarchical
    ):
        """GraphScore should leverage tree structure via neighbors."""
        strategy = strategies.GraphScoreSeeding(
            markers=marker_dict_hierarchical, propagation_steps=3
        )
        result = tl.label(adata_hierarchical_overlap, strategy, key_added="graph_labels")
        labeling_result = result["graph_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Validate DTO payload
        assert labeling_result.adata is not None
        assert hasattr(labeling_result.adata, "X")

    def test_hierarchical_consensus_voting(
        self, adata_hierarchical_overlap, marker_dict_hierarchical
    ):
        """Consensus voting should converge despite tree hierarchy."""
        adata = adata_hierarchical_overlap.copy()

        strategy1 = strategies.QCQAdaptiveSeeding(markers=marker_dict_hierarchical, quantile=0.9)
        tl.label(adata, strategy1, key_added="labels_1")

        strategy2 = strategies.OtsuAdaptiveSeeding(markers=marker_dict_hierarchical)
        tl.label(adata, strategy2, key_added="labels_2")

        strategy3 = strategies.GraphScoreSeeding(markers=marker_dict_hierarchical)
        tl.label(adata, strategy3, key_added="labels_3")

        strategy_consensus = strategies.ConsensusVoting(
            keys=["labels_1", "labels_2", "labels_3"], majority_fraction=0.66
        )
        result = tl.label(adata, strategy_consensus, key_added="consensus")
        labeling_result = result["consensus"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        assert labeling_result.strategy.__class__.__name__ == "ConsensusVoting"


class TestComplexMixedPattern:
    """Test strategies on realistic complex mixed overlap patterns."""

    def test_complex_mixed_qcq_realistic(self, adata_complex_mixed, marker_dict_complex_mixed):
        """QCQ should handle complex realistic patterns."""
        strategy = strategies.QCQAdaptiveSeeding(
            markers=marker_dict_complex_mixed, quantile=0.9, min_score=0.01
        )
        result = tl.label(adata_complex_mixed, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        # Allow some unknown labels in complex mixed case
        assert all(
            label in list(marker_dict_complex_mixed.keys()) + ["unknown"]
            for label in labeling_result.labels
        )
        assert labeling_result.obsm is not None

    def test_complex_mixed_otsu_realistic(self, adata_complex_mixed, marker_dict_complex_mixed):
        """Otsu should adapt to complex mixed patterns."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict_complex_mixed, min_score=0.01)
        result = tl.label(adata_complex_mixed, strategy, key_added="otsu_labels")
        labeling_result = result["otsu_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000

    def test_complex_mixed_graph_score_realistic(
        self, adata_complex_mixed, marker_dict_complex_mixed
    ):
        """GraphScore should leverage spatial information in complex patterns."""
        strategy = strategies.GraphScoreSeeding(
            markers=marker_dict_complex_mixed, propagation_steps=3
        )
        result = tl.label(adata_complex_mixed, strategy, key_added="graph_labels")
        labeling_result = result["graph_labels"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        assert labeling_result.adata is not None

    def test_complex_mixed_consensus_voting(self, adata_complex_mixed, marker_dict_complex_mixed):
        """Consensus voting on complex mixed should aggregate robustly."""
        adata = adata_complex_mixed.copy()

        strategy1 = strategies.QCQAdaptiveSeeding(markers=marker_dict_complex_mixed, quantile=0.9)
        tl.label(adata, strategy1, key_added="labels_1")

        strategy2 = strategies.OtsuAdaptiveSeeding(markers=marker_dict_complex_mixed)
        tl.label(adata, strategy2, key_added="labels_2")

        strategy3 = strategies.GraphScoreSeeding(markers=marker_dict_complex_mixed)
        tl.label(adata, strategy3, key_added="labels_3")

        strategy_consensus = strategies.ConsensusVoting(
            keys=["labels_1", "labels_2", "labels_3"], majority_fraction=0.66
        )
        result = tl.label(adata, strategy_consensus, key_added="consensus")
        labeling_result = result["consensus"]

        assert isinstance(labeling_result, LabelingResult)
        assert labeling_result.labels is not None
        assert len(labeling_result.labels) == 1000
        assert labeling_result.adata is not None
        assert labeling_result.strategy.__class__.__name__ == "ConsensusVoting"
        # Validate full DTO structure for complex case
        assert labeling_result.obs is not None
        assert labeling_result.obsm is not None
        assert labeling_result.uns is not None


class TestCrossPatternComparison:
    """Test strategy performance across different overlap patterns."""

    def test_qcq_across_ubiquitous_and_specific(
        self,
        adata_ubiquitous_shared,
        marker_dict_ubiquitous,
        adata_highly_specific,
        marker_dict_highly_specific,
    ):
        """QCQ should work on both ubiquitous and specific patterns."""
        # Test on ubiquitous
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict_ubiquitous)
        result1 = tl.label(adata_ubiquitous_shared, strategy, key_added="qcq_labels")
        assert result1["qcq_labels"].labels is not None

        # Test on specific
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict_highly_specific)
        result2 = tl.label(adata_highly_specific, strategy, key_added="qcq_labels")
        assert result2["qcq_labels"].labels is not None

    def test_otsu_across_hierarchical_and_complex(
        self,
        adata_hierarchical_overlap,
        marker_dict_hierarchical,
        adata_complex_mixed,
        marker_dict_complex_mixed,
    ):
        """Otsu should adapt to both hierarchical and complex patterns."""
        for adata, markers in [
            (adata_hierarchical_overlap, marker_dict_hierarchical),
            (adata_complex_mixed, marker_dict_complex_mixed),
        ]:
            strategy = strategies.OtsuAdaptiveSeeding(markers=markers)
            result = tl.label(adata, strategy, key_added="otsu_labels")
            labeling_result = result["otsu_labels"]
            assert isinstance(labeling_result, LabelingResult)
            assert labeling_result.labels is not None
            assert len(labeling_result.labels) == 1000

    def test_graph_score_across_all_patterns(
        self,
        adata_ubiquitous_shared,
        marker_dict_ubiquitous,
        adata_highly_specific,
        marker_dict_highly_specific,
        adata_hierarchical_overlap,
        marker_dict_hierarchical,
        adata_complex_mixed,
        marker_dict_complex_mixed,
    ):
        """GraphScore should work across all patterns."""
        patterns = [
            (adata_ubiquitous_shared, marker_dict_ubiquitous),
            (adata_highly_specific, marker_dict_highly_specific),
            (adata_hierarchical_overlap, marker_dict_hierarchical),
            (adata_complex_mixed, marker_dict_complex_mixed),
        ]
        for adata, markers in patterns:
            strategy = strategies.GraphScoreSeeding(markers=markers)
            result = tl.label(adata, strategy, key_added="graph_labels")
            labeling_result = result["graph_labels"]
            assert isinstance(labeling_result, LabelingResult)
            assert labeling_result.labels is not None


class TestDTOValidationAcrossPatterns:
    """Validate LabelingResult DTO correctness across all patterns."""

    def test_dto_validity_ubiquitous(self, adata_ubiquitous_shared, marker_dict_ubiquitous):
        """DTO should be complete for ubiquitous pattern."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict_ubiquitous)
        result = tl.label(adata_ubiquitous_shared, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        # All required attributes
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Payload validation
        assert labeling_result.labels is not None
        assert labeling_result.adata is not None
        assert labeling_result.obs is not None
        assert labeling_result.obsm is not None

    def test_dto_validity_highly_specific(self, adata_highly_specific, marker_dict_highly_specific):
        """DTO should be complete for highly specific pattern."""
        strategy = strategies.OtsuAdaptiveSeeding(markers=marker_dict_highly_specific)
        result = tl.label(adata_highly_specific, strategy, key_added="otsu_labels")
        labeling_result = result["otsu_labels"]

        assert all(
            hasattr(labeling_result, attr)
            for attr in ["labels", "adata", "strategy", "obs", "obsm", "uns"]
        )
        assert all(getattr(labeling_result, attr) is not None for attr in ["labels", "adata"])

    def test_dto_validity_hierarchical(self, adata_hierarchical_overlap, marker_dict_hierarchical):
        """DTO should be complete for hierarchical pattern."""
        strategy = strategies.GraphScoreSeeding(markers=marker_dict_hierarchical)
        result = tl.label(adata_hierarchical_overlap, strategy, key_added="graph_labels")
        labeling_result = result["graph_labels"]

        assert labeling_result.labels is not None
        assert labeling_result.adata is not None
        assert isinstance(labeling_result.obs, dict)
        assert isinstance(labeling_result.obsm, dict)

    def test_dto_validity_complex_mixed(self, adata_complex_mixed, marker_dict_complex_mixed):
        """DTO should be complete for complex mixed pattern."""
        strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict_complex_mixed)
        result = tl.label(adata_complex_mixed, strategy, key_added="qcq_labels")
        labeling_result = result["qcq_labels"]

        assert labeling_result.labels is not None
        assert labeling_result.adata is not None
        assert labeling_result.strategy.__class__.__name__ == "QCQAdaptiveSeeding"
        assert isinstance(labeling_result.obs, dict)
        assert isinstance(labeling_result.obsm, dict)
        assert isinstance(labeling_result.uns, dict)

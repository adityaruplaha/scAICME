"""
Tests for Phase 1.5: Consensus Voting Strategy.

Tests the ConsensusVoting strategy which aggregates multiple label vectors
to generate high-confidence consensus seeds.
"""

import pandas as pd
import pytest

from scAICME import strategies, tl


class TestConsensusVoting:
    """Test suite for ConsensusVoting strategy."""

    def test_basic_execution(self, synthetic_adata):
        """Test that ConsensusVoting runs without error and produces valid DTO."""
        # Manually create three label columns for 1000 cells
        synthetic_adata.obs["labels_1"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )
        synthetic_adata.obs["labels_2"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )
        synthetic_adata.obs["labels_3"] = pd.Series(
            ["unknown"] * 500 + ["Class B"] * 400 + ["unknown"] * 100,
            index=synthetic_adata.obs_names,
        )

        strategy = strategies.ConsensusVoting(
            keys=["labels_1", "labels_2", "labels_3"], majority_fraction=0.66
        )

        result = tl.label(synthetic_adata, strategy, key_added="consensus")
        labeling_result = result["consensus"]

        # Validate result dict
        assert "consensus" in result
        assert labeling_result is not None

        # Validate DTO structure
        assert hasattr(labeling_result, "labels")
        assert hasattr(labeling_result, "adata")
        assert hasattr(labeling_result, "strategy")
        assert hasattr(labeling_result, "obs")
        assert hasattr(labeling_result, "obsm")
        assert hasattr(labeling_result, "uns")

        # Verify references
        assert labeling_result.adata is synthetic_adata
        assert labeling_result.strategy is strategy

        # Validate labels written to obs
        assert "consensus" in synthetic_adata.obs

        # Validate voting metadata in payload
        assert "agreement_fraction" in labeling_result.obs
        assert "valid_voters" in labeling_result.obs
        assert "is_confident" in labeling_result.obs

    def test_unanimous_agreement(self, synthetic_adata):
        """Test unanimous agreement (majority_fraction=1.0)."""
        # All agree on Class A for cells 0-299
        synthetic_adata.obs["m1"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )
        synthetic_adata.obs["m2"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )
        synthetic_adata.obs["m3"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )

        strategy = strategies.ConsensusVoting(keys=["m1", "m2", "m3"], majority_fraction=1.0)

        tl.label(synthetic_adata, strategy, key_added="consensus_unanimous")

        # All three agree on Class A for cells 0-299
        assert synthetic_adata.obs.loc["cell_0", "consensus_unanimous"] == "Class A", (
            "Cell 0 should be Class A with unanimous agreement"
        )
        assert synthetic_adata.obs.loc["cell_300", "consensus_unanimous"] == "Class B", (
            "Cell 300 should be Class B with unanimous agreement"
        )

    def test_majority_voting(self, synthetic_adata):
        """Test simple majority voting (majority_fraction=0.51)."""
        # Two Class A, one unknown for cells 0-499
        synthetic_adata.obs["m1"] = pd.Series(
            ["Class A"] * 500 + ["unknown"] * 500, index=synthetic_adata.obs_names
        )
        synthetic_adata.obs["m2"] = pd.Series(
            ["Class A"] * 500 + ["unknown"] * 500, index=synthetic_adata.obs_names
        )
        synthetic_adata.obs["m3"] = pd.Series(
            ["unknown"] * 500 + ["Class B"] * 500, index=synthetic_adata.obs_names
        )

        strategy = strategies.ConsensusVoting(keys=["m1", "m2", "m3"], majority_fraction=0.51)

        tl.label(synthetic_adata, strategy, key_added="consensus_majority")

        # Cells 0-499: 2/2 valid votes for Class A -> Class A (100% agreement)
        assert synthetic_adata.obs.loc["cell_0", "consensus_majority"] == "Class A", (
            "Cell 0 should be Class A with 2/2 votes"
        )

        # Cells 500-999: 1 valid vote for Class B out of 1 valid vote -> Class B (100% agreement)
        assert synthetic_adata.obs.loc["cell_500", "consensus_majority"] == "Class B", (
            "Cell 500 should be Class B with 1/1 valid vote"
        )

    def test_supermajority_required(self, synthetic_adata):
        """Test supermajority requirement (majority_fraction=0.66)."""
        # Two Class A, one Class B for cells 0-299
        synthetic_adata.obs["m1"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )
        synthetic_adata.obs["m2"] = pd.Series(
            ["Class A"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )
        synthetic_adata.obs["m3"] = pd.Series(
            ["Class B"] * 300 + ["Class B"] * 300 + ["unknown"] * 400,
            index=synthetic_adata.obs_names,
        )

        strategy = strategies.ConsensusVoting(keys=["m1", "m2", "m3"], majority_fraction=0.66)

        tl.label(synthetic_adata, strategy, key_added="consensus_super")

        # Cells 0-299: 2/3 votes for Class A -> Class A (0.66 exactly)
        assert synthetic_adata.obs.loc["cell_0", "consensus_super"] == "Class A", (
            "Cell 0 should be Class A with 2/3 supermajority"
        )

    def test_unknown_abstention(self, synthetic_adata):
        """Test that 'unknown' votes are ignored (abstentions)."""
        # m1 and m2 agree on Class A, m3 abstains
        synthetic_adata.obs["m1"] = pd.Series(
            ["Class A"] * 500 + ["unknown"] * 500, index=synthetic_adata.obs_names
        )
        synthetic_adata.obs["m2"] = pd.Series(
            ["Class A"] * 500 + ["unknown"] * 500, index=synthetic_adata.obs_names
        )
        synthetic_adata.obs["m3"] = pd.Series(["unknown"] * 1000, index=synthetic_adata.obs_names)

        strategy = strategies.ConsensusVoting(
            keys=["m1", "m2", "m3"], majority_fraction=0.51, unknown_label="unknown"
        )

        tl.label(synthetic_adata, strategy, key_added="consensus_abstain")

        # Cells 0-499: 2/2 valid votes (m3 abstains) -> Class A
        assert synthetic_adata.obs.loc["cell_0", "consensus_abstain"] == "Class A", (
            "Cell 0 should be Class A (abstentions ignored)"
        )

        # Cells 500-999: 0 valid votes -> unknown
        assert synthetic_adata.obs.loc["cell_500", "consensus_abstain"] == "unknown", (
            "Cell 500 should be unknown (all voted unknown)"
        )

    def test_all_abstain(self, synthetic_adata):
        """Test behavior when all voters abstain."""
        synthetic_adata.obs["m1"] = pd.Series(["unknown"] * 1000, index=synthetic_adata.obs_names)
        synthetic_adata.obs["m2"] = pd.Series(["unknown"] * 1000, index=synthetic_adata.obs_names)
        synthetic_adata.obs["m3"] = pd.Series(["unknown"] * 1000, index=synthetic_adata.obs_names)

        strategy = strategies.ConsensusVoting(keys=["m1", "m2", "m3"], majority_fraction=0.66)

        tl.label(synthetic_adata, strategy, key_added="consensus_all_abstain")

        # All cells should be unknown
        assert all(synthetic_adata.obs["consensus_all_abstain"] == "unknown"), (
            "All cells should be unknown when all voters abstain"
        )

    def test_missing_keys_error(self, synthetic_adata):
        """Test that missing keys raise ValueError."""
        synthetic_adata.obs["m1"] = "T-Cell"

        strategy = strategies.ConsensusVoting(
            keys=["m1", "nonexistent_key"], majority_fraction=0.66
        )

        with pytest.raises(ValueError, match="not found"):
            tl.label(synthetic_adata, strategy, key_added="consensus_error")

    def test_empty_keys_error(self):
        """Test that empty keys list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            strategies.ConsensusVoting(keys=[], majority_fraction=0.66)

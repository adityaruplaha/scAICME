"""Behavioral tests for GCNSeeding and the evaluation helpers."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scaicme import evaluation, strategies, tl


@pytest.fixture
def scored(synthetic_adata, marker_dict):
    """Synthetic data with a QCQ score matrix to propagate."""
    strategy = strategies.QCQAdaptiveSeeding(markers=marker_dict, quantile=0.6, min_confidence=0.2)
    tl.label(synthetic_adata, strategy, key_added="q")
    return synthetic_adata


class TestGCNSeeding:
    def test_basic_execution_and_outputs(self, scored, marker_dict):
        strategy = strategies.GCNSeeding(
            markers=marker_dict,
            initial_scores_key="q_scores",
            min_score=0.2,
            delta=0.1,
            min_cells_floor=20,
        )
        tl.label(scored, strategy, key_added="g")
        obs = scored.obs
        assert set(obs["g"].unique()) <= set(marker_dict) | {"unknown"}
        assert (obs["g"] != "unknown").sum() > 0
        for col in ("g_max_score", "g_top1", "g_margin", "g_is_confident"):
            assert col in obs
        assert scored.obsm["g_scores"].shape == (scored.n_obs, len(marker_dict))
        uns = scored.uns["g_uns"]
        assert uns["min_cells_used"] == max(20, int(np.floor(0.005 * scored.n_obs)), 5)
        assert 0 < uns["gate_pass_fraction"] <= 1
        # Diffused scores are row-stochastic (Y0 is row-normalized, A_hat mixes rows).
        assert obs.loc[obs["g"] != "unknown", "g_max_score"].between(0, 1).all()

    def test_accepts_type_list_and_dataframe_scores(self, scored, marker_dict):
        types = list(marker_dict)
        scored.obsm["q_df"] = pd.DataFrame(
            scored.obsm["q_scores"], index=scored.obs_names, columns=types
        )
        a = strategies.GCNSeeding(markers=types, initial_scores_key="q_scores", min_cells_floor=20)
        b = strategies.GCNSeeding(markers=types, initial_scores_key="q_df", min_cells_floor=20)
        tl.label(scored, a, key_added="ga")
        tl.label(scored, b, key_added="gb")
        assert (scored.obs["ga"] == scored.obs["gb"]).all()

    def test_target_frac_caps_and_keeps_most_confident(self, scored, marker_dict):
        uncapped = strategies.GCNSeeding(
            markers=marker_dict,
            initial_scores_key="q_scores",
            target_frac=None,
            min_score=0.2,
            delta=0.05,
            min_cells_floor=5,
        )
        capped = strategies.GCNSeeding(
            markers=marker_dict,
            initial_scores_key="q_scores",
            target_frac=0.1,
            min_score=0.2,
            delta=0.05,
            min_cells_floor=5,
        )
        tl.label(scored, uncapped, key_added="u")
        tl.label(scored, capped, key_added="c")
        n_u = (scored.obs["u"] != "unknown").sum()
        n_c = (scored.obs["c"] != "unknown").sum()
        assert n_u > 100 and n_c <= 100 < n_u
        kept = scored.obs["c"] != "unknown"
        dropped = (scored.obs["u"] != "unknown") & ~kept
        assert (
            scored.obs.loc[kept, "u_max_score"].min()
            >= scored.obs.loc[dropped, "u_max_score"].max()
        )

    def test_size_floor_drops_small_types(self, scored, marker_dict):
        strategy = strategies.GCNSeeding(
            markers=marker_dict, initial_scores_key="q_scores", min_cells_floor=10_000
        )
        tl.label(scored, strategy, key_added="g")
        assert (scored.obs["g"] == "unknown").all()
        assert (scored.obs["g_max_score"] == 0).all()

    def test_unknown_label_configurable(self, scored, marker_dict):
        strategy = strategies.GCNSeeding(
            markers=marker_dict,
            initial_scores_key="q_scores",
            min_cells_floor=10_000,
            unknown_label="unlabeled",
        )
        tl.label(scored, strategy, key_added="g")
        assert (scored.obs["g"] == "unlabeled").all()

    def test_missing_inputs_raise(self, scored, marker_dict):
        with pytest.raises(ValueError, match="Initial scores key"):
            strategies.GCNSeeding(markers=marker_dict, initial_scores_key="nope").execute_on(scored)
        with pytest.raises(ValueError, match="Graph key"):
            strategies.GCNSeeding(
                markers=marker_dict, initial_scores_key="q_scores", obsp_key="nope"
            ).execute_on(scored)
        with pytest.raises(ValueError, match="shape"):
            strategies.GCNSeeding(
                markers=list(marker_dict)[:3], initial_scores_key="q_scores"
            ).execute_on(scored)


class TestEvaluation:
    @pytest.fixture
    def labeled(self):
        adata = AnnData(X=np.zeros((6, 1)))
        adata.obs_names = [f"c{i}" for i in range(6)]
        adata.obs["ref"] = ["A", "A", "B", "B", "C", "Unknown"]
        adata.obs["pred"] = ["A", "B", "B", "B", "unlabeled", "C"]
        adata.obs["agree"] = [1.0, 0.4, 0.8, 0.6, 0.2, 1.0]
        return adata

    def test_compare_labels_coverage_and_modes(self, labeled):
        out = evaluation.compare_labels(labeled, "pred", "ref")
        assert out["coverage_pred"] == pytest.approx(5 / 6)
        assert out["coverage_ref"] == pytest.approx(5 / 6)
        assert out["coverage_both"] == pytest.approx(4 / 6)
        assert out["all_n_eval"] == 6 and out["labeled_n_eval"] == 4
        assert out["labeled_acc"] == pytest.approx(3 / 4)
        only = evaluation.compare_labels(labeled, "pred", "ref", evaluate_on="labeled")
        assert "all_acc" not in only and "labeled_acc" in only
        with pytest.raises(ValueError):
            evaluation.compare_labels(labeled, "pred", "ref", evaluate_on="some")

    def test_compare_many_skips_missing_keys(self, labeled):
        df = evaluation.compare_many(labeled, ["pred", "missing"], "ref")
        assert list(df["pred_col"]) == ["pred"]

    def test_flag_rare(self, labeled):
        flags = evaluation.flag_rare(labeled, "pred", "agree", max_agreement=0.4, tiny_frac=0.5)
        # tiny_cut = max(5, int(0.5 * 6)) = 5 -> every type is tiny -> all flagged.
        assert flags.all()
        flags = evaluation.flag_rare(labeled, "pred", "agree", max_agreement=0.4, tiny_frac=0.0)
        # tiny_cut = 5 still (min_tiny) -> all tiny; lower min_tiny to isolate the agreement rule.
        flags = evaluation.flag_rare(
            labeled, "pred", "agree", max_agreement=0.4, tiny_frac=0.0, min_tiny=1
        )
        assert list(flags) == [False, True, False, False, True, False]

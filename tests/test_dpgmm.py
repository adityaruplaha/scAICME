import numpy as np
import pandas as pd

from scAICME import strategies


def test_dpgmm_uses_precomputed_initial_scores(synthetic_adata, marker_dict):
    """DPGMM should consume a prior seed score matrix when one is provided."""
    adata = synthetic_adata.copy()
    cell_types = list(marker_dict.keys())

    seed_scores = pd.DataFrame(
        {
            cell_type: np.full(adata.n_obs, (idx + 1) / 10.0)
            for idx, cell_type in enumerate(cell_types)
        },
        index=adata.obs_names,
    )
    adata.obsm["seed_scores_matrix"] = seed_scores.values

    strategy = strategies.DPGMMClusteredSmoothing(
        markers=marker_dict,
        initial_scores_key="seed_scores_matrix",
        min_confidence=0.0,
        min_cells_per_gene=1,
        min_expressed_markers=1,
        min_cell_enrichment=0.0,
    )

    seen_initial_scores = []

    def fake_fit(X, initial_scores=None):
        assert initial_scores is not None
        seen_initial_scores.append(round(float(np.unique(initial_scores)[0]), 3))
        return np.asarray(initial_scores, dtype=float), {
            "converged": True,
            "n_total_components": 1,
            "n_signal_components": 1,
            "background_component_idx": None,
            "signal_component_indices": [0],
            "component_sizes": [adata.n_obs],
            "component_mean_sums": [0.0],
            "marker_score_thresholds": [],
            "marker_score_cluster_means": [float(np.unique(initial_scores)[0])],
            "marker_score_source": "initial_scores",
        }

    strategy._fit_single_ctype_adaptive = fake_fit  # type: ignore[method-assign]

    result = strategy.execute_on(adata)

    assert set(seen_initial_scores) == {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
    np.testing.assert_allclose(result.obsm["posterior_probabilities"].values, seed_scores.values)
    assert (result.labels != "unknown").all()

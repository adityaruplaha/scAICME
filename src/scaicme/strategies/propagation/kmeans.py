from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class KMeansPropagation(BaseMLPropagation):
    """
    Propagates labels using K-Means clustering followed by majority-vote seed assignment.

    Parameters
    ----------
    seed_key : str
        Column in `adata.obs` containing seed labels.
    obsm_key : str, default "X_pca"
        Key in `adata.obsm` containing the features used for clustering.
    unknown_label : str, default "unknown"
        Label used for unlabeled cells in the seed column.
    keep_seeds : bool, default True
        Whether to keep seed labels unchanged in the final output.
    n_clusters : int | None, default None
        Total number of clusters to form. If `None`, defaults to the number of unique
        non-unknown classes discovered in the seed column.
    scale_features : bool, default True
        Whether to standardize the feature matrix (`StandardScaler`) across all cells prior
        to K-Means clustering.
    random_state : int | None, default None
        Random seed for reproducible K-Means initialization.
    min_seed_conf : float, default 0.0
        Minimum confidence threshold for initial seed cells to be included in majority vote.
    conf_key : str, default "max_confidence"
        Key in `adata.obs` holding initial seed confidence scores when `min_seed_conf > 0`.
    min_conf : float, default 0.0
        Post-propagation confidence threshold; predictions below this score are set to `unknown_label`.
    max_pcs : int | None, default None
        If provided, slices `adata.obsm[obsm_key]` to the top `max_pcs` features.
    """

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        n_clusters: int | None = None,
        scale_features: bool = True,
        random_state: int | None = None,
        min_seed_conf: float = 0.0,
        conf_key: str = "max_confidence",
        min_conf: float = 0.0,
        max_pcs: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seed_key=seed_key,
            obsm_key=obsm_key,
            unknown_label=unknown_label,
            keep_seeds=keep_seeds,
            min_seed_conf=min_seed_conf,
            conf_key=conf_key,
            min_conf=min_conf,
            max_pcs=max_pcs,
            **kwargs,
        )
        self.n_clusters = n_clusters
        self.scale_features = scale_features
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "kmeans_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        if self.scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        if self.n_clusters is not None:
            n_clusters = self.n_clusters
        else:
            unique_classes = np.unique(y_train)
            n_clusters = len(unique_classes)
            if n_clusters == 0:
                raise ValueError("No valid classes found in seed data to infer n_clusters.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        cluster_ids = kmeans.fit_predict(X_scaled)

        # Majority vote matching per cluster
        preds = np.full(adata.n_obs, self.unknown_label, dtype=object)
        is_labeled_arr = is_labeled.to_numpy()
        y_raw_arr = y_raw.to_numpy()

        for k in range(n_clusters):
            mask = (cluster_ids == k) & is_labeled_arr
            if np.any(mask):
                valid_seeds = y_raw_arr[mask]
                mode_res = pd.Series(valid_seeds).mode()
                if not mode_res.empty:
                    preds[cluster_ids == k] = mode_res.iloc[0]

        # Calculate pseudo-probability membership confidence based on distance to cluster centers
        distances = kmeans.transform(X_scaled)
        # Shift negative distances if any or use exponential similarity
        sim = np.exp(-distances + np.min(distances, axis=1, keepdims=True))
        probs = sim / np.sum(sim, axis=1, keepdims=True)
        max_probs = probs.max(axis=1)

        # Zero confidence for clusters assigned to unknown_label
        max_probs[preds == self.unknown_label] = 0.0

        final_labels = pd.Series(preds, index=adata.obs_names)
        final_labels = self._apply_min_conf(final_labels, max_probs, is_labeled, y_raw)

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={
                "confidence": pd.Series(max_probs, index=adata.obs_names),
                "kmeans_cluster": pd.Series(cluster_ids, index=adata.obs_names),
            },
            obsm={"cluster_distances": pd.DataFrame(distances, index=adata.obs_names)},
            uns={
                "cluster_centers": kmeans.cluster_centers_,
                "fraction_propagated": float((~is_labeled).mean()),
            },
        )

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class KMeansPropagation(BaseMLPropagation):
    """
    Propagates labels using K-Means clustering followed by majority-vote seed assignment.

    All cells are clustered in feature space. Each cluster takes the majority label of
    the seeds it contains; the cluster's confidence is that majority's share of the
    whole cluster (seed purity over all members). A cluster with no seeds is assigned
    the class whose seed centroid is nearest to the cluster center, with confidence 0.

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
        Number of clusters to form. If `None`, uses
        ``min(10, max(8, int(sqrt(n_cells / 2))))``.
    n_init : int | str, default 20
        Number of K-Means initializations (or ``"auto"``).
    max_iter : int, default 500
        Maximum K-Means iterations per initialization.
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
        n_init: int | str = 20,
        max_iter: int = 500,
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
        self.n_init = n_init
        self.max_iter = max_iter
        self.scale_features = scale_features
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "kmeans_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        if self.scale_features:
            X_scaled = StandardScaler().fit_transform(X)
        else:
            X_scaled = X

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = min(10, max(8, int(np.sqrt(adata.n_obs / 2))))

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        cluster_ids = kmeans.fit_predict(X_scaled)

        # Seed-class centroids (in the clustered feature space) for seedless clusters.
        is_labeled_arr = is_labeled.to_numpy()
        y_raw_arr = y_raw.to_numpy()
        known_types = np.unique(y_train)
        centroids = np.vstack(
            [X_scaled[is_labeled_arr & (y_raw_arr == t)].mean(axis=0) for t in known_types]
        )

        cluster_to_type: dict[int, str] = {}
        cluster_conf: dict[int, float] = {}
        for k in np.unique(cluster_ids):
            members = np.where(cluster_ids == k)[0]
            seeds_in_k = y_raw_arr[members][is_labeled_arr[members]]
            if len(seeds_in_k) > 0:
                # Counter keeps first-seen order on ties, matching a sequential vote.
                maj_label, maj_count = Counter(seeds_in_k).most_common(1)[0]
                cluster_to_type[k] = maj_label
                cluster_conf[k] = maj_count / len(members)
            else:
                nearest = pairwise_distances(
                    kmeans.cluster_centers_[k].reshape(1, -1), centroids
                ).argmin()
                cluster_to_type[k] = known_types[nearest]
                cluster_conf[k] = 0.0

        preds = np.array([cluster_to_type[k] for k in cluster_ids], dtype=object)
        max_probs = np.array([cluster_conf[k] for k in cluster_ids], dtype=float)

        final_labels = pd.Series(preds, index=adata.obs_names)
        final_labels = self._apply_min_conf(final_labels, max_probs, is_labeled, y_raw)

        distances = kmeans.transform(X_scaled)
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
                "n_clusters": int(n_clusters),
                "fraction_propagated": float((~is_labeled).mean()),
            },
        )

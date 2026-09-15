from typing import Any

import pandas as pd
from anndata import AnnData
from sklearn.neighbors import KNeighborsClassifier

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class KNNPropagation(BaseMLPropagation):
    """
    Propagates labels using a k-Nearest Neighbors classifier trained on the seeds.

    Parameters
    ----------
    seed_key : str
        Column in `adata.obs` containing seed labels.
    obsm_key : str, default "X_pca"
        Key in `adata.obsm` containing the features used for classification.
    unknown_label : str, default "unknown"
        Label used for unlabeled cells in the seed column.
    keep_seeds : bool, default True
        Whether to keep seed labels unchanged in the final output.
    n_neighbors : int, default 15
        Number of neighbors to use for k-nearest neighbors classification.
    weights : str, default "distance"
        Weight function used in prediction ("uniform" or "distance").
    min_seed_conf : float, default 0.0
        Minimum confidence threshold for initial seed cells to be included in training.
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
        n_neighbors: int = 15,
        weights: str = "distance",
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
        self.n_neighbors = n_neighbors
        self.weights = weights

    @property
    def name(self) -> str:
        return "knn_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        n_neighbors = min(self.n_neighbors, len(X_train))
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=self.weights)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X)
        preds, max_probs = self._labels_from_proba(probs, clf.classes_)

        final_labels = pd.Series(preds, index=adata.obs_names)
        final_labels = self._apply_min_conf(final_labels, max_probs, is_labeled, y_raw)

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={"confidence": pd.Series(max_probs, index=adata.obs_names)},
            obsm={
                "probabilities": pd.DataFrame(probs, index=adata.obs_names, columns=clf.classes_)
            },
            uns={"fraction_propagated": float((~is_labeled).mean())},
        )

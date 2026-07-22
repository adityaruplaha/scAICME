from typing import Any

import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestCentroid

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class NearestCentroidPropagation(BaseMLPropagation):
    """
    Propagates labels by assigning cells to the nearest seed centroid.

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
    metric : str, default "euclidean"
        Metric to use for centroid distance calculation.
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
        metric: str = "euclidean",
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
        self.metric = metric

    @property
    def name(self) -> str:
        return "centroid_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        clf = NearestCentroid(metric=self.metric)
        clf.fit(X_train, y_train)

        preds = clf.predict(X)

        final_labels = pd.Series(preds, index=adata.obs_names)
        if self.keep_seeds:
            final_labels[is_labeled] = y_raw[is_labeled]

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            uns={"metric_used": self.metric, "fraction_propagated": float((~is_labeled).mean())},
        )

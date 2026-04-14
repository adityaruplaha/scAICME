import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestCentroid

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class NearestCentroidPropagation(BaseMLPropagation):
    """
    Propagates labels by assigning cells to the nearest seed centroid.
    """

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(seed_key, obsm_key, unknown_label, keep_seeds)
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

        # Note: NearestCentroid does not output predict_proba in sklearn
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            uns={"metric_used": self.metric, "fraction_propagated": float((~is_labeled).mean())},
        )

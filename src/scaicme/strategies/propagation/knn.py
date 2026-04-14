import pandas as pd
from anndata import AnnData
from sklearn.neighbors import KNeighborsClassifier

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class KNNPropagation(BaseMLPropagation):
    """
    Propagates labels using a k-Nearest Neighbors classifier trained on the seeds.
    """

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        n_neighbors: int = 15,
        weights: str = "distance",
        **kwargs,
    ):
        super().__init__(seed_key, obsm_key, unknown_label, keep_seeds)
        self.n_neighbors = n_neighbors
        self.weights = weights

    @property
    def name(self) -> str:
        return "knn_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        # Hard dependency on sklearn
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        clf.fit(X_train, y_train)

        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        max_probs = probs.max(axis=1)

        final_labels = pd.Series(preds, index=adata.obs_names)
        if self.keep_seeds:
            final_labels[is_labeled] = y_raw[is_labeled]

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

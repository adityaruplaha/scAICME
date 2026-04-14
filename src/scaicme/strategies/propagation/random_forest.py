import pandas as pd
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class RandomForestPropagation(BaseMLPropagation):
    """
    Propagates labels using a Random Forest classifier trained on the seeds.
    """

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        n_estimators: int = 100,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(seed_key, obsm_key, unknown_label, keep_seeds)
        self.n_estimators = n_estimators
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "rf_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # Automatically use all available CPU cores
        )
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

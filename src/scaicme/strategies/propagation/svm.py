import pandas as pd
from anndata import AnnData
from sklearn.svm import SVC

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class SVMPropagation(BaseMLPropagation):
    """
    Propagates labels using a Support Vector Machine classifier trained on the seeds.

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
    kernel : str, default "rbf"
        Kernel type to be used in the SVM.
    c : float, default 1.0
        Regularization parameter. Smaller values increase the regularization strength.
    gamma : str or float, default "scale"
        Kernel coefficient for "rbf", "poly" and "sigmoid".
    class_weight : dict | None, default None
        Class weights to handle class imbalance.
    probability : bool, default True
        Whether to enable probability estimates via cross-validation.
    random_state : int | None, default None
        Random seed for probability estimation when enabled.
    """

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        kernel: str = "rbf",
        c: float = 1.0,
        gamma: str | float = "scale",
        class_weight: dict | None = None,
        probability: bool = True,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(seed_key, obsm_key, unknown_label, keep_seeds)
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "svm_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)

        clf = SVC(
            kernel=self.kernel,
            C=self.c,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.random_state,
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X)

        final_labels = pd.Series(preds, index=adata.obs_names)
        if self.keep_seeds:
            final_labels[is_labeled] = y_raw[is_labeled]

        obs = {}
        obsm = {}
        if self.probability:
            probs = clf.predict_proba(X)
            max_probs = probs.max(axis=1)
            obs["confidence"] = pd.Series(max_probs, index=adata.obs_names)
            obsm["probabilities"] = pd.DataFrame(
                probs, index=adata.obs_names, columns=clf.classes_
            )

        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs=obs,
            obsm=obsm,
            uns={"fraction_propagated": float((~is_labeled).mean())},
        )

from typing import Tuple

import pandas as pd
from anndata import AnnData
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from ..base import LabelingResult
from .ml_base import BaseMLPropagation


class NeuralNetworkPropagation(BaseMLPropagation):
    """
    Propagates labels using a shallow neural network classifier trained on the seeds.

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
    hidden_layer_sizes : tuple, default (64, 32)
        Size of the hidden layers.
    activation : str, default "relu"
        Activation function for the hidden layers.
    solver : str, default "adam"
        Optimizer for weight updates.
    alpha : float, default 0.0001
        L2 regularization term.
    learning_rate_init : float, default 0.001
        Initial learning rate for training.
    max_iter : int, default 300
        Maximum number of training iterations.
    early_stopping : bool, default True
        Whether to use early stopping with a validation split.
    random_state : int | None, default None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 300,
        early_stopping: bool = True,
        random_state: int | None = None,
        **kwargs,
    ):
        super().__init__(seed_key, obsm_key, unknown_label, keep_seeds)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "nn_prop"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        X, X_train, y_train, y_raw, is_labeled = self._prepare_data(adata)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        clf.fit(X_train, y_train_encoded)

        preds = label_encoder.inverse_transform(clf.predict(X))
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
                "probabilities": pd.DataFrame(
                    probs, index=adata.obs_names, columns=label_encoder.classes_
                )
            },
            uns={"fraction_propagated": float((~is_labeled).mean())},
        )

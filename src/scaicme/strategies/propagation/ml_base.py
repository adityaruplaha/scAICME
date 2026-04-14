from abc import ABC

from anndata import AnnData

from ..base import BaseLabelingStrategy


class BaseMLPropagation(BaseLabelingStrategy, ABC):
    """Internal base class handling the boilerplate for sklearn-based label propagation."""

    def __init__(
        self,
        seed_key: str,
        obsm_key: str = "X_pca",
        unknown_label: str = "unknown",
        keep_seeds: bool = True,
        **kwargs,
    ):
        self.seed_key = seed_key
        self.obsm_key = obsm_key
        self.unknown_label = unknown_label
        self.keep_seeds = keep_seeds

    def _prepare_data(self, adata: AnnData):
        if self.seed_key not in adata.obs:
            raise ValueError(f"Seed key '{self.seed_key}' not found in adata.obs")
        if self.obsm_key not in adata.obsm:
            raise ValueError(f"Feature key '{self.obsm_key}' not found in adata.obsm")

        X = adata.obsm[self.obsm_key]
        y_raw = adata.obs[self.seed_key].astype(str)

        is_labeled = y_raw != self.unknown_label
        if not is_labeled.any():
            raise ValueError("No labeled cells found in the seed column to train the model.")

        X_train = X[is_labeled]
        y_train = y_raw[is_labeled]

        return X, X_train, y_train, y_raw, is_labeled

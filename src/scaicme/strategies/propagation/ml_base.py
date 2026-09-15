from abc import ABC
from typing import Any, Tuple

import numpy as np
import pandas as pd
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
        min_seed_conf: float = 0.0,
        conf_key: str = "max_confidence",
        min_conf: float = 0.0,
        max_pcs: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.seed_key = seed_key
        self.obsm_key = obsm_key
        self.unknown_label = unknown_label
        self.keep_seeds = keep_seeds
        self.min_seed_conf = min_seed_conf
        self.conf_key = conf_key
        self.min_conf = min_conf
        self.max_pcs = max_pcs

    def _prepare_data(
        self, adata: AnnData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        if self.seed_key not in adata.obs:
            raise ValueError(f"Seed key '{self.seed_key}' not found in adata.obs")
        if self.obsm_key not in adata.obsm:
            raise ValueError(f"Feature key '{self.obsm_key}' not found in adata.obsm")

        X = np.asarray(adata.obsm[self.obsm_key])
        if self.max_pcs is not None and X.ndim == 2 and X.shape[1] > self.max_pcs:
            X = X[:, : self.max_pcs]

        y_raw = adata.obs[self.seed_key].astype(str)
        is_labeled = y_raw != self.unknown_label

        if self.min_seed_conf > 0.0:
            actual_conf_key = None
            candidates = [
                self.conf_key,
                f"{self.seed_key}_{self.conf_key}",
                f"{self.seed_key}_max_confidence",
                f"{self.seed_key}_max_score",
                f"{self.seed_key}_margin",
            ]
            for candidate in candidates:
                if candidate in adata.obs:
                    actual_conf_key = candidate
                    break
            if actual_conf_key is None:
                raise ValueError(
                    f"Seed confidence key '{self.conf_key}' (or prefixed candidates) not found in adata.obs when min_seed_conf > 0."
                )
            seed_conf = adata.obs[actual_conf_key].to_numpy(dtype=float, na_value=0.0)
            is_labeled = is_labeled & (seed_conf >= self.min_seed_conf)

        if not is_labeled.any():
            raise ValueError("No labeled cells found in the seed column to train the model.")

        X_train = X[is_labeled]
        y_train = y_raw[is_labeled].to_numpy()

        return X, X_train, y_train, y_raw, is_labeled

    @staticmethod
    def _labels_from_proba(
        probs: np.ndarray, classes: np.ndarray | pd.Index | list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Derive predicted labels and confidences from one probability matrix.

        Using ``argmax`` of the same matrix that supplies the confidence keeps the
        label and its confidence consistent (``SVC.predict`` can disagree with
        ``predict_proba`` when Platt scaling is enabled).
        """
        probs = np.asarray(probs)
        preds = np.asarray(classes, dtype=object)[probs.argmax(axis=1)]
        return preds, probs.max(axis=1)

    def _apply_min_conf(
        self,
        final_labels: pd.Series,
        max_probs: np.ndarray | pd.Series,
        is_labeled: pd.Series,
        y_raw: pd.Series,
    ) -> pd.Series:
        """Applies minimum post-propagation prediction confidence filtering."""
        if self.min_conf > 0.0:
            low_conf_mask = np.asarray(max_probs) < self.min_conf
            final_labels[low_conf_mask] = self.unknown_label
        if self.keep_seeds:
            final_labels[is_labeled] = y_raw[is_labeled]
        return final_labels

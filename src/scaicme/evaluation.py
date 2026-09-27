"""Evaluation helpers: agreement with a reference annotation and rare-group flagging."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)

DEFAULT_IGNORE = ("unknown", "unlabeled", "Unknown")


_METRIC_KEYS = (
    "ARI",
    "NMI",
    "macroF1",
    "acc",
    "specificity_macro",
    "sensitivity_macro",
)


def _specificity_sensitivity_macro(y_ref: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Macro one-vs-rest specificity (TNR) and sensitivity (TPR).

    Classes are every label appearing in either vector; a class for which a rate is
    undefined (no negatives, or no positives in the reference) is skipped rather than
    counted as zero, so the macro average is over the classes where it is defined.
    """
    classes = np.unique(np.concatenate([y_ref, y_pred]))
    specificities, sensitivities = [], []
    for cls in classes:
        ref_bin = (y_ref == cls).astype(int)
        pred_bin = (y_pred == cls).astype(int)
        tn, fp, fn, tp = confusion_matrix(ref_bin, pred_bin, labels=[0, 1]).ravel()
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
    return float(np.nanmean(specificities)), float(np.nanmean(sensitivities))


def _metrics(y_ref: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_pred.size == 0:
        return {**dict.fromkeys(_METRIC_KEYS, np.nan), "n_eval": 0}
    specificity, sensitivity = _specificity_sensitivity_macro(y_ref, y_pred)
    return {
        "ARI": adjusted_rand_score(y_ref, y_pred),
        "NMI": normalized_mutual_info_score(y_ref, y_pred),
        "macroF1": f1_score(y_ref, y_pred, average="macro", zero_division=0),
        "acc": accuracy_score(y_ref, y_pred),
        "specificity_macro": specificity,
        "sensitivity_macro": sensitivity,
        "n_eval": int(y_pred.size),
    }


def compare_labels(
    adata: AnnData,
    pred_key: str,
    ref_key: str,
    ignore_labels: Iterable[str] = DEFAULT_IGNORE,
    evaluate_on: str = "both",
) -> Dict[str, object]:
    """Compare a prediction column against a reference column.

    Returns ARI, NMI, macro-F1, accuracy, macro one-vs-rest specificity and
    sensitivity, plus coverage. With ``evaluate_on="all"`` the metrics use every cell;
    with ``"labeled"`` only cells where both columns carry a label outside
    `ignore_labels`; ``"both"`` reports both, prefixed ``all_`` and ``labeled_``.
    Macro-F1, accuracy, specificity and sensitivity assume the two columns share a
    label vocabulary; ARI and NMI do not.
    """
    if evaluate_on not in ("all", "labeled", "both"):
        raise ValueError("evaluate_on must be 'all', 'labeled' or 'both'.")

    y_pred = adata.obs[pred_key].astype(str).to_numpy()
    y_ref = adata.obs[ref_key].astype(str).to_numpy()
    ignore = list(map(str, ignore_labels))
    pred_labeled = ~np.isin(y_pred, ignore)
    ref_labeled = ~np.isin(y_ref, ignore)
    both = pred_labeled & ref_labeled

    out: Dict[str, object] = {
        "pred_col": pred_key,
        "ref_col": ref_key,
        "coverage_pred": float(pred_labeled.mean()),
        "coverage_ref": float(ref_labeled.mean()),
        "coverage_both": float(both.mean()),
        "n_total": int(len(y_pred)),
    }
    if evaluate_on in ("all", "both"):
        out.update({f"all_{k}": v for k, v in _metrics(y_ref, y_pred).items()})
    if evaluate_on in ("labeled", "both"):
        out.update({f"labeled_{k}": v for k, v in _metrics(y_ref[both], y_pred[both]).items()})
    return out


def compare_many(
    adata: AnnData,
    pred_keys: List[str],
    ref_key: str,
    ignore_labels: Iterable[str] = DEFAULT_IGNORE,
) -> pd.DataFrame:
    """Run :func:`compare_labels` for every existing key in `pred_keys`; one row each."""
    rows = [
        compare_labels(adata, k, ref_key, ignore_labels=ignore_labels)
        for k in pred_keys
        if k in adata.obs
    ]
    return pd.DataFrame(rows)


def flag_rare(
    adata: AnnData,
    label_key: str,
    agreement_key: str,
    max_agreement: float = 0.4,
    tiny_frac: float = 0.005,
    min_tiny: int = 5,
) -> pd.Series:
    """Flag cells whose consensus is weak or whose consensus type is tiny.

    A cell is flagged when its agreement fraction is at most `max_agreement`, or when
    its label belongs to a type with fewer than ``max(min_tiny, int(tiny_frac * n))``
    cells. Returns a boolean Series aligned to ``adata.obs_names``.
    """
    n = adata.n_obs
    tiny_cut = max(min_tiny, int(tiny_frac * n))
    low = adata.obs[agreement_key].to_numpy(dtype=float) <= max_agreement
    counts = adata.obs[label_key].value_counts()
    tiny_types = set(counts[counts < tiny_cut].index.tolist())
    tiny = adata.obs[label_key].isin(tiny_types).to_numpy()
    return pd.Series(low | tiny, index=adata.obs_names, name="rare_flag")


def cluster_quality(adata: AnnData, label_key: str, obsm_key: str = "X_pca") -> Dict[str, object]:
    """Internal cluster-quality scores for a label column in an embedding.

    Returns silhouette, Calinski-Harabasz and Davies-Bouldin scores computed on
    ``adata.obsm[obsm_key]``, all NaN when the column holds fewer than two labels.
    These measure geometric separation in the embedding, not agreement with any
    reference annotation. Silhouette is O(n^2) in the number of cells.
    """
    X = np.asarray(adata.obsm[obsm_key])
    labels = adata.obs[label_key].astype(str).to_numpy()
    if len(np.unique(labels)) < 2:
        return {
            "method": label_key,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
    return {
        "method": label_key,
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }

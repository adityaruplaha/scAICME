from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Set, Union

import numpy as np
import pandas as pd
from anndata import AnnData

ICME_LABEL_PREFIX = "_icme_label"  # Default prefix for storing results into AnnData


@dataclass
class LabelingResult:
    """A container for the results of a cell labeling strategy execution.

    This class stores the output of a labeling strategy along with the necessary
    context to write these results back into an AnnData object.

    Attributes
    ----------
    adata : AnnData
        The AnnData object from which this result was derived.
    strategy : LabelingStrategy
        The labeling strategy used to generate this result.
    labels : pd.Series
        The main labels assigned by the strategy to each cell.
    obs : Dict[str, pd.Series], default={}
        A dictionary of auxiliary observation-level data to be written to .obs with derived keys.
    obsm : Dict[str, Union[np.ndarray, pd.DataFrame]], default={}
        A dictionary of auxiliary multi-dimensional observation data to be written to .obsm with derived keys.
    uns : Dict[str, Any], default={}
        A dictionary of additional unstructured metadata to be written to .uns.

    Methods
    -------
    write_in(key=None)
        Commit the results of this labeling to the AnnData object under a unique key.
    """

    # Inputs from which this result was derived
    adata: AnnData
    strategy: BaseLabelingStrategy

    # Payload
    labels: pd.Series
    obs: Dict[str, pd.Series] = field(default_factory=dict)
    obsm: Dict[str, Union[np.ndarray, pd.DataFrame]] = field(default_factory=dict)
    uns: Dict[str, Any] = field(default_factory=dict)

    def write_in(self, key: str | None = None) -> str:
        """Commit the results of this labeling to the AnnData object under a unique key.

        Parameters
        ----------
        key : str | None, default=None
            A unique key under which to save the main labels in .obs. If None, a unique
            key will be generated based on the strategy name. The main labels will be
            saved under this key, and any auxiliary data will be saved under subkeys
            derived from this base key (see Notes).

        Returns
        -------
        str
            The key under which the results were saved in the AnnData object.

        Notes
        -----
        This method organizes the output into several AnnData components:
        - Main labels are stored in .obs[key]
        - Auxiliary .obs data is stored with derived subkeys (i.e., .obs["{key}_{suffix}"])
        - Auxiliary .obsm data is stored with derived subkeys (i.e., .obsm["{key}_{suffix}"])
        - Strategy parameters and metadata are stored in .uns["{key}_params"] and
          .uns["{key}_uns"] respectively.

        If the auto-generated key already exists, a numeric suffix is appended to
        ensure uniqueness.
        """

        # Determine a unique key to save this result under, if not provided.
        base = key
        if base is None:
            base = f"{ICME_LABEL_PREFIX}_{self.strategy.name}"
            key = base
            c = 1
            while key in self.adata.obs:
                key = f"{base}_{c}"
                c += 1

        # Write main labels to .obs
        self.adata.obs[key] = self.labels.astype(str)

        # Write any auxiliary .obs (e.g. key_uncertainty)
        for suffix, series in self.obs.items():
            self.adata.obs[f"{key}_{suffix}"] = series

        # Write any auxiliary .obsm (e.g. key_probs)
        for suffix, matrix in self.obsm.items():
            aux_key = f"{key}_{suffix}"
            val = matrix.values if isinstance(matrix, pd.DataFrame) else matrix
            self.adata.obsm[aux_key] = val

        # Write the strategy parameters to .uns for provenance
        self.adata.uns[f"{key}_params"] = {
            "strategy": self.strategy.name,
            "params": {
                k: v
                for k, v in self.strategy.__dict__.items()
                if not k.startswith("_")
                # Exclude large/uninformative parameters
                and k not in self.strategy._repr_exclude
            },
        }

        # Write any additional (unstructured) metadata to .uns
        if self.uns:
            self.adata.uns[f"{key}_uns"] = self.uns

        return key


class BaseLabelingStrategy(ABC):
    """
    This class defines the interface that all cell labeling strategies must implement.

    Provides common functionality for strategy identification and representation.

    Attributes
    ----------
    _repr_exclude : Set[str], default={"markers"}
        A set of attribute names to exclude or abbreviate in the string representation.

    Methods
    -------
    name
        A unique internal short-name identifying the strategy (e.g., 'adaptive').
        Must be implemented by subclasses.
    execute_on(adata : AnnData) -> LabelingResult
        Execute the labeling strategy on the provided AnnData object.
        Must be implemented by subclasses.

    Notes
    -----
    - Each strategy must implement the abstract `execute_on` and `name` members.
    - The base class provides automatic __repr__ support for debugging and logging.
    - The __repr__ method includes all non-private attributes by default, except those
      specified in _repr_exclude, which are abbreviated to prevent excessively verbose output.
    - Subclasses with complex parameters (e.g., large marker gene lists, hardware
      resource handles) should override __repr__ for more informative output.
    - Non-private attributes are automatically included in the string representation
      unless specified in _repr_exclude.
    """

    _repr_exclude: Set[str] = {"markers"}

    @property
    @abstractmethod
    def name(self) -> str:
        """Internal short-name (e.g., 'adaptive'). Must be unique."""
        pass

    @abstractmethod
    def execute_on(self, adata: AnnData) -> LabelingResult:
        """Execute the labeling strategy on the provided AnnData object and return a LabelingResult."""
        pass

    def __repr__(self) -> str:
        params = [f"name='{self.name}'"]
        for k, v in sorted(self.__dict__.items()):
            if k.startswith("_"):
                continue

            if k in self._repr_exclude:
                val_str = "[...]"
            else:
                val_str = repr(v)
                if len(val_str) > 100:
                    val_str = val_str[:97] + "..."

            params.append(f"{k}={val_str}")

        return f"{self.__class__.__name__}({', '.join(params)})"

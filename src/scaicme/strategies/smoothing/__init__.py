"""Smoothing strategies."""

from .dpgmm import DPGMMClusteredSmoothing
from .gcn import GCNSmoothing

__all__ = [
    "GCNSmoothing",
    "DPGMMClusteredSmoothing",
]

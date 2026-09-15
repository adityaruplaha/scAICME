"""Semi-supervised analysis of scRNA-seq data."""

from . import evaluation, pp, strategies, tl

__version__ = "0.1.1"

__all__ = ["tl", "pp", "evaluation", "strategies", "__version__"]

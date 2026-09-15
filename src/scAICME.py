"""Compatibility shim exposing the scaicme package as scAICME."""

from __future__ import annotations

import sys

from scaicme import __version__, evaluation, pp, strategies, tl

__all__ = ["tl", "pp", "evaluation", "strategies", "__version__"]

__path__ = []

sys.modules.setdefault(__name__ + ".tl", tl)
sys.modules.setdefault(__name__ + ".pp", pp)
sys.modules.setdefault(__name__ + ".evaluation", evaluation)
sys.modules.setdefault(__name__ + ".strategies", strategies)

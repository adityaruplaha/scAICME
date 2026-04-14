"""Compatibility shim exposing the scaicme package as scAICME."""

from __future__ import annotations

import sys

from scaicme import __version__, strategies, tl

__all__ = ["tl", "strategies", "__version__"]

__path__ = []

sys.modules.setdefault(__name__ + ".tl", tl)
sys.modules.setdefault(__name__ + ".strategies", strategies)

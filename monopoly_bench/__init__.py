"""Isolated MonopolyZero benchmark for the repository's ppo-plus-v2 engine."""

from __future__ import annotations

import sys

# Legacy modules live outside this package.  Suppress bytecode before importing
# them so benchmark runs never write caches into the existing implementations.
sys.dont_write_bytecode = True

from .config import BenchmarkConfig  # noqa: E402
from .contracts import SearchResult  # noqa: E402

__all__ = ["BenchmarkConfig", "SearchResult"]

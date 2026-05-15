"""Shared pytest fixtures — cache mapgen calls across tests.

Mapgen is the dominant cost of the test suite (each call runs OpenSimplex
over the full grid + Dijkstra three times). The contract tests only need a
representative validated map per (size, map_type), so we generate each one
*once* per session and hand out the cached record.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from cogniland.nav import generate_map
from cogniland.nav.mapgen import MapRecord


@lru_cache(maxsize=None)
def _cached_map(size: int, map_type: str, seed: int) -> MapRecord:
    return generate_map(size=size, map_type=map_type, seed=seed, max_retries=200)


@pytest.fixture(scope="session")
def cached_map():
    """Return a function that lazily generates and caches map records."""
    return _cached_map

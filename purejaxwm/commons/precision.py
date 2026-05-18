"""Shared mixed-precision helper.

`resolve_dtype` is one universal contract across all purejaxwm baselines:
`cfg.compute_dtype` (string) → `jnp.dtype`.
"""
from __future__ import annotations

import jax.numpy as jnp


def resolve_dtype(name: str) -> jnp.dtype:
    """Map a config string to a concrete jnp dtype.

    Standard values: 'float32', 'bfloat16', 'float16'. All baselines in purejaxwm
    must honour `cfg.compute_dtype` via this helper to preserve apples-to-apples
    comparability; a 200M DreamerV3 in fp32 won't fit on a 32GB RTX 5090.
    """
    mapping = {
        "float32": jnp.float32, "fp32": jnp.float32,
        "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16,
        "float16": jnp.float16, "fp16": jnp.float16,
    }
    if name not in mapping:
        raise ValueError(
            f"unknown compute_dtype {name!r}; expected one of {sorted(mapping)}"
        )
    return mapping[name]

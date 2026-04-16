"""XLA/JAX environment setup — must be called before any ``import jax``."""


def setup_environment():
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

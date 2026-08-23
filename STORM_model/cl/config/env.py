import multiprocessing
import os

from dotenv import load_dotenv


def _configure_multiprocessing(method: str = "spawn", force: bool = True) -> None:
    """
    Sets the multiprocessing start method.

    Args:
        method: Start method ("fork", "spawn", "forkserver"). Defaults to "spawn"
                for compatibility with JAX/CUDA.
        force: If True, forces the start method even if it has been set.
    """
    available_methods = multiprocessing.get_all_start_methods()
    if method not in available_methods:
        raise ValueError(
            f"Invalid multiprocessing method: {method}. "
            f"Available methods: {available_methods}",
        )

    multiprocessing.set_start_method(method, force=force)


def setup_environment() -> None:
    """
    Loads .env file and sets up JAX/XLA environment variables.
    This function should be called at the start of the program.

    Sets up:
    - JAX memory management (prevents OOM)
    - JAX determinism (for reproducibility)
    - Thread counts for CPU libraries
    - Multiprocessing configuration
    """
    load_dotenv()

    # JAX Memory Management
    # Fix weird OOM: https://github.com/google/jax/discussions/6332#discussioncomment-1279991
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # JAX CPU parallelism
    # Fix threading issues with JAX
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )

    # JAX Determinism
    # Fix CUDNN non-determinism: https://github.com/google/jax/issues/4823#issuecomment-952835771
    os.environ.setdefault(
        "TF_XLA_FLAGS",
        "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
    )
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

    # Suppress XLA compilation warnings (autotuning messages)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL

    # CPU Library Thread Counts (for numpy, etc.)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    # Configure multiprocessing (spawn is safer with JAX/CUDA)
    _configure_multiprocessing(
        method=os.environ.get("MULTIPROCESSING_METHOD", "spawn"),
        force=True,
    )

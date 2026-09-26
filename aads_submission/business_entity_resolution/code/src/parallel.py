"""
Multi-core helpers.

Why this exists: the pipeline's per-pair work (string similarity, token-set
maths, text normalization) is plain Python/C code that runs on ONE core by
default -- on a many-core SageMaker instance that leaves nearly all cores
idle. Two different mechanisms are used depending on the work:

  1. RapidFuzz's own C++ thread pool (process.cpdist(..., workers=N)) for the
     fuzzy-string scorers -- no Python overhead, no pickling (see features.py).
  2. A fork-based process pool (this module) for the pure-Python parts
     (per-row text cleaning, per-pair token features), which the GIL would
     otherwise serialize.

The process pool is created ONCE, early in main(), while the parent process
is still small: forking after multi-GB dataframes / an OpenMP-using library
(torch, faiss, xgboost) are live is both slow and a known source of hangs,
so this is deliberately started before any of them. Workers are stateless --
every task carries its own (pickled) inputs -- so there is no shared-memory
coupling to get wrong. If 'fork' isn't available (e.g. Windows) or n_jobs<=1
everything transparently runs serially with identical results.
"""
import os
import multiprocessing as mp
from typing import Callable, List, Optional, Sequence

_POOL = None
_N_JOBS = 1


def available_cpus() -> int:
    """CPUs this process may actually use (respects taskset/cgroup affinity where exposed)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS / Windows
        return max(1, os.cpu_count() or 1)


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """None / 0 / negative -> all available CPUs; otherwise the requested count (>=1)."""
    if n_jobs is None or n_jobs <= 0:
        return available_cpus()
    return max(1, int(n_jobs))


def init_pool(n_jobs: Optional[int]) -> int:
    """Creates the shared worker pool (idempotent). Returns the resolved worker count."""
    global _POOL, _N_JOBS
    _N_JOBS = resolve_n_jobs(n_jobs)
    if _POOL is not None or _N_JOBS <= 1:
        return _N_JOBS
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        _N_JOBS = 1
        return _N_JOBS
    _POOL = ctx.Pool(processes=_N_JOBS)
    return _N_JOBS


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


def n_workers() -> int:
    return _N_JOBS


def map_chunks(func: Callable, chunks: Sequence) -> List:
    """
    Applies func to every chunk, in parallel when a pool exists, otherwise
    serially. Result order always matches chunk order.
    """
    if _POOL is None or len(chunks) <= 1:
        return [func(c) for c in chunks]
    return _POOL.map(func, chunks, chunksize=1)


def split_ranges(n: int, min_chunk: int, max_chunks: Optional[int] = None) -> List[tuple]:
    """
    Splits range(n) into contiguous (start, end) slices, at most 2x n_workers
    of them (a little over-decomposed so one slow chunk doesn't stall the
    rest) and never smaller than min_chunk (below which pickling overhead
    outweighs the gain).
    """
    if n <= 0:
        return []
    cap = max_chunks or max(1, 2 * _N_JOBS)
    n_chunks = max(1, min(cap, n // max(1, min_chunk)))
    size = -(-n // n_chunks)  # ceil
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def configure_native_threads(n_jobs: Optional[int]):
    """
    Points the C/C++ thread pools of the numeric libraries at n_jobs cores.
    Safe to call late: torch/faiss expose runtime setters. Each is a no-op if
    the library isn't installed, so this never becomes a hard dependency.
    """
    n = resolve_n_jobs(n_jobs)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        import faiss
        faiss.omp_set_num_threads(n)
    except Exception:
        pass
    return n

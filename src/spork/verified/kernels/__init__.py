"""
spork.verified.kernels — Stile-verified analogs of ``spork.kernels``.

Each entry point declares its dims, traces the kernel with the typed
primitives from ``spork.verified``, fires per-tile verification at
trace time, and fires the bind-time coverage check before returning a
``BoundKernel``.

Currently exported:
  - ``matmul(M, N, K, dtype=float32)`` — verified MPP matmul2d with
    K-loop accumulation. Per-tile equivalence (``ParametricReduce``
    over K-tiles folds into the spec's full-K reduction) + grid
    coverage.
"""

from .attention import attention
from .matmul import matmul


__all__ = [
    "attention",
    "matmul",
]

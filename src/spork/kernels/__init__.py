"""
spork.kernels — a library of pre-built Metal compute kernels.

Each function in this module takes the relevant compile-time dimensions
and parameters, generates a kernel specialized for those settings,
computes suitable launch parameters, and returns a ``BoundKernel`` that
can be called directly with the numpy array arguments — no need to
specify a grid or threadgroup size at the call site.

    matmul = sk.kernels.matmul(1024, 1024, 1024)
    matmul(C, A, B)

    add = sk.kernels.matrix_add((1024, 1024))
    add(C, A, B)
"""

from .causal_gqa import causal_gqa
from .matmul import matmul
from .matmul_oneshot import matmul_oneshot
from .matrix_add import matrix_add

__all__ = [
    "causal_gqa",
    "matmul",
    "matmul_oneshot",
    "matrix_add",
]

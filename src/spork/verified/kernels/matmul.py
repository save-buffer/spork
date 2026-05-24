"""
Verified analog of ``sk.kernels.matmul``.

Per-tile equivalence: each threadgroup's ``coop.store`` is verified
against the spec restricted to its tile (the ``ParametricReduce`` over
the K-loop folds into the spec's full-K reduction).

Coverage: ``.bind(grid, threadgroup)`` enumerates over the threadgroup
grid axes and the K-loop SymbolicInt, unions per-axis intervals, and
checks the union equals each declared output dim.

Differences from ``sk.kernels.matmul``:
  - Linear tile traversal only. Z-order needs non-affine bit-twiddle
    which stile's affine machinery can't track.
  - Each call enters its own ``stile.scope()`` so per-call dim
    declarations don't collide.
"""

import stile

import spork as sk
from spork.jit import BoundKernel

from .. import (
    DevicePointer,
    OutputSpec,
    dim,
    jit,
    matmul2d,
    range as skv_range,
)


_TM = 64
_TN = 64
_TK = 64
_SIMDGROUPS = 4


def matmul(
    M_size : int,
    N_size : int,
    K_size : int,
    dtype  = None,
) -> BoundKernel:
    """
    Verified tile-walking matmul. Computes ``out = A @ B`` with
    ``A : (M, K)``, ``B : (K, N)``, ``out : (M, N)``.

    Constraints:
      - ``M_size``, ``N_size`` multiples of 64 (``TM`` = ``TN`` = 64)
      - ``K_size`` multiple of 64 (``TK`` = 64)
    """
    if dtype is None:
        dtype = sk.dt.float32

    if M_size % _TM != 0 or N_size % _TN != 0 or K_size % _TK != 0:
        raise ValueError(
            f"skv.kernels.matmul: M and N must be multiples of {_TM} "
            f"and K a multiple of {_TK}; got M={M_size}, N={N_size}, "
            f"K={K_size}"
        )

    with stile.scope():
        M = dim('M', M_size)
        N = dim('N', N_size)
        K = dim('K', K_size)

        @jit(out_spec=OutputSpec("(M K, K N -> M N)", st=(M, N)))
        def matmul_kernel(
            out : DevicePointer[dtype, (M, N)],
            A   : DevicePointer[dtype, (M, K)],
            B   : DevicePointer[dtype, (K, N)],
            bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
        ):
            op = matmul2d(_TM, _TN, _TK, simdgroups=_SIMDGROUPS)
            out_tile = out.slice((_TM, _TN), (bid.y * _TM, bid.x * _TN))
            # Allocate the accumulator. ``get_destination`` needs typed
            # tile slices for type inference; their concrete K-offset
            # doesn't matter because the coop starts at zero and the
            # K-loop overwrites it.
            a_seed = A.slice((_TM, _TK), (bid.y * _TM, 0))
            b_seed = B.slice((_TK, _TN), (0, bid.x * _TN))
            coop = op.get_destination(a_seed, b_seed, dtype)
            for k_idx in skv_range(K_size // _TK):
                a_tile = A.slice((_TM, _TK), (bid.y * _TM, k_idx * _TK))
                b_tile = B.slice((_TK, _TN), (k_idx * _TK, bid.x * _TN))
                op.run(a_tile, b_tile, coop)
            coop.store(out_tile)

        return matmul_kernel.bind(
            grid=(N_size // _TN, M_size // _TM, 1),
            threadgroup=(32 * _SIMDGROUPS, 1, 1),
        )

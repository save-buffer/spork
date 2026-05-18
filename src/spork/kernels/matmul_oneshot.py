from .. import dtypes as dt
from ..jit import BoundKernel, jit
from ..tracer import matmul2d, tensor
from ..types import (
    DevicePointer,
    ThreadgroupPositionInGrid,
    Uint2,
)
from .matmul import _SIMDGROUPS, _TM, _TN


def matmul_oneshot(
    M     : int,
    N     : int,
    K     : int,
    dtype : dt.Dtype = dt.float32,
) -> BoundKernel:
    """
    Raw MPP ``matmul2d`` per output tile, with no Python-side K-tiling and
    no custom traversal — each threadgroup computes one TM × TN output tile
    in a single cooperative call by setting the descriptor's K to the full
    problem K. Useful as the simplest possible MPP baseline.

    Shapes:
      - A : (M, K)
      - B : (K, N)
      - out: (M, N)

    Constraints:
      - ``M`` and ``N`` must be multiples of 64 (the per-threadgroup tile
        size).
      - ``K`` is only bounded by MPP's per-tile resource budget (registers
        + threadgroup memory). In practice, expect this kernel to fail to
        compile or dispatch once K gets too large; the loop-tiled
        ``sk.kernels.matmul`` handles arbitrary K.
    """
    if M % _TM != 0 or N % _TN != 0:
        raise ValueError(
            f"matmul_oneshot: M and N must be multiples of {_TM}; got M={M}, N={N}"
        )
    if K <= 0:
        raise ValueError(f"matmul_oneshot: K must be positive; got K={K}")

    @jit
    def matmul_oneshot_kernel(
        out : DevicePointer[dtype],
        A   : DevicePointer[dtype],
        B   : DevicePointer[dtype],
        bid : Uint2[ThreadgroupPositionInGrid],
    ):
        tA = tensor(A,   dtype, (K, M))
        tB = tensor(B,   dtype, (N, K))
        tC = tensor(out, dtype, (N, M))

        im = bid.x * _TM
        in_ = bid.y * _TN

        op = matmul2d(_TM, _TN, K, simdgroups=_SIMDGROUPS)
        coop_c = op.get_destination(tA, tB, dtype)

        tile_a = tA.slice((K, _TM), (0, im))
        tile_b = tB.slice((_TN, K), (in_, 0))
        op.run(tile_a, tile_b, coop_c)

        tile_c = tC.slice((_TN, _TM), (in_, im))
        coop_c.store(tile_c)

    return matmul_oneshot_kernel.bind(
        grid=(M // _TM, N // _TN, 1),
        threadgroup=(32 * _SIMDGROUPS, 1, 1),
    )

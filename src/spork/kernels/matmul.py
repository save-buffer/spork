from .. import dtypes as dt
from ..jit import BoundKernel, jit
from ..tracer import (
    local,
    matmul2d,
    range,
    tensor,
    threadgroup_barrier,
)
from ..types import (
    DevicePointer,
    ThreadgroupPositionInGrid,
    Uint3,
)


# Cooperative-tensor tile sizes. These match Apple's recommended defaults for
# matmul2d on M-series GPUs; if Apple ships a tuner we can swap these out per
# dtype.
_TM = 64
_TN = 64
_TK = 128
_SIMDGROUPS = 4


def _is_pow2(n : int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def matmul(
    M     : int,
    N     : int,
    K     : int,
    dtype : dt.Dtype = dt.float32,
) -> BoundKernel:
    """
    Tiled ``out = A @ B`` using MetalPerformancePrimitives ``matmul2d``
    cooperative-tensor blocks, with Z-order tile dispatch for L2 locality.

    Shapes:
      - A : (M, K)
      - B : (K, N)
      - out: (M, N)

    Constraints (for this initial implementation):
      - ``M``, ``N`` must be multiples of 64 (the per-threadgroup tile size)
      - ``K`` must be a multiple of 128 (the K-tile size)
      - ``M // 64`` and ``N // 64`` must each be powers of two (so the
        Z-order curve covers exactly all tiles)

    Future tile-size or fallback paths can be added without changing the
    public API.
    """
    if M % _TM != 0 or N % _TN != 0 or K % _TK != 0:
        raise ValueError(
            f"matmul: M and N must be multiples of {_TM} and K a multiple "
            f"of {_TK}; got M={M}, N={N}, K={K}"
        )
    m_tiles = M // _TM
    n_tiles = N // _TN
    if not (_is_pow2(m_tiles) and _is_pow2(n_tiles)):
        raise ValueError(
            f"matmul: M//{_TM} and N//{_TN} must each be powers of two "
            f"(Z-order tile dispatch); got {m_tiles} and {n_tiles}"
        )

    @jit
    def matmul_kernel(
        out  : DevicePointer[dtype],
        A    : DevicePointer[dtype],
        B    : DevicePointer[dtype],
        tgid : Uint3[ThreadgroupPositionInGrid],
    ):
        tA = tensor(A,   dtype, (K, M))
        tB = tensor(B,   dtype, (N, K))
        tC = tensor(out, dtype, (N, M))

        tile_id = tgid.x

        # Z-order decode of tile_id into (ix, iy) via bit interleaving.
        ix = local(dt.uint32, tile_id & 0x55555555)
        ix.assign((ix | (ix >> 1)) & 0x33333333)
        ix.assign((ix | (ix >> 2)) & 0x0F0F0F0F)
        ix.assign((ix | (ix >> 4)) & 0x00FF00FF)
        ix.assign((ix | (ix >> 8)) & 0x0000FFFF)

        iy = local(dt.uint32, (tile_id >> 1) & 0x55555555)
        iy.assign((iy | (iy >> 1)) & 0x33333333)
        iy.assign((iy | (iy >> 2)) & 0x0F0F0F0F)
        iy.assign((iy | (iy >> 4)) & 0x00FF00FF)
        iy.assign((iy | (iy >> 8)) & 0x0000FFFF)

        im = ix * _TM
        in_ = iy * _TN

        op = matmul2d(_TM, _TN, _TK, simdgroups=_SIMDGROUPS)
        coop_c = op.get_destination(tA, tB, dtype)

        for i in range(K // _TK):
            threadgroup_barrier("none")
            ik = i * _TK
            tile_a = tA.slice((_TK, _TM), (ik, im))
            tile_b = tB.slice((_TN, _TK), (in_, ik))
            op.run(tile_a, tile_b, coop_c)

        tile_c = tC.slice((_TN, _TM), (in_, im))
        coop_c.store(tile_c)

    num_tile_groups = m_tiles * n_tiles
    return matmul_kernel.bind(
        grid=(num_tile_groups, 1, 1),
        threadgroup=(32 * _SIMDGROUPS, 1, 1),
    )

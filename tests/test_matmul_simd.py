import numpy as np

import spork as sk


def test_matmul_simd():
    """
    One simdgroup per output element; the K-dimension dot product is split
    across the 32 lanes, then a simd_sum reduces to a single value that lane 0
    writes to global memory.
    """
    M = N = 64
    K = 128

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    expected = A @ B

    @sk.jit
    def matmul_simd(
        out        : sk.DevicePointer[sk.dt.float32],
        A          : sk.DevicePointer[sk.dt.float32],
        B          : sk.DevicePointer[sk.dt.float32],
        K          : sk.Uint,
        N          : sk.Uint,
        bid        : sk.Uint2[sk.ThreadgroupPositionInGrid],
        warp_size  : sk.Uint[sk.ThreadsPerSimdgroup],
        thread_idx : sk.Uint[sk.ThreadIndexInSimdgroup],
    ):
        row = bid.y
        col = bid.x

        partial = sk.local(sk.dt.float32, 0.0)
        for k in sk.range(thread_idx, K, warp_size):
            partial += A[row * K + k] * B[k * N + col]

        total = sk.simd_sum(partial)
        with sk.if_(thread_idx == 0):
            out[row * N + col] = total

    matmul_simd[
        (N, M, 1),       # one threadgroup per (row, col)
        (32, 1, 1),      # one simdgroup of 32 lanes per threadgroup
    ](
        C,
        A,
        B,
        K,
        N,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)

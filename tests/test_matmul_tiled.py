import numpy as np

import spork as sk


def test_matmul_tiled():
    M = N = K = 128
    TILE = 16

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    expected = A @ B

    @sk.jit
    def matmul_tiled(
        out : sk.DevicePointer[sk.dt.float32],
        A   : sk.DevicePointer[sk.dt.float32],
        B   : sk.DevicePointer[sk.dt.float32],
        M   : sk.Uint,
        N   : sk.Uint,
        K   : sk.Uint,
        tid : sk.Uint2[sk.ThreadPositionInThreadgroup],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        A_tile = sk.threadgroup(sk.dt.float32, (TILE, TILE))
        B_tile = sk.threadgroup(sk.dt.float32, (TILE, TILE))

        row = bid.y * TILE + tid.y
        col = bid.x * TILE + tid.x

        acc = sk.local(sk.dt.float32, 0.0)
        for k_tile in sk.range(0, K, TILE):
            A_tile[tid.y, tid.x] = A[row * K + (k_tile + tid.x)]
            B_tile[tid.y, tid.x] = B[(k_tile + tid.y) * N + col]
            sk.threadgroup_barrier()

            for k in sk.range(TILE):
                acc += A_tile[tid.y, k] * B_tile[k, tid.x]
            sk.threadgroup_barrier()

        out[row * N + col] = acc

    matmul_tiled[
        (N // TILE, M // TILE, 1),
        (TILE, TILE, 1),
    ](
        C,
        A,
        B,
        M,
        N,
        K,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)

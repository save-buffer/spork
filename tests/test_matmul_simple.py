import numpy as np

import spork as sk


def test_matmul_simple():
    M = N = K = 256

    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, K).astype(np.float32)
    C = np.zeros((M, K), dtype=np.float32)

    expected = A @ B

    @sk.jit
    def matmul_simple(
        out       : sk.DevicePointer[sk.dt.float32],
        A         : sk.DevicePointer[sk.dt.float32],
        B         : sk.DevicePointer[sk.dt.float32],
        N         : sk.Uint,
        gid       : sk.Uint2[sk.ThreadPositionInGrid],
        grid_size : sk.Uint2[sk.ThreadsPerGrid],
    ):
        M = grid_size.x
        K = grid_size.y

        im = gid.x
        ik = gid.y

        result = sk.local(sk.dt.float32, 0.0)
        for in_ in sk.range(N):
            ia = im * M + in_
            ib = in_ * N + ik
            result += A[ia] * B[ib]

        iout = im * M + ik
        out[iout] = result

    matmul_simple[
        (M // 32, K // 32, 1),
        (32, 32, 1),
    ](
        C,
        A,
        B,
        N,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)

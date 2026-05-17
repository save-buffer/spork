import numpy as np

import spork as sk


def test_matrix_add():
    shape = (1024, 1024)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    expected = A + B

    @sk.jit
    def matrix_add(
        out   : sk.DevicePointer[sk.dt.float32],
        A     : sk.DevicePointer[sk.dt.float32],
        B     : sk.DevicePointer[sk.dt.float32],
        index : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[index] = A[index] + B[index]

    matrix_add[
        (int(np.prod(shape)) // 128, 1, 1),
        (128, 1, 1),
    ](
        C,
        A,
        B,
    )

    np.testing.assert_allclose(C, expected)


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


def test_matmul_simd():
    """
    One simdgroup per output element; the K-dim dot product is split across
    the 32 lanes, then simd_sum reduces to a single value lane 0 writes out.
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
        (N, M, 1),
        (32, 1, 1),
    ](
        C,
        A,
        B,
        K,
        N,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)


def test_matmul_mpp():
    """
    Tiled matmul using MetalPerformancePrimitives matmul2d cooperative tensors.
    One threadgroup of 4 simdgroups produces one TM x TN output tile; the K
    dimension is consumed in TK-sized chunks. Tiles are dispatched in Z-order
    for L2 locality, matching the reference kernel in pymetal/kernels/matmul.metal.
    """
    M = N = K = 256
    TM = TN = 64
    TK = 128

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    expected = A @ B

    @sk.jit
    def matmul_mpp(
        out  : sk.DevicePointer[sk.dt.float32],
        A    : sk.DevicePointer[sk.dt.float32],
        B    : sk.DevicePointer[sk.dt.float32],
        tgid : sk.Uint3[sk.ThreadgroupPositionInGrid],
    ):
        tA = sk.tensor(A,   sk.dt.float32, (K, M))
        tB = sk.tensor(B,   sk.dt.float32, (N, K))
        tC = sk.tensor(out, sk.dt.float32, (N, M))

        tile_id = tgid.x

        # Z-order decode of tile_id into (ix, iy) using bit interleaving.
        # Materializing ix/iy as locals keeps the generated source readable
        # and matches the original Metal kernel.
        ix = sk.local(sk.dt.uint32, tile_id & 0x55555555)
        ix.assign((ix | (ix >> 1)) & 0x33333333)
        ix.assign((ix | (ix >> 2)) & 0x0F0F0F0F)
        ix.assign((ix | (ix >> 4)) & 0x00FF00FF)
        ix.assign((ix | (ix >> 8)) & 0x0000FFFF)

        iy = sk.local(sk.dt.uint32, (tile_id >> 1) & 0x55555555)
        iy.assign((iy | (iy >> 1)) & 0x33333333)
        iy.assign((iy | (iy >> 2)) & 0x0F0F0F0F)
        iy.assign((iy | (iy >> 4)) & 0x00FF00FF)
        iy.assign((iy | (iy >> 8)) & 0x0000FFFF)

        im = ix * TM
        in_ = iy * TN

        op = sk.matmul2d(TM, TN, TK, simdgroups=4)
        coop_c = op.get_destination(tA, tB, sk.dt.float32)

        for i in sk.range(K // TK):
            sk.threadgroup_barrier("none")
            ik = i * TK
            tile_a = tA.slice((TK, TM), (ik, im))
            tile_b = tB.slice((TN, TK), (in_, ik))
            op.run(tile_a, tile_b, coop_c)

        tile_c = tC.slice((TN, TM), (in_, im))
        coop_c.store(tile_c)

    num_tile_groups = (M // TM) * (N // TN)

    matmul_mpp[
        (num_tile_groups, 1, 1),
        (32 * 4, 1, 1),
    ](
        C,
        A,
        B,
    )

    np.testing.assert_allclose(C, expected, atol=1e-2, rtol=1e-2)

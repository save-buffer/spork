import os

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


def test_sigmoid():
    """
    Exercises math intrinsics (sk.exp) by computing 1 / (1 + exp(-x)).
    """
    shape = (4096,)
    A = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    expected = 1.0 / (1.0 + np.exp(-A))

    @sk.jit
    def sigmoid(
        out : sk.DevicePointer[sk.dt.float32],
        A   : sk.DevicePointer[sk.dt.float32],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[i] = 1.0 / (1.0 + sk.exp(-A[i]))

    sigmoid[
        (shape[0] // 128, 1, 1),
        (128, 1, 1),
    ](C, A)

    np.testing.assert_allclose(C, expected, atol=1e-5, rtol=1e-5)


def test_histogram():
    """
    Each thread reads one value from `values`, computes its bin, and bumps
    the bin counter with sk.atomic_fetch_add on a `device atomic_uint *`.
    """
    N = 4096
    NBINS = 16

    np.random.seed(0)
    values = np.random.randint(0, NBINS * 4, size=N).astype(np.uint32)
    counts = np.zeros(NBINS, dtype=np.uint32)

    expected = np.bincount(values % NBINS, minlength=NBINS).astype(np.uint32)

    @sk.jit
    def histogram(
        counts : sk.DevicePointer[sk.dt.atomic_uint32],
        values : sk.DevicePointer[sk.dt.uint32],
        nbins  : sk.Uint,
        gid    : sk.Uint[sk.ThreadPositionInGrid],
    ):
        val = values[gid]
        bin_idx = val % nbins
        sk.atomic_fetch_add(counts, bin_idx, 1)

    histogram[
        (N // 128, 1, 1),
        (128, 1, 1),
    ](counts, values, NBINS)

    np.testing.assert_array_equal(counts, expected)


def test_profile_smoke():
    """
    Smoke test for sk.profile. Only runs when MTL_CAPTURE_ENABLED=1 is set in
    the environment — Metal refuses to start a capture otherwise.
    """
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        return

    shape = (1024,)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    @sk.jit
    def add(
        out : sk.DevicePointer[sk.dt.float32],
        A   : sk.DevicePointer[sk.dt.float32],
        B   : sk.DevicePointer[sk.dt.float32],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[i] = A[i] + B[i]

    with sk.profile(name="test_profile_smoke", open_in_xcode=False) as path:
        add[(shape[0] // 32, 1, 1), (32, 1, 1)](C, A, B)

    assert os.path.exists(path)
    np.testing.assert_allclose(C, A + B)


def test_device_fn_and_control_flow():
    """
    Exercises @sk.device_fn (newton_sqrt), sk.while_, sk.break_, and
    sk.if_().else_() in one kernel.
    """
    N = 4096
    A = np.random.rand(N).astype(np.float32) * 100.0  # positive inputs
    # Sprinkle a few negatives to hit the else branch
    A[::128] = -A[::128]
    C = np.zeros(N, dtype=np.float32)

    expected = np.where(A < 0.0, 0.0, np.sqrt(np.abs(A))).astype(np.float32)

    @sk.device_fn
    def newton_sqrt(x : sk.dt.float32) -> sk.dt.float32:
        guess = sk.local(sk.dt.float32, x * 0.5)
        i = sk.local(sk.dt.uint32, 0)
        with sk.while_(i < 20):
            new_guess = 0.5 * (guess + x / guess)
            with sk.if_(sk.fabs(new_guess - guess) < 1e-6):
                sk.break_()
            guess.assign(new_guess)
            i += 1
        return guess

    @sk.jit
    def sqrt_kernel(
        out : sk.DevicePointer[sk.dt.float32],
        A   : sk.DevicePointer[sk.dt.float32],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        x = A[i]
        with sk.if_(x < 0.0) as branch:
            out[i] = 0.0
        with branch.else_():
            out[i] = newton_sqrt(x)

    sqrt_kernel[(N // 128, 1, 1), (128, 1, 1)](C, A)

    np.testing.assert_allclose(C, expected, atol=1e-4, rtol=1e-4)


def test_source_map_captures_user_locs():
    """
    Verify the JittedKernel's source_map points generated-Metal lines back at
    the originating Python source locations.
    """
    THIS_FILE = __file__

    @sk.jit
    def add_two(
        out : sk.DevicePointer[sk.dt.float32],
        A   : sk.DevicePointer[sk.dt.float32],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        x = A[i] + 1.0
        out[i] = x + 1.0  # this line should appear in source_map

    smap = add_two.source_map
    assert smap is not None and len(smap) > 0, "source_map should be populated"
    # All mapped locations should point at this test file
    files = {loc[0] for loc in smap.values()}
    assert files == {THIS_FILE}, (
        f"Expected all source_map entries to point at {THIS_FILE}, got {files}"
    )


def test_compile_error_rewriter():
    """
    Unit test for the regex-based compile-error rewriter — confirms it injects
    a Python source location header when Metal references a mapped line.
    """
    from spork.runtime import _rewrite_compile_error

    metal_error = (
        "program_source:7:23: error: use of undeclared identifier 'foo'\n"
        "    out[i] = foo + 1;\n"
        "              ^\n"
    )
    rewritten = _rewrite_compile_error(metal_error, {7: ("/path/user.py", 42)})
    assert "Python source locations" in rewritten
    assert "user.py:42" in rewritten
    assert "generated line 7" in rewritten
    # Original error must still be present
    assert "use of undeclared identifier" in rewritten

    # No map → passthrough
    assert _rewrite_compile_error(metal_error, None) == metal_error
    assert _rewrite_compile_error(metal_error, {}) == metal_error

    # Map entries that aren't referenced → passthrough
    assert (
        _rewrite_compile_error(metal_error, {99: ("/x.py", 1)}) == metal_error
    )


def test_bind_eliminates_launch_boilerplate():
    """
    JittedKernel.bind(grid, threadgroup) returns a BoundKernel that's just
    callable; the launch geometry is baked in.
    """
    shape = (512,)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    @sk.jit
    def add(
        out : sk.DevicePointer[sk.dt.float32],
        A   : sk.DevicePointer[sk.dt.float32],
        B   : sk.DevicePointer[sk.dt.float32],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[i] = A[i] + B[i]

    bound = add.bind(grid=(shape[0] // 32, 1, 1), threadgroup=(32, 1, 1))
    assert isinstance(bound, sk.BoundKernel)
    assert bound.grid == (shape[0] // 32, 1, 1)
    assert bound.threadgroup == (32, 1, 1)
    bound(C, A, B)
    np.testing.assert_allclose(C, A + B)


def test_kernels_matrix_add():
    """
    sk.kernels.matrix_add picks its own launch geometry and runs directly.
    """
    shape = (1024, 1024)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    add = sk.kernels.matrix_add(shape)
    add(C, A, B)
    np.testing.assert_allclose(C, A + B)


def test_kernels_matmul():
    """
    sk.kernels.matmul is the MPP matmul2d kernel with Z-order tile dispatch,
    pre-bundled with launch params derived from M/N/K.
    """
    M = N = K = 256
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul = sk.kernels.matmul(M, N, K)
    assert matmul.grid == (4 * 4, 1, 1)
    assert matmul.threadgroup == (128, 1, 1)
    matmul(C, A, B)
    np.testing.assert_allclose(C, A @ B, atol=1e-2, rtol=1e-2)


def test_kernels_matmul_validation():
    """
    sk.kernels.matmul raises with a clear message for unsupported shapes.
    """
    import pytest
    # K=200 is not a multiple of 128
    with pytest.raises(ValueError, match="multiple"):
        sk.kernels.matmul(256, 256, 200)
    # M=192 (=3*64): multiple of 64, but 192//64=3 is not a power of two
    with pytest.raises(ValueError, match="power"):
        sk.kernels.matmul(192, 256, 256)

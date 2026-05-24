import numpy as np
import pytest
import stile

import spork as sk
from spork import verified as skv


@pytest.fixture(autouse=True)
def _stile_scope():
    """
    Each test runs inside its own ``stile.scope()`` so dim and tensor
    registries are isolated — tests can re-declare ``dim('M', ...)``
    with different sizes without collisions.
    """
    with stile.scope():
        yield


def test_verified_matmul_single_tile():
    """
    End-to-end verified matmul at a size that fits in one MPP tile.
    Exercises: skv.DevicePointer[dtype, shape], @skv.jit, skv.tensor /
    .slice, skv.matmul2d, TypedMatmulOp.get_destination / run,
    TypedCooperativeTensor.store (verification fires here), then
    actual dispatch and a correctness check against numpy.
    """
    M_size = N_size = K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        a_tile   = A.slice((M_size, K_size), (0, 0))
        b_tile   = B.slice((K_size, N_size), (0, 0))
        out_tile = out.slice((M_size, N_size), (0, 0))
        op = skv.matmul2d(M_size, N_size, K_size, simdgroups=4)
        coop = op.get_destination(a_tile, b_tile, sk.dt.float32)
        op.run(a_tile, b_tile, coop)
        coop.store(out_tile)

    np.random.seed(0)
    A_arr = np.random.randn(M_size, K_size).astype(np.float32)
    B_arr = np.random.randn(K_size, N_size).astype(np.float32)
    C_arr = np.zeros((M_size, N_size), dtype=np.float32)
    matmul.bind(grid=(1, 1, 1), threadgroup=(128, 1, 1))(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr @ B_arr, atol=1e-2, rtol=1e-2)


def test_verified_matmul_tiled_symbolic_offsets():
    """
    Tiled matmul: 4 threadgroups, each computing a 64x64 tile of a
    128x128 output. Slice offsets are bid.y * TM / bid.x * TN — exercises
    the TypedScalarTracer + TypedVectorTracer machinery and proves the
    verifier handles per-tile symbolic offsets that refine to Sliced
    dims.
    """
    TM = TN = 64
    M_size = 128
    N_size = 128
    K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_tiled(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        a_tile   = A.slice((TM, K_size),   (bid.y * TM, 0))
        b_tile   = B.slice((K_size, TN),   (0, bid.x * TN))
        out_tile = out.slice((TM, TN),     (bid.y * TM, bid.x * TN))
        op = skv.matmul2d(TM, TN, K_size, simdgroups=4)
        coop = op.get_destination(a_tile, b_tile, sk.dt.float32)
        op.run(a_tile, b_tile, coop)
        coop.store(out_tile)

    np.random.seed(0)
    A_arr = np.random.randn(M_size, K_size).astype(np.float32)
    B_arr = np.random.randn(K_size, N_size).astype(np.float32)
    C_arr = np.zeros((M_size, N_size), dtype=np.float32)
    matmul_tiled.bind(
        grid=(N_size // TN, M_size // TM, 1),
        threadgroup=(128, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr @ B_arr, atol=1e-2, rtol=1e-2)


def test_verified_matmul_coverage_rejects_undersized_grid():
    """
    Tiled matmul whose grid only covers half the output (M-axis grid =
    1 instead of 2) should be rejected at .bind() before any GPU
    dispatch, with a clear coverage error.
    """
    TM = TN = 64
    M_size = 128
    N_size = 128
    K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_tiled(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        a_tile   = A.slice((TM, K_size),   (bid.y * TM, 0))
        b_tile   = B.slice((K_size, TN),   (0, bid.x * TN))
        out_tile = out.slice((TM, TN),     (bid.y * TM, bid.x * TN))
        op = skv.matmul2d(TM, TN, K_size, simdgroups=4)
        coop = op.get_destination(a_tile, b_tile, sk.dt.float32)
        op.run(a_tile, b_tile, coop)
        coop.store(out_tile)

    # Grid (N/TN, M/TM, 1) = (2, 2, 1) would be correct.
    # Use (2, 1, 1) — only writes the top half of M.
    with pytest.raises(ValueError, match="cover"):
        matmul_tiled.bind(grid=(2, 1, 1), threadgroup=(128, 1, 1))


def test_verified_matmul_coverage_rejects_oversized_grid():
    """
    Oversized grid (overlapping writes) should also be rejected — every
    output position should be written exactly once, not multiple times.
    """
    TM = TN = 64
    M_size = 128
    N_size = 128
    K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_tiled(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        a_tile   = A.slice((TM, K_size),   (bid.y * TM, 0))
        b_tile   = B.slice((K_size, TN),   (0, bid.x * TN))
        out_tile = out.slice((TM, TN),     (bid.y * TM, bid.x * TN))
        op = skv.matmul2d(TM, TN, K_size, simdgroups=4)
        coop = op.get_destination(a_tile, b_tile, sk.dt.float32)
        op.run(a_tile, b_tile, coop)
        coop.store(out_tile)

    # Grid (2, 3, 1) overruns the M axis: bid.y ∈ {0, 1, 2} writes
    # tiles at M-offsets {0, 64, 128} but M only has 128 elements
    # (offset 128 walks off the end).
    with pytest.raises(ValueError, match="cover"):
        matmul_tiled.bind(grid=(2, 3, 1), threadgroup=(128, 1, 1))


def test_verified_matmul_persistent_runtime_loop():
    """
    Persistent-M matmul: each threadgroup walks the M-axis via a
    skv.range loop and stores one (TM, TN) tile per iteration.
    Grid is (N/TN, 1, 1); coverage enumerates over bid.x AND the loop
    variable. Both per-tile verification and coverage check must
    succeed; the kernel then dispatches and matches numpy.
    """
    TM = TN = 64
    M_size = 128
    N_size = 128
    K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_persistent_m(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        for m_idx in skv.range(M_size // TM):
            a_tile   = A.slice((TM, K_size), (m_idx * TM, 0))
            b_tile   = B.slice((K_size, TN), (0, bid.x * TN))
            out_tile = out.slice((TM, TN),   (m_idx * TM, bid.x * TN))
            op = skv.matmul2d(TM, TN, K_size, simdgroups=4)
            coop = op.get_destination(a_tile, b_tile, sk.dt.float32)
            op.run(a_tile, b_tile, coop)
            coop.store(out_tile)

    np.random.seed(0)
    A_arr = np.random.randn(M_size, K_size).astype(np.float32)
    B_arr = np.random.randn(K_size, N_size).astype(np.float32)
    C_arr = np.zeros((M_size, N_size), dtype=np.float32)
    matmul_persistent_m.bind(
        grid=(N_size // TN, 1, 1),
        threadgroup=(128, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr @ B_arr, atol=1e-2, rtol=1e-2)


def test_verified_matmul_persistent_runtime_loop_rejects_undersized_loop():
    """
    Same persistent-M structure, but the loop only covers half the M
    tiles. Coverage should reject at .bind() before any GPU dispatch.
    """
    TM = TN = 64
    M_size = 128
    N_size = 128
    K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_persistent_m_short(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        # Bug: loops only over half of M.
        for m_idx in skv.range(M_size // TM // 2):
            a_tile   = A.slice((TM, K_size), (m_idx * TM, 0))
            b_tile   = B.slice((K_size, TN), (0, bid.x * TN))
            out_tile = out.slice((TM, TN),   (m_idx * TM, bid.x * TN))
            op = skv.matmul2d(TM, TN, K_size, simdgroups=4)
            coop = op.get_destination(a_tile, b_tile, sk.dt.float32)
            op.run(a_tile, b_tile, coop)
            coop.store(out_tile)

    with pytest.raises(ValueError, match="cover"):
        matmul_persistent_m_short.bind(
            grid=(N_size // TN, 1, 1),
            threadgroup=(128, 1, 1),
        )


def test_verified_matmul_k_loop():
    """
    K-loop matmul: each threadgroup computes one (TM, TN) output tile
    by accumulating over K-chunks via skv.range. The verifier must
    recognize that ParametricReduce-over-K-tiles equals the spec's
    full-K reduction (interval-merging on sum reductions).
    """
    TM = TN = 64
    TK = 64
    M_size = N_size = 128
    K_size = 128
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_k_loop(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        op = skv.matmul2d(TM, TN, TK, simdgroups=4)
        out_tile = out.slice((TM, TN), (bid.y * TM, bid.x * TN))
        # Allocate the accumulator using the first K-tile for type
        # inference (its concrete slice doesn't matter — coop starts
        # at zero anyway).
        a_seed = A.slice((TM, TK), (bid.y * TM, 0))
        b_seed = B.slice((TK, TN), (0, bid.x * TN))
        coop = op.get_destination(a_seed, b_seed, sk.dt.float32)
        for k_idx in skv.range(K_size // TK):
            a_tile = A.slice((TM, TK), (bid.y * TM, k_idx * TK))
            b_tile = B.slice((TK, TN), (k_idx * TK, bid.x * TN))
            op.run(a_tile, b_tile, coop)
        coop.store(out_tile)

    np.random.seed(0)
    A_arr = np.random.randn(M_size, K_size).astype(np.float32)
    B_arr = np.random.randn(K_size, N_size).astype(np.float32)
    C_arr = np.zeros((M_size, N_size), dtype=np.float32)
    matmul_k_loop.bind(
        grid=(N_size // TN, M_size // TM, 1),
        threadgroup=(128, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr @ B_arr, atol=1e-2, rtol=1e-2)


def test_verified_matmul_rejects_wrong_kernel():
    """
    The verifier must reject a kernel whose expression doesn't normalize
    to the declared output spec. Reuse the matmul scaffold but feed A
    in as both operands — should fail before any GPU dispatch.
    """
    M_size = N_size = K_size = 64
    M = skv.dim('M', M_size)
    N = skv.dim('N', N_size)
    K = skv.dim('K', K_size)

    @skv.jit(out_spec=skv.OutputSpec("(M K, K N -> M N)", st=(M, N)))
    def matmul_wrong(
        out : skv.DevicePointer[sk.dt.float32, (M, N)],
        A   : skv.DevicePointer[sk.dt.float32, (M, K)],
        B   : skv.DevicePointer[sk.dt.float32, (K, N)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
    ):
        a_tile   = A.slice((M_size, K_size), (0, 0))
        out_tile = out.slice((M_size, N_size), (0, 0))
        op = skv.matmul2d(M_size, N_size, K_size, simdgroups=4)
        coop = op.get_destination(a_tile, a_tile, sk.dt.float32)
        op.run(a_tile, a_tile, coop)  # A @ A, not A @ B — should be caught
        coop.store(out_tile)

    with pytest.raises(ValueError, match="does not match spec"):
        _ = matmul_wrong.metal_source  # trace fires verification

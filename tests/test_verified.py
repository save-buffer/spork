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


def test_skv_kernels_matmul():
    """
    spork.verified.kernels.matmul: the canonical verified MPP matmul,
    end-to-end. Each call enters its own stile.scope() so per-call dim
    declarations don't collide across tests.
    """
    import spork.verified.kernels as skvk

    M = N = K = 128
    np.random.seed(0)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    mm = skvk.matmul(M, N, K)
    mm(C, A, B)
    np.testing.assert_allclose(C, A @ B, atol=1e-2, rtol=1e-2)


def test_skv_kernels_matmul_validation():
    """
    skv.kernels.matmul rejects misaligned dims with a clear error.
    """
    import spork.verified.kernels as skvk

    with pytest.raises(ValueError, match="multiple"):
        skvk.matmul(128, 128, 100)
    with pytest.raises(ValueError, match="multiple"):
        skvk.matmul(120, 128, 128)


def test_verified_elementwise_add():
    """
    Element-level verified kernel: each thread reads A[i], B[i], adds,
    writes out[i]. Exercises typed pointer __getitem__/__setitem__,
    TypedScalarValue arithmetic, per-element coverage via ThreadPositionInGrid.
    """
    N_size = 1024
    N = skv.dim('N', N_size)

    @skv.jit(out_spec=skv.OutputSpec("(N + N -> N)", st=(N,)))
    def add(
        out : skv.DevicePointer[sk.dt.float32, (N,)],
        A   : skv.DevicePointer[sk.dt.float32, (N,)],
        B   : skv.DevicePointer[sk.dt.float32, (N,)],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[i] = A[i] + B[i]

    np.random.seed(0)
    A_arr = np.random.randn(N_size).astype(np.float32)
    B_arr = np.random.randn(N_size).astype(np.float32)
    C_arr = np.zeros(N_size, dtype=np.float32)
    add.bind(
        grid=(N_size // 128, 1, 1),
        threadgroup=(128, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr + B_arr)


def test_verified_elementwise_add_rejects_undersized_grid():
    """
    Element-level kernel with grid = (4, 1, 1) and threadgroup =
    (128, 1, 1) covers only 4*128 = 512 of 1024 positions. Coverage
    check should reject before any GPU dispatch.
    """
    N_size = 1024
    N = skv.dim('N', N_size)

    @skv.jit(out_spec=skv.OutputSpec("(N + N -> N)", st=(N,)))
    def add(
        out : skv.DevicePointer[sk.dt.float32, (N,)],
        A   : skv.DevicePointer[sk.dt.float32, (N,)],
        B   : skv.DevicePointer[sk.dt.float32, (N,)],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[i] = A[i] + B[i]

    with pytest.raises(ValueError, match="cover"):
        add.bind(grid=(4, 1, 1), threadgroup=(128, 1, 1))


def test_verified_elementwise_with_exp():
    """
    Exercises a typed math intrinsic (skv.exp) inside an element kernel.
    Computes ``out[i] = exp(A[i])`` — the kernel's ExprType composes to
    ``UnaryOp("exp", Tensor(N))``, matching the spec.
    """
    N_size = 1024
    N = skv.dim('N', N_size)

    @skv.jit(out_spec=skv.OutputSpec("exp(N)", st=(N,)))
    def expk(
        out : skv.DevicePointer[sk.dt.float32, (N,)],
        A   : skv.DevicePointer[sk.dt.float32, (N,)],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[i] = skv.exp(A[i])

    np.random.seed(0)
    A_arr = np.random.randn(N_size).astype(np.float32)
    C_arr = np.zeros(N_size, dtype=np.float32)
    expk.bind(
        grid=(N_size // 128, 1, 1),
        threadgroup=(128, 1, 1),
    )(C_arr, A_arr)
    np.testing.assert_allclose(C_arr, np.exp(A_arr), atol=1e-5, rtol=1e-5)


def test_verified_typed_local_accumulator():
    """
    skv.local: per-thread typed accumulator. Computes A + B by
    initializing a local to A[i] then += B[i].
    """
    N_size = 1024
    N = skv.dim('N', N_size)

    @skv.jit(out_spec=skv.OutputSpec("(N + N -> N)", st=(N,)))
    def add_via_local(
        out : skv.DevicePointer[sk.dt.float32, (N,)],
        A   : skv.DevicePointer[sk.dt.float32, (N,)],
        B   : skv.DevicePointer[sk.dt.float32, (N,)],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        acc = skv.local(sk.dt.float32, A[i])
        acc += B[i]
        out[i] = acc

    np.random.seed(0)
    A_arr = np.random.randn(N_size).astype(np.float32)
    B_arr = np.random.randn(N_size).astype(np.float32)
    C_arr = np.zeros(N_size, dtype=np.float32)
    add_via_local.bind(
        grid=(N_size // 128, 1, 1),
        threadgroup=(128, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr + B_arr)


def test_verified_typed_threadgroup_array():
    """
    skv.threadgroup: a typed threadgroup-memory scratch array. Each
    thread writes A[i] + B[i] into scratch, then reads its own slot
    back into the output (exercises typed [] read/write).
    """
    TGROUP = 128
    N_size = TGROUP
    N = skv.dim('N', N_size)

    @skv.jit(out_spec=skv.OutputSpec("(N + N -> N)", st=(N,)))
    def add_via_tg(
        out : skv.DevicePointer[sk.dt.float32, (N,)],
        A   : skv.DevicePointer[sk.dt.float32, (N,)],
        B   : skv.DevicePointer[sk.dt.float32, (N,)],
        i   : sk.Uint[sk.ThreadPositionInGrid],
    ):
        scratch = skv.threadgroup(sk.dt.float32, (N,))
        scratch[i] = A[i] + B[i]
        sk.threadgroup_barrier()
        out[i] = scratch[i]

    np.random.seed(0)
    A_arr = np.random.randn(N_size).astype(np.float32)
    B_arr = np.random.randn(N_size).astype(np.float32)
    C_arr = np.zeros(N_size, dtype=np.float32)
    add_via_tg.bind(
        grid=(1, 1, 1),
        threadgroup=(TGROUP, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr + B_arr)


def test_verified_if_thread_zero_write_pattern():
    """
    A common GQA-style pattern: each threadgroup's thread 0 does the
    writeback for its tile. Without skv.if_, coverage would (a) fail
    because the per-thread enumeration of tid would expect ALL threads
    to participate, OR (b) silently miscount. With skv.if_(tid.x == 0),
    coverage restricts tid.x to 0 for stores inside the block, so the
    per-axis intervals reflect the gated execution correctly.
    """
    TGROUP = 32
    TILE = 4
    N_size = 16  # 4 threadgroups × 4 elements/tile
    N = skv.dim('N', N_size)

    @skv.jit(out_spec=skv.OutputSpec("(N + N -> N)", st=(N,)))
    def add_thread0_writes(
        out : skv.DevicePointer[sk.dt.float32, (N,)],
        A   : skv.DevicePointer[sk.dt.float32, (N,)],
        B   : skv.DevicePointer[sk.dt.float32, (N,)],
        bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
        tid : sk.Uint2[sk.ThreadPositionInThreadgroup],
    ):
        with skv.if_(tid.x == 0):
            for i in skv.range(TILE):
                out[bid.x * TILE + i] = A[bid.x * TILE + i] + B[bid.x * TILE + i]

    np.random.seed(0)
    A_arr = np.random.randn(N_size).astype(np.float32)
    B_arr = np.random.randn(N_size).astype(np.float32)
    C_arr = np.zeros(N_size, dtype=np.float32)
    add_thread0_writes.bind(
        grid=(N_size // TILE, 1, 1),
        threadgroup=(TGROUP, 1, 1),
    )(C_arr, A_arr, B_arr)
    np.testing.assert_allclose(C_arr, A_arr + B_arr)

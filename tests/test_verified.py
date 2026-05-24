import numpy as np
import pytest

import spork as sk
from spork import verified as skv


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

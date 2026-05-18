import numpy as np
import pytest

import spork as sk


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


def test_kernels_matmul_oneshot():
    """
    sk.kernels.matmul_oneshot uses a single MPP call per output tile (no
    K-loop), with the descriptor's K equal to the full problem K.
    """
    M = N = 256
    K = 128
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul = sk.kernels.matmul_oneshot(M, N, K)
    assert matmul.grid == (M // 64, N // 64, 1)
    assert matmul.threadgroup == (128, 1, 1)
    matmul(C, A, B)
    np.testing.assert_allclose(C, A @ B, atol=1e-2, rtol=1e-2)


def test_kernels_matmul_linear_traversal():
    """
    traversal='linear' produces a working kernel and lifts the power-of-two
    constraint on M//64 and N//64.
    """
    # M=192 (=3*64) — not a power-of-two tile count
    M, N, K = 192, 256, 128
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul = sk.kernels.matmul(M, N, K, traversal="linear")
    matmul(C, A, B)
    np.testing.assert_allclose(C, A @ B, atol=1e-2, rtol=1e-2)


def test_kernels_matmul_validation():
    """
    sk.kernels.matmul raises with a clear message for unsupported shapes.
    """
    # K=200 is not a multiple of 128
    with pytest.raises(ValueError, match="multiple"):
        sk.kernels.matmul(256, 256, 200)
    # M=192 (=3*64): multiple of 64, but 192//64=3 is not a power of two
    with pytest.raises(ValueError, match="power"):
        sk.kernels.matmul(192, 256, 256)
    # Linear lifts the power-of-two constraint
    sk.kernels.matmul(192, 256, 256, traversal="linear")  # should not raise
    # Bogus traversal name rejected
    with pytest.raises(ValueError, match="traversal"):
        sk.kernels.matmul(256, 256, 256, traversal="spiral")


def _causal_gqa_numpy(Q, K, V):
    """
    Numpy reference for causal GQA. Q/K/V are (heads, ctx, dhead) with
    Q's head dim a multiple of K/V's. Returns the same shape as Q.
    """
    nq, qctx, dhead = Q.shape
    nkv, nctx, _ = K.shape
    assert nq % nkv == 0
    nq_per_kv = nq // nkv

    O = np.zeros_like(Q)
    causal = np.arange(nctx)[None, :] > np.arange(qctx)[:, None]
    scale = 1.0 / np.sqrt(dhead)
    for h in range(nq):
        kvh = h // nq_per_kv
        scores = (Q[h] @ K[kvh].T) * scale  # (qctx, nctx)
        scores = np.where(causal, -np.inf, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        P = np.exp(scores)
        P /= P.sum(axis=-1, keepdims=True)
        O[h] = P @ V[kvh]
    return O


def test_kernels_causal_gqa():
    """
    Fused FlashAttention-2 with MPP matmul2d for Q@K^T and P@V. Tests
    against a numpy reference at small dims.
    """
    nq, nkv = 4, 2
    qctx, nctx = 64, 128
    dhead = 128

    np.random.seed(0)
    Q = np.random.randn(nq, qctx, dhead).astype(np.float32)
    K = np.random.randn(nkv, nctx, dhead).astype(np.float32)
    V = np.random.randn(nkv, nctx, dhead).astype(np.float32)
    O = np.zeros((nq, qctx, dhead), dtype=np.float32)

    expected = _causal_gqa_numpy(Q, K, V)

    gqa = sk.kernels.causal_gqa(nq, nkv, qctx, nctx, dhead)
    assert gqa.grid == (nq, qctx // 64, 1)
    assert gqa.threadgroup == (128, 1, 1)
    gqa(O, Q, K, V)

    np.testing.assert_allclose(O, expected, atol=1e-3, rtol=1e-3)


def test_kernels_causal_gqa_validation():
    """
    causal_gqa raises clearly for unsupported shape combinations.
    """
    with pytest.raises(ValueError, match="multiple of nkv"):
        sk.kernels.causal_gqa(nq=3, nkv=2, qctx=64, nctx=128, dhead=128)
    with pytest.raises(ValueError, match="block_m"):
        sk.kernels.causal_gqa(nq=4, nkv=2, qctx=33, nctx=128, dhead=128)
    with pytest.raises(ValueError, match="block_n"):
        sk.kernels.causal_gqa(nq=4, nkv=2, qctx=64, nctx=100, dhead=128)

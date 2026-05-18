import math

from .. import dtypes as dt
from ..jit import BoundKernel, jit
from ..tracer import (
    exp as sk_exp,
    if_,
    local,
    matmul2d,
    range,
    tensor,
    threadgroup,
    threadgroup_barrier,
)
from ..types import (
    DevicePointer,
    ThreadPositionInThreadgroup,
    ThreadgroupPositionInGrid,
    Uint,
    Uint2,
)


# Sentinel for masked scores. Metal lacks a portable "-INFINITY" literal in
# our IR yet; a sufficiently-large negative value flushes ``exp()`` to zero
# and contributes nothing to the row sum.
_NEG_INF = -1.0e30

_SIMDGROUPS = 4
_THREADGROUP_SIZE = 32 * _SIMDGROUPS


def causal_gqa(
    nq      : int,
    nkv     : int,
    qctx    : int,
    nctx    : int,
    dhead   : int,
    dtype   : dt.Dtype = dt.float32,
    *,
    block_m : int = 64,
    block_n : int = 64,
) -> BoundKernel:
    """
    Causal Grouped-Query Attention, fused FlashAttention-2 style.

    Uses MPP ``matmul2d`` cooperative tensors for both the Q @ K^T scores
    matmul and the P @ V output matmul; runs the online-softmax pass on
    threadgroup-memory scratch between them. Each threadgroup processes one
    ``(q_head, q_tile)`` pair, producing ``block_m`` rows of output for that
    query head.

    Shapes (all row-major, C-contiguous):
      Q   : (nq, qctx, dhead)
      K   : (nkv, nctx, dhead)
      V   : (nkv, nctx, dhead)
      out : (nq, qctx, dhead)

    ``nq`` must be a multiple of ``nkv`` (each KV head serves
    ``nq // nkv`` consecutive query heads).

    Constraints (current implementation):
      - ``qctx % block_m == 0`` and ``nctx % block_n == 0``
      - ``dhead``, ``block_m``, ``block_n`` all valid MPP ``matmul2d`` tile
        sizes (64 and 128 are known-good)

    The softmax pass currently runs on a single thread per threadgroup;
    correctness comes first, perf optimization is a follow-up.
    """
    if nq % nkv != 0:
        raise ValueError(
            f"causal_gqa: nq must be a multiple of nkv (got nq={nq}, nkv={nkv})"
        )
    if qctx % block_m != 0:
        raise ValueError(
            f"causal_gqa: qctx must be a multiple of block_m (got qctx={qctx}, "
            f"block_m={block_m})"
        )
    if nctx % block_n != 0:
        raise ValueError(
            f"causal_gqa: nctx must be a multiple of block_n (got nctx={nctx}, "
            f"block_n={block_n})"
        )

    nq_per_kv = nq // nkv
    n_kv_tiles = nctx // block_n
    inv_sqrt_d = 1.0 / math.sqrt(dhead)

    @jit
    def causal_gqa_kernel(
        out : DevicePointer[dtype],
        Q   : DevicePointer[dtype],
        K   : DevicePointer[dtype],
        V   : DevicePointer[dtype],
        bid : Uint2[ThreadgroupPositionInGrid],
        tid : Uint[ThreadPositionInThreadgroup],
    ):
        iqhead = bid.x
        iqtile = bid.y
        ikvhead = iqhead // nq_per_kv

        # MPP tensor views over device pointers. We treat each (head, ctx,
        # dhead) tensor as a 2-D matrix of shape (head*ctx, dhead) — MPP's
        # extents convention puts the inner/contiguous dim first.
        tQ = tensor(Q, dtype, (dhead, nq  * qctx))
        tK = tensor(K, dtype, (dhead, nkv * nctx))
        tV = tensor(V, dtype, (dhead, nkv * nctx))

        # Threadgroup-memory scratch.
        scores_tg = threadgroup(dtype, (block_m, block_n))
        pv_tg     = threadgroup(dtype, (block_m, dhead))
        O_tg      = threadgroup(dtype, (block_m, dhead))
        max_tg    = threadgroup(dtype, (block_m,))
        sum_tg    = threadgroup(dtype, (block_m,))

        tS  = tensor(scores_tg, dtype, (block_n, block_m))
        tPV = tensor(pv_tg,     dtype, (dhead,   block_m))

        # Initialize running stats and accumulator on thread 0.
        with if_(tid == 0):
            for i in range(block_m):
                max_tg[i] = _NEG_INF
                sum_tg[i] = 0.0
                for d in range(dhead):
                    O_tg[i, d] = 0.0
        threadgroup_barrier()

        op_s  = matmul2d(block_m, block_n, dhead,  simdgroups=_SIMDGROUPS)
        op_pv = matmul2d(block_m, dhead,  block_n, simdgroups=_SIMDGROUPS)

        # Q-tile slice (same every iteration).
        q_offset = iqhead * qctx + iqtile * block_m
        q_tile = tQ.slice((dhead, block_m), (0, q_offset))

        for j in range(n_kv_tiles):
            kv_offset = ikvhead * nctx + j * block_n

            # ----- S = Q @ K^T -----
            coop_s = op_s.get_destination(tQ, tK, dtype)
            k_tile = tK.slice((dhead, block_n), (0, kv_offset))
            op_s.run(q_tile, k_tile, coop_s)
            coop_s.store(tS.slice((block_n, block_m), (0, 0)))
            threadgroup_barrier()

            # ----- Causal mask + online softmax (thread 0 serial) -----
            with if_(tid == 0):
                for i in range(block_m):
                    q_pos = iqtile * block_m + i

                    # Apply causal mask + 1/sqrt(d) scale, find row max
                    # of the freshly-computed S block.
                    block_max = local(dtype, _NEG_INF)
                    for jj in range(block_n):
                        k_pos = j * block_n + jj
                        s = local(dtype, scores_tg[i, jj] * inv_sqrt_d)
                        with if_(k_pos > q_pos):
                            s.assign(_NEG_INF)
                        scores_tg[i, jj] = s
                        with if_(s > block_max):
                            block_max.assign(s)

                    old_max = max_tg[i]
                    new_max = local(dtype, old_max)
                    with if_(block_max > old_max):
                        new_max.assign(block_max)

                    # Scale factor for the running stats and accumulator.
                    alpha = local(dtype, sk_exp(old_max - new_max))

                    # Convert S → P (in place) and accumulate row sum.
                    # ``p`` must be materialized: otherwise the second use
                    # would re-read the now-updated scores_tg cell and
                    # compute exp(p - new_max) instead of exp(s - new_max).
                    block_sum = local(dtype, 0.0)
                    for jj in range(block_n):
                        p = local(dtype, sk_exp(scores_tg[i, jj] - new_max))
                        scores_tg[i, jj] = p
                        block_sum += p

                    sum_tg[i] = alpha * sum_tg[i] + block_sum
                    max_tg[i] = new_max

                    # Rescale running O by alpha (we'll add P @ V below).
                    for d in range(dhead):
                        O_tg[i, d] = alpha * O_tg[i, d]
            threadgroup_barrier()

            # ----- PV = P @ V -----
            coop_pv = op_pv.get_destination(tS, tV, dtype)
            v_tile  = tV.slice((dhead, block_n), (0, kv_offset))
            p_tile  = tS.slice((block_n, block_m), (0, 0))
            op_pv.run(p_tile, v_tile, coop_pv)
            coop_pv.store(tPV.slice((dhead, block_m), (0, 0)))
            threadgroup_barrier()

            # Add PV into running O (thread 0 serial).
            with if_(tid == 0):
                for i in range(block_m):
                    for d in range(dhead):
                        O_tg[i, d] = O_tg[i, d] + pv_tg[i, d]
            threadgroup_barrier()

        # Normalize by running sum and write O to device (thread 0 serial).
        with if_(tid == 0):
            for i in range(block_m):
                inv = local(dtype, 1.0 / sum_tg[i])
                out_row = local(dt.uint32, iqhead * qctx + iqtile * block_m + i)
                for d in range(dhead):
                    out[out_row * dhead + d] = O_tg[i, d] * inv

    return causal_gqa_kernel.bind(
        grid=(nq, qctx // block_m, 1),
        threadgroup=(_THREADGROUP_SIZE, 1, 1),
    )

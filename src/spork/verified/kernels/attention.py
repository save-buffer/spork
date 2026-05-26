"""
Verified non-causal attention.

Spec::

    (softmax[nctx](
        (qctx dhead, nctx dhead -> qctx nctx) / sqrt(D)
     ), nctx dhead -> qctx dhead)

Strategy:

  1. ``Q @ K^T`` via MPP ``matmul2d`` (with transpose_b=True since K is
     stored as ``(nctx, dhead)`` row-major). Result lives in a typed
     cooperative tensor whose stile expression is the parsed
     ``(qctx dhead, nctx dhead -> qctx nctx)`` einsum.
  2. Cooperative-tensor ``.store`` writes the scores to a typed
     threadgroup-memory tile. The verifier checks the coop's einsum
     ExprType matches the tile's declared type.
  3. **Softmax on thread 0** using raw spork loops over the tile (since
     we don't yet have tile-level typed reductions). The body
     subtracts each row's max, exps, normalizes by the row sum.
  4. ``skv.with_type`` (trust-me) asserts that ``scores_tg`` now holds
     ``softmax[nctx](QK / sqrt(D))``. The verifier doesn't check the
     raw softmax body matches; the user is asserting it.
  5. ``P @ V`` via MPP ``matmul2d``; the verifier composes
     ``softmax(QK/sqrt(D)) @ V`` and matches it against the spec.
  6. Final coop store to the output with full grid coverage.

Constraints:
  - Single-threadgroup kernel: dims must fit MPP tile budget.
    ``qctx`` <= 64, ``nctx`` <= 64, ``dhead`` <= 128 typically work.
  - Threadgroup-memory budget: ``qctx * nctx * 4`` bytes <= ~30 KB.

Follow-ups before this is "real" attention:
  - Tile the qctx and nctx dims so the kernel handles arbitrary sizes.
  - Build tile-level typed reductions (``.max(dim)`` / ``.sum(dim)``) so
    the softmax body is verified, not trust-me'd.
  - Add causal mask via ``skv.if_`` or a typed mask primitive.
"""

import math

import stile

import spork as sk
from spork.jit import BoundKernel

from .. import (
    DevicePointer,
    OutputSpec,
    dim,
    exp,
    if_,
    jit,
    local,
    matmul2d,
    maximum,
    threadgroup,
    with_type,
)


def attention(
    qctx_size  : int,
    nctx_size  : int,
    dhead_size : int,
    dtype      = None,
) -> BoundKernel:
    """
    Verified single-threadgroup attention. See module docstring for the
    spec and strategy.
    """
    if dtype is None:
        dtype = sk.dt.float32

    with stile.scope():
        qctx  = dim('qctx',  qctx_size)
        nctx  = dim('nctx',  nctx_size)
        dhead = dim('dhead', dhead_size)

        QK_SPEC = "(qctx dhead, nctx dhead -> qctx nctx)"
        P_SPEC = (
            f"softmax[nctx]("
            f"(qctx dhead, nctx dhead -> qctx nctx) / sqrt({dhead_size})"
            f")"
        )
        OUT_SPEC = (
            f"(softmax[nctx]("
            f"(qctx dhead, nctx dhead -> qctx nctx) / sqrt({dhead_size})"
            f"), nctx dhead -> qctx dhead)"
        )

        inv_sqrt_d = 1.0 / math.sqrt(dhead_size)

        @jit(out_spec=OutputSpec(OUT_SPEC, st=(qctx, dhead)))
        def attn(
            out : DevicePointer[dtype, (qctx, dhead)],
            Q   : DevicePointer[dtype, (qctx, dhead)],
            K   : DevicePointer[dtype, (nctx, dhead)],
            V   : DevicePointer[dtype, (nctx, dhead)],
            tid : sk.Uint2[sk.ThreadPositionInThreadgroup],
        ):
            # ----- Q @ K^T -----
            op_s = matmul2d(
                qctx_size, nctx_size, dhead_size,
                simdgroups=4, transpose_b=True,
            )
            q_tile = Q.slice((qctx_size, dhead_size), (0, 0))
            k_tile = K.slice((nctx_size, dhead_size), (0, 0))
            coop_s = op_s.get_destination(q_tile, k_tile, dtype)
            op_s.run(q_tile, k_tile, coop_s)

            # Spill scores to a typed threadgroup-memory tile.
            scores_tg = threadgroup(dtype, (qctx, nctx))
            scores_view = with_type(
                scores_tg, QK_SPEC, st=(qctx, nctx), dtype=dtype,
            )
            coop_s.store(scores_view.slice((qctx_size, nctx_size), (0, 0)))
            sk.threadgroup_barrier()

            # ----- Softmax on thread 0 (trust-me body) -----
            with if_(tid.x == 0):
                for q in range(qctx_size):
                    # Row max of (scores * inv_sqrt_d).
                    row_max = local(sk.dt.float32, -1.0e30)
                    for n in range(nctx_size):
                        s = scores_tg[q, n] * inv_sqrt_d
                        row_max.assign(maximum(row_max, s))
                    # exp(s - max) in place, accumulate row sum.
                    # ``e`` is materialized into a local so that
                    # ``scores_tg[q, n] = e`` writing back doesn't make
                    # ``row_sum += e`` re-read the now-overwritten cell.
                    row_sum = local(sk.dt.float32, 0.0)
                    for n in range(nctx_size):
                        s = scores_tg[q, n] * inv_sqrt_d - row_max
                        e = local(sk.dt.float32, exp(s))
                        scores_tg[q, n] = e
                        row_sum += e
                    # Normalize by row sum.
                    inv = local(sk.dt.float32, 1.0 / row_sum)
                    for n in range(nctx_size):
                        scores_tg[q, n] = scores_tg[q, n] * inv
            sk.threadgroup_barrier()

            # ----- Assert P = softmax(QK / sqrt(D)) -----
            tP = with_type(scores_tg, P_SPEC, st=(qctx, nctx), dtype=dtype)

            # ----- P @ V → coop_o -----
            op_pv = matmul2d(qctx_size, dhead_size, nctx_size, simdgroups=4)
            p_tile = tP.slice((qctx_size, nctx_size), (0, 0))
            v_tile = V.slice((nctx_size, dhead_size), (0, 0))
            coop_pv = op_pv.get_destination(p_tile, v_tile, dtype)
            op_pv.run(p_tile, v_tile, coop_pv)

            # ----- Store output -----
            out_tile = out.slice((qctx_size, dhead_size), (0, 0))
            coop_pv.store(out_tile)

        return attn.bind(grid=(1, 1, 1), threadgroup=(128, 1, 1))

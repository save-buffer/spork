#!/usr/bin/env python
"""
Profile sk.kernels.causal_gqa under sk.profile and open the trace in Xcode.

Usage:
    MTL_CAPTURE_ENABLED=1 uv run python bench/causal_gqa.py
    MTL_CAPTURE_ENABLED=1 uv run python bench/causal_gqa.py 1024
    MTL_CAPTURE_ENABLED=1 uv run python bench/causal_gqa.py 8 2 128 512 64

Runs the kernel once, verifies correctness against a numpy reference,
captures a single GPU trace, and opens it in Xcode. Read kernel runtime,
occupancy, memory traffic, etc. from the Metal debugger.

Argument forms:
    (no args)         → defaults: nq=8 nkv=2 qctx=128 nctx=512 dhead=64
    1 arg: nctx       → other dims defaulted, only nctx varies
    5 args            → nq nkv qctx nctx dhead

Constraints to keep the kernel within the 32 KB threadgroup-memory cap:
    block_m * dhead * 2 + block_m * block_n + 2 * block_m ≤ ~8000 floats.
    With default block_m=block_n=32, dhead up to ~96 fits.

If ``MTL_CAPTURE_ENABLED=1`` is not set in the environment, Metal will
refuse to start a capture and the script will tell you what to do.
"""

import os
import sys

import numpy as np

import spork as sk


def _causal_gqa_numpy(Q, K, V):
    """
    Numpy reference matching the spork kernel's semantics.
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
        scores = (Q[h] @ K[kvh].T) * scale
        scores = np.where(causal, -np.inf, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        P = np.exp(scores)
        P /= P.sum(axis=-1, keepdims=True)
        O[h] = P @ V[kvh]
    return O


def run(nq : int, nkv : int, qctx : int, nctx : int, dhead : int) -> None:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        print(
            "MTL_CAPTURE_ENABLED is not set. Re-run with:\n"
            f"    MTL_CAPTURE_ENABLED=1 uv run python {sys.argv[0]} "
            f"{nq} {nkv} {qctx} {nctx} {dhead}",
            file=sys.stderr,
        )
        sys.exit(1)

    np.random.seed(0)
    Q = np.random.randn(nq, qctx, dhead).astype(np.float32)
    K = np.random.randn(nkv, nctx, dhead).astype(np.float32)
    V = np.random.randn(nkv, nctx, dhead).astype(np.float32)
    O = np.zeros((nq, qctx, dhead), dtype=np.float32)

    # Full (non-causal) FLOP count — useful for back-of-envelope throughput.
    # Causal halves the effective work, but standard attention bench numbers
    # ignore that.
    flops = nq * (4 * qctx * nctx * dhead + 3 * qctx * nctx)

    gqa = sk.kernels.causal_gqa(nq, nkv, qctx, nctx, dhead)
    print(
        f"causal_gqa  nq={nq} nkv={nkv} qctx={qctx} nctx={nctx} dhead={dhead}  "
        f"({flops / 1e9:.3f} GFLOP)"
    )
    print(f"  grid={gqa.grid}  threadgroup={gqa.threadgroup}")

    trace_name = f"causal_gqa_q{nq}kv{nkv}_qc{qctx}_nc{nctx}_d{dhead}"
    with sk.profile(name=trace_name) as trace_path:
        gqa(O, Q, K, V)

    expected = _causal_gqa_numpy(Q, K, V)
    max_err = float(np.max(np.abs(O - expected)))
    if np.allclose(O, expected, atol=1e-3, rtol=1e-3):
        print(f"  correctness: OK  (max abs error vs numpy: {max_err:.2e})")
    else:
        print(f"  correctness: MISMATCH  (max abs error vs numpy: {max_err:.2e})")
        sys.exit(2)

    print(f"  trace: {trace_path}  (opening in Xcode...)")


def main() -> None:
    args = [int(a) for a in sys.argv[1:]]
    if len(args) == 0:
        run(nq=8, nkv=2, qctx=128, nctx=512, dhead=64)
    elif len(args) == 1:
        run(nq=8, nkv=2, qctx=128, nctx=args[0], dhead=64)
    elif len(args) == 5:
        run(*args)
    else:
        print(
            "Usage: causal_gqa.py | causal_gqa.py [nctx] | "
            "causal_gqa.py [nq nkv qctx nctx dhead]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

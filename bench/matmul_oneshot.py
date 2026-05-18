#!/usr/bin/env python
"""
Profile sk.kernels.matmul_oneshot under sk.profile and open the trace in Xcode.

Usage:
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul_oneshot.py
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul_oneshot.py 256
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul_oneshot.py 256 256 128

Raw MPP ``matmul2d`` per output tile, no K-loop. Runs the kernel once,
verifies correctness against numpy, captures a single GPU trace, and opens
it in Xcode. Read kernel runtime, occupancy, memory traffic, etc. from the
Metal debugger.

If ``MTL_CAPTURE_ENABLED=1`` is not set in the environment, Metal will
refuse to start a capture and the script will tell you what to do.
"""

import os
import sys

import numpy as np

import spork as sk


def run(M : int, N : int, K : int) -> None:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        print(
            "MTL_CAPTURE_ENABLED is not set. Re-run with:\n"
            f"    MTL_CAPTURE_ENABLED=1 uv run python {sys.argv[0]} "
            f"{M} {N} {K}",
            file=sys.stderr,
        )
        sys.exit(1)

    np.random.seed(0)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul = sk.kernels.matmul_oneshot(M, N, K)
    flops = 2 * M * N * K
    print(f"matmul_oneshot  M={M} N={N} K={K}  ({flops/1e9:.2f} GFLOP)")
    print(f"  grid={matmul.grid}  threadgroup={matmul.threadgroup}")

    trace_name = f"matmul_oneshot_M{M}_N{N}_K{K}"
    with sk.profile(name=trace_name) as trace_path:
        matmul(C, A, B)

    expected = A @ B
    max_err = float(np.max(np.abs(C - expected)))
    if np.allclose(C, expected, atol=1e-2, rtol=1e-2):
        print(f"  correctness: OK  (max abs error vs numpy: {max_err:.2e})")
    else:
        print(f"  correctness: MISMATCH  (max abs error vs numpy: {max_err:.2e})")
        sys.exit(2)

    print(f"  trace: {trace_path}  (opening in Xcode...)")


def main() -> None:
    args = [int(a) for a in sys.argv[1:]]
    if len(args) == 0:
        run(256, 256, 128)
    elif len(args) == 1:
        run(args[0], args[0], 128)
    elif len(args) == 3:
        run(*args)
    else:
        print(
            "Usage: matmul_oneshot.py [N] | matmul_oneshot.py [M N K]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Profile sk.kernels.matmul under sk.profile and open the trace in Xcode.

Usage:
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul.py
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul.py 1024
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul.py 2048 2048 2048
    MTL_CAPTURE_ENABLED=1 uv run python bench/matmul.py 1024 1024 1024 linear

Runs the matmul once, verifies correctness against numpy, captures a single
GPU trace, and opens it in Xcode. Read kernel runtime, occupancy, memory
traffic, etc. from the Metal debugger.

The optional final argument selects the tile traversal pattern
(``zorder`` (default) or ``linear``).

If ``MTL_CAPTURE_ENABLED=1`` is not set in the environment, Metal will
refuse to start a capture and the script will tell you what to do.
"""

import os
import sys

import numpy as np

import spork as sk


def run(M : int, N : int, K : int, traversal : str) -> None:
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        print(
            "MTL_CAPTURE_ENABLED is not set. Re-run with:\n"
            f"    MTL_CAPTURE_ENABLED=1 uv run python {sys.argv[0]} "
            f"{M} {N} {K} {traversal}",
            file=sys.stderr,
        )
        sys.exit(1)

    np.random.seed(0)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul = sk.kernels.matmul(M, N, K, traversal=traversal)
    flops = 2 * M * N * K
    print(f"matmul  M={M} N={N} K={K}  traversal={traversal}  ({flops/1e9:.2f} GFLOP)")
    print(f"  grid={matmul.grid}  threadgroup={matmul.threadgroup}")

    trace_name = f"matmul_M{M}_N{N}_K{K}_{traversal}"
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


def _parse_args(argv : list) -> tuple:
    """
    Split positional dimension ints from an optional trailing traversal name.
    Returns (dim_ints, traversal).
    """
    dims : list = []
    traversal = "zorder"
    for a in argv:
        if a.isdigit() or (a.startswith("-") and a[1:].isdigit()):
            dims.append(int(a))
        else:
            traversal = a
    return dims, traversal


def main() -> None:
    dims, traversal = _parse_args(sys.argv[1:])
    if len(dims) == 0:
        run(1024, 1024, 1024, traversal)
    elif len(dims) == 1:
        run(dims[0], dims[0], dims[0], traversal)
    elif len(dims) == 3:
        run(dims[0], dims[1], dims[2], traversal)
    else:
        print(
            "Usage: matmul.py [N] [traversal] | matmul.py [M N K] [traversal]",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

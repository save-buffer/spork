import math
from typing import Sequence

from .. import dtypes as dt
from ..jit import BoundKernel, jit
from ..types import (
    DevicePointer,
    ThreadPositionInGrid,
    Uint,
)


def matrix_add(
    shape : Sequence[int],
    dtype : dt.Dtype = dt.float32,
) -> BoundKernel:
    """
    Elementwise add: ``out = A + B`` over arrays of the given shape.

    The shape is only used to pick a sensible launch geometry; the kernel
    itself treats inputs as flat 1-D buffers.

    Threadgroup size defaults to 128 but is rounded down to the largest
    power of two that divides ``prod(shape)``, so any shape with a
    power-of-two element count works.
    """
    n = int(math.prod(shape))
    if n <= 0:
        raise ValueError(f"matrix_add: shape {tuple(shape)!r} has zero elements")

    tg = 128
    while n % tg != 0 and tg > 1:
        tg //= 2

    @jit
    def matrix_add_kernel(
        out : DevicePointer[dtype],
        A   : DevicePointer[dtype],
        B   : DevicePointer[dtype],
        i   : Uint[ThreadPositionInGrid],
    ):
        out[i] = A[i] + B[i]

    return matrix_add_kernel.bind(
        grid=(n // tg, 1, 1),
        threadgroup=(tg, 1, 1),
    )

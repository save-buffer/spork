import numpy as np

import spork as sk


def test_matrix_add():
    shape = (1024, 1024)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    expected = A + B

    @sk.jit
    def matrix_add(
        out   : sk.DevicePointer[sk.dt.float32],
        A     : sk.DevicePointer[sk.dt.float32],
        B     : sk.DevicePointer[sk.dt.float32],
        index : sk.Uint[sk.ThreadPositionInGrid],
    ):
        out[index] = A[index] + B[index]


    matrix_add[
        (int(np.prod(shape)) // 128, 1, 1),
        (128, 1, 1),
    ](
        C,
        A,
        B,
    )

    np.testing.assert_allclose(
        C,
        expected,
    )

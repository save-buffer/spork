"""
spork.verified — Stile-verified versions of spork primitives and kernels.

This subpackage requires the optional ``verified`` extra::

    pip install 'spork-metal[verified]'

which pulls in ``stile-verifier``. Importing ``spork.verified`` without
stile installed raises an ImportError pointing at the right install
command.

API surface (current scaffolding):

  - ``skv.dim(name, size)`` — re-export of stile's dim factory.
  - ``skv.OutputSpec(spec, st, dt=None)`` — declares the expected output.
  - ``skv.DevicePointer[dtype, shape]`` — typed pointer parameter; the
    shape is a tuple of stile dims.
  - ``@skv.jit(out_spec=...)`` — verified-kernel decorator.
  - ``skv.tensor(...)`` — wrap a spork pointer / threadgroup array as a
    typed tensor view.
  - ``TypedTensorHandle.assign(...)`` — fires per-tile verification.

Coverage tracking across grid (proving all threadgroups together write
every output element exactly once) is the planned follow-up; will fire
at ``.bind(grid=...)``.
"""

try:
    import stile as _stile  # noqa: F401
except ImportError as e:
    raise ImportError(
        "spork.verified requires the 'verified' extra. "
        "Install with:\n"
        "    pip install 'spork-metal[verified]'\n"
        "or, with uv:\n"
        "    uv add 'spork-metal[verified]'"
    ) from e


from ._backend import (
    DevicePointer,
    OutputSpec,
    TypedDevicePointerSpec,
    dim,
)
from .jit import jit
from .primitives import (
    TypedCooperativeTensor,
    TypedMatmulOp,
    TypedTensorHandle,
    TypedTileSlice,
    matmul2d,
    tensor,
)


__all__ = [
    "dim",
    "OutputSpec",
    "DevicePointer",
    "TypedDevicePointerSpec",
    "jit",
    "tensor",
    "matmul2d",
    "TypedTensorHandle",
    "TypedTileSlice",
    "TypedMatmulOp",
    "TypedCooperativeTensor",
]

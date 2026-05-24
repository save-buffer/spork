"""
spork.verified._backend — typed-primitive wrappers over spork's tracer,
mirroring the structure of stile.jax.pallas.

Design (per discussion):
  - Shapes live on type annotations: ``DevicePointer[dtype, shape]``.
  - One decorator: ``@verified.jit(out_spec=OutputSpec(...))``.
  - Per-tile verification fires at ``tOut.assign(value)`` during trace;
    coverage verification fires later at ``.bind(grid=..., tg=...)``
    (follow-up).

Internal layout mirrors stile's existing backends:
  - Data structures (OutputSpec, DevicePointer 2-arg, dim re-export) here.
  - jit decorator + typed kernel-param injection in jit.py.
  - Typed primitive wrappers (tensor, slice, matmul2d, exp, sum, ...) in
    primitives.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

# Stile core. Imported here (not lazily) — guarded at the package level
# by spork/verified/__init__.py's import gate.
from stile import dim as _stile_dim
from stile.type import FullDim, ShapeType

from .. import dtypes as dt
from ..types import DevicePointerSpec


# Re-exports so users don't have to import stile separately for the
# common cases.
dim = _stile_dim


@dataclass
class OutputSpec:
    """
    Declares the *expected* output of a verified spork kernel.

    Attributes:
      spec  : a stile-spec-language string describing the value the output
        should hold (e.g. ``"(M K, K N -> M N)"``).
      st    : the output's ShapeType (the dim signature, possibly sliced).
        Doesn't have to match ``spec`` verbatim — per-tile slice
        overrides happen via ``override_dims_in_type``.
      dtype : the output's spork dtype (defaults to float32).
    """
    spec  : str
    st    : ShapeType
    dtype : Optional[dt.Dtype] = None


@dataclass(frozen=True)
class TypedDevicePointerSpec:
    """
    The parameter spec for ``skv.DevicePointer[dtype, shape]``: a regular
    spork DevicePointer plus a tuple of stile dims describing the tensor
    view this parameter should be wrapped as.
    """
    dtype : dt.Dtype
    shape : Tuple[FullDim, ...]


class _DevicePointer:
    """
    Typed analog of ``spork.DevicePointer``. Subscript form:

        skv.DevicePointer[sk.dt.float32, (M, K)]

    Resolves to a ``TypedDevicePointerSpec`` that the ``@skv.jit``
    decorator picks up during kernel-signature inspection. The
    underlying spork ``DevicePointer[dtype]`` is still what's emitted
    in the generated Metal — the stile shape is purely meta.
    """

    def __class_getitem__(cls, args):
        if not isinstance(args, tuple) or len(args) != 2:
            raise TypeError(
                "skv.DevicePointer expects two subscripts: "
                "skv.DevicePointer[dtype, shape]"
            )
        dtype_arg, shape_arg = args
        if not isinstance(dtype_arg, dt.Dtype):
            raise TypeError(
                f"skv.DevicePointer dtype must be a spork dtype, got {dtype_arg!r}"
            )
        if isinstance(shape_arg, FullDim):
            shape_arg = (shape_arg,)
        elif not isinstance(shape_arg, tuple):
            shape_arg = tuple(shape_arg)
        for d in shape_arg:
            if not isinstance(d, FullDim):
                raise TypeError(
                    f"skv.DevicePointer shape must be a tuple of stile FullDim "
                    f"(from skv.dim(name, size)), got element {d!r}"
                )
        return TypedDevicePointerSpec(dtype=dtype_arg, shape=shape_arg)


DevicePointer = _DevicePointer


def _untyped_pointer_spec(typed : TypedDevicePointerSpec) -> DevicePointerSpec:
    """
    Lower a typed pointer spec to the plain spork DevicePointerSpec — what
    the spork tracer + codegen need. The stile shape is dropped at this
    point; @skv.jit keeps it separately for trace-time wrapping.
    """
    return DevicePointerSpec(dtype=typed.dtype)

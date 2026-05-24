"""
Typed-primitive wrappers over spork's tracer.

These are the per-op analogs of ``spork.tracer.tensor``, ``.slice``,
``ThreadgroupArray``, ``matmul2d``, etc. — but each one carries a stile
``Type`` alongside the spork IR it emits, so the chain composes into a
verifiable expression.

Only the minimum needed to land a first verified kernel ships here.
Additional primitives (typed simd, typed matmul2d, typed threadgroup
arrays, etc.) follow in subsequent commits.
"""

from typing import Optional, Tuple

from stile.type import (
    FullDim,
    ShapeType,
    Tensor,
    Type,
    as_int,
    dim_full_dim,
    dim_size,
)
from stile.specification import parse_spec_into_type
from stile.verification import verify_types_equivalent

from .. import dtypes as dt
from .. import tracer as _spork_tracer
from ._backend import OutputSpec


class TypedTensorHandle:
    """
    A spork ``TensorHandle`` paired with a stile ``Type``.

    The wrapped ``TensorHandle`` is what spork's codegen and MPP ops
    operate on; the ``Type`` is what the verifier sees.
    """

    __slots__ = ("_handle", "_type")

    def __init__(
        self,
        handle : _spork_tracer.TensorHandle,
        type   : Type,
    ):
        self._handle = handle
        self._type = type

    @property
    def type(self) -> Type:
        return self._type

    @property
    def handle(self) -> _spork_tracer.TensorHandle:
        return self._handle

    def assign(self, value : "TypedTensorHandle | _ValueTyped") -> None:
        """
        Verify ``value``'s stile type matches this output's expected type,
        then no-op on the spork side (the value has already been written
        through the underlying tensor handle during its computation).

        Mirrors ``stile.jax.pallas.TypedOutputRef.assign``: the
        verification fires before any memory effect; if types don't
        normalize to the same expression, an exception is raised before
        the kernel is allowed to compile.
        """
        if not isinstance(value, (TypedTensorHandle, _ValueTyped)):
            raise TypeError(
                "TypedTensorHandle.assign expects a typed value (typed "
                "tensor handle or a typed scalar/array), got "
                f"{type(value).__name__}"
            )
        if not verify_types_equivalent(self._type, value._type):
            raise ValueError(
                "Verified kernel output does not match spec.\n"
                f"  Expected: {self._type.et}\n"
                f"  Actual  : {value._type.et}"
            )


class _ValueTyped:
    """
    A typed in-flight value (the result of a typed op chain). Carries a
    stile ``Type`` and a placeholder reference to whatever the spork
    side produced — opaque to the verifier.

    Used as the right-hand side of ``TypedTensorHandle.assign(...)``.
    """

    __slots__ = ("_type", "_payload")

    def __init__(self, type : Type, payload):
        self._type = type
        self._payload = payload


def tensor(
    typed_pointer,
    shape : Optional[Tuple[FullDim, ...]] = None,
    *,
    dtype : Optional[dt.Dtype] = None,
    expected_spec : Optional[str] = None,
) -> TypedTensorHandle:
    """
    Wrap a spork pointer tracer (or threadgroup array) as a typed tensor
    view.

    For kernel parameters declared as ``skv.DevicePointer[dt, shape]``,
    ``@skv.jit`` already knows the shape and auto-wraps the parameter as
    a TypedTensorHandle — you typically just use the parameter directly.
    Call ``skv.tensor(...)`` explicitly for threadgroup-memory arrays or
    for re-wrapping with an overridden shape.

    If ``expected_spec`` is given, the returned handle's ExprType is
    seeded from the parsed spec (useful for outputs whose ExprType
    should be the *spec's* expression, not the trivial input tensor).
    Otherwise the ExprType is the bare input ``Tensor(name)``.
    """
    if isinstance(typed_pointer, TypedTensorHandle):
        return typed_pointer  # already wrapped

    shape_tuple = _normalize_shape(shape)
    if dtype is None:
        # Try to infer from the underlying pointer
        underlying_dtype = getattr(typed_pointer, "_dtype", None)
        if underlying_dtype is None:
            raise TypeError(
                "skv.tensor: cannot infer dtype; pass dtype=... explicitly"
            )
        dtype = underlying_dtype

    # Build the underlying spork TensorHandle.
    int_shape = tuple(as_int(dim_size(d)) for d in shape_tuple)
    handle = _spork_tracer.tensor(typed_pointer, dtype, int_shape)

    if expected_spec is not None:
        spec_type = parse_spec_into_type(expected_spec)
        et = spec_type.et
    else:
        # Bare-tensor ExprType: just an opaque Tensor name carrying the shape.
        name = handle._name
        et = Tensor(name)

    return TypedTensorHandle(
        handle=handle,
        type=Type(st=shape_tuple, et=et, dt=None),
    )


def _normalize_shape(shape) -> Tuple[FullDim, ...]:
    if shape is None:
        raise TypeError("skv.tensor: shape is required when wrapping a raw pointer")
    if isinstance(shape, FullDim):
        return (shape,)
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    for d in shape:
        if not isinstance(d, FullDim):
            raise TypeError(
                f"skv.tensor: shape must be a tuple of stile FullDim, "
                f"got element {d!r}"
            )
    return shape


def make_output_handle(
    ptr,
    spec : OutputSpec,
) -> TypedTensorHandle:
    """
    Build the typed handle the kernel will call ``.assign(...)`` on.
    The handle's ExprType is the parsed spec — so ``assign(value)``
    compares value's ExprType against the spec's ExprType.

    Used internally by ``@skv.jit``.
    """
    spec_type = parse_spec_into_type(spec.spec)
    output_dtype = spec.dtype or _infer_dtype_for_output(ptr)
    int_shape = tuple(as_int(dim_size(d)) for d in spec.st)
    handle = _spork_tracer.tensor(ptr, output_dtype, int_shape)
    return TypedTensorHandle(
        handle=handle,
        type=Type(st=spec.st, et=spec_type.et, dt=output_dtype),
    )


def _infer_dtype_for_output(ptr) -> dt.Dtype:
    underlying = getattr(ptr, "_dtype", None)
    if underlying is None:
        raise TypeError(
            "Verified kernel output: cannot infer dtype; set OutputSpec(dt=...)"
        )
    return underlying

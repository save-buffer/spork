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

import stile.type as t
from stile.type import (
    Constant,
    FullDim,
    ShapeType,
    Tensor,
    Type,
    as_int,
    dim_full_dim,
    dim_name,
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

    # Build the underlying spork TensorHandle. Stile uses math order
    # (rows, cols, ...); MPP's extents are memory-order (innermost-stride
    # first), which for row-major numpy buffers is the reverse.
    int_shape = tuple(as_int(dim_size(d)) for d in shape_tuple)
    spork_shape = tuple(reversed(int_shape))
    handle = _spork_tracer.tensor(typed_pointer, dtype, spork_shape)

    if expected_spec is not None:
        spec_type = parse_spec_into_type(expected_spec)
        et = spec_type.et
    else:
        # Bare-tensor ExprType: a fresh Tensor leaf carrying the dim
        # tuple. The auto-generated name (_tensor_<n>) is irrelevant to
        # verification — stile's normalize canonicalizes leaves by their
        # dim signatures.
        et = Tensor(dims=shape_tuple)

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
    spork_shape = tuple(reversed(int_shape))  # math → MPP memory order
    handle = _spork_tracer.tensor(ptr, output_dtype, spork_shape)
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


# ---------------------------------------------------------------------------
# Typed slices
# ---------------------------------------------------------------------------


class TypedTileSlice:
    """
    A typed analog of spork's ``TileSlice`` — opaque to the MPP runtime,
    but carries a stile ``Type`` so downstream typed ops compose into a
    verifiable expression.
    """

    __slots__ = ("_slice", "_type")

    def __init__(
        self,
        slice_handle : _spork_tracer.TileSlice,
        type         : Type,
    ):
        self._slice = slice_handle
        self._type = type

    @property
    def type(self) -> Type:
        return self._type


def _slice_type(parent_type : Type, tile_shape, offsets) -> Type:
    """
    Compute the stile Type that results from slicing ``parent_type`` with
    the given per-axis tile_shape and offsets. Only Python-int offsets
    are supported in this first cut; symbolic (Tracer) offsets pass
    through without refining the dim (the slice is still emitted, just
    not tracked in the stile shape).
    """
    out = parent_type
    for parent_dim, tile_size, offset in zip(parent_type.st, tile_shape, offsets):
        parent_size = as_int(dim_size(parent_dim))
        tile_size = int(tile_size)
        if isinstance(offset, int):
            if offset == 0 and tile_size == parent_size:
                continue  # full-axis slice — no Sliced wrapping needed
            out = out.slice(dim_full_dim(parent_dim), offset, offset + tile_size)
        # Tracer offsets: skip refinement for now. The MPP slice still
        # emits correctly; stile just sees the parent dim until we wire
        # LoopVariable bindings in a later patch.
    return out


# Patch TypedTensorHandle to add a .slice method without redefining the class
# (keeps the implementation alongside the other typed primitives).
def _typed_slice(self, tile_shape, offsets) -> TypedTileSlice:
    # User passes tile_shape + offsets in math order (rows, cols, ...);
    # spork's slice wants MPP memory order (inner first). Reverse at the
    # boundary so stile-side bookkeeping stays in math order.
    spork_tile_shape = tuple(reversed(tile_shape))
    spork_offsets = tuple(reversed(offsets))
    slice_handle = self._handle.slice(spork_tile_shape, spork_offsets)
    return TypedTileSlice(
        slice_handle=slice_handle,
        type=_slice_type(self._type, tile_shape, offsets),
    )


TypedTensorHandle.slice = _typed_slice


# ---------------------------------------------------------------------------
# Typed matmul2d + cooperative tensor
# ---------------------------------------------------------------------------


class TypedCooperativeTensor:
    """
    A typed analog of spork's ``CooperativeTensor`` — wraps the MPP coop
    handle with the stile ``Type`` of the value currently accumulated
    inside it. Starts at the zero tensor; each ``op.run(...)`` adds an
    einsum contribution. ``.store(out_tile)`` runs the verifier.
    """

    __slots__ = ("_coop", "_type", "_dtype")

    def __init__(
        self,
        coop  : _spork_tracer.CooperativeTensor,
        type  : Type,
        dtype : dt.Dtype,
    ):
        self._coop = coop
        self._type = type
        self._dtype = dtype

    @property
    def type(self) -> Type:
        return self._type

    def store(self, out_tile : TypedTileSlice) -> None:
        """
        Verify the accumulated expression matches the output tile's
        expected type, then emit the spork store. Mirrors
        ``TypedTensorHandle.assign``.
        """
        if not isinstance(out_tile, TypedTileSlice):
            raise TypeError(
                "TypedCooperativeTensor.store expects a TypedTileSlice, got "
                f"{type(out_tile).__name__}"
            )
        if not verify_types_equivalent(out_tile._type, self._type):
            raise ValueError(
                "Verified matmul output does not match spec.\n"
                f"  Expected: {out_tile._type.et}\n"
                f"  Actual  : {self._type.et}"
            )
        self._coop.store(out_tile._slice)


class TypedMatmulOp:
    """
    Typed analog of ``MatmulOp``. ``get_destination`` returns a
    TypedCooperativeTensor initialized to zero; ``run`` accumulates an
    ``einsum`` contribution into the coop's stile type.
    """

    __slots__ = ("_op", "_M", "_N", "_K")

    def __init__(
        self,
        op  : _spork_tracer.MatmulOp,
        M   : int,
        N   : int,
        K   : int,
    ):
        self._op = op
        self._M, self._N, self._K = M, N, K

    def get_destination(
        self,
        a     : TypedTileSlice,
        b     : TypedTileSlice,
        dtype : dt.Dtype,
    ) -> TypedCooperativeTensor:
        """
        Allocate a fresh cooperative accumulator. The stile type is the
        zero tensor with shape (rhs of einsum(a, b)). ``run`` will then
        replace this with the actual accumulated expression.
        """
        einstr, _rhs_dims = _matmul_einsum_string(a.type, b.type)
        # We need the shape of the result. Easiest: build the einsum
        # type and look at its shape.
        result_type = t.einsum(a.type, b.type, einstr)
        zero_type = Type(st=result_type.st, et=Constant(0.0), dt=dtype)

        # Spork side: get_destination wants the source TensorHandles, which
        # are stashed on the TileSlice's _source_tensor.
        spork_a_tensor = a._slice._source_tensor.handle if isinstance(
            a._slice._source_tensor, TypedTensorHandle
        ) else a._slice._source_tensor
        spork_b_tensor = b._slice._source_tensor.handle if isinstance(
            b._slice._source_tensor, TypedTensorHandle
        ) else b._slice._source_tensor
        coop = self._op.get_destination(spork_a_tensor, spork_b_tensor, dtype)
        return TypedCooperativeTensor(coop, zero_type, dtype)

    def run(
        self,
        a    : TypedTileSlice,
        b    : TypedTileSlice,
        coop : TypedCooperativeTensor,
    ) -> None:
        """
        Accumulate ``a @ b`` into ``coop``. Stile type:
        ``coop.type = coop.type + einsum(a, b)``.
        """
        einstr, _ = _matmul_einsum_string(a.type, b.type)
        contribution = t.einsum(a.type, b.type, einstr)
        # If coop is currently zero, replace with the contribution; else
        # add. (Constant(0) + x == x after normalize, but skipping the
        # rebuild keeps the ExprType cleaner.)
        if isinstance(coop._type.et, Constant) and coop._type.et.value == 0.0:
            coop._type = contribution
        else:
            coop._type = coop._type + contribution
        self._op.run(a._slice, b._slice, coop._coop)


def matmul2d(
    M           : int,
    N           : int,
    K           : int,
    *,
    simdgroups  : int = 4,
    transpose_a : bool = False,
    transpose_b : bool = False,
    transpose_c : bool = False,
    mode        : str = "multiply_accumulate",
) -> TypedMatmulOp:
    """
    Verified analog of ``sk.matmul2d``. Returns a TypedMatmulOp that
    threads stile types through ``get_destination`` / ``run`` /
    ``store``.
    """
    op = _spork_tracer.matmul2d(
        M, N, K,
        simdgroups=simdgroups,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        transpose_c=transpose_c,
        mode=mode,
    )
    return TypedMatmulOp(op, M, N, K)


def _matmul_einsum_string(a_type : Type, b_type : Type) -> tuple:
    """
    Infer the einsum string for A @ B from the operand shape types.
    Shared dim names are reduction; A-only + B-only dim names are output
    (in that order). Returns (einstr, rhs_dim_names_list).
    """
    a_names = [dim_name(d) for d in a_type.st]
    b_names = [dim_name(d) for d in b_type.st]
    shared = [n for n in a_names if n in b_names]
    if not shared:
        raise ValueError(
            "skv.matmul2d: operand shapes share no dim; can't infer "
            f"reduction axis (A dims={a_names}, B dims={b_names})"
        )
    a_only = [n for n in a_names if n not in shared]
    b_only = [n for n in b_names if n not in shared]
    rhs = a_only + b_only
    einstr = f"{' '.join(a_names)}, {' '.join(b_names)} -> {' '.join(rhs)}"
    return einstr, rhs


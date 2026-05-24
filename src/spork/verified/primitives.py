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
from stile.indexing import (
    AffineExpr,
    SymbolicInt,
    to_affine,
)
from stile.specification import parse_spec_into_type
from stile.verification import verify_types_equivalent

from .. import dtypes as dt
from .. import tracer as _spork_tracer
from . import _coverage
from ._backend import OutputSpec


class TypedTensorHandle:
    """
    A spork ``TensorHandle`` paired with a stile ``Type``.

    The wrapped ``TensorHandle`` is what spork's codegen and MPP ops
    operate on; the ``Type`` is what the verifier sees.

    ``_is_output`` flags the kernel's verified output. Set by
    ``make_output_handle``; propagated through ``.slice`` to the
    resulting ``TypedTileSlice`` so ``.assign`` / ``.store`` know to
    record their writes for the bind-time coverage check.
    """

    __slots__ = ("_handle", "_type", "_is_output")

    def __init__(
        self,
        handle    : _spork_tracer.TensorHandle,
        type      : Type,
        is_output : bool = False,
    ):
        self._handle = handle
        self._type = type
        self._is_output = is_output

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
        if self._is_output:
            _coverage.record_store(self, self._type.st)


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
        is_output=True,
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
    verifiable expression. ``_is_output`` propagates from the parent
    ``TypedTensorHandle`` so ``coop.store(out_tile)`` knows to record
    the write for the coverage check.
    """

    __slots__ = ("_slice", "_type", "_is_output")

    def __init__(
        self,
        slice_handle : _spork_tracer.TileSlice,
        type         : Type,
        is_output    : bool = False,
    ):
        self._slice = slice_handle
        self._type = type
        self._is_output = is_output

    @property
    def type(self) -> Type:
        return self._type


def _slice_type(parent_type : Type, tile_shape, offsets) -> Type:
    """
    Compute the stile Type that results from slicing ``parent_type`` with
    the given per-axis tile_shape and offsets.

    Offset handling, by type:
      - Python ``int``: refines the dim to ``Sliced(d, off, off+size)``;
        a full-coverage int offset (0, full-size) is a no-op.
      - ``TypedScalarTracer``: extracts the stile ``SymbolicInt`` /
        ``AffineExpr`` from ``._sym`` and uses it as the start of the
        Sliced dim — the verifier sees a symbolic-but-tracked slice.
      - Raw spork ``Tracer``: passes through unrefined (we don't know
        the symbolic identity of an arbitrary spork expression).
    """
    out = parent_type
    for parent_dim, tile_size, offset in zip(parent_type.st, tile_shape, offsets):
        parent_size = as_int(dim_size(parent_dim))
        tile_size = int(tile_size)

        sym_start = _offset_to_symbolic(offset)
        if sym_start is None:
            # Unknown symbolic; can't refine. Spork side still slices correctly.
            continue
        if isinstance(sym_start, int):
            if sym_start == 0 and tile_size == parent_size:
                continue  # full-axis slice — no Sliced wrapping needed
        out = out.slice(dim_full_dim(parent_dim), sym_start, sym_start + tile_size)
    return out


def _offset_to_symbolic(offset):
    """
    Lower a slice offset into a stile ``SymbolicIndex`` (int /
    SymbolicInt / AffineExpr), or None if it can't be tracked.
    """
    if isinstance(offset, int):
        return offset
    if isinstance(offset, TypedScalarTracer):
        return offset._sym
    return None


# Patch TypedTensorHandle to add a .slice method without redefining the class
# (keeps the implementation alongside the other typed primitives).
def _typed_slice(self, tile_shape, offsets) -> TypedTileSlice:
    # User passes tile_shape + offsets in math order (rows, cols, ...);
    # spork's slice wants MPP memory order (inner first). Reverse at the
    # boundary so stile-side bookkeeping stays in math order. Also
    # unwrap any TypedScalarTracer offsets to their underlying spork
    # Tracer before handing to spork.
    spork_offsets_math = tuple(
        o._tracer if isinstance(o, TypedScalarTracer) else o
        for o in offsets
    )
    spork_tile_shape = tuple(reversed(tile_shape))
    spork_offsets = tuple(reversed(spork_offsets_math))
    slice_handle = self._handle.slice(spork_tile_shape, spork_offsets)
    return TypedTileSlice(
        slice_handle=slice_handle,
        type=_slice_type(self._type, tile_shape, offsets),
        is_output=self._is_output,
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
        if out_tile._is_output:
            _coverage.record_store(out_tile, out_tile._type.st)
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


# ---------------------------------------------------------------------------
# Typed scalar / vector tracers — LoopVariable-aware
# ---------------------------------------------------------------------------


class TypedScalarTracer:
    """
    A spork uint scalar Tracer paired with a stile ``SymbolicInt`` or
    ``AffineExpr``. Arithmetic ops (``* int``, ``+``, ``-``) compose on
    both sides — the spork tracer goes into the emitted Metal, the
    stile expression refines slice offsets the verifier can reason
    about.

    Used wherever a grid-position component (e.g. ``bid.x``) flows into
    a slice offset or other index expression.
    """

    __slots__ = ("_tracer", "_sym")

    def __init__(self, tracer : _spork_tracer.Tracer, sym):
        self._tracer = tracer
        self._sym = sym  # SymbolicInt | AffineExpr | int

    def __mul__(self, k : int) -> "TypedScalarTracer":
        if not isinstance(k, int):
            raise TypeError(
                "TypedScalarTracer can only be multiplied by a Python int "
                f"(got {type(k).__name__}); stile AffineExprs only admit "
                "compile-time integer coefficients."
            )
        return TypedScalarTracer(self._tracer * k, to_affine(self._sym) * k)

    __rmul__ = __mul__

    def __add__(self, other) -> "TypedScalarTracer":
        if isinstance(other, TypedScalarTracer):
            return TypedScalarTracer(
                self._tracer + other._tracer,
                to_affine(self._sym) + to_affine(other._sym),
            )
        if isinstance(other, int):
            return TypedScalarTracer(
                self._tracer + other,
                to_affine(self._sym) + other,
            )
        raise TypeError(
            f"Cannot add TypedScalarTracer + {type(other).__name__}"
        )

    __radd__ = __add__

    def __sub__(self, other) -> "TypedScalarTracer":
        if isinstance(other, TypedScalarTracer):
            return TypedScalarTracer(
                self._tracer - other._tracer,
                to_affine(self._sym) - to_affine(other._sym),
            )
        if isinstance(other, int):
            return TypedScalarTracer(
                self._tracer - other,
                to_affine(self._sym) - other,
            )
        raise TypeError(
            f"Cannot subtract {type(other).__name__} from TypedScalarTracer"
        )

    def __rsub__(self, other) -> "TypedScalarTracer":
        if isinstance(other, int):
            return TypedScalarTracer(
                other - self._tracer,
                other - to_affine(self._sym),
            )
        raise TypeError(
            f"Cannot subtract TypedScalarTracer from {type(other).__name__}"
        )

    def __neg__(self) -> "TypedScalarTracer":
        return TypedScalarTracer(-self._tracer, -to_affine(self._sym))


class TypedVectorTracer:
    """
    Typed wrapper around a spork ``VectorTracer`` (uint2/uint3/...).
    Each component (``.x`` / ``.y`` / ``.z`` / ``.w``) returns a
    ``TypedScalarTracer`` carrying a per-component ``SymbolicInt`` so
    derived slice offsets are tracked by the verifier.
    """

    __slots__ = ("_vec", "_sym_components")

    _FIELDS = ("x", "y", "z", "w")

    def __init__(self, vec_tracer : _spork_tracer.VectorTracer, var_name_prefix : str):
        self._vec = vec_tracer
        self._sym_components : dict[str, SymbolicInt] = {}
        for i, field in enumerate(self._FIELDS):
            if i < self._vec._vec_size:
                self._sym_components[field] = SymbolicInt(
                    name=f"_{var_name_prefix}_{field}",
                )

    def __getattr__(self, name : str) -> TypedScalarTracer:
        if name in self._FIELDS:
            if name not in self._sym_components:
                raise AttributeError(
                    f"Vector of size {self._vec._vec_size} has no field "
                    f".{name}"
                )
            return TypedScalarTracer(
                tracer=getattr(self._vec, name),
                sym=self._sym_components[name],
            )
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Typed runtime loop
# ---------------------------------------------------------------------------


_loop_var_counter = [0]


def _next_loop_var_name() -> str:
    _loop_var_counter[0] += 1
    return f"_loop_{_loop_var_counter[0]}"


def range(*args):
    """
    Verified analog of ``sk.range``. Emits a Metal for-loop on the
    spork side and yields a ``TypedScalarTracer`` whose stile component
    is a fresh ``SymbolicInt`` registered with the active kernel's
    ``loop_var_ranges``. The coverage check enumerates over the loop's
    ``(lo, hi, step)`` range when this symbol appears in a store's
    slice bounds.

    Signatures mirror Python's ``range``:

        for i in skv.range(end):                ...
        for i in skv.range(start, end):         ...
        for i in skv.range(start, end, step):   ...

    For static (compile-time-unrolled) loops, use Python's built-in
    ``range`` — each iteration's body is traced separately with
    concrete-int values, and stores get recorded with concrete offsets.
    """
    if len(args) == 1:
        lo, hi, step = 0, args[0], 1
    elif len(args) == 2:
        lo, hi, step = args[0], args[1], 1
    elif len(args) == 3:
        lo, hi, step = args
    else:
        raise TypeError(f"skv.range expects 1-3 args, got {len(args)}")

    for v, label in ((lo, "lo"), (hi, "hi"), (step, "step")):
        if not isinstance(v, int):
            raise TypeError(
                f"skv.range bounds must be Python ints for now (coverage "
                f"enumeration needs concrete ranges); got {label}={v!r}"
            )
    return _TypedRange(lo, hi, step)


class _TypedRange:
    """
    Iterator wrapper that delegates body-stmt push/pop to spork's
    ``_RangeLoop`` while yielding a ``TypedScalarTracer`` (so slice
    offsets composed from the loop var carry the right stile
    ``SymbolicInt``).
    """

    __slots__ = ("_lo", "_hi", "_step")

    def __init__(self, lo : int, hi : int, step : int):
        self._lo, self._hi, self._step = lo, hi, step

    def __iter__(self):
        sk_range = _spork_tracer.range(self._lo, self._hi, self._step)
        # Drive the spork _RangeLoop generator manually so we can wrap
        # its yield as a typed scalar AND register the SymbolicInt with
        # the active state.
        spork_iter = iter(sk_range)
        spork_tracer_var = next(spork_iter)
        sym = SymbolicInt(name=_next_loop_var_name())
        state = _coverage._active_state.get()
        if state is not None:
            state.loop_var_ranges[sym.name] = (self._lo, self._hi, self._step)
        typed_var = TypedScalarTracer(spork_tracer_var, sym)
        try:
            yield typed_var
        finally:
            try:
                next(spork_iter)
            except StopIteration:
                pass


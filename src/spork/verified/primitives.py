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
    BinaryOp,
    Constant,
    FullDim,
    ShapeType,
    Sliced,
    Tensor,
    Type,
    UnaryOp,
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


# Save the builtin ``range`` since this module also defines ``range``
# (the verified runtime-loop primitive). Any Python-level for loops in
# this module must use ``_py_range``.
_py_range = range


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

    __slots__ = ("_handle", "_type", "_is_output", "_ptr")

    def __init__(
        self,
        handle    : _spork_tracer.TensorHandle,
        type      : Type,
        is_output : bool = False,
        ptr       = None,
    ):
        self._handle = handle
        self._type = type
        self._is_output = is_output
        # The original device-pointer parameter (PointerTracer) or
        # threadgroup array; needed for element-wise indexing via
        # __getitem__ / __setitem__, since MPP ``tensor`` wrappers
        # aren't subscriptable in Metal.
        self._ptr = ptr

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
        ptr=typed_pointer,
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
        ptr=ptr,
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

        If we're inside one or more ``skv.range`` loops, snapshot
        ``coop``'s pre-mutation type into each loop's snapshots so the
        loop-exit wrap can compute the per-iter delta and emit a
        ``ParametricReduce``.
        """
        _coverage.on_coop_touched(coop)
        einstr, _ = _matmul_einsum_string(a.type, b.type)
        contribution = t.einsum(a.type, b.type, einstr)
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
        # its yield as a typed scalar AND push an ActiveLoop frame for
        # accumulator snapshotting.
        spork_iter = iter(sk_range)
        spork_tracer_var = next(spork_iter)
        sym = SymbolicInt(name=_next_loop_var_name())
        state = _coverage._active_state.get()
        if state is not None:
            state.loop_var_ranges[sym.name] = (self._lo, self._hi, self._step)
        loop = _coverage.ActiveLoop(
            sym=sym, lo=self._lo, hi=self._hi, step=self._step,
        )
        _coverage._active_loops.append(loop)
        typed_var = TypedScalarTracer(spork_tracer_var, sym)
        try:
            yield typed_var
        finally:
            # 1) Finalize spork's for-loop body (close the Metal scope).
            try:
                next(spork_iter)
            except StopIteration:
                pass
            # 2) Pop the active loop and wrap any accumulators that
            #    were touched in this body as ParametricReduce.
            popped = _coverage._active_loops.pop()
            _coverage.wrap_loop_accumulators(popped)


# ---------------------------------------------------------------------------
# Typed scalar values + element-level tensor reads / writes
# ---------------------------------------------------------------------------


class TypedScalarValue:
    """
    A typed scalar VALUE — the result of reading a single element from a
    typed tensor, or of arithmetic between typed scalars.

    Distinct from ``TypedScalarTracer``, which carries a stile
    ``SymbolicInt`` for slice-offset arithmetic. ``TypedScalarValue``
    carries a stile ``Type`` with shape ``()`` and an ``ExprType`` that
    composes via stile's ``BinaryOp`` / ``UnaryOp`` constructors.

    Backed on the spork side by an ordinary ``Tracer`` (or a numeric
    literal lifted into one) so the kernel still emits Metal for the
    arithmetic.
    """

    __slots__ = ("_tracer", "_type")

    def __init__(self, tracer, type : Type):
        self._tracer = tracer
        self._type = type

    @property
    def type(self) -> Type:
        return self._type

    def _binop(self, op : str, other) -> "TypedScalarValue":
        other_tracer, other_et = _lower_value(other)
        new_tracer = _spork_binop(self._tracer, op, other_tracer)
        new_et = BinaryOp(op=op, lhs=self._type.et, rhs=other_et)
        return TypedScalarValue(new_tracer, Type(st=(), et=new_et, dt=self._type.dt))

    def __add__(self, other):     return self._binop("+", other)
    def __radd__(self, other):    return _lift_value(other)._binop("+", self)
    def __sub__(self, other):     return self._binop("-", other)
    def __rsub__(self, other):    return _lift_value(other)._binop("-", self)
    def __mul__(self, other):     return self._binop("*", other)
    def __rmul__(self, other):    return _lift_value(other)._binop("*", self)
    def __truediv__(self, other): return self._binop("/", other)
    def __rtruediv__(self, other):return _lift_value(other)._binop("/", self)
    def __neg__(self):            return TypedScalarValue(
        -self._tracer,
        Type(st=(), et=BinaryOp(op="-", lhs=Constant(0.0), rhs=self._type.et), dt=self._type.dt),
    )


def _lower_value(x) -> tuple:
    """
    Coerce ``x`` to ``(spork_tracer_or_value, stile_et)`` for use as the
    rhs of a typed scalar binop. Accepts TypedScalarValue, plain Python
    int/float, and TypedScalarTracer (its symbolic index is taken as
    the value's expression — useful when an index is used inside a
    value-arithmetic context, though it's an unusual mix).
    """
    if isinstance(x, TypedScalarValue):
        return x._tracer, x._type.et
    if isinstance(x, TypedScalarTracer):
        return x._tracer, Constant(0.0)  # treat as opaque value
    if isinstance(x, (int, float, bool)):
        return x, Constant(float(x) if not isinstance(x, bool) else x)
    raise TypeError(f"Can't lower {type(x).__name__} to a typed scalar value")


def _lift_value(x) -> "TypedScalarValue":
    """Lift a Python literal into a TypedScalarValue carrying a Constant ET."""
    if isinstance(x, TypedScalarValue):
        return x
    if isinstance(x, (int, float)):
        return TypedScalarValue(x, Type(st=(), et=Constant(float(x)), dt=None))
    raise TypeError(f"Can't lift {type(x).__name__} to TypedScalarValue")


def _spork_binop(lhs, op : str, rhs):
    """Apply a Python binary op to the underlying spork tracers."""
    if op == "+":   return lhs + rhs
    if op == "-":   return lhs - rhs
    if op == "*":   return lhs * rhs
    if op == "/":   return lhs / rhs
    raise ValueError(f"Unsupported binop {op!r}")


# Typed __getitem__ / __setitem__ on TypedTensorHandle
# ---------------------------------------------------------------------------


def _typed_getitem(self, index) -> TypedScalarValue:
    """
    Read a single element of the typed tensor by ``(i, j, ...)`` (math
    order). Returns a TypedScalarValue whose stile ExprType is the same
    Tensor leaf as the handle.
    """
    if self._ptr is None:
        raise TypeError(
            "Typed __getitem__ requires the typed tensor to have a "
            "backing device pointer (got None — was it constructed via "
            "an MPP-only path?)"
        )
    indices = _normalize_indices(index)
    flat_idx = _flatten_indices(indices, self._type.st)
    # Use the raw spork PointerTracer's __getitem__ for the load —
    # MPP's tensor wrapper isn't subscriptable in Metal.
    spork_scalar = self._ptr[flat_idx]
    return TypedScalarValue(
        spork_scalar,
        Type(st=(), et=self._type.et, dt=self._handle._dtype),
    )


def _typed_setitem(self, index, value) -> None:
    """
    Write a single element. Records a per-element store into the active
    state's stored_slices so the bind-time coverage check sees the
    element-level coverage too.
    """
    if self._ptr is None:
        raise TypeError(
            "Typed __setitem__ requires the typed tensor to have a "
            "backing device pointer (got None)"
        )
    indices = _normalize_indices(index)
    if len(indices) != len(self._type.st):
        raise TypeError(
            f"Typed tensor __setitem__ expects {len(self._type.st)} "
            f"indices (one per declared dim), got {len(indices)}"
        )

    # Per-element coverage: each axis becomes Sliced(dim, idx, idx+1).
    sliced_shape = tuple(
        _per_element_sliced(parent_dim, idx)
        for parent_dim, idx in zip(self._type.st, indices)
    )
    if self._is_output:
        _coverage.record_store(self, sliced_shape)

    # Spork side: flatten the math-order indices to a single offset and
    # store via the raw PointerTracer.
    flat_idx = _flatten_indices(indices, self._type.st)
    val_expr = (
        value._tracer if isinstance(value, (TypedScalarValue, TypedScalarTracer))
        else value
    )
    self._ptr[flat_idx] = val_expr


def _flatten_indices(indices, shape):
    """
    Compute the row-major flat offset for ``indices`` given the
    math-order ``shape`` (a tuple of stile dims). Each index can be a
    Python int, a TypedScalarTracer, or a raw spork Tracer; the result
    is a spork Tracer expression (or Python int) suitable for direct
    use as a PointerTracer subscript.
    """
    sizes = [as_int(dim_size(d)) for d in shape]
    strides = [1] * len(sizes)
    for i in _py_range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]

    flat = None
    for idx, stride in zip(indices, strides):
        idx_tracer = (
            idx._tracer if isinstance(idx, TypedScalarTracer) else idx
        )
        term = idx_tracer * stride if stride != 1 else idx_tracer
        flat = term if flat is None else flat + term
    return flat if flat is not None else 0


def _normalize_indices(index) -> tuple:
    if isinstance(index, tuple):
        return index
    return (index,)


def _per_element_sliced(parent_dim, idx) -> "Sliced | FullDim":
    """One-element slice ``Sliced(dim, idx, idx+1)``."""
    if isinstance(idx, int):
        sym_idx = idx
    elif isinstance(idx, TypedScalarTracer):
        sym_idx = idx._sym
    else:
        # Unknown index — treat as full coverage on this axis (sound).
        return parent_dim
    return Sliced(dim=dim_full_dim(parent_dim), start=sym_idx, end=sym_idx + 1)


TypedTensorHandle.__getitem__ = _typed_getitem
TypedTensorHandle.__setitem__ = _typed_setitem


# ---------------------------------------------------------------------------
# Typed math intrinsics (operate on TypedScalarValue)
# ---------------------------------------------------------------------------


def _math_intrinsic(spork_fn, stile_op_name : str, x : TypedScalarValue) -> TypedScalarValue:
    if not isinstance(x, TypedScalarValue):
        raise TypeError(
            f"skv math intrinsic expects a TypedScalarValue, got "
            f"{type(x).__name__}"
        )
    new_tracer = spork_fn(x._tracer)
    new_et = UnaryOp(op=stile_op_name, child=x._type.et)
    return TypedScalarValue(new_tracer, Type(st=(), et=new_et, dt=x._type.dt))


def exp(x : TypedScalarValue) -> TypedScalarValue:
    return _math_intrinsic(_spork_tracer.exp, "exp", x)


def sqrt(x : TypedScalarValue) -> TypedScalarValue:
    return _math_intrinsic(_spork_tracer.sqrt, "sqrt", x)


def sin(x : TypedScalarValue) -> TypedScalarValue:
    return _math_intrinsic(_spork_tracer.sin, "sin", x)


def cos(x : TypedScalarValue) -> TypedScalarValue:
    return _math_intrinsic(_spork_tracer.cos, "cos", x)


# ---------------------------------------------------------------------------
# Typed mutable local scalars
# ---------------------------------------------------------------------------


class TypedLocal(TypedScalarValue):
    """
    A typed mutable local scalar — wraps a spork ``Local`` (from
    ``sk.local``) with a stile ``Type`` that updates on each assign /
    compound assign.

    Reads compose like ``TypedScalarValue`` (inherits ``+``/``*``/etc.).
    The underlying spork ``Local`` is what the kernel actually mutates;
    this typed wrapper just tracks the current value's stile expression
    alongside, so downstream reads carry the right ExprType.
    """

    __slots__ = ("_spork_local", "_dtype")

    def __init__(
        self,
        spork_local : _spork_tracer.Local,
        dtype       : dt.Dtype,
        init_et,
    ):
        super().__init__(
            tracer=spork_local,
            type=Type(st=(), et=init_et, dt=dtype),
        )
        self._spork_local = spork_local
        self._dtype = dtype

    def _assigned_et(self, value):
        if isinstance(value, TypedScalarValue):
            return value._type.et
        if isinstance(value, TypedScalarTracer):
            return Constant(0.0)
        if isinstance(value, (int, float)):
            return Constant(float(value))
        raise TypeError(f"Can't assign {type(value).__name__} to TypedLocal")

    def _value_tracer(self, value):
        if isinstance(value, (TypedScalarValue, TypedScalarTracer)):
            return value._tracer
        return value

    def assign(self, value) -> None:
        """
        Re-assign the local to ``value``. Updates both spork-side
        emission and the typed wrapper's current ExprType.
        """
        self._spork_local.assign(self._value_tracer(value))
        self._type = Type(st=(), et=self._assigned_et(value), dt=self._dtype)

    def _iaccum(self, op : str, value) -> "TypedLocal":
        self._spork_local._update(op, self._value_tracer(value))
        new_et = BinaryOp(
            op=op.rstrip("="),  # "+=" → "+"
            lhs=self._type.et,
            rhs=self._assigned_et(value),
        )
        self._type = Type(st=(), et=new_et, dt=self._dtype)
        return self

    def __iadd__(self, other): return self._iaccum("+=", other)
    def __isub__(self, other): return self._iaccum("-=", other)
    def __imul__(self, other): return self._iaccum("*=", other)
    def __itruediv__(self, other): return self._iaccum("/=", other)


def local(dtype : dt.Dtype, init) -> TypedLocal:
    """
    Allocate a typed mutable local scalar initialized to ``init``.

    Emits ``<dtype> v = init;`` on the spork side and returns a
    ``TypedLocal`` that supports reads (composes via inherited
    ``TypedScalarValue`` arithmetic) and writes (``.assign``, ``+=``,
    ``-=``, ``*=``, ``/=``).
    """
    init_tracer = init._tracer if isinstance(init, (TypedScalarValue, TypedScalarTracer)) else init
    spork_local = _spork_tracer.local(dtype, init_tracer)
    init_et = (
        init._type.et if isinstance(init, TypedScalarValue)
        else Constant(0.0) if isinstance(init, TypedScalarTracer)
        else Constant(float(init))
    )
    return TypedLocal(spork_local, dtype, init_et)


# ---------------------------------------------------------------------------
# Typed threadgroup-memory arrays
# ---------------------------------------------------------------------------


class TypedThreadgroupArray:
    """
    A typed wrapper around ``sk.threadgroup`` — a fixed-size array in
    threadgroup-shared memory, indexable via ``[i]`` or ``[i, j, ...]``.

    Reads return ``TypedScalarValue`` with the array's declared dim
    tuple as the ExprType's Tensor leaf (so verification through
    intermediate threadgroup-scratch still sees a sensible expression).
    Writes pass straight through. No coverage tracking on writes —
    threadgroup memory is per-kernel scratch, not user-visible output.

    Also supports being wrapped as an MPP ``tensor`` (via ``skv.tensor``)
    so cooperative-tensor ``.store(tg_view.slice(...))`` continues to
    work on typed threadgroup arrays.
    """

    __slots__ = ("_tg", "_dtype", "_shape", "_dtype_obj")

    def __init__(
        self,
        spork_tg : _spork_tracer.ThreadgroupArray,
        dtype    : dt.Dtype,
        shape    : tuple,
    ):
        self._tg = spork_tg
        self._dtype = dtype
        self._shape = shape  # tuple of stile FullDims

    @property
    def shape(self) -> tuple:
        return self._shape

    def __getitem__(self, index) -> TypedScalarValue:
        # Extract spork-side indices, unwrapping typed scalars.
        if isinstance(index, tuple):
            spork_idx = tuple(
                i._tracer if isinstance(i, (TypedScalarValue, TypedScalarTracer)) else i
                for i in index
            )
        else:
            spork_idx = (
                index._tracer if isinstance(index, (TypedScalarValue, TypedScalarTracer))
                else index
            )
        spork_val = self._tg[spork_idx]
        # Type: an element of "this scratch array", carried as a Tensor
        # leaf whose dims are the array's declared shape.
        return TypedScalarValue(
            spork_val,
            Type(st=(), et=Tensor(dims=self._shape), dt=self._dtype),
        )

    def __setitem__(self, index, value) -> None:
        if isinstance(index, tuple):
            spork_idx = tuple(
                i._tracer if isinstance(i, (TypedScalarValue, TypedScalarTracer)) else i
                for i in index
            )
        else:
            spork_idx = (
                index._tracer if isinstance(index, (TypedScalarValue, TypedScalarTracer))
                else index
            )
        val_expr = (
            value._tracer if isinstance(value, (TypedScalarValue, TypedScalarTracer))
            else value
        )
        self._tg[spork_idx] = val_expr

    # MPP compatibility — when passed to skv.tensor, expose the same
    # ``_expr`` and ``_shape`` slots the wrapping code reads from
    # spork's ``ThreadgroupArray``.
    @property
    def _expr(self):
        return self._tg._expr


def threadgroup(dtype : dt.Dtype, shape) -> TypedThreadgroupArray:
    """
    Allocate a typed threadgroup-memory array of the given shape.

    Shape is a tuple of stile dims (``skv.dim(name, size)``); the
    spork side gets the concrete integer sizes.
    """
    if isinstance(shape, FullDim):
        shape = (shape,)
    elif not isinstance(shape, tuple):
        shape = tuple(shape)
    for d in shape:
        if not isinstance(d, FullDim):
            raise TypeError(
                f"skv.threadgroup shape must be a tuple of skv.dim(...), "
                f"got element {d!r}"
            )
    int_shape = tuple(as_int(dim_size(d)) for d in shape)
    spork_tg = _spork_tracer.threadgroup(dtype, int_shape)
    return TypedThreadgroupArray(spork_tg, dtype, shape)


import contextlib
import contextvars
import inspect
import os
import subprocess
import sys
from tempfile import mkdtemp
from typing import Callable, Dict, List, Optional

from Foundation import NSURL
from Metal import (
    MTLCaptureDescriptor,
    MTLCaptureDestinationGPUTraceDocument,
    MTLCaptureManager,
)

from . import dtypes as dt
from . import ir
from . import runtime
from .types import DevicePointerSpec


_SPORK_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _caller_loc() -> Optional[tuple]:
    """
    Walk up the call stack and return the (filename, lineno) of the first
    frame that lives *outside* the spork package directory.

    Returns None if the entire stack is internal (e.g. invoked from a REPL
    with code that's hard to attribute).
    """
    frame = sys._getframe(1)
    while frame is not None:
        fname = frame.f_code.co_filename
        if not fname.startswith(_SPORK_PACKAGE_DIR):
            return (fname, frame.f_lineno)
        frame = frame.f_back
    return None


_builder : contextvars.ContextVar[Optional["KernelBuilder"]] = contextvars.ContextVar(
    "_spork_builder", default=None
)


def current_builder() -> "KernelBuilder":
    b = _builder.get()
    if b is None:
        raise RuntimeError(
            "spork operations may only be used inside a @sk.jit-decorated kernel"
        )
    return b


class KernelBuilder:
    def __init__(self, name : str):
        self.name : str = name
        self.params : List[ir.Param] = []
        self.stmts : List[ir.Stmt] = []
        self.includes : List[tuple] = [("metal_stdlib", True)]
        self.usings : List[str] = ["metal"]
        self.device_functions : List["DeviceFn"] = []
        self._dfn_names : set = set()
        self._name_counters : Dict[str, int] = {}

    def add_stmt(self, stmt : ir.Stmt) -> None:
        if getattr(stmt, "loc", None) is None:
            stmt.loc = _caller_loc()
        self.stmts.append(stmt)

    def fresh_name(self, prefix : str = "v") -> str:
        n = self._name_counters.get(prefix, 0)
        self._name_counters[prefix] = n + 1
        return f"{prefix}{n}"

    def add_include(self, path : str, system : bool = True) -> None:
        entry = (path, system)
        if entry not in self.includes:
            self.includes.append(entry)

    def add_using(self, namespace : str) -> None:
        if namespace not in self.usings:
            self.usings.append(namespace)

    def add_device_fn(self, df : "DeviceFn") -> None:
        """
        Register a DeviceFn as a dependency of this kernel. Traces it
        lazily if needed and recursively pulls in transitive dependencies.
        Also merges the device fn's includes/usings into this builder.
        """
        df._ensure_traced()
        for dep in df._deps:
            self.add_device_fn(dep)
        if df._name in self._dfn_names:
            return
        self._dfn_names.add(df._name)
        self.device_functions.append(df)
        for inc in df._includes:
            if inc not in self.includes:
                self.includes.append(inc)
        for ns in df._usings:
            if ns not in self.usings:
                self.usings.append(ns)


def _to_expr(value) -> ir.Expr:
    if isinstance(value, Tracer):
        return value._expr
    if isinstance(value, _OpaqueHandle):
        return value._expr
    if isinstance(value, VectorTracer):
        raise TypeError(
            "Cannot use a vector value as a scalar — access a field like .x first"
        )
    if isinstance(value, PointerTracer):
        raise TypeError(
            "Cannot use a pointer as a scalar value — index it with ptr[i] or "
            "wrap it with sk.tensor(ptr, dtype, shape)"
        )
    if isinstance(value, bool):
        return ir.Const(value)
    if isinstance(value, (int, float)):
        return ir.Const(value)
    raise TypeError(
        f"Cannot convert {value!r} (type {type(value).__name__}) to a spork expression"
    )


class _OpaqueHandle:
    """
    Base class for handles to values with C++-opaque types (cooperative
    tensors, tile slices, etc.). Subclasses set ``_expr`` to the IR
    expression that references them by name.
    """

    _expr : ir.Expr


def _result_dtype(a : "Tracer", other) -> Optional[dt.Dtype]:
    if isinstance(other, Tracer):
        if a._dtype is not None and other._dtype is not None:
            if a._dtype.is_float or other._dtype.is_float:
                return a._dtype if a._dtype.is_float else other._dtype
            return a._dtype
        return a._dtype or other._dtype
    return a._dtype


class Tracer:
    """
    Represents a scalar value in a traced kernel.
    """

    __slots__ = ("_expr", "_dtype")

    def __init__(self, expr : ir.Expr, dtype : Optional[dt.Dtype]):
        self._expr = expr
        self._dtype = dtype

    def _binop(self, op : str, other, *, swap : bool = False) -> "Tracer":
        rhs = _to_expr(other)
        if swap:
            expr = ir.BinOp(op, rhs, self._expr)
        else:
            expr = ir.BinOp(op, self._expr, rhs)
        return Tracer(expr, _result_dtype(self, other))

    def _cmp(self, op : str, other) -> "Tracer":
        rhs = _to_expr(other)
        return Tracer(ir.BinOp(op, self._expr, rhs), dt.bool_)

    def __add__(self, other):      return self._binop("+", other)
    def __radd__(self, other):     return self._binop("+", other, swap=True)
    def __sub__(self, other):      return self._binop("-", other)
    def __rsub__(self, other):     return self._binop("-", other, swap=True)
    def __mul__(self, other):      return self._binop("*", other)
    def __rmul__(self, other):     return self._binop("*", other, swap=True)
    def __truediv__(self, other):  return self._binop("/", other)
    def __rtruediv__(self, other): return self._binop("/", other, swap=True)
    def __floordiv__(self, other): return self._binop("/", other)
    def __rfloordiv__(self, other):return self._binop("/", other, swap=True)
    def __mod__(self, other):      return self._binop("%", other)
    def __rmod__(self, other):     return self._binop("%", other, swap=True)
    def __and__(self, other):      return self._binop("&", other)
    def __rand__(self, other):     return self._binop("&", other, swap=True)
    def __or__(self, other):       return self._binop("|", other)
    def __ror__(self, other):      return self._binop("|", other, swap=True)
    def __xor__(self, other):      return self._binop("^", other)
    def __rxor__(self, other):     return self._binop("^", other, swap=True)
    def __lshift__(self, other):   return self._binop("<<", other)
    def __rshift__(self, other):   return self._binop(">>", other)

    def __lt__(self, other): return self._cmp("<", other)
    def __le__(self, other): return self._cmp("<=", other)
    def __gt__(self, other): return self._cmp(">", other)
    def __ge__(self, other): return self._cmp(">=", other)
    def __eq__(self, other): return self._cmp("==", other)
    def __ne__(self, other): return self._cmp("!=", other)

    def __neg__(self):    return Tracer(ir.UnaryOp("-", self._expr), self._dtype)
    def __pos__(self):    return self
    def __invert__(self): return Tracer(ir.UnaryOp("~", self._expr), self._dtype)

    def __bool__(self):
        raise TypeError(
            "Cannot use a traced spork value in a Python boolean context "
            "(e.g. `if`, `and`, `or`). Use kernel-level control flow instead."
        )

    def __hash__(self):
        return id(self)


class PointerTracer:
    """
    Represents a device pointer in a traced kernel.
    """

    __slots__ = ("_expr", "_dtype", "_builder")

    def __init__(self, expr : ir.Expr, dtype : dt.Dtype, builder : KernelBuilder):
        self._expr = expr
        self._dtype = dtype
        self._builder = builder

    def __getitem__(self, index) -> Tracer:
        idx = _to_expr(index)
        return Tracer(ir.Load(self._expr, idx), self._dtype)

    def __setitem__(self, index, value) -> None:
        idx = _to_expr(index)
        val = _to_expr(value)
        self._builder.add_stmt(ir.Store(self._expr, idx, val))
        if isinstance(self._expr, ir.Var):
            for p in self._builder.params:
                if p.kind == "pointer" and p.name == self._expr.name:
                    p.written = True
                    break


class VectorTracer:
    """
    Represents a vector-typed value (uint2/uint3/int2/...) in a traced kernel.

    Component access (``v.x``, ``v.y``, ``v.z``, ``v.w``) yields a scalar Tracer.
    """

    __slots__ = ("_expr", "_dtype", "_vec_size")

    _FIELDS = ("x", "y", "z", "w")

    def __init__(self, expr : ir.Expr, elem_dtype : dt.Dtype, vec_size : int):
        self._expr = expr
        self._dtype = elem_dtype
        self._vec_size = vec_size

    def __getattr__(self, name : str) -> Tracer:
        if name in self._FIELDS:
            idx = self._FIELDS.index(name)
            if idx >= self._vec_size:
                raise AttributeError(
                    f"Vector of size {self._vec_size} has no field .{name}"
                )
            return Tracer(ir.Member(self._expr, name), self._dtype)
        raise AttributeError(name)


class Local(Tracer):
    """
    A mutable local variable declared with ``sk.local``.

    Reads behave like a normal scalar Tracer. Compound updates
    (``local += x``, ``local -= x``, etc.) emit an Update statement.
    """

    __slots__ = ("_builder", "_name")

    def __init__(self, name : str, dtype : dt.Dtype, builder : KernelBuilder):
        super().__init__(ir.Var(name), dtype)
        self._builder = builder
        self._name = name

    def _update(self, op : str, value) -> "Local":
        self._builder.add_stmt(ir.Update(self._name, op, _to_expr(value)))
        return self

    def __iadd__(self, other):      return self._update("+=", other)
    def __isub__(self, other):      return self._update("-=", other)
    def __imul__(self, other):      return self._update("*=", other)
    def __itruediv__(self, other):  return self._update("/=", other)
    def __ifloordiv__(self, other): return self._update("/=", other)
    def __imod__(self, other):      return self._update("%=", other)
    def __iand__(self, other):      return self._update("&=", other)
    def __ior__(self, other):       return self._update("|=", other)
    def __ixor__(self, other):      return self._update("^=", other)
    def __ilshift__(self, other):   return self._update("<<=", other)
    def __irshift__(self, other):   return self._update(">>=", other)

    def assign(self, value) -> None:
        """
        Assign a new value to this local: ``self = value;``.
        """
        self._builder.add_stmt(ir.Update(self._name, "=", _to_expr(value)))


def local(dtype : dt.Dtype, init) -> Local:
    """
    Declare a mutable local variable in the current kernel.

    Emits ``<dtype> name = <init>;`` and returns a Local handle that supports
    compound assignment (``+=``, ``-=``, ...) and ``.assign(value)``.
    """
    builder = current_builder()
    name = builder.fresh_name("v")
    init_expr = _to_expr(init)
    builder.add_stmt(ir.Assign(name, dtype.metal, init_expr))
    return Local(name, dtype, builder)


def range(*args):
    """
    Emit a ``for`` loop. Usage:

        for i in sk.range(end): ...
        for i in sk.range(start, end): ...
        for i in sk.range(start, end, step): ...

    Bounds and step may be Python ints or spork Tracers.
    """
    if len(args) == 1:
        start_expr = ir.Const(0)
        end_expr = _to_expr(args[0])
        step_expr = ir.Const(1)
    elif len(args) == 2:
        start_expr = _to_expr(args[0])
        end_expr = _to_expr(args[1])
        step_expr = ir.Const(1)
    elif len(args) == 3:
        start_expr = _to_expr(args[0])
        end_expr = _to_expr(args[1])
        step_expr = _to_expr(args[2])
    else:
        raise TypeError(f"sk.range expects 1-3 arguments, got {len(args)}")
    return _RangeLoop(start_expr, end_expr, step_expr)


class _RangeLoop:
    __slots__ = ("_start", "_end", "_step")

    def __init__(self, start : ir.Expr, end : ir.Expr, step : ir.Expr):
        self._start = start
        self._end = end
        self._step = step

    def __iter__(self):
        builder = current_builder()
        loop_var = builder.fresh_name("i")
        saved_stmts = builder.stmts
        builder.stmts = []
        try:
            yield Tracer(ir.Var(loop_var), dt.uint32)
        finally:
            body = builder.stmts
            builder.stmts = saved_stmts
            builder.add_stmt(ir.ForLoop(
                var_name=loop_var,
                start=self._start,
                end=self._end,
                step=self._step,
                body=body,
            ))


class ThreadgroupArray:
    """
    A threadgroup-memory array. Supports both single-axis and tuple subscripts:

        a[i]        # 1-D
        a[i, j]     # 2-D, sugar for a[i][j]
        a[i, j, k]  # 3-D, ...
    """

    __slots__ = ("_name", "_dtype", "_shape", "_builder", "_expr")

    def __init__(
        self,
        name    : str,
        dtype   : dt.Dtype,
        shape   : tuple,
        builder : KernelBuilder,
    ):
        self._name = name
        self._dtype = dtype
        self._shape = shape
        self._builder = builder
        self._expr = ir.Var(name)

    def _indices(self, index):
        if isinstance(index, tuple):
            return tuple(_to_expr(i) for i in index)
        return (_to_expr(index),)

    def __getitem__(self, index) -> Tracer:
        indices = self._indices(index)
        expr : ir.Expr = self._expr
        for idx in indices:
            expr = ir.Load(expr, idx)
        return Tracer(expr, self._dtype)

    def __setitem__(self, index, value) -> None:
        indices = self._indices(index)
        *outer, last = indices
        ptr_expr : ir.Expr = self._expr
        for idx in outer:
            ptr_expr = ir.Load(ptr_expr, idx)
        self._builder.add_stmt(ir.Store(ptr_expr, last, _to_expr(value)))


def threadgroup(dtype : dt.Dtype, shape) -> ThreadgroupArray:
    """
    Declare a threadgroup-memory array of the given fixed shape.

    Shape elements must be Python ints (Metal requires constexpr sizes).
    """
    builder = current_builder()
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            raise TypeError(
                f"threadgroup array shape must be positive Python ints, got {shape!r}"
            )
    name = builder.fresh_name("s")
    builder.add_stmt(ir.ThreadgroupDecl(
        name=name,
        metal_type=dtype.metal,
        shape=list(shape),
    ))
    return ThreadgroupArray(name, dtype, shape, builder)


_VALID_MEM_SCOPES = {"none", "device", "threadgroup", "threadgroup_imageblock", "texture"}


def _barrier(func : str, scopes : tuple) -> None:
    if not scopes:
        scopes = ("threadgroup",)
    for s in scopes:
        if s not in _VALID_MEM_SCOPES:
            raise ValueError(
                f"Unknown memory scope {s!r}; expected one of {sorted(_VALID_MEM_SCOPES)}"
            )
    flag_text = " | ".join(f"mem_flags::mem_{s}" for s in scopes)
    builder = current_builder()
    builder.add_stmt(ir.ExprStmt(ir.Call(func, [ir.Raw(flag_text)])))


def threadgroup_barrier(*scopes : str) -> None:
    """
    Emit ``threadgroup_barrier(mem_flags::mem_<scope> | ...)``.

    Defaults to ``mem_threadgroup`` when no scopes are given.
    """
    _barrier("threadgroup_barrier", scopes)


def simdgroup_barrier(*scopes : str) -> None:
    """
    Emit ``simdgroup_barrier(mem_flags::mem_<scope> | ...)``.

    Defaults to ``mem_threadgroup`` when no scopes are given.
    """
    _barrier("simdgroup_barrier", scopes)


def _simd_op(func : str, *args, result_dtype : Optional[dt.Dtype] = None) -> Tracer:
    """
    Emit a simd collective as ``<T> simdN = func(args...);`` at the current
    statement position.

    Collectives must be called by every lane in the simdgroup. Materializing
    the result into a local prevents subsequent uses (e.g. inside an ``if``
    block) from re-inlining the call and accidentally putting the collective
    in divergent control flow.
    """
    builder = current_builder()
    arg_exprs = [_to_expr(a) for a in args]
    if result_dtype is not None:
        dtype = result_dtype
    else:
        first = args[0]
        dtype = first._dtype if isinstance(first, Tracer) else None
        if dtype is None:
            dtype = dt.float32
    name = builder.fresh_name("simd")
    builder.add_stmt(ir.Assign(name, dtype.metal, ir.Call(func, arg_exprs)))
    return Tracer(ir.Var(name), dtype)


def simd_sum(x) -> Tracer:     return _simd_op("simd_sum", x)
def simd_product(x) -> Tracer: return _simd_op("simd_product", x)
def simd_max(x) -> Tracer:     return _simd_op("simd_max", x)
def simd_min(x) -> Tracer:     return _simd_op("simd_min", x)
def simd_and(x) -> Tracer:     return _simd_op("simd_and", x)
def simd_or(x) -> Tracer:      return _simd_op("simd_or", x)
def simd_xor(x) -> Tracer:     return _simd_op("simd_xor", x)
def simd_all(x) -> Tracer:     return _simd_op("simd_all", x, result_dtype=dt.bool_)
def simd_any(x) -> Tracer:     return _simd_op("simd_any", x, result_dtype=dt.bool_)
def simd_prefix_inclusive_sum(x) -> Tracer:     return _simd_op("simd_prefix_inclusive_sum", x)
def simd_prefix_exclusive_sum(x) -> Tracer:     return _simd_op("simd_prefix_exclusive_sum", x)
def simd_prefix_inclusive_product(x) -> Tracer: return _simd_op("simd_prefix_inclusive_product", x)
def simd_prefix_exclusive_product(x) -> Tracer: return _simd_op("simd_prefix_exclusive_product", x)

def simd_broadcast(x, lane) -> Tracer:    return _simd_op("simd_broadcast", x, lane)
def simd_shuffle(x, lane) -> Tracer:      return _simd_op("simd_shuffle", x, lane)
def simd_shuffle_up(x, delta) -> Tracer:  return _simd_op("simd_shuffle_up", x, delta)
def simd_shuffle_down(x, delta) -> Tracer:return _simd_op("simd_shuffle_down", x, delta)
def simd_shuffle_xor(x, mask) -> Tracer:  return _simd_op("simd_shuffle_xor", x, mask)


class _IfBlock:
    """
    Context manager produced by ``sk.if_(cond)``.

    On enter, swaps in a fresh statement list so any kernel statements emitted
    inside the ``with`` block become the if-body. On exit, restores the outer
    statement list and appends an IfStmt. Holds a reference to the appended
    IfStmt so a paired ``.else_()`` block can mutate its ``else_body``.
    """

    __slots__ = ("_cond", "_builder", "_saved", "_if_stmt")

    def __init__(self, cond : ir.Expr):
        self._cond = cond
        self._builder : Optional[KernelBuilder] = None
        self._saved = None
        self._if_stmt : Optional[ir.IfStmt] = None

    def __enter__(self):
        builder = current_builder()
        self._builder = builder
        self._saved = builder.stmts
        builder.stmts = []
        return self

    def __exit__(self, exc_type, exc, tb):
        body = self._builder.stmts
        self._builder.stmts = self._saved
        self._if_stmt = ir.IfStmt(cond=self._cond, then_body=body, else_body=None)
        self._builder.add_stmt(self._if_stmt)
        return False

    def else_(self) -> "_ElseBlock":
        """
        Open the matching ``else { ... }`` block. Must immediately follow
        the ``with sk.if_(...)`` block — no statements may be emitted in
        between.
        """
        if self._if_stmt is None:
            raise RuntimeError(
                "sk.if_(...).else_() can only be used after the if-block "
                "has been exited."
            )
        return _ElseBlock(self)


class _ElseBlock:
    __slots__ = ("_if_block", "_builder", "_saved")

    def __init__(self, if_block : _IfBlock):
        self._if_block = if_block
        self._builder : Optional[KernelBuilder] = None
        self._saved = None

    def __enter__(self):
        builder = current_builder()
        if not builder.stmts or builder.stmts[-1] is not self._if_block._if_stmt:
            raise RuntimeError(
                "sk.else_ block must immediately follow its matching sk.if_ "
                "block (no statements in between)."
            )
        self._builder = builder
        self._saved = builder.stmts
        builder.stmts = []
        return self

    def __exit__(self, exc_type, exc, tb):
        body = self._builder.stmts
        self._builder.stmts = self._saved
        self._if_block._if_stmt.else_body = body
        return False


def if_(cond) -> _IfBlock:
    """
    Emit an ``if (cond) { ... }`` block.

    Usage::

        with sk.if_(thread_idx == 0):
            out[i] = total

    Pair with ``.else_()`` for an else branch::

        with sk.if_(x > 0) as branch:
            out[i] = x
        with branch.else_():
            out[i] = -x
    """
    return _IfBlock(_to_expr(cond))


class _WhileBlock:
    """
    Context manager produced by ``sk.while_(cond)``. Pushes a fresh statement
    list for the loop body; on exit, wraps the captured body in a WhileLoop.
    """

    __slots__ = ("_cond", "_builder", "_saved")

    def __init__(self, cond : ir.Expr):
        self._cond = cond
        self._builder : Optional[KernelBuilder] = None
        self._saved = None

    def __enter__(self):
        builder = current_builder()
        self._builder = builder
        self._saved = builder.stmts
        builder.stmts = []
        return self

    def __exit__(self, exc_type, exc, tb):
        body = self._builder.stmts
        self._builder.stmts = self._saved
        self._builder.add_stmt(ir.WhileLoop(cond=self._cond, body=body))
        return False


def while_(cond) -> _WhileBlock:
    """
    Emit a ``while (cond) { ... }`` loop.

    The condition is evaluated every iteration, so updates to locals
    referenced in ``cond`` take effect each iteration.

    Usage::

        i = sk.local(sk.dt.uint32, 0)
        with sk.while_(i < n):
            ...
            i += 1
    """
    return _WhileBlock(_to_expr(cond))


def break_() -> None:
    """
    Emit a ``break;`` statement (exit the innermost enclosing loop).
    """
    current_builder().add_stmt(ir.Break())


def continue_() -> None:
    """
    Emit a ``continue;`` statement (jump to the next iteration of the
    innermost enclosing loop).
    """
    current_builder().add_stmt(ir.Continue())


# ---------------------------------------------------------------------------
# MetalPerformancePrimitives: tensor views, matmul2d, cooperative tensors
# ---------------------------------------------------------------------------


class TensorHandle(_OpaqueHandle):
    """
    A handle to an ``mpp::tensor_ops::tensor`` view over a device pointer
    with compile-time extents.
    """

    __slots__ = ("_name", "_dtype", "_shape", "_builder", "_expr", "_source_pointer_name")

    def __init__(
        self,
        name                 : str,
        dtype                : dt.Dtype,
        shape                : tuple,
        builder              : KernelBuilder,
        source_pointer_name  : Optional[str] = None,
    ):
        self._name = name
        self._dtype = dtype
        self._shape = shape
        self._builder = builder
        self._expr = ir.Var(name)
        self._source_pointer_name = source_pointer_name

    def slice(self, tile_shape, offsets) -> "TileSlice":
        """
        Take a sub-tile of compile-time shape ``tile_shape`` starting at
        runtime offsets ``offsets``.

        Emits ``auto tileN = self.slice<W, H, ...>(off0, off1, ...);`` and
        returns an opaque handle referring to ``tileN``.
        """
        if not isinstance(tile_shape, tuple):
            tile_shape = tuple(tile_shape)
        if not isinstance(offsets, tuple):
            offsets = tuple(offsets)
        for d in tile_shape:
            if not isinstance(d, int) or d <= 0:
                raise TypeError(
                    f"slice tile_shape must be positive Python ints, got {tile_shape!r}"
                )
        offset_exprs = [_to_expr(o) for o in offsets]
        tile_name = self._builder.fresh_name("tile")
        method = ir.MethodCall(
            obj=ir.Var(self._name),
            method="slice",
            template_args=list(tile_shape),
            args=offset_exprs,
        )
        self._builder.add_stmt(ir.Assign(tile_name, "auto", method))
        return TileSlice(ir.Var(tile_name), self._dtype, tile_shape, self._builder, source_tensor=self)


class TileSlice(_OpaqueHandle):
    """
    An opaque handle to a tile produced by ``TensorHandle.slice``.

    Remembers the source TensorHandle so that downstream writes (e.g.
    ``coop.store(tile)``) can mark the underlying pointer parameter as
    written and trigger a buffer copy-back after dispatch.
    """

    __slots__ = ("_expr", "_dtype", "_shape", "_builder", "_source_tensor")

    def __init__(
        self,
        expr           : ir.Expr,
        dtype          : dt.Dtype,
        shape          : tuple,
        builder        : KernelBuilder,
        source_tensor  : Optional["TensorHandle"] = None,
    ):
        self._expr = expr
        self._dtype = dtype
        self._shape = shape
        self._builder = builder
        self._source_tensor = source_tensor


def tensor(ptr : PointerTracer, dtype : dt.Dtype, shape) -> TensorHandle:
    """
    Wrap a device pointer as an ``mpp::tensor_ops::tensor`` view with
    compile-time extents.

    Emits:

        extents<int, D0, D1, ...> extN;
        tensor tensorN(ptr, extN);
    """
    if not isinstance(ptr, PointerTracer):
        raise TypeError(
            f"sk.tensor expects a device pointer parameter, got {type(ptr).__name__}"
        )
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    for d in shape:
        if not isinstance(d, int) or d <= 0:
            raise TypeError(
                f"tensor shape must be positive Python ints, got {shape!r}"
            )
    builder = current_builder()
    builder.add_include("metal_tensor")
    builder.add_include("MetalPerformancePrimitives/MetalPerformancePrimitives.h")
    builder.add_using("mpp::tensor_ops")

    extents_name = builder.fresh_name("ext")
    shape_str = ", ".join(str(d) for d in shape)
    builder.add_stmt(ir.DefaultDecl(
        name=extents_name,
        metal_type=f"extents<int, {shape_str}>",
    ))
    tensor_name = builder.fresh_name("tensor")
    builder.add_stmt(ir.ConstructorDecl(
        name=tensor_name,
        metal_type="tensor",
        args=[ptr._expr, ir.Var(extents_name)],
    ))
    source_name = ptr._expr.name if isinstance(ptr._expr, ir.Var) else None
    return TensorHandle(tensor_name, dtype, shape, builder, source_pointer_name=source_name)


_MATMUL2D_MODES = {
    "multiply_accumulate" : "matmul2d_descriptor::mode::multiply_accumulate",
    "multiply"            : "matmul2d_descriptor::mode::multiply",
}


class MatmulOp:
    """
    Handle to an ``mpp::tensor_ops::matmul2d`` op with a fixed descriptor
    and execution policy.
    """

    __slots__ = ("_name", "_builder", "_tile_m", "_tile_n", "_tile_k")

    def __init__(
        self,
        name    : str,
        tile_m  : int,
        tile_n  : int,
        tile_k  : int,
        builder : KernelBuilder,
    ):
        self._name = name
        self._tile_m = tile_m
        self._tile_n = tile_n
        self._tile_k = tile_k
        self._builder = builder

    def get_destination(
        self,
        tensor_a : TensorHandle,
        tensor_b : TensorHandle,
        dtype    : dt.Dtype,
    ) -> "CooperativeTensor":
        """
        Allocate the accumulator cooperative tensor:

            auto coopN = opN.get_destination_cooperative_tensor<
                decltype(tensorA), decltype(tensorB), <dtype>>();
        """
        coop_name = self._builder.fresh_name("coop")
        method = ir.MethodCall(
            obj=ir.Var(self._name),
            method="get_destination_cooperative_tensor",
            template_args=[
                ir.Raw(f"decltype({tensor_a._name})"),
                ir.Raw(f"decltype({tensor_b._name})"),
                ir.Raw(dtype.metal),
            ],
            args=[],
        )
        self._builder.add_stmt(ir.Assign(coop_name, "auto", method))
        return CooperativeTensor(coop_name, dtype, self._builder)

    def run(self, tile_a : TileSlice, tile_b : TileSlice, coop : "CooperativeTensor") -> None:
        """
        Emit ``opN.run(tile_a, tile_b, coop);``.
        """
        call = ir.MethodCall(
            obj=ir.Var(self._name),
            method="run",
            template_args=[],
            args=[_to_expr(tile_a), _to_expr(tile_b), _to_expr(coop)],
        )
        self._builder.add_stmt(ir.ExprStmt(call))


class CooperativeTensor(_OpaqueHandle):
    """
    Handle to a cooperative tensor (matmul2d accumulator).
    """

    __slots__ = ("_name", "_dtype", "_builder", "_expr")

    def __init__(self, name : str, dtype : dt.Dtype, builder : KernelBuilder):
        self._name = name
        self._dtype = dtype
        self._builder = builder
        self._expr = ir.Var(name)

    def store(self, tile : TileSlice) -> None:
        """
        Emit ``self.store(tile);``.

        If the tile slices a tensor that wraps a kernel pointer parameter,
        mark that parameter as written so the runtime copies the buffer back
        into the user's numpy array after dispatch.
        """
        if isinstance(tile, TileSlice) and tile._source_tensor is not None:
            ptr_name = tile._source_tensor._source_pointer_name
            if ptr_name is not None:
                for p in self._builder.params:
                    if p.kind == "pointer" and p.name == ptr_name:
                        p.written = True
                        break
        call = ir.MethodCall(
            obj=ir.Var(self._name),
            method="store",
            template_args=[],
            args=[_to_expr(tile)],
        )
        self._builder.add_stmt(ir.ExprStmt(call))


def matmul2d(
    m            : int,
    n            : int,
    k            : int,
    *,
    simdgroups   : int = 4,
    transpose_a  : bool = False,
    transpose_b  : bool = False,
    transpose_c  : bool = False,
    mode         : str = "multiply_accumulate",
) -> MatmulOp:
    """
    Declare an ``mpp::tensor_ops::matmul2d`` op with the given tile sizes
    and execution policy.

    Emits:

        constexpr auto descN = matmul2d_descriptor(M, N, K,
            transA, transB, transC, <mode>);
        matmul2d<descN, execution_simdgroups<S>> opN;
    """
    if mode not in _MATMUL2D_MODES:
        raise ValueError(
            f"Unknown matmul2d mode {mode!r}; expected one of {sorted(_MATMUL2D_MODES)}"
        )
    for v, label in ((m, "m"), (n, "n"), (k, "k"), (simdgroups, "simdgroups")):
        if not isinstance(v, int) or v <= 0:
            raise TypeError(f"matmul2d {label} must be a positive Python int, got {v!r}")

    builder = current_builder()
    builder.add_include("metal_compute")
    builder.add_include("metal_tensor")
    builder.add_include("MetalPerformancePrimitives/MetalPerformancePrimitives.h")
    builder.add_using("mpp::tensor_ops")

    desc_name = builder.fresh_name("desc")
    desc_call = ir.Call("matmul2d_descriptor", [
        ir.Const(m), ir.Const(n), ir.Const(k),
        ir.Const(transpose_a), ir.Const(transpose_b), ir.Const(transpose_c),
        ir.Raw(_MATMUL2D_MODES[mode]),
    ])
    builder.add_stmt(ir.Assign(desc_name, "constexpr auto", desc_call))

    op_name = builder.fresh_name("op")
    builder.add_stmt(ir.DefaultDecl(
        name=op_name,
        metal_type=f"matmul2d<{desc_name}, execution_simdgroups<{simdgroups}>>",
    ))
    return MatmulOp(op_name, m, n, k, builder)


# ---------------------------------------------------------------------------
# Math intrinsics + cast
# ---------------------------------------------------------------------------


def _math_call(func : str, *args, result_dtype : Optional[dt.Dtype] = None) -> Tracer:
    """
    Pure-expression intrinsic — emits ``func(args...)`` without materializing.
    Result dtype defaults to the first arg's dtype.
    """
    arg_exprs = [_to_expr(a) for a in args]
    if result_dtype is None:
        first = args[0]
        result_dtype = first._dtype if isinstance(first, Tracer) else None
    return Tracer(ir.Call(func, arg_exprs), result_dtype)


# Unary float math
def exp(x)   : return _math_call("exp", x)
def exp2(x)  : return _math_call("exp2", x)
def log(x)   : return _math_call("log", x)
def log2(x)  : return _math_call("log2", x)
def log10(x) : return _math_call("log10", x)
def sqrt(x)  : return _math_call("sqrt", x)
def rsqrt(x) : return _math_call("rsqrt", x)
def sin(x)   : return _math_call("sin", x)
def cos(x)   : return _math_call("cos", x)
def tan(x)   : return _math_call("tan", x)
def asin(x)  : return _math_call("asin", x)
def acos(x)  : return _math_call("acos", x)
def atan(x)  : return _math_call("atan", x)
def sinh(x)  : return _math_call("sinh", x)
def cosh(x)  : return _math_call("cosh", x)
def tanh(x)  : return _math_call("tanh", x)
def floor(x) : return _math_call("floor", x)
def ceil(x)  : return _math_call("ceil", x)
def round(x) : return _math_call("round", x)
def trunc(x) : return _math_call("trunc", x)
def fabs(x)  : return _math_call("fabs", x)
def abs(x)   : return _math_call("abs", x)
def sign(x)  : return _math_call("sign", x)

# Two-arg
def pow(x, y)   : return _math_call("pow", x, y)
def fmod(x, y)  : return _math_call("fmod", x, y)
def atan2(y, x) : return _math_call("atan2", y, x)
def fmin(x, y)  : return _math_call("fmin", x, y)
def fmax(x, y)  : return _math_call("fmax", x, y)
def min(x, y)   : return _math_call("min", x, y)
def max(x, y)   : return _math_call("max", x, y)

# Three-arg
def clamp(x, lo, hi) : return _math_call("clamp", x, lo, hi)
def fma(a, b, c)     : return _math_call("fma", a, b, c)

# Fast (approximate, lower-precision) variants — Metal's metal::fast:: namespace
def fast_exp(x)   : return _math_call("fast::exp", x)
def fast_log(x)   : return _math_call("fast::log", x)
def fast_log2(x)  : return _math_call("fast::log2", x)
def fast_sqrt(x)  : return _math_call("fast::sqrt", x)
def fast_rsqrt(x) : return _math_call("fast::rsqrt", x)
def fast_sin(x)   : return _math_call("fast::sin", x)
def fast_cos(x)   : return _math_call("fast::cos", x)
def fast_tan(x)   : return _math_call("fast::tan", x)
def fast_pow(x, y): return _math_call("fast::pow", x, y)


def cast(value, dtype : dt.Dtype) -> Tracer:
    """
    Emit ``static_cast<dtype>(value)``.
    """
    return Tracer(ir.Cast(dtype, _to_expr(value)), dtype)


# ---------------------------------------------------------------------------
# Atomic operations
# ---------------------------------------------------------------------------


_VALID_MEM_ORDERS = {"relaxed"}  # Metal currently only exposes memory_order_relaxed


def _mem_order(order : str) -> ir.Raw:
    if order not in _VALID_MEM_ORDERS:
        raise ValueError(
            f"Unknown memory order {order!r}; expected one of {sorted(_VALID_MEM_ORDERS)}"
        )
    return ir.Raw(f"memory_order_{order}")


def _atomic_addr(ptr : PointerTracer, idx) -> ir.AddrOf:
    if not isinstance(ptr, PointerTracer):
        raise TypeError(
            "atomic operations expect a sk.DevicePointer (typically with an "
            "atomic dtype), got %r" % (type(ptr).__name__,)
        )
    return ir.AddrOf(ir.Load(ptr._expr, _to_expr(idx)))


def _mark_pointer_written(ptr : PointerTracer) -> None:
    if isinstance(ptr._expr, ir.Var):
        builder = current_builder()
        for p in builder.params:
            if p.kind == "pointer" and p.name == ptr._expr.name:
                p.written = True
                break


def _atomic_rmw(func : str, ptr : PointerTracer, idx, value, order : str) -> Tracer:
    """
    Atomic read-modify-write. Always emits a local that captures the prior
    value, so the side effect happens even if the caller ignores the result.
    """
    builder = current_builder()
    order_expr = _mem_order(order)
    addr = _atomic_addr(ptr, idx)
    val_expr = _to_expr(value)
    call = ir.Call(func, [addr, val_expr, order_expr])
    underlying = dt.underlying(ptr._dtype) if ptr._dtype is not None else None
    metal_type = underlying.metal if underlying is not None else "auto"
    name = builder.fresh_name("atom")
    builder.add_stmt(ir.Assign(name, metal_type, call))
    _mark_pointer_written(ptr)
    return Tracer(ir.Var(name), underlying)


def atomic_load(ptr : PointerTracer, idx, order : str = "relaxed") -> Tracer:
    """
    Atomically load ``ptr[idx]``.
    """
    builder = current_builder()
    addr = _atomic_addr(ptr, idx)
    call = ir.Call("atomic_load_explicit", [addr, _mem_order(order)])
    underlying = dt.underlying(ptr._dtype) if ptr._dtype is not None else None
    metal_type = underlying.metal if underlying is not None else "auto"
    name = builder.fresh_name("atomload")
    builder.add_stmt(ir.Assign(name, metal_type, call))
    return Tracer(ir.Var(name), underlying)


def atomic_store(ptr : PointerTracer, idx, value, order : str = "relaxed") -> None:
    """
    Atomically store ``value`` into ``ptr[idx]``.
    """
    addr = _atomic_addr(ptr, idx)
    call = ir.Call("atomic_store_explicit", [addr, _to_expr(value), _mem_order(order)])
    current_builder().add_stmt(ir.ExprStmt(call))
    _mark_pointer_written(ptr)


def atomic_fetch_add(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_add_explicit", ptr, idx, value, order)


def atomic_fetch_sub(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_sub_explicit", ptr, idx, value, order)


def atomic_fetch_and(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_and_explicit", ptr, idx, value, order)


def atomic_fetch_or(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_or_explicit", ptr, idx, value, order)


def atomic_fetch_xor(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_xor_explicit", ptr, idx, value, order)


def atomic_fetch_min(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_min_explicit", ptr, idx, value, order)


def atomic_fetch_max(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_fetch_max_explicit", ptr, idx, value, order)


def atomic_exchange(ptr, idx, value, order : str = "relaxed") -> Tracer:
    return _atomic_rmw("atomic_exchange_explicit", ptr, idx, value, order)


# ---------------------------------------------------------------------------
# GPU trace capture (profiler)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def profile(name : str = "spork_trace", open_in_xcode : bool = True):
    """
    Capture all GPU work submitted inside the ``with`` block to a
    ``.gputrace`` document, then optionally open it in Xcode.

    Requires the environment variable ``MTL_CAPTURE_ENABLED=1`` to be set
    before launching Python; otherwise Metal refuses to start a capture.

    Usage::

        with sk.profile():
            matmul[(g, 1, 1), (32, 1, 1)](C, A, B)
    """
    queue = runtime.get_command_queue()
    manager = MTLCaptureManager.sharedCaptureManager()
    if not manager.supportsDestination_(MTLCaptureDestinationGPUTraceDocument):
        raise RuntimeError(
            "GPU trace capture is not enabled. Set MTL_CAPTURE_ENABLED=1 in "
            "the environment before launching Python."
        )

    trace_path = os.path.join(mkdtemp(), f"{name}.gputrace")
    trace_url = NSURL.fileURLWithPath_(trace_path)
    desc = MTLCaptureDescriptor()
    desc.setCaptureObject_(queue)
    desc.setDestination_(MTLCaptureDestinationGPUTraceDocument)
    desc.setOutputURL_(trace_url)

    _, error = manager.startCaptureWithDescriptor_error_(desc, None)
    if error:
        raise RuntimeError(f"Failed to start capture: {error.localizedDescription()}")

    try:
        yield trace_path
    finally:
        manager.stopCapture()
        if open_in_xcode:
            subprocess.run(["open", trace_path])


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------


class DeviceFn:
    """
    A reusable device-side function, traced once on first call.

    Parameter annotations must each be either ``sk.DevicePointer[dtype]`` or
    a ``dt.Dtype`` (for scalar-by-value parameters). The return annotation
    determines the function's return type: omitted (or ``None``) means
    ``void``; a ``dt.Dtype`` means the body must return a value of that type.

    When called from inside a ``@sk.jit`` kernel (or from another device fn),
    the device fn is registered as a dependency of the enclosing kernel so
    its source is emitted alongside.
    """

    __slots__ = (
        "_fn", "_name", "_sig", "_traced", "_tracing",
        "_param_strs", "_return_type", "_return_dtype",
        "_stmts", "_includes", "_usings", "_deps",
    )

    def __init__(self, fn : Callable):
        self._fn = fn
        self._name : str = fn.__name__
        self._sig = inspect.signature(fn)
        self._traced : bool = False
        self._tracing : bool = False
        self._param_strs : List[str] = []
        self._return_type : str = "void"
        self._return_dtype : Optional[dt.Dtype] = None
        self._stmts : List[ir.Stmt] = []
        self._includes : List[tuple] = []
        self._usings : List[str] = []
        self._deps : List["DeviceFn"] = []

    def _ensure_traced(self) -> None:
        if self._traced:
            return
        if self._tracing:
            raise RuntimeError(
                f"Device fn '{self._name}' is recursive; Metal does not support "
                "recursion in device functions"
            )
        self._tracing = True
        try:
            self._do_trace()
            self._traced = True
        finally:
            self._tracing = False

    def _do_trace(self) -> None:
        builder = KernelBuilder(self._name)
        tracer_args = []
        for param_name, param in self._sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"Device fn '{self._name}': *args/**kwargs not supported"
                )
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                raise TypeError(
                    f"Device fn '{self._name}': parameter '{param_name}' is missing "
                    "a type annotation"
                )
            if isinstance(ann, DevicePointerSpec):
                self._param_strs.append(f"device {ann.dtype.metal} *{param_name}")
                tracer_args.append(PointerTracer(ir.Var(param_name), ann.dtype, builder))
            elif isinstance(ann, dt.Dtype):
                self._param_strs.append(f"{ann.metal} {param_name}")
                tracer_args.append(Tracer(ir.Var(param_name), ann))
            else:
                raise TypeError(
                    f"Device fn '{self._name}': parameter '{param_name}' has "
                    f"unsupported annotation {ann!r}. Expected sk.DevicePointer[...] "
                    "or a spork dtype."
                )

        ret_ann = self._sig.return_annotation
        if ret_ann is inspect.Parameter.empty or ret_ann is None or ret_ann is type(None):
            self._return_type = "void"
            self._return_dtype = None
        elif isinstance(ret_ann, dt.Dtype):
            self._return_type = ret_ann.metal
            self._return_dtype = ret_ann
        else:
            raise TypeError(
                f"Device fn '{self._name}': return annotation must be a spork dtype "
                f"or omitted, got {ret_ann!r}"
            )

        token = _builder.set(builder)
        try:
            result = self._fn(*tracer_args)
        finally:
            _builder.reset(token)

        if self._return_dtype is not None:
            if result is None:
                raise TypeError(
                    f"Device fn '{self._name}' declared return type "
                    f"{self._return_dtype.name} but returned None"
                )
            builder.add_stmt(ir.Return(_to_expr(result)))

        self._stmts = builder.stmts
        self._includes = list(builder.includes)
        self._usings = list(builder.usings)
        self._deps = list(builder.device_functions)

    def __call__(self, *args):
        builder = current_builder()
        builder.add_device_fn(self)
        if len(args) != len(self._param_strs):
            raise TypeError(
                f"Device fn '{self._name}' expected {len(self._param_strs)} "
                f"argument(s), got {len(args)}"
            )
        arg_exprs = [_to_expr(a) for a in args]
        call = ir.Call(self._name, arg_exprs)
        if self._return_dtype is None:
            builder.add_stmt(ir.ExprStmt(call))
            return None
        return Tracer(call, self._return_dtype)


def device_fn(fn : Callable) -> DeviceFn:
    """
    Decorator marking a function as a reusable device function.

    The function is traced once on its first call from inside a kernel and
    emitted as a standalone Metal function above the calling kernel. Device
    functions may call other device functions and use any kernel-body
    primitives (sk.local, sk.range, sk.if_, sk.while_, simd intrinsics,
    barriers, etc.).

    Usage::

        @sk.device_fn
        def square(x : sk.dt.float32) -> sk.dt.float32:
            return x * x

        @sk.jit
        def kernel(out : sk.DevicePointer[sk.dt.float32],
                   A   : sk.DevicePointer[sk.dt.float32],
                   i   : sk.Uint[sk.ThreadPositionInGrid]):
            out[i] = square(A[i])
    """
    return DeviceFn(fn)

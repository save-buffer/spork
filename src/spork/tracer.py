import contextvars
from typing import Dict, List, Optional

from . import dtypes as dt
from . import ir


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
        self._name_counters : Dict[str, int] = {}

    def add_stmt(self, stmt : ir.Stmt) -> None:
        self.stmts.append(stmt)

    def fresh_name(self, prefix : str = "v") -> str:
        n = self._name_counters.get(prefix, 0)
        self._name_counters[prefix] = n + 1
        return f"{prefix}{n}"


def _to_expr(value) -> ir.Expr:
    if isinstance(value, Tracer):
        return value._expr
    if isinstance(value, VectorTracer):
        raise TypeError(
            "Cannot use a vector value as a scalar — access a field like .x first"
        )
    if isinstance(value, PointerTracer):
        raise TypeError("Cannot use a pointer as a scalar value")
    if isinstance(value, bool):
        return ir.Const(value)
    if isinstance(value, (int, float)):
        return ir.Const(value)
    raise TypeError(
        f"Cannot convert {value!r} (type {type(value).__name__}) to a spork expression"
    )


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

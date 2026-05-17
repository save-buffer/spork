import contextvars
from typing import List, Optional

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

    def add_stmt(self, stmt : ir.Stmt) -> None:
        self.stmts.append(stmt)


def _to_expr(value) -> ir.Expr:
    if isinstance(value, Tracer):
        return value._expr
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

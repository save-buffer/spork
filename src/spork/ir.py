from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

from . import dtypes as dt
from .types import ThreadAttribute


class Expr:
    pass


@dataclass
class Var(Expr):
    name : str


@dataclass
class Const(Expr):
    value : Union[int, float, bool]


@dataclass
class Load(Expr):
    ptr   : Expr
    index : Expr


@dataclass
class Member(Expr):
    """
    Access a field on a vector or struct expression, e.g. ``gid.x``.
    """

    operand : Expr
    field   : str


@dataclass
class BinOp(Expr):
    op  : str
    lhs : Expr
    rhs : Expr


@dataclass
class UnaryOp(Expr):
    op      : str
    operand : Expr


@dataclass
class Cast(Expr):
    dtype   : dt.Dtype
    operand : Expr


@dataclass
class Call(Expr):
    """
    A function-call expression: ``func(arg0, arg1, ...)``.
    """

    func : str
    args : List["Expr"] = field(default_factory=list)


@dataclass
class Raw(Expr):
    """
    A verbatim chunk of Metal source embedded as an expression.

    Useful for things like ``mem_flags::mem_threadgroup`` that don't deserve
    their own IR node.
    """

    text : str


class Stmt:
    pass


@dataclass
class Store(Stmt):
    ptr   : Expr
    index : Expr
    value : Expr


@dataclass
class Assign(Stmt):
    """
    Declare-and-assign a local: ``<metal_type> name = <expr>;``.
    """

    name       : str
    metal_type : str
    value      : Expr


@dataclass
class Update(Stmt):
    """
    Re-assign an existing local: ``name <op>= <expr>;`` (op may be ``=``).
    """

    name  : str
    op    : str
    value : Expr


@dataclass
class ForLoop(Stmt):
    """
    A ``for (uint i = start; i < end; i += step) { ... }`` loop.
    """

    var_name : str
    start    : Expr
    end      : Expr
    step     : Expr
    body     : List[Stmt] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    """
    An ``if (cond) { ... } else { ... }`` statement; ``else_body`` may be None.
    """

    cond      : Expr
    then_body : List[Stmt] = field(default_factory=list)
    else_body : Optional[List[Stmt]] = None


@dataclass
class ExprStmt(Stmt):
    """
    A statement whose effect is just evaluating an expression
    (typically a Call): ``<expr>;``.
    """

    expr : Expr


@dataclass
class ThreadgroupDecl(Stmt):
    """
    A threadgroup memory array declaration:
    ``threadgroup <metal_type> <name>[D0][D1]...;``.

    All dimensions must be Python ints (Metal requires constexpr sizes).
    """

    name       : str
    metal_type : str
    shape      : List[int]


@dataclass
class Param:
    name       : str
    kind       : str
    dtype      : dt.Dtype
    metal_name : Optional[str] = None
    vec_size   : int = 1
    attribute  : Optional[Type[ThreadAttribute]] = None
    written    : bool = False

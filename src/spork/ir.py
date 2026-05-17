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
class Param:
    name       : str
    kind       : str
    dtype      : dt.Dtype
    metal_name : Optional[str] = None
    vec_size   : int = 1
    attribute  : Optional[Type[ThreadAttribute]] = None
    written    : bool = False

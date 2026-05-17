from dataclasses import dataclass
from typing import Optional, Type, Union

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
class Param:
    name       : str
    kind       : str
    dtype      : dt.Dtype
    metal_name : Optional[str] = None
    attribute  : Optional[Type[ThreadAttribute]] = None
    written    : bool = False

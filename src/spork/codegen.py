from typing import List

from . import ir
from .tracer import KernelBuilder


_PREC = {
    "||" : 1,
    "&&" : 2,
    "|"  : 3,
    "^"  : 4,
    "&"  : 5,
    "==" : 6, "!=" : 6,
    "<"  : 7, "<=" : 7, ">" : 7, ">=" : 7,
    "<<" : 8, ">>" : 8,
    "+"  : 9, "-"  : 9,
    "*"  : 10, "/" : 10, "%" : 10,
}


def _format_const(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if "." not in text and "e" not in text and "n" not in text:
            text += ".0"
        return text + "f"
    raise TypeError(f"Cannot format constant of type {type(value).__name__}: {value!r}")


def format_expr(expr : ir.Expr, parent_prec : int = 0) -> str:
    if isinstance(expr, ir.Var):
        return expr.name
    if isinstance(expr, ir.Const):
        return _format_const(expr.value)
    if isinstance(expr, ir.Load):
        return f"{format_expr(expr.ptr, 100)}[{format_expr(expr.index, 0)}]"
    if isinstance(expr, ir.Member):
        return f"{format_expr(expr.operand, 100)}.{expr.field}"
    if isinstance(expr, ir.BinOp):
        prec = _PREC.get(expr.op, 0)
        s = f"{format_expr(expr.lhs, prec)} {expr.op} {format_expr(expr.rhs, prec + 1)}"
        if prec < parent_prec:
            s = f"({s})"
        return s
    if isinstance(expr, ir.UnaryOp):
        return f"{expr.op}{format_expr(expr.operand, 11)}"
    if isinstance(expr, ir.Cast):
        return f"static_cast<{expr.dtype.metal}>({format_expr(expr.operand, 0)})"
    raise TypeError(f"Unknown expression node: {type(expr).__name__}")


def _format_for_step(var_name : str, step : ir.Expr) -> str:
    if isinstance(step, ir.Const) and step.value == 1:
        return f"{var_name}++"
    if isinstance(step, ir.Const) and step.value == -1:
        return f"{var_name}--"
    return f"{var_name} += {format_expr(step, 0)}"


def format_stmt(stmt : ir.Stmt, indent : int = 4) -> str:
    pad = " " * indent
    if isinstance(stmt, ir.Store):
        return (
            f"{pad}{format_expr(stmt.ptr, 100)}[{format_expr(stmt.index, 0)}] = "
            f"{format_expr(stmt.value, 0)};"
        )
    if isinstance(stmt, ir.Assign):
        return f"{pad}{stmt.metal_type} {stmt.name} = {format_expr(stmt.value, 0)};"
    if isinstance(stmt, ir.Update):
        return f"{pad}{stmt.name} {stmt.op} {format_expr(stmt.value, 0)};"
    if isinstance(stmt, ir.ForLoop):
        body_lines = format_stmts(stmt.body, indent + 4)
        body = body_lines if body_lines else f"{pad}    // empty"
        return (
            f"{pad}for (uint {stmt.var_name} = {format_expr(stmt.start, 0)}; "
            f"{stmt.var_name} < {format_expr(stmt.end, 0)}; "
            f"{_format_for_step(stmt.var_name, stmt.step)})\n"
            f"{pad}{{\n"
            f"{body}\n"
            f"{pad}}}"
        )
    raise TypeError(f"Unknown statement node: {type(stmt).__name__}")


def format_stmts(stmts : List[ir.Stmt], indent : int = 4) -> str:
    return "\n".join(format_stmt(s, indent) for s in stmts)


def _format_param(p : ir.Param, buffer_idx : int) -> tuple[str, int]:
    if p.kind == "pointer":
        return (
            f"    device {p.dtype.metal} *{p.name} [[buffer({buffer_idx})]]",
            buffer_idx + 1,
        )
    if p.kind == "constant":
        return (
            f"    constant {p.metal_name or p.dtype.metal} &{p.name} [[buffer({buffer_idx})]]",
            buffer_idx + 1,
        )
    if p.kind == "attribute":
        assert p.attribute is not None
        return (
            f"    {p.metal_name or p.dtype.metal} {p.name} [[{p.attribute.metal_attr}]]",
            buffer_idx,
        )
    raise ValueError(f"Unknown param kind: {p.kind}")


def emit_kernel(builder : KernelBuilder) -> str:
    param_lines = []
    buffer_idx = 0
    for p in builder.params:
        line, buffer_idx = _format_param(p, buffer_idx)
        param_lines.append(line)

    params_block = ",\n".join(param_lines)
    body = format_stmts(builder.stmts, indent=4)
    if not body:
        body = "    // empty kernel"

    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        f"kernel void {builder.name}(\n"
        f"{params_block})\n"
        "{\n"
        f"{body}\n"
        "}\n"
    )

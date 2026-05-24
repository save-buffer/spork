"""
``@skv.jit`` — the verified-kernel decorator.

Wraps ``@sk.jit`` so that during the kernel's trace, parameters declared
as ``skv.DevicePointer[dtype, shape]`` are presented to the kernel body
as ``TypedTensorHandle`` instances. Untyped params (thread-position
attributes, plain scalars) pass through unchanged.

Per-tile verification fires when the kernel calls ``tOut.assign(...)``
on its typed output handle. Coverage tracking — proving that all
threadgroups together write every output element exactly once — fires
later at ``.bind(grid=..., threadgroup=...)`` (follow-up work).
"""

import functools
import inspect
from typing import Callable, Optional

from ..jit import jit as _spork_jit
from .. import tracer as _spork_tracer
from ..types import DevicePointerSpec
from ._backend import OutputSpec, TypedDevicePointerSpec, _untyped_pointer_spec
from .primitives import (
    TypedScalarTracer,
    TypedTensorHandle,
    TypedVectorTracer,
    make_output_handle,
    tensor as _typed_tensor,
)
from stile.indexing import SymbolicInt


# Sentinel: the name of the output parameter in a verified kernel. By
# convention, the first parameter is the output (matches the rest of
# spork). Subclasses / future versions may want a more explicit
# annotation; for now we just take param 0.
_OUTPUT_PARAM_INDEX = 0


def jit(
    *,
    out_spec : OutputSpec,
    output_param : Optional[str] = None,
) -> Callable:
    """
    Decorator for verified spork kernels.

    Usage::

        @skv.jit(out_spec=OutputSpec("(M K, K N -> M N)", st=(M, N)))
        def matmul(
            out : skv.DevicePointer[sk.dt.float32, (M, N)],
            A   : skv.DevicePointer[sk.dt.float32, (M, K)],
            B   : skv.DevicePointer[sk.dt.float32, (K, N)],
            bid : sk.Uint2[sk.ThreadgroupPositionInGrid],
        ):
            tA = skv.tensor(A)
            tB = skv.tensor(B)
            # ... compute `result` (a TypedTensorHandle or typed value) ...
            out.assign(result)

    ``output_param`` may name the parameter that's the verified output;
    defaults to the first parameter.
    """
    if not isinstance(out_spec, OutputSpec):
        raise TypeError(
            f"@skv.jit out_spec must be a skv.OutputSpec, got {type(out_spec).__name__}"
        )

    def decorator(fn : Callable):
        sig = inspect.signature(fn)
        params = list(sig.parameters.items())
        if not params:
            raise TypeError(f"@skv.jit kernel '{fn.__name__}' has no parameters")

        if output_param is None:
            output_name = params[_OUTPUT_PARAM_INDEX][0]
        else:
            output_name = output_param
            if output_name not in dict(params):
                raise TypeError(
                    f"@skv.jit kernel '{fn.__name__}' has no parameter "
                    f"named {output_name!r}"
                )

        # Walk the annotations and build a substitution table mapping
        # parameter name → (untyped-spec-to-pass-to-sk.jit, stile-shape).
        # We replace each typed annotation with the underlying spork
        # DevicePointerSpec before sk.jit sees it.
        typed_params : dict = {}  # name → TypedDevicePointerSpec
        for name, param in params:
            ann = param.annotation
            if isinstance(ann, TypedDevicePointerSpec):
                typed_params[name] = ann

        if output_name not in typed_params:
            raise TypeError(
                f"@skv.jit kernel '{fn.__name__}': output parameter "
                f"{output_name!r} must be annotated as "
                "skv.DevicePointer[dtype, shape]"
            )

        # Build the kernel function that sk.jit will see: same signature
        # as the user's, but with typed annotations replaced by plain
        # sk.DevicePointer specs, and wrapping incoming spork PointerTracer
        # values as TypedTensorHandles before delegating to the user body.
        @functools.wraps(fn)
        def inner(*tracer_args):
            # tracer_args come in the same order as the parameters; we
            # wrap each one that has a typed annotation, plus auto-wrap
            # vector/scalar grid-position attribute params so their
            # components carry stile SymbolicInts the slice machinery
            # can use as Sliced offsets.
            wrapped : list = []
            for (name, _param), arg in zip(params, tracer_args):
                if name in typed_params:
                    spec = typed_params[name]
                    if name == output_name:
                        wrapped.append(make_output_handle(arg, out_spec))
                    else:
                        wrapped.append(
                            _typed_tensor(arg, spec.shape, dtype=spec.dtype)
                        )
                elif isinstance(arg, _spork_tracer.VectorTracer):
                    wrapped.append(TypedVectorTracer(arg, var_name_prefix=name))
                elif isinstance(arg, _spork_tracer.Tracer):
                    wrapped.append(TypedScalarTracer(
                        tracer=arg,
                        sym=SymbolicInt(name=f"_{name}"),
                    ))
                else:
                    wrapped.append(arg)
            return fn(*wrapped)

        # Construct the underlying spork-jitted kernel. We have to swap
        # the typed annotations out of inner's signature so spork's own
        # parameter-spec dispatch (which expects DevicePointerSpec) is
        # happy.
        inner.__signature__ = sig.replace(parameters=[
            param.replace(annotation=_untyped_pointer_spec(typed_params[name]))
            if name in typed_params else param
            for name, param in params
        ])

        return _spork_jit(inner)

    return decorator

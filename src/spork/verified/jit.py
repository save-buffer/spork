"""
``@skv.jit`` — the verified-kernel decorator.

Wraps ``@sk.jit`` so that during the kernel's trace, parameters declared
as ``skv.DevicePointer[dtype, shape]`` are presented to the kernel body
as ``TypedTensorHandle`` instances and grid-position attribute params
(``Uint2[ThreadgroupPositionInGrid]`` etc.) are presented as
``TypedVectorTracer`` / ``TypedScalarTracer`` whose components carry
stile ``SymbolicInt``s registered against their grid axis.

Per-tile verification fires when the kernel calls ``tOut.assign(...)``
or ``coop.store(out_tile)``. Bind-time coverage verification — proving
the union of per-store slices across all grid threadgroups covers the
declared output shape exactly once — fires at ``.bind(grid=...,
threadgroup=...)``.
"""

import functools
import inspect
from typing import Callable, Optional

from stile.indexing import SymbolicInt

from ..jit import BoundKernel, jit as _spork_jit
from .. import tracer as _spork_tracer
from ..types import (
    DevicePointerSpec,
    ScalarParamSpec,
    ThreadPositionInGrid,
    ThreadgroupPositionInGrid,
)
from ._backend import OutputSpec, TypedDevicePointerSpec, _untyped_pointer_spec
from . import _coverage
from .primitives import (
    TypedScalarTracer,
    TypedTensorHandle,
    TypedVectorTracer,
    make_output_handle,
    tensor as _typed_tensor,
)


# Attribute classes whose component values correspond to grid axes —
# they vary across the dispatched work and can partition the output.
# Each entry maps the attribute class to its GridAxisInfo "kind"
# string for the coverage check's enumeration.
_GRID_ATTR_KINDS = {
    ThreadgroupPositionInGrid : "tgid",
    ThreadPositionInGrid      : "gid",
}


_OUTPUT_PARAM_INDEX = 0


class VerifiedJittedKernel:
    """
    Wraps a spork ``JittedKernel`` with the verified-kernel state
    captured during trace. ``.bind(grid, threadgroup)`` runs the
    bind-time coverage check before producing a ``BoundKernel``.

    Otherwise proxies the underlying kernel transparently
    (``.metal_source``, ``.source_map``, etc.).
    """

    __slots__ = ("_kernel", "_state")

    def __init__(self, kernel, state : _coverage.VerifiedKernelState):
        self._kernel = kernel
        self._state = state

    def bind(self, grid, threadgroup) -> BoundKernel:
        # Ensure we have a trace (and thus stored_slices populated).
        self._kernel._ensure_compiled()
        _coverage.check_coverage(
            self._state,
            tuple(int(x) for x in grid),
            tuple(int(x) for x in threadgroup),
        )
        return self._kernel.bind(grid, threadgroup)

    def __getitem__(self, dispatch_spec):
        # Skip the coverage check when launched via subscript — the
        # subscript form goes through spork's normal _Launcher; users
        # who want coverage should call .bind explicitly. (We can
        # tighten this later if it's noisy.)
        return self._kernel[dispatch_spec]

    @property
    def metal_source(self) -> str:
        return self._kernel.metal_source

    @property
    def source_map(self):
        return self._kernel.source_map

    @property
    def name(self) -> str:
        return self._kernel.name


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
            ...
            out.assign(result)   # per-tile verification
            # → .bind(grid=..., tg=...) verifies grid covers the output

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

        # Per-parameter typed-spec table.
        typed_params : dict = {}
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

        # Identify grid-position attribute params and figure out which
        # vector component → which grid axis & kind. For
        # ``bid : Uint2[ThreadgroupPositionInGrid]``, vector
        # components (x, y, z) map to grid axes (0, 1, 2) with kind
        # "tgid". For ``i : Uint[ThreadPositionInGrid]`` (scalar), the
        # single value maps to axis 0 with kind "gid" (per-thread range).
        grid_param_components : dict[str, tuple] = {}
        for name, param in params:
            ann = param.annotation
            if isinstance(ann, ScalarParamSpec) and ann.attribute is not None:
                kind = None
                for attr_cls, k in _GRID_ATTR_KINDS.items():
                    if issubclass(ann.attribute, attr_cls):
                        kind = k
                        break
                if kind is None:
                    continue
                if ann.vec_size > 1:
                    components = ["x", "y", "z", "w"][:ann.vec_size]
                else:
                    components = [None]
                grid_param_components[name] = (components, kind)

        # State that lives across the trace and feeds .bind's coverage
        # check. Populated by the wrapper below + `record_store` calls
        # during `coop.store` / `.assign`.
        state = _coverage.VerifiedKernelState(
            output_ptr_name=output_name,
            output_shape=out_spec.st,
        )

        @functools.wraps(fn)
        def inner(*tracer_args):
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
                    typed_vec = TypedVectorTracer(arg, var_name_prefix=name)
                    if name in grid_param_components:
                        components, kind = grid_param_components[name]
                        for axis, comp in enumerate(components):
                            if comp is None:
                                continue
                            sym = typed_vec._sym_components[comp]
                            state.pid_axes[sym.name] = _coverage.GridAxisInfo(
                                axis=axis, kind=kind,
                            )
                    wrapped.append(typed_vec)
                elif isinstance(arg, _spork_tracer.Tracer):
                    sym = SymbolicInt(name=f"_{name}")
                    if name in grid_param_components:
                        _components, kind = grid_param_components[name]
                        state.pid_axes[sym.name] = _coverage.GridAxisInfo(
                            axis=0, kind=kind,
                        )
                    wrapped.append(TypedScalarTracer(tracer=arg, sym=sym))
                else:
                    wrapped.append(arg)
            token = _coverage._active_state.set(state)
            try:
                return fn(*wrapped)
            finally:
                _coverage._active_state.reset(token)

        inner.__signature__ = sig.replace(parameters=[
            param.replace(annotation=_untyped_pointer_spec(typed_params[name]))
            if name in typed_params else param
            for name, param in params
        ])

        jitted = _spork_jit(inner)
        return VerifiedJittedKernel(jitted, state)

    return decorator

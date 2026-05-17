import inspect
from typing import Callable, Optional

from . import ir
from . import runtime
from . import tracer as _tracer
from .codegen import emit_kernel
from .tracer import KernelBuilder, PointerTracer, Tracer
from .types import DevicePointerSpec, ScalarParamSpec


class JittedKernel:
    """
    A function decorated with ``@sk.jit``.

    Calling ``kernel[grid, threadgroup](*args)`` traces (on first use),
    compiles, and dispatches the kernel.
    """

    def __init__(self, fn : Callable, name : Optional[str] = None):
        self._fn = fn
        self.name : str = name or fn.__name__
        self._sig = inspect.signature(fn)
        self._builder : Optional[KernelBuilder] = None
        self._source : Optional[str] = None
        self._pipeline = None

    def _trace(self) -> KernelBuilder:
        builder = KernelBuilder(self.name)
        tracer_args = []
        for param_name, param in self._sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"Kernel '{self.name}': *args/**kwargs not supported"
                )
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                raise TypeError(
                    f"Kernel '{self.name}': parameter '{param_name}' is missing a type annotation"
                )
            if isinstance(ann, DevicePointerSpec):
                builder.params.append(ir.Param(
                    name=param_name,
                    kind="pointer",
                    dtype=ann.dtype,
                ))
                tracer_args.append(PointerTracer(ir.Var(param_name), ann.dtype, builder))
            elif isinstance(ann, ScalarParamSpec):
                kind = "attribute" if ann.attribute is not None else "constant"
                builder.params.append(ir.Param(
                    name=param_name,
                    kind=kind,
                    dtype=ann.dtype,
                    metal_name=ann.metal_name,
                    attribute=ann.attribute,
                ))
                tracer_args.append(Tracer(ir.Var(param_name), ann.dtype))
            else:
                raise TypeError(
                    f"Kernel '{self.name}': parameter '{param_name}' has unsupported "
                    f"annotation {ann!r}. Expected sk.DevicePointer[...] or sk.Uint[...] etc."
                )

        token = _tracer._builder.set(builder)
        try:
            self._fn(*tracer_args)
        finally:
            _tracer._builder.reset(token)

        return builder

    def _ensure_compiled(self):
        if self._pipeline is not None:
            return
        self._builder = self._trace()
        self._source = emit_kernel(self._builder)
        library = runtime.compile_source(self._source)
        self._pipeline = runtime.make_pipeline(library, self.name)

    def __getitem__(self, dispatch_spec) -> "_Launcher":
        if not (isinstance(dispatch_spec, tuple) and len(dispatch_spec) == 2):
            raise TypeError(
                "Kernel must be subscripted with (grid_size, threadgroup_size), "
                f"got {dispatch_spec!r}"
            )
        grid_size, tg_size = dispatch_spec
        return _Launcher(self, grid_size, tg_size)

    @property
    def metal_source(self) -> str:
        if self._source is None:
            self._builder = self._trace()
            self._source = emit_kernel(self._builder)
        return self._source


class _Launcher:
    __slots__ = ("_kernel", "_grid_size", "_threadgroup_size")

    def __init__(self, kernel : JittedKernel, grid_size, threadgroup_size):
        self._kernel = kernel
        self._grid_size = tuple(grid_size)
        self._threadgroup_size = tuple(threadgroup_size)

    def __call__(self, *args):
        self._kernel._ensure_compiled()
        runtime.dispatch(
            self._kernel._pipeline,
            self._kernel._builder.params,
            args,
            self._grid_size,
            self._threadgroup_size,
        )


def jit(fn : Callable) -> JittedKernel:
    """
    Decorator that turns a Python function into a Metal kernel.

    The function body is traced once on first launch using the type annotations
    on its parameters; the resulting Metal source is compiled and dispatched.
    """
    return JittedKernel(fn)

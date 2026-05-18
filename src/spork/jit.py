import inspect
from typing import Callable, Optional

from . import ir
from . import runtime
from . import tracer as _tracer
from .codegen import emit_kernel
from .tracer import KernelBuilder, PointerTracer, Tracer, VectorTracer
from .types import DevicePointerSpec, ScalarParamSpec, _ScalarTypeBase


def _annotation_to_spec(ann, kernel_name : str, param_name : str):
    """
    Normalize a parameter annotation into a DevicePointerSpec or ScalarParamSpec.

    A bare scalar class (e.g. ``sk.Uint``) becomes a constant ScalarParamSpec.
    A subscripted form (e.g. ``sk.Uint[ThreadPositionInGrid]``) is already
    a ScalarParamSpec.
    """
    if isinstance(ann, DevicePointerSpec):
        return ann
    if isinstance(ann, ScalarParamSpec):
        return ann
    if inspect.isclass(ann) and issubclass(ann, _ScalarTypeBase):
        return ScalarParamSpec(
            dtype=ann._dtype,
            metal_name=ann._metal_name,
            vec_size=ann._vec_size,
            attribute=None,
        )
    raise TypeError(
        f"Kernel '{kernel_name}': parameter '{param_name}' has unsupported "
        f"annotation {ann!r}. Expected sk.DevicePointer[...] or sk.Uint[...] etc."
    )


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
        self._source_map : Optional[dict] = None
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
            spec = _annotation_to_spec(ann, self.name, param_name)
            if isinstance(spec, DevicePointerSpec):
                builder.params.append(ir.Param(
                    name=param_name,
                    kind="pointer",
                    dtype=spec.dtype,
                ))
                tracer_args.append(PointerTracer(ir.Var(param_name), spec.dtype, builder))
            else:
                kind = "attribute" if spec.attribute is not None else "constant"
                builder.params.append(ir.Param(
                    name=param_name,
                    kind=kind,
                    dtype=spec.dtype,
                    metal_name=spec.metal_name,
                    vec_size=spec.vec_size,
                    attribute=spec.attribute,
                ))
                if spec.vec_size > 1:
                    tracer_args.append(VectorTracer(
                        ir.Var(param_name), spec.dtype, spec.vec_size,
                    ))
                else:
                    tracer_args.append(Tracer(ir.Var(param_name), spec.dtype))

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
        self._source, self._source_map = emit_kernel(self._builder)
        library = runtime.compile_source(self._source, source_map=self._source_map)
        self._pipeline = runtime.make_pipeline(library, self.name)

    def __getitem__(self, dispatch_spec) -> "_Launcher":
        if not (isinstance(dispatch_spec, tuple) and len(dispatch_spec) == 2):
            raise TypeError(
                "Kernel must be subscripted with (grid_size, threadgroup_size), "
                f"got {dispatch_spec!r}"
            )
        grid_size, tg_size = dispatch_spec
        return _Launcher(self, grid_size, tg_size)

    def bind(self, grid, threadgroup) -> "BoundKernel":
        """
        Return a BoundKernel that calls this kernel with ``grid`` and
        ``threadgroup`` already baked in. Eliminates the
        ``kernel[grid, tg](*args)`` boilerplate at each call site.

        Usage::

            bound = my_kernel.bind(grid=(8, 1, 1), threadgroup=(32, 1, 1))
            bound(C, A, B)
        """
        return BoundKernel(self, grid, threadgroup)

    @property
    def metal_source(self) -> str:
        if self._source is None:
            self._builder = self._trace()
            self._source, self._source_map = emit_kernel(self._builder)
        return self._source

    @property
    def source_map(self) -> Optional[dict]:
        """
        Mapping of generated-Metal line number (1-indexed) to a
        ``(python_filename, python_lineno)`` tuple.
        """
        if self._source is None:
            _ = self.metal_source
        return self._source_map


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


class BoundKernel:
    """
    A JittedKernel with its dispatch ``grid`` and ``threadgroup`` already
    bound. Call it directly with the kernel's pointer + constant arguments
    — no need to specify launch parameters at each call site.
    """

    __slots__ = ("_kernel", "_grid", "_threadgroup")

    def __init__(self, kernel : JittedKernel, grid, threadgroup):
        self._kernel = kernel
        self._grid = tuple(int(x) for x in grid)
        self._threadgroup = tuple(int(x) for x in threadgroup)

    def __call__(self, *args):
        self._kernel[self._grid, self._threadgroup](*args)

    @property
    def metal_source(self) -> str:
        return self._kernel.metal_source

    @property
    def source_map(self):
        return self._kernel.source_map

    @property
    def grid(self) -> tuple:
        return self._grid

    @property
    def threadgroup(self) -> tuple:
        return self._threadgroup

    @property
    def name(self) -> str:
        return self._kernel.name

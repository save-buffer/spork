from typing import List, Sequence

import numpy as np
import objc
from Metal import (
    MTL4CompilerDescriptor,
    MTL4LibraryDescriptor,
    MTLCommandBufferStatusError,
    MTLCompileOptions,
    MTLCreateSystemDefaultDevice,
    MTLResourceStorageModeShared,
    MTLSizeMake,
)

from . import dtypes as dt
from . import ir


# PyObjC lacks metadata for MTL4 APIs — register the error: param as an
# output NSError** so we get back (result, error) tuples.
objc.registerMetaDataForSelector(
    b"NSObject",
    b"newLibraryWithDescriptor:error:",
    {"arguments": {3: {"type_modifier": b"o"}}},
)
objc.registerMetaDataForSelector(
    b"NSObject",
    b"newCompilerWithDescriptor:error:",
    {"arguments": {3: {"type_modifier": b"o"}}},
)


_device = None
_command_queue = None


def get_device():
    global _device
    if _device is None:
        _device = MTLCreateSystemDefaultDevice()
        if _device is None:
            raise RuntimeError("Metal is not supported on this device")
    return _device


def get_command_queue():
    global _command_queue
    if _command_queue is None:
        _command_queue = get_device().newCommandQueue()
    return _command_queue


def compile_source(source : str, enable_logging : bool = False):
    device = get_device()

    options = MTLCompileOptions()
    options.setEnableLogging_(enable_logging)
    options.setLanguageVersion_(0x40000)

    compiler_desc = MTL4CompilerDescriptor()
    compiler, error = device.newCompilerWithDescriptor_error_(compiler_desc, None)
    if error:
        raise RuntimeError(f"Failed to create MTL4 compiler: {error.localizedDescription()}")

    lib_desc = MTL4LibraryDescriptor()
    lib_desc.setSource_(source)
    lib_desc.setOptions_(options)

    library, error = compiler.newLibraryWithDescriptor_error_(lib_desc, None)
    if error:
        raise RuntimeError(
            f"Failed to compile kernel:\n{error.localizedDescription()}\n\nSource:\n{source}"
        )
    return library


def make_pipeline(library, kernel_name : str):
    device = get_device()
    fn = library.newFunctionWithName_(kernel_name)
    if not fn:
        raise RuntimeError(f"Kernel '{kernel_name}' not found in library")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
    if error:
        raise RuntimeError(f"Pipeline creation failed: {error.localizedDescription()}")
    return pipeline


def _expect_pointer_arg(arr, param : ir.Param) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"Argument for pointer parameter '{param.name}' must be a numpy array, "
            f"got {type(arr).__name__}"
        )
    expected = param.dtype
    actual = dt.from_numpy(arr.dtype)
    if actual != expected:
        raise TypeError(
            f"Argument for pointer parameter '{param.name}' has dtype {actual.name}, "
            f"expected {expected.name}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"Argument for pointer parameter '{param.name}' must be C-contiguous"
        )
    return arr


def _numpy_dtype_for(d : dt.Dtype):
    mapping = {
        dt.float32 : np.float32,
        dt.float16 : np.float16,
        dt.int32   : np.int32,
        dt.uint32  : np.uint32,
        dt.int64   : np.int64,
        dt.uint64  : np.uint64,
        dt.bool_   : np.bool_,
    }
    if d not in mapping:
        raise TypeError(f"No numpy dtype for spork dtype {d.name}")
    return mapping[d]


def dispatch(
    pipeline,
    params : Sequence[ir.Param],
    user_args : Sequence,
    grid_size : tuple,
    threadgroup_size : tuple,
) -> None:
    device = get_device()
    command_queue = get_command_queue()

    bindable = [p for p in params if p.kind in ("pointer", "constant")]
    if len(user_args) != len(bindable):
        raise TypeError(
            f"Kernel expected {len(bindable)} arguments "
            f"(pointer + constant params), got {len(user_args)}"
        )

    bindings : List[tuple] = []
    buffer_idx = 0
    arg_iter = iter(user_args)
    for p in params:
        if p.kind == "pointer":
            arr = _expect_pointer_arg(next(arg_iter), p)
            buf = device.newBufferWithBytes_length_options_(
                arr.tobytes(),
                arr.nbytes,
                MTLResourceStorageModeShared,
            )
            if not buf:
                raise RuntimeError(
                    f"Failed to allocate buffer for '{p.name}' ({arr.nbytes} bytes)"
                )
            bindings.append((buffer_idx, buf, arr, p))
            buffer_idx += 1
        elif p.kind == "constant":
            value = next(arg_iter)
            np_dtype = np.dtype(_numpy_dtype_for(p.dtype))
            scalar = np.asarray(value).astype(np_dtype, copy=False)
            buf = device.newBufferWithBytes_length_options_(
                scalar.tobytes(),
                scalar.nbytes,
                MTLResourceStorageModeShared,
            )
            bindings.append((buffer_idx, buf, None, p))
            buffer_idx += 1

    cmd_buf = command_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for idx, buf, _arr, _p in bindings:
        encoder.setBuffer_offset_atIndex_(buf, 0, idx)

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        MTLSizeMake(*(int(x) for x in grid_size)),
        MTLSizeMake(*(int(x) for x in threadgroup_size)),
    )
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
    if cmd_buf.status() == MTLCommandBufferStatusError:
        raise RuntimeError(f"Kernel command returned error: {cmd_buf.error()}")

    for _idx, buf, arr, p in bindings:
        if arr is None:
            continue
        if not p.written:
            continue
        raw = bytes(buf.contents().as_buffer(arr.nbytes))
        out = np.frombuffer(raw, dtype=arr.dtype, count=arr.size).reshape(arr.shape)
        arr[...] = out

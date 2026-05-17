from dataclasses import dataclass
from typing import Optional, Type

from . import dtypes as dt


class ThreadAttribute:
    """
    Marker for a Metal kernel attribute that the runtime fills in.

    Subclasses set ``metal_attr`` to the Metal attribute spelling
    (e.g. ``thread_position_in_grid``).
    """

    metal_attr : str = ""


class ThreadPositionInGrid(ThreadAttribute):
    metal_attr = "thread_position_in_grid"


class ThreadPositionInThreadgroup(ThreadAttribute):
    metal_attr = "thread_position_in_threadgroup"


class ThreadgroupPositionInGrid(ThreadAttribute):
    metal_attr = "threadgroup_position_in_grid"


class ThreadsPerThreadgroup(ThreadAttribute):
    metal_attr = "threads_per_threadgroup"


class ThreadsPerGrid(ThreadAttribute):
    metal_attr = "threads_per_grid"


class ThreadsPerSimdgroup(ThreadAttribute):
    metal_attr = "threads_per_simdgroup"


class ThreadIndexInSimdgroup(ThreadAttribute):
    metal_attr = "thread_index_in_simdgroup"


class SimdgroupIndexInThreadgroup(ThreadAttribute):
    metal_attr = "simdgroup_index_in_threadgroup"


@dataclass(frozen=True)
class DevicePointerSpec:
    """
    A parameter spec: ``device <dtype> *name [[buffer(i)]]``.
    """

    dtype : dt.Dtype


@dataclass(frozen=True)
class ScalarParamSpec:
    """
    A scalar parameter spec.

    If ``attribute`` is set, the parameter is filled by the Metal runtime
    (e.g. thread_position_in_grid). Otherwise it is a ``constant T &`` buffer
    sourced from a Python value at launch time.
    """

    dtype      : dt.Dtype
    metal_name : str
    attribute  : Optional[Type[ThreadAttribute]] = None


class _DevicePointer:
    def __class_getitem__(cls, dtype : dt.Dtype) -> DevicePointerSpec:
        if not isinstance(dtype, dt.Dtype):
            raise TypeError(
                f"DevicePointer[...] expects a spork dtype, got {dtype!r}"
            )
        return DevicePointerSpec(dtype=dtype)


DevicePointer = _DevicePointer


def _scalar_factory(dtype : dt.Dtype, metal_name : str):
    class _ScalarType:
        _dtype      = dtype
        _metal_name = metal_name

        def __class_getitem__(cls, attribute):
            if not (isinstance(attribute, type) and issubclass(attribute, ThreadAttribute)):
                raise TypeError(
                    f"{metal_name}[...] expects a ThreadAttribute subclass, "
                    f"got {attribute!r}"
                )
            return ScalarParamSpec(
                dtype=cls._dtype,
                metal_name=cls._metal_name,
                attribute=attribute,
            )

    _ScalarType.__name__ = metal_name.capitalize()
    return _ScalarType


Uint  = _scalar_factory(dt.uint32, "uint")
Uint2 = _scalar_factory(dt.uint32, "uint2")
Uint3 = _scalar_factory(dt.uint32, "uint3")
Int   = _scalar_factory(dt.int32, "int")
Int2  = _scalar_factory(dt.int32, "int2")
Int3  = _scalar_factory(dt.int32, "int3")

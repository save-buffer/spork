# Spork: A Tracing DSL for Apple GPUs

Spork is a DSL for writing kernels for Apple GPUs. You can think of it as a Python wrapper
on top of Metal, Apple's GPU language. This makes development more convenient, since it makes
the kernels live in the same source language as most ML code, and makes it easy to verify 
correctness against NumPy. 

Unlike Triton, Spork is a tracing DSL rather than a parsing-based DSL. This means that a Spork
program is fundamentally a Python program that creates a Metal program. Spork can use any Python
libraries to aid in metaprogramming. 

## Example: Matrix Addition
One of the most basic kernels is a matrix-addition kernel. In Numpy, you could write
```python
shape = (1024, 1024)
A = np.random.randn(*shape).astype(np.float32)
B = np.random.randn(*shape).astype(np.float32)
out = A + B
```

A Metal kernel to perform this operation would look like this:
```cpp
#include <metal_stdlib>
using namespace metal;

kernel void matrix_add(
    device float *out [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    out[index] = A[index] + B[index];
}
```

Notice that it exists in a separate source file and requires special code to compile, link to,
and invoke. An equivalent Spork kernel looks like this:
```python
@sk.jit
def matrix_add(
    out   : sk.DevicePointer[sk.dt.float32],
    A     : sk.DevicePointer[sk.dt.float32],
    B     : sk.DevicePointer[sk.dt.float32],
    index : sk.Uint[sk.ThreadPositionInGrid],
):
    out[index] = A[index] + B[index]
```

Notice that we have a direct correspondance here between the Spork kernel and the Metal kernel. We
declare the types of our inputs, and we can take the attribute parameters that we take in Metal. 


Then to actually invoke and run kernel, we simply call it from Python with Numpy arrays, and 
like magic it runs your kernel!

```python
matrix_add[
    (int(np.prod(shape)) // 128, 1, 1),
    (128, 1, 1),
](
    C,
    A,
    B,
)
```
Notice that in the brackets, we provide two parameters. The first is the Grid size, and the second
is the Warpgroup size. In the parentheses, we supply the Numpy tensors we wish to use. Notice
that here we're taking a pointer to `C`, which is allocated via Numpy but written to from the 
Spork kernel.


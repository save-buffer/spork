
# Changelog

All notable changes to spork-metal are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- **`spork.verified`** subpackage scaffold + **`[verified]` extra**. Installs
  ``stile-verifier`` via ``pip install 'spork-metal[verified]'``; importing
  ``spork.verified`` without that extra raises an ImportError pointing at the
  right install command. The subpackage will host the Stile-typed primitive
  wrappers and spec-verified analogs of ``spork.kernels`` (matmul, causal_gqa,
  …) in follow-up changes.

### Added

- **`spork.kernels` subpackage** — a library of pre-built kernels that take
  the relevant compile-time dimensions/parameters, generate a specialized
  kernel, compute the launch geometry, and return a callable with the
  ``grid`` + ``threadgroup`` already bound. No more
  ``kernel[grid, tg](*args)`` boilerplate at call sites.
  - ``sk.kernels.matmul(M, N, K, dtype=dt.float32)`` — MPP ``matmul2d``
    cooperative-tensor matmul with Z-order tile dispatch (TM=TN=64, TK=128;
    requires M%64==N%64==K%128==0 and (M/64, N/64) powers of two for now;
    clear `ValueError` otherwise).
  - ``sk.kernels.matrix_add(shape, dtype=dt.float32)`` — elementwise add,
    picks the largest power-of-two threadgroup size that divides
    ``prod(shape)``.
- **`JittedKernel.bind(grid, threadgroup)`** returns a new `BoundKernel` —
  a callable that wraps the kernel with the dispatch geometry baked in.
  Forwards ``metal_source``, ``source_map``, ``grid``, ``threadgroup``,
  ``name`` for introspection.
- **`sk.BoundKernel`** is now part of the public surface.
- **`sk.kernels.matmul(..., traversal=...)`** keyword argument. Accepts
  ``"zorder"`` (default; Morton bit-interleave) or ``"linear"`` (row-major,
  ``tid % m_tiles`` / ``tid / m_tiles``). Linear has no bit-twiddle overhead
  and lifts the power-of-two constraint on tile counts; Z-order trades a
  few instructions per threadgroup for better L2 locality on large
  matrices.
- **`bench/matmul.py`** dev script — runs ``sk.kernels.matmul`` once,
  checks correctness against numpy, and opens a ``.gputrace`` in Xcode for
  perf inspection. Accepts dimensions and an optional traversal name.
- **`sk.kernels.matmul_oneshot(M, N, K, dtype=dt.float32)`** — single-call
  MPP matmul2d per output tile, no Python-side K-loop and no custom
  traversal (the descriptor's K equals the full problem K). Useful as the
  simplest possible MPP baseline for perf comparisons; limited to small K
  by MPP's per-tile resource budget.
- **`bench/matmul_oneshot.py`** dev script — same shape as
  ``bench/matmul.py`` but for ``sk.kernels.matmul_oneshot``.
- **`sk.kernels.causal_gqa(nq, nkv, qctx, nctx, dhead, dtype, *, block_m,
  block_n)`** — fused causal Grouped-Query Attention in FlashAttention-2
  style. Uses MPP ``matmul2d`` cooperative tensors for both Q@K^T and P@V;
  runs the online-softmax pass (causal mask, row max/sum tracking,
  rescaling of the running output) on threadgroup-memory scratch between
  the two matmuls. Each threadgroup processes one ``(q_head, q_tile)``
  pair. First version's softmax runs on a single thread per threadgroup
  for clarity — perf-tuning is a follow-up.

### Changed

- **`sk.tensor(...)`** now accepts a ``sk.threadgroup(...)`` array (not
  just a device-pointer parameter) so MPP cooperative tensors can be
  stored to / sliced over threadgroup-memory scratch. Required for fused
  attention-style kernels that need to inspect MPP results between two
  matmuls.


## [0.4.0] — 2026-05-17

### Added

- **Source-mapped compile errors.** Every emitted IR statement now records
  the Python `(filename, lineno)` of the call site that produced it (via a
  stack-walk that skips spork's own frames). Codegen builds a per-line source
  map alongside the generated Metal source, and `runtime.compile_source`
  uses it to prepend a header to Metal compile errors that translates each
  referenced `program_source:LINE` to the originating Python location.
- **`JittedKernel.source_map`** property — exposes the mapping for
  introspection (line in generated Metal → `(python_filename, python_lineno)`).

### Changed

- `codegen.emit_kernel` now returns `(source, source_map)` instead of just
  `source`. Internal API only; the user-facing `JittedKernel.metal_source`
  property is unchanged.
- `runtime.compile_source` accepts an optional `source_map` keyword argument
  used to rewrite compile errors.


## [0.3.0] — 2026-05-16

### Added

- **`@sk.device_fn`** decorator for reusable device-side functions. Parameters
  annotated with `sk.DevicePointer[dtype]` or a `dt.Dtype`; return type taken
  from the `->` annotation (omitted means `void`). Traced lazily on first
  call, emitted as a free function above the kernel, deduplicated across call
  sites, and transitively pulls in any device fns it calls.
- **`sk.while_(cond)`** context manager — emits a `while (cond) { ... }` loop;
  the condition is re-evaluated each iteration.
- **`sk.break_()`** and **`sk.continue_()`** — emit `break;` / `continue;` for
  loop control flow.
- **`with sk.if_(cond) as branch: ... ; with branch.else_(): ...`** — adds an
  else branch to an existing if. Must immediately follow its matching `with
  sk.if_` block.

### IR

- New `ir.Return`, `ir.WhileLoop`, `ir.Break`, `ir.Continue` nodes.

### Internal

- `KernelBuilder` gained a `device_functions` list and an `add_device_fn`
  method that handles lazy tracing, topo-sorted dependency insertion, and
  merging of the device fn's `#include`s and `using` directives back into the
  kernel's.


## [0.2.0] — 2026-05-16

### Added

- **Math intrinsics**: `sk.exp`, `sk.exp2`, `sk.log`, `sk.log2`, `sk.log10`,
  `sk.sqrt`, `sk.rsqrt`, `sk.sin`, `sk.cos`, `sk.tan`, `sk.asin`, `sk.acos`,
  `sk.atan`, `sk.atan2`, `sk.sinh`, `sk.cosh`, `sk.tanh`, `sk.floor`,
  `sk.ceil`, `sk.round`, `sk.trunc`, `sk.fabs`, `sk.abs`, `sk.sign`, `sk.pow`,
  `sk.fmod`, `sk.fmin`, `sk.fmax`, `sk.min`, `sk.max`, `sk.clamp`, `sk.fma`.
- **Fast math variants**: `sk.fast_exp`, `sk.fast_log`, `sk.fast_log2`,
  `sk.fast_sqrt`, `sk.fast_rsqrt`, `sk.fast_sin`, `sk.fast_cos`, `sk.fast_tan`,
  `sk.fast_pow` (lower-precision, hardware-accelerated variants from
  `metal::fast::`).
- **`sk.cast(value, dtype)`**: emits `static_cast<dtype>(value)`.
- **Atomic dtypes**: `dt.atomic_uint32`, `dt.atomic_int32`, `dt.atomic_float32`.
  A `sk.DevicePointer[dt.atomic_uint32]` parameter declares
  `device atomic_uint *name` in the generated kernel; the runtime accepts the
  underlying numpy dtype (`np.uint32` for `atomic_uint32`, etc.) since the
  memory layout matches.
- **Atomic operations**: `sk.atomic_load`, `sk.atomic_store`,
  `sk.atomic_fetch_add`, `sk.atomic_fetch_sub`, `sk.atomic_fetch_and`,
  `sk.atomic_fetch_or`, `sk.atomic_fetch_xor`, `sk.atomic_fetch_min`,
  `sk.atomic_fetch_max`, `sk.atomic_exchange`. All RMW ops materialize the
  prior-value return into a local so the side effect emits even if the result
  is ignored, and mark the underlying pointer parameter as written so the
  runtime copies the buffer back after dispatch.
- **`sk.profile(name=..., open_in_xcode=True)`**: context manager that wraps
  GPU work in an `MTLCaptureManager` session, writes a `.gputrace` document,
  and optionally opens it in Xcode for performance analysis. Requires
  `MTL_CAPTURE_ENABLED=1` in the environment.

### IR

- New `ir.AddrOf(expr)` node, emitted as `&<expr>`, used by atomics to take the
  address of an indexed element.

### Internal

- `runtime._expect_pointer_arg` now validates against the *underlying* dtype of
  the param (via `dt.underlying`), so atomic-typed parameters accept the
  matching non-atomic numpy dtype.


## [0.1.1] — 2026-05-16

### Added

- Install instructions in the README (`pip install spork-metal`,
  `uv add spork-metal`).
- `.github/workflows/publish.yml` — GitHub Actions workflow that builds and
  publishes to PyPI on any `v*` tag, using Trusted Publishing (OIDC, no
  long-lived tokens). Includes a tag/`pyproject.toml`-version sanity check
  before publishing.


## [0.1.0] — 2026-05-16

### Added

- **`@sk.jit`** decorator that traces a Python function into Metal source on
  first call and dispatches it via PyObjC/MTL4.
- **Launch syntax**: `kernel[grid_size, threadgroup_size](*args)`. Arguments
  correspond 1:1 with `DevicePointer` and constant scalar params; attribute
  params are filled by the Metal runtime.
- **Pointer parameters**: `sk.DevicePointer[sk.dt.<dtype>]` → `device T *name`.
  Numpy arrays passed as arguments are written back into in place after
  dispatch if the kernel mutated them.
- **Scalar / vector parameters**: `sk.Uint`, `sk.Uint2`, `sk.Uint3`,
  `sk.Int`, `sk.Int2`, `sk.Int3`. Bare form (no subscript) declares a
  `constant T &` parameter sourced from a Python value at launch;
  subscripted form (`sk.Uint[ThreadPositionInGrid]`) declares an attribute
  parameter filled by Metal.
- **Thread attribute markers**: `ThreadPositionInGrid`,
  `ThreadPositionInThreadgroup`, `ThreadgroupPositionInGrid`,
  `ThreadsPerThreadgroup`, `ThreadsPerGrid`, `ThreadsPerSimdgroup`,
  `ThreadIndexInSimdgroup`, `SimdgroupIndexInThreadgroup`.
- **Dtypes**: `dt.float32`, `dt.float16`, `dt.bfloat16`, `dt.int32`,
  `dt.uint32`, `dt.int64`, `dt.uint64`, `dt.bool_`.
- **Local variables**: `sk.local(dtype, init)` declares a mutable local that
  supports compound assignment (`+=`, `-=`, `*=`, `/=`, etc.) and
  `.assign(value)`.
- **Control flow**:
  - `sk.range(end)` / `sk.range(start, end)` / `sk.range(start, end, step)`
    generator-based for-loops that materialize as `for (uint i = ...; ... ; ...)`
    in Allman braces.
  - `sk.if_(cond)` context manager for `if (cond) { ... }` blocks.
- **Threadgroup memory**: `sk.threadgroup(dtype, shape)` declares
  `threadgroup T name[D0][D1]...;` and returns a handle that supports tuple
  subscripts (`a[i, j]` desugars to `a[i][j]`).
- **Barriers**: `sk.threadgroup_barrier(*scopes)` and
  `sk.simdgroup_barrier(*scopes)`. Scope strings combine into
  `mem_flags::mem_<scope> | ...`. Defaults to `threadgroup`.
- **Simd intrinsics**:
  - Reductions: `simd_sum`, `simd_product`, `simd_max`, `simd_min`, `simd_and`,
    `simd_or`, `simd_xor`, `simd_all`, `simd_any`.
  - Prefix scans: `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`,
    `simd_prefix_inclusive_product`, `simd_prefix_exclusive_product`.
  - Movement: `simd_broadcast`, `simd_shuffle`, `simd_shuffle_up`,
    `simd_shuffle_down`, `simd_shuffle_xor`.
  - All materialize into a local at the call site so that subsequent uses
    don't accidentally place the collective in divergent control flow.
- **MetalPerformancePrimitives**:
  - `sk.tensor(ptr, dtype, shape)` wraps a device pointer as
    `mpp::tensor_ops::tensor` with compile-time `extents`.
  - `tensor_handle.slice((W, H), (x, y))` produces an opaque tile.
  - `sk.matmul2d(M, N, K, *, simdgroups=4, transpose_a=False,
    transpose_b=False, transpose_c=False, mode="multiply_accumulate")`
    declares a `matmul2d<desc, execution_simdgroups<S>>` op.
  - `op.get_destination(tA, tB, dtype)` allocates the cooperative tensor
    accumulator; `op.run(tile_a, tile_b, coop)` executes;
    `coop.store(tile_c)` writes back.
  - Provenance tracking: `coop.store(tile)` flags the source pointer
    parameter as written so the runtime copies the buffer back after
    dispatch.
- **`kernel.metal_source`** property for inspecting the generated Metal source
  without dispatching.
- **Generated source style**: Allman braces throughout, precedence-aware
  parenthesization, `f` suffix on float literals, and `u`/`ull` suffixes on
  large positive integer literals so bitmask constants past 2³¹ keep their
  unsigned type.

[Unreleased]: https://github.com/save-buffer/spork/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/save-buffer/spork/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/save-buffer/spork/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/save-buffer/spork/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/save-buffer/spork/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/save-buffer/spork/releases/tag/v0.1.0

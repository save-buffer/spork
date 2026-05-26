
# Changelog

All notable changes to spork-metal are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- **`spork.verified`** subpackage scaffold + **`[verified]` extra**. Installs
  ``stile-verifier`` via ``pip install 'spork-metal[verified]'``; importing
  ``spork.verified`` without that extra raises an ImportError pointing at the
  right install command.
- **`skv.dim`, `skv.OutputSpec`, `skv.DevicePointer[dtype, shape]`** —
  typed kernel parameters. Shapes are tuples of stile dims declared via
  ``skv.dim(name, size)``; the spec language is shared with stile.
- **`@skv.jit(out_spec=...)`** — verified-kernel decorator that wraps
  ``@sk.jit``. At trace time it walks the kernel signature, swaps each
  typed ``DevicePointer`` annotation for the underlying spork
  ``DevicePointerSpec``, and presents the kernel body with
  ``TypedTensorHandle`` wrappers carrying stile ``Type`` info. Untyped
  params (thread-position attributes) pass through unchanged.
- **`skv.tensor`, `TypedTensorHandle`** — typed analog of
  ``sk.tensor`` / ``TensorHandle``. ``.assign(value)`` runs
  ``verify_types_equivalent`` (per-tile, fires at trace time, no
  launch-param awareness — coverage tracking across grid is the
  planned follow-up).
- **`TypedTensorHandle.slice((shape), (offsets))`**, **`TypedTileSlice`** —
  typed slicing. The user expresses shapes/offsets in math order
  (rows, cols, ...); the backend reverses to MPP's memory order
  (inner first) at the spork boundary. Python-int offsets refine the
  stile ShapeType via ``Type.slice``; symbolic (Tracer) offsets pass
  through unrefined for now (LoopVariable bindings are the next step).
- **`skv.matmul2d`**, **`TypedMatmulOp`**, **`TypedCooperativeTensor`** —
  typed MPP matmul. ``op.run`` accumulates an ``einsum`` contribution
  into the coop's stile ``Type``; ``coop.store(out_tile)`` runs
  ``verify_types_equivalent`` against the output tile's spec. The
  einsum string is inferred from the operands' stile dim names.

First end-to-end verified kernel works: a 64×64×64 matmul that the
verifier accepts before any GPU dispatch, runs correctly, and rejects
a deliberately-wrong variant (``A @ A`` instead of ``A @ B``) with a
clear ``does not match spec`` error.

- **`TypedScalarTracer`** / **`TypedVectorTracer`** — typed analogs of
  spork's scalar and vector tracers. Each carries the spork ``Tracer``
  AND a stile ``SymbolicInt`` / ``AffineExpr``. ``@skv.jit`` auto-wraps
  attribute params (e.g. ``Uint2[ThreadgroupPositionInGrid]``) so
  ``bid.x``, ``bid.x * TILE``, ``bid.x * TILE + j`` all flow through
  the verifier as symbolic affine expressions.
- **`_slice_type`** now refines stile ``Sliced`` dims when slice
  offsets are ``TypedScalarTracer``s — so per-tile verification on
  tiled kernels sees the right symbolic slice (rather than passing
  through unrefined).
- **Tiled verified matmul works**: 128×128×64 with 4 threadgroups,
  each handling a 64×64 output tile via ``A.slice((TM, K), (bid.y * TM,
  0))`` etc. Verifier accepts the symbolic-offset tiles, dispatch
  produces correct output.
- **Required dep bump**: ``stile-verifier>=0.1.3``. The
  ``spork.verified`` test suite now wraps each test in ``with
  stile.scope():`` (new stile-0.1.3 API) so dim/tensor registries don't
  collide across tests.

- **`skv.maximum(a, b)`** — typed binary max on ``TypedScalarValue``;
  ExprType wrapped as ``BinaryOp("max", a_et, b_et)``.
- **`skv.with_type(handle, spec_str, st, dtype=None)`** — "trust-me"
  escape hatch that overrides a handle's stile ``Type`` by parsing a
  spec string. Used when part of a computation is opaque to the
  verifier (e.g. a threadgroup-memory softmax built from raw loops);
  the user asserts the result's symbolic type. Downstream typed ops
  compose against the assumed type.
- **`spork.verified.kernels.attention(qctx, nctx, dhead)`** — verified
  non-causal attention against the canonical stile spec
  ``(softmax[nctx]((qctx dhead, nctx dhead -> qctx nctx) / sqrt(D)),
  nctx dhead -> qctx dhead)``. Uses MPP ``matmul2d`` for Q@K^T and P@V
  (verified end-to-end), with a thread-0 softmax on threadgroup-memory
  scratch (currently trust-me'd via ``with_type``; tile-level typed
  reductions to verify the softmax body itself are a follow-up).

- **`skv.if_(predicate)`** + ``TypedPredicate``. ``TypedScalarTracer
  == int`` (only for bare grid-position / loop-var SymbolicInts —
  not derived AffineExprs) produces a ``TypedPredicate`` carrying both
  the spork bool tracer and a stile constraint
  ``{SymbolicInt name → int value}``. ``skv.if_(predicate)`` is a
  context manager that pushes the constraint onto
  ``_active_if_constraints`` and emits the underlying ``sk.if_``;
  stores recorded inside snapshot the constraint into their
  ``StoredSlice.constraints``. Coverage enumeration restricts the
  named SymbolicInt to the constraint value for those stores,
  enabling patterns like ``with skv.if_(tid.x == 0):`` (thread-0-only
  writeback) where coverage would otherwise treat every thread as
  writing.
- **``ThreadPositionInThreadgroup``** now registered as a ``tid``-kind
  grid axis (range = ``threadgroup[axis]``).
  locals. ``TypedLocal`` subclasses ``TypedScalarValue`` (inherits
  reads + arithmetic) and adds ``.assign(value)``, ``+=``, ``-=``,
  ``*=``, ``/=``. Underlying spork ``Local`` is what actually mutates;
  the typed wrapper tracks the current value's ExprType for downstream
  reads.
- **`skv.threadgroup(dtype, shape)`** + ``TypedThreadgroupArray`` —
  typed wrapper around ``sk.threadgroup``. Reads return
  ``TypedScalarValue`` whose ExprType is a Tensor leaf with the
  array's declared dim tuple (so verification through intermediate
  scratch sees a sensible expression). Writes pass through. Compatible
  with ``skv.tensor(...)`` wrapping for MPP-cooperative-tensor stores
  into threadgroup scratch.

- **`spork.verified.kernels.matmul(M, N, K, dtype=float32)`** — the
  canonical verified MPP matmul, tile-walking with K-loop accumulation.
  Each call enters its own ``stile.scope()`` so dim declarations don't
  collide across calls. Per-tile verification (ParametricReduce over
  K-tiles folds into the spec's full-K reduction) + grid coverage.
- **Element-level typed primitives**:
  - ``TypedScalarValue`` — a typed scalar value (distinct from
    ``TypedScalarTracer`` which is for slice offsets). Carries a stile
    ``Type`` with shape ``()`` and an ``ExprType``; supports
    ``+``/``-``/``*``/``/``/``-``.
  - ``TypedTensorHandle.__getitem__((i, j, ...))`` returns
    ``TypedScalarValue`` reading from the underlying device pointer
    (math-order indices flatten to row-major).
  - ``TypedTensorHandle.__setitem__((i, j, ...), value)`` writes via
    the underlying device pointer AND records a per-element store
    (Sliced(dim, idx, idx+1) per axis) into ``stored_slices`` so the
    bind-time coverage check verifies every element is written exactly
    once.
  - ``skv.exp``, ``skv.sqrt``, ``skv.sin``, ``skv.cos`` — typed math
    intrinsics on ``TypedScalarValue``; ExprType wrapped as
    ``UnaryOp("exp"/...)``.
- **Coverage tracking now also handles ``ThreadPositionInGrid``** (not
  just ``ThreadgroupPositionInGrid``). Each grid-position SymbolicInt
  is registered with a ``GridAxisInfo(axis, kind)``; ``kind="tgid"``
  enumerates over ``grid[axis]``, ``kind="gid"`` enumerates over
  ``grid[axis] * threadgroup[axis]``. The coverage check now takes
  both ``grid`` and ``threadgroup`` from ``.bind()``.

- **ParametricReduce on accumulating cooperative tensors**.
  ``TypedMatmulOp.run`` now snapshots its ``coop`` arg into every
  active ``skv.range`` frame; on loop exit, the wrapper walks the
  snapshots and replaces each touched coop's type with
  ``snap + ParametricReduce(sym, lo, hi, "sum", delta)``. Stile's
  normalize folds the ``Constant(0) + ParametricReduce(...)`` case to
  the bare reduction, so a K-tile accumulating matmul verifies against
  the full-K spec without any user-side annotation.
- **End-to-end verified K-loop matmul** (in tests): each threadgroup
  computes one ``(TM, TN)`` output tile by accumulating ``op.run`` over
  K-chunks via ``skv.range``. Verifier accepts the per-tile
  ``ParametricReduce`` as equivalent to the spec's full-K reduction,
  coverage check accepts the grid, dispatch matches numpy.

- **Runtime loops + coverage**. ``skv.range(start, end, step)`` is the
  verified analog of ``sk.range`` — emits a Metal ``for`` and yields a
  ``TypedScalarTracer`` whose ``SymbolicInt`` is registered in
  ``VerifiedKernelState.loop_var_ranges`` so the bind-time coverage
  check enumerates over the loop's iteration range. Distinct from
  Python's builtin ``range``: that one is the static / compile-time
  loop (each iteration's body is traced separately, stores get
  concrete-int offsets). Both are now supported; pick the right one for
  the kernel's intent.
- ``_coverage._enumerate_substitutions`` now takes the full
  ``VerifiedKernelState`` and walks both ``pid_axes`` (grid indices)
  and ``loop_var_ranges`` (runtime-loop indices).

- **Bind-time coverage check**. ``@skv.jit`` now returns a
  ``VerifiedJittedKernel`` that records every store into the output
  during trace (with the destination's symbolic ``Sliced`` shape) and
  registers each grid-position SymbolicInt against its grid axis.
  Calling ``.bind(grid, threadgroup)`` substitutes those symbolic
  bounds over the grid range, unions per-axis intervals, and raises
  ``ValueError`` if the union doesn't equal each declared output dim's
  full extent. Catches:
    - undersized grid (positions left unwritten),
    - oversized grid (overlapping or out-of-bounds writes),
    - off-by-one constant offsets,
  all before any GPU work is dispatched. Mirrors
  ``stile.triton._core._check_coverage_at_launch``.

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

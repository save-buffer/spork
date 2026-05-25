"""
Bind-time coverage tracking + check for verified spork kernels.

Each verified kernel records, during trace, every store into its declared
output tensor as a tuple of ``Sliced`` dims (the output tile's ShapeType).
It also records which stile ``SymbolicInt`` names correspond to which
grid axis (e.g. ``_bid_x → 0`` for ``bid : Uint2[ThreadgroupPositionInGrid]``).

At ``.bind(grid=..., threadgroup=...)`` time, the coverage checker
substitutes each ``SymbolicInt`` over its grid range, resolves the slice
bounds to concrete ints, unions the resulting per-axis intervals across
all stores × all substitutions, and verifies the union equals the full
declared output dim. Anything less raises ``ValueError`` before the GPU
is touched.

This is a port of ``stile.triton._core._check_coverage_at_launch`` — same
algorithm, adapted to spork's bind-time entry point.
"""

import contextvars
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from stile.indexing import AffineExpr, SymbolicInt, to_affine
from stile.type import BinaryOp, ParametricReduce, Sliced, ShapeType, Type


@dataclass
class GridAxisInfo:
    """
    How to enumerate a grid-position SymbolicInt at .bind() time:
    ``axis`` is the axis index, ``kind`` selects the enumeration range.
    """
    axis : int
    kind : str  # "tgid" | "gid" — threadgroup_position vs thread_position_in_grid

    def iter_values(self, grid : tuple, threadgroup : tuple) -> "list[int]":
        if self.axis >= len(grid):
            raise ValueError(
                f"GridAxisInfo: axis {self.axis} out of range for grid {grid!r}"
            )
        if self.kind == "tgid":
            return list(range(int(grid[self.axis])))
        if self.kind == "gid":
            return list(range(int(grid[self.axis]) * int(threadgroup[self.axis])))
        if self.kind == "tid":
            return list(range(int(threadgroup[self.axis])))
        raise ValueError(f"Unknown GridAxisInfo kind {self.kind!r}")


@dataclass
class StoredSlice:
    """
    One recorded write to the verified output. ``sliced_shape`` is the
    destination's per-axis Sliced/FullDim dims (as captured by the typed
    setitem or coop.store path). ``constraints`` records the
    ``{SymbolicInt name → int value}`` set that was active when this
    store happened — i.e. ``skv.if_(typed_scalar == value)`` blocks
    that gated the store. Coverage enumeration uses these to restrict
    substitutions for that store.
    """
    sliced_shape : tuple
    constraints  : dict = field(default_factory=dict)


@dataclass
class VerifiedKernelState:
    """
    Per-kernel verifier state populated during trace; consulted at
    ``.bind(grid=..., threadgroup=...)`` for the coverage check.

    ``pid_axes`` maps a grid-position SymbolicInt name → a
    ``GridAxisInfo`` describing how to enumerate that symbol at bind
    time (which axis, and whether it's a per-threadgroup index that
    ranges over ``grid[axis]`` or a per-thread index that ranges over
    ``grid[axis] * threadgroup[axis]``).

    ``loop_var_ranges`` maps a runtime-loop SymbolicInt name →
    ``(lo, hi, step)`` Python ints describing the loop's iteration range.
    Populated by ``skv.range`` when it's used inside the kernel body.
    """
    output_ptr_name : Optional[str] = None
    output_shape   : ShapeType = ()
    stored_slices  : List[StoredSlice] = field(default_factory=list)
    pid_axes       : dict = field(default_factory=dict)
    loop_var_ranges : dict = field(default_factory=dict)


# Active state during tracing of a verified kernel. Set by
# ``@skv.jit``'s wrapper; consulted by ``_record_store`` from
# ``coop.store`` / ``TypedTensorHandle.assign``.
_active_state : contextvars.ContextVar[Optional[VerifiedKernelState]] = (
    contextvars.ContextVar("_skv_active_state", default=None)
)


# ---------------------------------------------------------------------------
# Active-loop stack — for ParametricReduce wrapping of accumulators
# ---------------------------------------------------------------------------


@dataclass
class ActiveLoop:
    """
    A live ``skv.range`` loop. ``snapshots`` maps each typed cooperative
    tensor first touched in this loop's body → its ``Type`` at the
    moment of first touch (i.e. just before the body's first
    contribution applied).
    """
    sym  : SymbolicInt
    lo   : int
    hi   : int
    step : int
    snapshots : dict = field(default_factory=dict)


# Stack of active runtime loops. Pushed by ``skv.range`` on body entry,
# popped on body exit; the ``primitives`` module consults this for
# accumulator snapshotting.
_active_loops : List[ActiveLoop] = []


# Stack of active ``skv.if_(predicate)`` constraint dicts. Each entry
# is a ``{SymbolicInt name → int value}`` map gated by the if-block.
# Stores made inside one or more if-blocks snapshot the union of the
# stack into their StoredSlice.constraints for the coverage check.
_active_if_constraints : List[dict] = []


def _current_if_constraints() -> dict:
    merged : dict = {}
    for frame in _active_if_constraints:
        merged.update(frame)
    return merged


def on_coop_touched(coop) -> None:
    """
    Called by typed accumulating ops (e.g. ``TypedMatmulOp.run``) before
    they mutate a ``TypedCooperativeTensor``. For every loop on the
    stack that hasn't yet snapshotted this coop, record its current
    type — that's the type we'll need on loop exit to compute the
    per-iter delta.
    """
    for loop in _active_loops:
        if id(coop) not in loop.snapshots:
            loop.snapshots[id(coop)] = (coop, coop._type)


def wrap_loop_accumulators(loop : ActiveLoop) -> None:
    """
    Called at the end of a ``skv.range`` body (after the spork-side
    ForLoop has been finalized). For each coop that was touched
    during the body, replace its type with
    ``snap + ParametricReduce(sym, lo, hi, "sum", delta)`` where
    ``delta = current_type - snap``. Stile's normalize then folds the
    Constant(0)/zero-init case to just the ParametricReduce.
    """
    for coop, snap_type in loop.snapshots.values():
        delta_et = BinaryOp(op="-", lhs=coop._type.et, rhs=snap_type.et)
        wrapped_et = BinaryOp(
            op="+",
            lhs=snap_type.et,
            rhs=ParametricReduce(
                loop_var=loop.sym,
                lo=loop.lo,
                hi=loop.hi,
                op="sum",
                body=delta_et,
            ),
        )
        coop._type = Type(st=coop._type.st, et=wrapped_et, dt=coop._type.dt)


def record_store(output_handle, sliced_shape : ShapeType) -> None:
    """
    Called by ``coop.store(out_tile)`` / ``TypedTensorHandle.assign(...)`` /
    typed ``__setitem__`` when the destination is the verified output.
    Appends a ``StoredSlice`` (destination shape + any active
    ``skv.if_`` constraints) to the active state's stored_slices list.
    """
    state = _active_state.get()
    if state is None:
        return  # not in a verified-kernel trace; nothing to record
    if output_handle is None:
        return
    if state.output_ptr_name is None:
        return
    state.stored_slices.append(StoredSlice(
        sliced_shape=tuple(sliced_shape),
        constraints=_current_if_constraints(),
    ))


def check_coverage(
    state       : VerifiedKernelState,
    grid        : tuple,
    threadgroup : tuple,
) -> None:
    """
    Verify the union of per-store slices covers the declared output
    shape on every axis. Raises ``ValueError`` with a clear message on
    mismatch.

    Stores whose offsets involve SymbolicInts not in ``pid_axes`` are
    treated as unresolvable; their slices are skipped (no false
    positives, but also no coverage guarantee for those axes).
    """
    if state.output_ptr_name is None or not state.output_shape:
        return  # nothing declared

    # For each axis (by dim name), collect concrete [lo, hi) intervals
    # from every store × every substitution.
    per_axis_intervals : dict[str, list[tuple[int, int]]] = {
        _dim_name(d) : [] for d in state.output_shape
    }

    for stored in state.stored_slices:
        sliced_tuple = stored.sliced_shape
        # Collect SymbolicInts appearing across this store's bounds
        # (FullDim has none; only Sliced bounds can be symbolic).
        atoms : set[SymbolicInt] = set()
        for sliced in sliced_tuple:
            if not isinstance(sliced, Sliced):
                continue
            for bound in (sliced.start, sliced.end):
                if isinstance(bound, int):
                    continue
                for atom, _ in to_affine(bound).terms:
                    atoms.add(atom)

        # Enumerate substitutions, restricted by any active-if
        # constraints recorded at the time of the store.
        substitutions = _enumerate_substitutions(
            atoms, state, grid, threadgroup,
            constraints=stored.constraints,
        )
        if substitutions is None:
            continue

        for sub in substitutions:
            for sliced in sliced_tuple:
                name = _dim_name(sliced)
                if isinstance(sliced, Sliced):
                    lo_i = _resolve(sliced.start, sub)
                    hi_i = _resolve(sliced.end, sub)
                    if lo_i is None or hi_i is None:
                        continue
                    per_axis_intervals[name].append((lo_i, hi_i))
                else:
                    # FullDim — covers the entire axis trivially.
                    per_axis_intervals[name].append((0, int(sliced.size)))

    # Compare union to [0, dim.size) on every declared output dim.
    for declared_dim in state.output_shape:
        name = _dim_name(declared_dim)
        intervals = per_axis_intervals.get(name, [])
        if not intervals:
            # No store constrained this axis. If the output ShapeType
            # declares a non-trivial size for it, the kernel left it
            # uncovered.
            full = _dim_size(declared_dim)
            raise ValueError(
                f"Verified kernel: output dim `{name}` (size {full}) "
                "has no store covering it — every position should be "
                "written exactly once. Either widen the output-tile "
                f"slice on the `{name}` axis or fix the kernel's "
                "write pattern."
            )
        merged = _union_intervals(intervals)
        full = _dim_size(declared_dim)
        if merged != [(0, full)]:
            raise ValueError(
                f"Verified kernel: stores cover {merged!r} along output "
                f"dim `{name}` (size {full}); expected [(0, {full})]. "
                "Either widen the output-tile slice bounds or adjust "
                "the grid so every position is written exactly once."
            )
        # Note: this only catches UNDER-coverage. Over-coverage
        # (overlapping writes from multiple threads) is harder to
        # detect with per-axis intervals alone, since for multi-axis
        # tile stores the per-axis sums naturally exceed the axis size
        # via differentiation on other axes. Proper overlap detection
        # would need N-D rectangle disjointness — a follow-up.


def _enumerate_substitutions(
    atoms : "set[SymbolicInt]",
    state : "VerifiedKernelState",
    grid : tuple,
    threadgroup : tuple,
    constraints : "Optional[dict]" = None,
) -> "Optional[list[dict]]":
    """
    Cartesian product of values for each known SymbolicInt. If
    ``constraints`` restricts a SymbolicInt name to a specific int
    (e.g. from an enclosing ``skv.if_(tid.x == 0)``), use only that
    value instead of enumerating the full range. Returns None if any
    atom is unknown.
    """
    if not atoms:
        return [{}]
    constraints = constraints or {}
    sorted_atoms = sorted(atoms, key=lambda a: a.name)
    results : list[dict] = [{}]
    for atom in sorted_atoms:
        name = atom.name
        if name in constraints:
            values = [int(constraints[name])]
        elif name in state.pid_axes:
            info = state.pid_axes[name]
            try:
                values = info.iter_values(grid, threadgroup)
            except ValueError:
                return None
        elif name in state.loop_var_ranges:
            lo, hi, step = state.loop_var_ranges[name]
            values = list(range(int(lo), int(hi), int(step)))
        else:
            return None
        results = [{**s, atom: v} for s in results for v in values]
    return results


def _resolve(expr, substitutions : dict) -> "Optional[int]":
    if isinstance(expr, int):
        return expr
    a = to_affine(expr)
    total = a.const
    for atom, coeff in a.terms:
        if atom not in substitutions:
            return None
        total += coeff * substitutions[atom]
    return total


def _union_intervals(intervals : "list[tuple[int, int]]") -> "list[tuple[int, int]]":
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged : list[list[int]] = [list(sorted_iv[0])]
    for lo, hi in sorted_iv[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(iv) for iv in merged]


def _dim_name(dim) -> str:
    """Helper: dim's underlying FullDim's name (Sliced or FullDim)."""
    if isinstance(dim, Sliced):
        return _dim_name(dim.dim)
    return dim.name


def _dim_size(dim) -> int:
    if isinstance(dim, Sliced):
        return _dim_size(dim.dim)
    return int(dim.size)

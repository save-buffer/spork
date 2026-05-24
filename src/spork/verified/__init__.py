"""
spork.verified — Stile-verified versions of spork primitives and kernels.

This subpackage requires the optional ``verified`` extra::

    pip install 'spork-metal[verified]'

which pulls in ``stile-verifier``. Importing ``spork.verified`` without
stile installed raises an ImportError pointing at the right install
command.

Once stile is in place, this package will host:

  - ``spork.verified._backend`` — typed-primitive wrappers over spork's
    tracer, mirroring how ``stile.jax`` / ``stile.triton`` wrap their
    underlying backends.
  - ``spork.verified.kernels`` — spec-verified analogs of
    ``spork.kernels`` (``matmul``, ``causal_gqa``, …), proving structural
    equivalence to a stile specification at trace time.

Today this module is just the import gate; the backend + verified-kernel
contents land in follow-up changes.
"""

try:
    import stile as _stile  # noqa: F401
except ImportError as e:
    raise ImportError(
        "spork.verified requires the 'verified' extra. "
        "Install with:\n"
        "    pip install 'spork-metal[verified]'\n"
        "or, with uv:\n"
        "    uv add 'spork-metal[verified]'"
    ) from e

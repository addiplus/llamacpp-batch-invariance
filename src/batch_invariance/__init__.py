"""batch_invariance -- per-cell batch-invariance verifier for OpenAI-compatible
local LLM servers (llama.cpp continuous batching and work-alikes).

It answers exactly one question, per ``(model, ctx, N)`` cell, against a REAL server:
does firing ``N`` requests CONCURRENTLY at a ``--parallel N`` server produce the SAME
scored outputs as running those same requests ONE AT A TIME? If identical, that cell's
batched outputs are safe to trust; if not, the tool emits a RED cert and the cell stays
serial-only.

This package is a VERIFIER only: it does not route, schedule, autoscale, or batch your
traffic. It launches at most one short-lived experiment server (PID-scoped teardown) and
tells you, after the fact, whether a given batched cell was output-identical to serial.

The public surface splits a PURE core (the diff/verdict logic + cert I/O, fully
unit-testable with zero GPU) from an IMPURE driver (owns the server lifecycle + real
HTTP). Bring your own server binary, work-set (``--workset`` JSON), and scorer
(``--scorer pkg.mod:fn``). Runtime dependencies: Python standard library only.

Import policy: the PURE pillars (``concurrent_dispatch``, ``invariance_diff``) are
imported eagerly (both are stdlib-only). The IMPURE driver / CLI and the pluggable
scorer / work-set seams are exposed LAZILY via :pep:`562` ``__getattr__`` so that simply
importing the package does not pull in the driver's subprocess/threading machinery until a
symbol is actually accessed.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

# ---- PURE pillars (eager; stdlib-only, no network / no GPU) ----
from . import concurrent_dispatch, invariance_diff
from .concurrent_dispatch import (
    CERT_REQUIRED_FIELDS,
    INVARIANT_FIELDS,
    VOLATILE_FIELDS,
    assert_score_invariant,
    cert_filename,
    cert_is_green,
    load_cert,
    write_cert_artifact,
)
from .invariance_diff import (
    ARM_A,
    ARM_B,
    ARM_C,
    STATUS_FAILED,
    STATUS_GREEN,
    STATUS_GREEN_UNVERIFIED,
    STATUS_GREEN_WITH_CAVEAT,
    build_cert,
    build_union_arm_c,
    compute_cobatch_coverage,
    compute_divergence_report,
    content_sha,
    count_ok_completions,
    decide_status,
    diff_arms,
    is_promotable,
    persist_cert,
)

# ---- LAZY surface: (attribute name) -> (submodule, symbol) ----
# Resolved on first access via __getattr__ so the impure driver/CLI and the pluggable
# scorer/work-set seams are not imported at package-import time.
_LAZY: dict[str, tuple[str, str]] = {
    # impure live driver
    "LiveInvarianceDriver": ("live_invariance", "LiveInvarianceDriver"),
    "GuardError": ("live_invariance", "GuardError"),
    "run_arm": ("live_invariance", "run_arm"),
    "score_one": ("live_invariance", "score_one"),
    "run_main": ("live_invariance", "run_main"),
    # CLI entry
    "cli_main": ("cli", "main"),
    "build_parser": ("cli", "build_parser"),
    # pluggable scorer seam
    "DEFAULT_SCORER": ("scorer_api", "DEFAULT_SCORER"),
    "resolve_scorer": ("scorer_api", "resolve_scorer"),
    "classify_transport_error": ("scorer_api", "classify_transport_error"),
    # pluggable work-set seam
    "load_workset": ("workset", "load_workset"),
    "build_body": ("workset", "build_body"),
    # HTTP edge helpers
    "http_post_json": ("http_harness", "http_post_json"),
    "tokenize_count": ("http_harness", "tokenize_count"),
    "wait_for_ready": ("http_harness", "wait_for_ready"),
}


def __getattr__(name: str):
    """Lazily import the impure driver / seam symbols on first access (:pep:`562`)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = target
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, symbol)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "__version__",
    "concurrent_dispatch",
    "invariance_diff",
    # concurrent_dispatch (pure)
    "CERT_REQUIRED_FIELDS",
    "INVARIANT_FIELDS",
    "VOLATILE_FIELDS",
    "assert_score_invariant",
    "cert_filename",
    "cert_is_green",
    "load_cert",
    "write_cert_artifact",
    # invariance_diff (pure)
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "STATUS_FAILED",
    "STATUS_GREEN",
    "STATUS_GREEN_UNVERIFIED",
    "STATUS_GREEN_WITH_CAVEAT",
    "build_cert",
    "build_union_arm_c",
    "compute_cobatch_coverage",
    "compute_divergence_report",
    "content_sha",
    "count_ok_completions",
    "decide_status",
    "diff_arms",
    "is_promotable",
    "persist_cert",
    # impure driver (lazy)
    "LiveInvarianceDriver",
    "GuardError",
    "run_arm",
    "score_one",
    "run_main",
    # CLI (lazy)
    "cli_main",
    "build_parser",
    # scorer seam (lazy)
    "DEFAULT_SCORER",
    "resolve_scorer",
    "classify_transport_error",
    # work-set seam (lazy)
    "load_workset",
    "build_body",
    # HTTP edge (lazy)
    "http_post_json",
    "tokenize_count",
    "wait_for_ready",
]

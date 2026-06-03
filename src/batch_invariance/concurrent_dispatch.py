#!/usr/bin/env python3
"""concurrent_dispatch.py -- score-invariance under concurrent dispatch (PURE brains).

This module is the I/O-free LOGIC core for the concurrent-dispatch invariance check.
Every public function here is referentially transparent (no subprocess, no network,
no GPU, no threads), so the invariant-field set and the score-invariance assertion are
instantly unit-testable with zero files and zero models. The atomic cert persistence
helpers are the sole filesystem touch points and are kept here so the producer, the
consumer, and the offline tests all share ONE cert schema.

WHAT CONCURRENT DISPATCH IS (background): an OpenAI-compatible local LLM server launched
with ``--parallel N`` and ``--cont-batching`` (llama.cpp continuous batching, ON by
default) merges N concurrent HTTP requests into ONE forward pass. Firing N requests
concurrently at one ``--parallel N`` server therefore batches them; the server side is
thread-safe (slots isolated, attention masking prevents cross-slot bleed, over-firing
QUEUES).

INVARIANCE CLAIM -- SCOPED HONESTLY (do NOT widen this to an absolute):
``assert_score_invariant`` proves a CLIENT-side plumbing invariant ONLY -- that the
worker pool, queue, scorer, /tokenize calibration, seeding and result paths do not
themselves corrupt scores under concurrency. It does NOT prove SERVER-side
score-invariance, and CANNOT offline: an offline mock returns content independent of
batch composition, so the offline assert can only ever pass (a structurally vacuous
pass against a canned mock; a mock divergence knob exists only to exercise the ABORT
path, it is NOT evidence of real invariance). Real continuous batching is NOT guaranteed
batch-invariant even at temp=0 -- batched matmul / RMSNorm / attention reductions can
reorder (llama.cpp #7052 observed 5-8 unique completions for one prompt at 8 slots /
temp=0 on H100/A100-class hardware; PR #16016 deterministic mode is OFF by default).

THEREFORE: SERVER-side score-invariance is UNPROVEN offline and MUST be confirmed by a
LIVE per-model real-server diff BEFORE any batched score is trusted. The live diff --
the same seeded work-set run serially then concurrently on the REAL server, both fed to
the gate, repeated several times over >=2 N values, with ``source=='live'`` on the
persisted cert -- is the sole arbiter. A green cert with ``source=='mock'`` is NOT
sufficient and must never promote (``cert_is_green`` defaults ``require_source='live'``).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Dispatch-invariance assertion. The KEY gate: a batched run and a serial run of
# the SAME seeds must produce IDENTICAL scores. Compares the SCORE-bearing fields
# per test_id, ignoring the volatile timing/GPU/timestamp fields that are EXPECTED
# to differ (wall_ms, decode_tok_s, prefill_tok_s, gpu_after, ts_utc).
# ---------------------------------------------------------------------------
# Fields that MUST match between a serial and a concurrent run of the same seed (the
# score is a deterministic function of these). prompt_tokens_measured is included
# because the /tokenize calibration must land on the SAME token target.
#
# This is the GENERIC default. The scorer contract returns (score, passed,
# failure_mode); a consumer that surfaces additional score-bearing fields can swap
# this tuple end-to-end (the cert helpers thread ``invariant_fields=`` through, and
# the diff functions iterate whatever tuple is supplied) without editing this module.
INVARIANT_FIELDS = (
    "score", "passed", "expected_answer",
    "prompt_tokens_measured", "failure_mode",
)
# Fields EXPECTED to differ (timing/throughput/host snapshot) -- explicitly ignored
# so the invariance check does not false-fail on them.
VOLATILE_FIELDS = (
    "wall_ms", "decode_tok_s", "prefill_tok_s", "gpu_after", "ts_utc",
    "completion_tokens", "prompt_tokens_actual", "thinking_token_multiplier",
    "timed_out", "calibration", "reason", "response_first_200",
)


def assert_score_invariant(seq_results: dict, conc_results: dict) -> None:
    """Raise AssertionError unless the two result maps are score-invariant.

    ``seq_results`` / ``conc_results`` are dicts keyed by ``test_id`` -> trial-result
    dict. Asserts:
      1. the SAME set of test_ids (none lost / duplicated by concurrency), AND
      2. for every test_id, every field in :data:`INVARIANT_FIELDS` is equal,
         while :data:`VOLATILE_FIELDS` (timing/GPU/timestamps) are ignored.

    This proves the CLIENT plumbing is invariant; only a live real-server diff proves
    the server-side slot masking is bit-perfect (delegated to the live driver).
    """
    seq_ids = set(seq_results)
    conc_ids = set(conc_results)
    if seq_ids != conc_ids:
        missing = seq_ids - conc_ids
        extra = conc_ids - seq_ids
        raise AssertionError(
            "dispatch-invariance: test_id sets differ "
            f"(only-sequential={sorted(missing)[:5]}, "
            f"only-concurrent={sorted(extra)[:5]})"
        )
    for tid in sorted(seq_ids):
        a = seq_results[tid]
        b = conc_results[tid]
        for fld in INVARIANT_FIELDS:
            av = a.get(fld)
            bv = b.get(fld)
            if av != bv:
                raise AssertionError(
                    f"dispatch-invariance: test_id={tid!r} field {fld!r} differs: "
                    f"sequential={av!r} vs concurrent={bv!r}"
                )


# ---------------------------------------------------------------------------
# Dispatch-invariance certification ARTIFACT (PURE I/O helpers). The producer writes
# one of these after running the seq-vs-conc canary; the consumer REFUSES to trust a
# batched cell unless a green, source=='live' cert exists. Kept here (atomic,
# injectable) so both sides + the offline tests share ONE schema.
# ---------------------------------------------------------------------------
CERT_REQUIRED_FIELDS = ("model", "ctx", "dispatch_n", "status", "source")


def cert_filename(model: str, ctx: int, dispatch_n: int) -> str:
    """Stable artifact filename for a (model, ctx, N) cert. Collision-free per cell."""
    safe_model = str(model).replace(os.sep, "_").replace("/", "_")
    return f"{safe_model}__ctx{int(ctx)}__N{int(dispatch_n)}.json"


def write_cert_artifact(
    cert_dir: str,
    model: str,
    ctx: int,
    dispatch_n: int,
    status: str,
    *,
    source: str,
    kv_label: str = "q8_0",
    n_sample: int = 0,
    mismatch: str | None = None,
    invariant_fields: tuple = INVARIANT_FIELDS,
    ts_utc: str | None = None,
) -> str:
    """Atomically write a dispatch-invariance cert JSON; return its path.

    ``status`` is 'green' (seq==conc) or 'failed' (a mismatch was found). ``source``
    is 'mock' (offline canary -- NEVER promotes) or 'live' (real-server diff -- the
    only provenance the promotion gate accepts). Atomic tempfile+os.replace so a
    concurrent reader never sees a partial file."""
    os.makedirs(cert_dir, exist_ok=True)
    payload = {
        "model": str(model),
        "ctx": int(ctx),
        "dispatch_n": int(dispatch_n),
        "kv_label": str(kv_label),
        "status": str(status),
        "source": str(source),
        "n_sample": int(n_sample),
        "invariant_fields": list(invariant_fields),
        "mismatch": mismatch,
        "ts_utc": ts_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = os.path.join(cert_dir, cert_filename(model, ctx, dispatch_n))
    fd, tmp = tempfile.mkstemp(dir=cert_dir, prefix=".cert-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def load_cert(cert_dir: str, model: str, ctx: int, dispatch_n: int) -> dict | None:
    """Load the cert for (model, ctx, N); None if absent / unreadable / malformed."""
    if not cert_dir or not os.path.isdir(cert_dir):
        return None
    path = os.path.join(cert_dir, cert_filename(model, ctx, dispatch_n))
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def cert_is_green(
    cert: dict | None,
    *,
    require_source: str | None = "live",
) -> bool:
    """True iff ``cert`` is a green dispatch-invariance cert of the required provenance.

    The promotion gate: a batched cell may be trusted ONLY when this returns True. By
    DEFAULT ``require_source='live'`` -- a green cert with ``source=='mock'`` is
    REJECTED (an offline canary can never smuggle a batched score in; the offline
    assert is structurally vacuous against a canned mock). Pass ``require_source=None``
    to accept any provenance (tests of the 'mock green' rejection use the default;
    nothing in production should)."""
    if not isinstance(cert, dict):
        return False
    if cert.get("status") != "green":
        return False
    if require_source is not None and cert.get("source") != require_source:
        return False
    return True

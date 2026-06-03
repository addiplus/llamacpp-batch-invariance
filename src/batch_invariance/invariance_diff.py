#!/usr/bin/env python3
"""invariance_diff.py -- PURE diff + cert logic for the live batch-invariance gate.

No network, no subprocess, no threads, no GPU. Consumes per-arm result maps
({test_id: result_dict}) produced by the live driver and produces (a) a full
divergence report and (b) a cert dict compatible with
``concurrent_dispatch.cert_is_green``.

WHY THIS MODULE EXISTS (the anti-vacuity hinge).  A naive dispatch-invariance
"certification" is structurally vacuous if ``assert_score_invariant`` only ever runs
against a mock whose completion content is independent of batch composition: the
assertion is then dead code that can only pass. The reference results come from the
SEQUENTIAL path (``--parallel 1``); batched concurrent dispatch is ``--parallel N`` +
overlapping requests = llama.cpp continuous batching. Web-confirmed prior: llama.cpp
#7052 (8 slots, temp=0 -> 5-8 unique completions on H100/A100-class hardware), Thinking
Machines "Defeating Nondeterminism in LLM Inference", llama.cpp PR #16016 (deterministic
mode OFF by default). The null hypothesis is therefore "batching DOES change outputs,"
and a GREEN must be *earned* against the real server. This module is the brains of that
gate: it diffs three arm maps and decides GREEN / RED / AMBER / UNVERIFIED, and it stamps
a cert whose ``status``/``source`` make a mock pass non-promotable by construction
(``cert_is_green`` default ``require_source='live'``).

SEAM (the reason this is split out and pure).  The live driver owns the experiment
server lifecycle, runs the arms over real HTTP, and builds per-arm result maps
``{test_id: result_dict}``. It hands those maps to THIS module. This module never
touches the network and never knows what an "arm" launched -- it only compares maps.
That split is what makes the gate logic unit-testable with zero GPU (and is what lets a
test suite prove RED-on-divergence, not merely identity).

DESIGN PRINCIPLE -- leniency is the only dangerous direction.  The gate criterion is
EXACT-SCORE MATCH over ``concurrent_dispatch.INVARIANT_FIELDS`` (score, passed,
expected_answer, prompt_tokens_measured, failure_mode). Tolerance is BANNED as a gate: a
lenient compare converts a real divergence into a false GREEN (the original sin). Exact
match can only OVER-report (safe: it triggers an investigation, never a silent
promotion). Token-level content (``content_sha``) and any logprob deltas are recorded as
DIAGNOSTICS and never gate (logprobs live in VOLATILE_FIELDS).
"""
from __future__ import annotations

import hashlib
import math

from .concurrent_dispatch import (  # noqa: F401  (re-exported for the driver + tests)
    INVARIANT_FIELDS,
    VOLATILE_FIELDS,
    assert_score_invariant,
    cert_filename,
    cert_is_green,
    load_cert,
    write_cert_artifact,
)

# Arm identifiers (stable strings used as dict keys everywhere).
ARM_A = "A"   # --parallel 1, serial dispatch    == the sequential reference results
ARM_B = "B"   # --parallel N, serial dispatch     (slot-allocation control)
ARM_C = "C"   # --parallel N, concurrent dispatch (the batched cell under test)

# Overlap thresholds for a reference-grade GREEN (see the truth table below). A "pass"
# with no observed co-batching is the single most dangerous false GREEN: the arm
# *looks* concurrent but never actually ran >1 request in one forward pass, so it
# proves nothing about batch-invariance.
MIN_CLIENT_OVERLAP_DEPTH = 2
MIN_SERVER_BUSY_SLOTS = 2

# Completion floor for a reference-grade GREEN (the SECOND vacuity, distinct from the
# canned-mock sin). INVARIANT_FIELDS includes ``failure_mode``, so two arms that BOTH
# FAIL IDENTICALLY (e.g. every trial http_timeout/empty under contention) compare
# EQUAL -> n_divergent==0 -> a "clean" AC that would otherwise mint a PROMOTABLE live
# GREEN on a cell where ZERO real completions occurred ("matching nothing" must not
# read as "invariant something"). The ONLY score-bearing trials are those whose
# ``failure_mode == 'ok'`` (genuinely scored). We require a minimum FRACTION of such
# trials in BOTH ARM_A and ARM_C before GREEN; below it the verdict is
# ``green_unverified`` (the safe direction -- over-reports as unverified, never silently
# promotes). This is the experiment-contention regime: if the experiment server runs
# concurrent with other load, a starved/wedged server times out per-request and a whole
# arm can be all-failure while still tripping overlap_ok (a stuck slot reports
# is_processing:true on /slots and the hung client intervals overlap). The floor closes
# that path.
MIN_OK_FRACTION = 0.8        # >= 80% of compared trials must be genuinely scored ('ok')
OK_FAILURE_MODE = "ok"       # the scorer contract's "scored fine" label

# Cert status vocabulary (superset of concurrent_dispatch's 'green'/'failed').
# IMPORTANT: only the literal "green" is accepted by concurrent_dispatch.cert_is_green.
# The two GREEN_* variants are deliberately NOT the literal "green", so cert_is_green
# returns False for them with ZERO change to concurrent_dispatch -- exactly the
# "not auto-promotable" semantics we want.
# promotable iff source=='live' AND overlap_ok
STATUS_GREEN = "green"
# score-level divergence -> RED
STATUS_FAILED = "failed"
# AC passed but no real overlap -> NOT reference-grade
STATUS_GREEN_UNVERIFIED = "green_unverified"
# AC passed, token-only divergence OR B anomaly -> sign-off
STATUS_GREEN_WITH_CAVEAT = "green_with_caveat"

# Divergence-class + anomaly vocabulary (diagnostics; never read by cert_is_green).
DIVCLASS_CO_BATCHING = "co_batching"
DIVCLASS_SLOT_ALLOC = "slot_allocation(+co_batching)"
ANOMALY_AC_AGREE_B_DISAGREES = "AC_agree_B_disagrees"
# Logit-drift AMBER subtype (additive; diagnostic only). When the logit-drift gate is
# ARMED (logit_drift_eps>0) a score-clean / token-only id whose per-id logit summary
# moved by MORE than eps is surfaced as this first-class AMBER subtype: "tokens differ
# but within logit eps" is the WEAKER caveat; "tokens AND logits drifted" is the stronger
# one. Never RED (logprobs are VOLATILE; they corroborate, they do not gate a score).
ANOMALY_TOKEN_AND_LOGIT_DRIFT = "token_and_logit_drift"

# Co-batching COVERAGE floor (the per-id attribution gate). The PEAK scalar
# (server_peak_busy_slots / client_max_overlap_depth) proves only "≥2 requests were
# co-resident at SOME instant in SOME pass" -- it does NOT prove WHICH ids actually shared
# a forward pass, so a GREEN earned on the peak alone can certify a cell where most ids
# ran effectively alone. The AUTHORITATIVE per-id signal is request-interval overlap: id i
# was co-batched iff some OTHER id j's [dispatch_ts, complete_ts] interval overlaps i's.
# We require a FRACTION of ids to have actually co-batched before a reference-grade GREEN;
# the peak remains only a corroborating FLOOR (Row 2 already demotes when the peak is < 2).
# FAIL CLOSED: if per-id dispatch_ts/complete_ts are missing/uncertain for ANY id the
# coverage is treated as NOT satisfied (-> green_unverified, never green) -- the safe
# direction (over-reports as unverified, never silently promotes).
MIN_COBATCH_COVERAGE_FRACTION = 0.80   # >= 80% of ids must have actually co-batched
# Below this id-count the coverage fraction is statistically meaningless, so when the
# coverage gate is CHECKED we additionally require at least this many ids (and N>=2, since
# a lone id can never co-batch). Scales the absolute requirement with N alongside the
# fraction. Purely a floor on the CHECKED path; never engaged when coverage is unchecked.
MIN_COBATCH_COVERAGE_IDS = 2
# The per-id request-interval keys the driver stamps. Captured as time.monotonic() floats
# at HTTP send / response receipt.
COBATCH_DISPATCH_TS_FIELD = "dispatch_ts"
COBATCH_COMPLETE_TS_FIELD = "complete_ts"

# Logit-drift gate default. OFF (0.0) reproduces a pure-forensic logprob diagnostic that
# never influences a verdict. A positive eps ARMS the corroboration so a GREEN means
# "token AND logit stable"; a token-only id whose logit summary drifted by > eps becomes
# AMBER (green_with_caveat), never RED.
DEFAULT_LOGIT_DRIFT_EPS = 0.0
# Optional per-id logprob summary fields the diff will read IF the driver ever plumbs them.
# Tried in order; first numeric wins. All live OUTSIDE INVARIANT_FIELDS. Inert by
# construction until both eps>0 AND the field is present.
LOGIT_SUMMARY_FIELDS = ("logit_sum", "logprob_sum", "mean_logprob", "logit_mean")

# Re-export for callers that want the content-bearing volatile set name without
# re-deriving it. content_sha() lives OUTSIDE INVARIANT_FIELDS by design (it is a
# diagnostic, not a gate) -- see the module docstring.
VOLATILE_PLUS_CONTENT = VOLATILE_FIELDS

# Generic name-substring hints that suggest a Mixture-of-Experts model. Used only as a
# heuristic flag on the cert's promotion_scope (never a gate): MoE models route experts
# per token, and batch composition changes which experts fire -- the textbook #7052
# regime -- so a dense-model GREEN says nothing about MoE expert-routing divergence.
_MOE_NAME_HINTS = ("moe", "a3b", "a4b", "mixtral", "80b")


# ---------------------------------------------------------------------------
# content_sha -- the token-level diagnostic. NOT a gate trigger by itself.
# ---------------------------------------------------------------------------
def content_sha(text: str | None) -> str:
    """Stable short sha of a completion string, for the token-level diagnostic.

    Returns the first 12 hex chars of ``sha256(text)``. ``None``/``''`` -> sha of
    ``''`` (a fixed, well-known constant), so a missing completion compares equal to
    an empty completion and never raises. Pure. Used to flag token-divergence (scores
    equal but content differs => AMBER); it is NEVER an INVARIANT_FIELD and NEVER a
    direct divergence trigger.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def _result_content(result: dict) -> str:
    """Best available completion text for the token diagnostic.

    Prefer the full ``'content'`` (most sensitive to token drift); fall back to the
    truncated ``'response_first_200'`` (what the driver always records); else ''.
    Pure; tolerant of a result dict missing both keys.
    """
    if not isinstance(result, dict):
        return ""
    val = result.get("content")
    if val is None:
        val = result.get("response_first_200")
    return val if isinstance(val, str) else ("" if val is None else str(val))


def _fill_key(result: dict) -> str:
    """Stable string bucket for the by_fill aggregate. Pure; tolerant of missing/None.

    Formats a numeric fill_ratio as ``f"{x:.2f}"`` (matches the cert schema's
    ``by_fill`` keys like "0.05"/"0.90"); a missing/non-numeric fill_ratio buckets
    under the literal ``"unknown"`` rather than raising.
    """
    if not isinstance(result, dict):
        return "unknown"
    fr = result.get("fill_ratio")
    try:
        return f"{float(fr):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _family_key(result: dict) -> str:
    """Stable string bucket for the by_family aggregate. Pure; tolerant of missing/None."""
    if not isinstance(result, dict):
        return "unknown"
    fam = result.get("family")
    return str(fam) if fam is not None else "unknown"


# ---------------------------------------------------------------------------
# _invariant_fields_differ -- the single shared score-divergence predicate.
# Both compute_divergence_report and build_union_arm_c use it so the "what counts
# as a score divergence" rule is defined in exactly ONE place.
# ---------------------------------------------------------------------------
def _invariant_fields_differ(rx: dict, ry: dict) -> bool:
    """True iff ANY INVARIANT_FIELD differs between two result dicts. Pure; tolerant."""
    rx = rx if isinstance(rx, dict) else {}
    ry = ry if isinstance(ry, dict) else {}
    for fld in INVARIANT_FIELDS:
        if rx.get(fld) != ry.get(fld):
            return True
    return False


# ---------------------------------------------------------------------------
# count_ok_completions -- the completion-floor primitive (closes the all-failure
# vacuity). A "real completion" is a trial the scorer genuinely SCORED, i.e.
# failure_mode == 'ok'. Everything else (http_timeout, empty, arm_deadline,
# error_other, oom, ...) carries score=0.0/passed=False by construction and is NOT
# evidence of an invariant *score* -- only of an invariant *failure*.
# ---------------------------------------------------------------------------
def count_ok_completions(arm_map: dict) -> dict:
    """Count genuinely-scored ('ok') vs failure trials in one {test_id: result} map.

    Returns ``{"n_total": int, "n_ok": int, "n_failure": int}``. A trial is "ok" iff
    its ``failure_mode`` equals :data:`OK_FAILURE_MODE` ('ok'); every other failure
    mode (or a missing/blank failure_mode) counts as a failure. Pure; tolerant of a
    None/non-dict map or result. This is the score-bearing-trial census the
    completion-floor gate consumes so an all-failure arm (which compares EQUAL on
    INVARIANT_FIELDS and therefore looks "clean") cannot mint a promotable GREEN.
    """
    arm_map = arm_map if isinstance(arm_map, dict) else {}
    n_ok = 0
    n_total = 0
    for r in arm_map.values():
        if not isinstance(r, dict):
            n_total += 1
            continue
        n_total += 1
        if r.get("failure_mode") == OK_FAILURE_MODE:
            n_ok += 1
    return {"n_total": n_total, "n_ok": n_ok, "n_failure": n_total - n_ok}


# ---------------------------------------------------------------------------
# Per-id co-batching COVERAGE (the request-interval attribution primitive).
# Turns the per-id [dispatch_ts, complete_ts] intervals the driver stamps into the
# AUTHORITATIVE "did this id actually share a forward pass with another in-flight id?"
# signal -- the thing the PEAK scalar (server_peak_busy_slots) cannot answer. Pure;
# import-pure (NO clock reads, NO I/O); deterministic given the maps.
# ---------------------------------------------------------------------------
def _ts_float(value) -> float | None:
    """Coerce a timestamp to float, or None if missing / non-numeric / non-finite.

    A None / unparseable / NaN / inf timestamp means the per-id request interval is
    UNKNOWN for that id, which the coverage gate treats as attribution-uncertain (fail
    closed). Pure; never raises."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _interval_overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    """True iff the closed request intervals [a0,a1] and [b0,b1] intersect. Pure.

    Two requests are co-batched if they were BOTH in flight at the same instant, i.e.
    their [dispatch_ts, complete_ts] intervals overlap. Uses ``<=`` (closed intervals)
    so an instantaneous touch at an endpoint counts as overlap -- the SAFE/conservative
    direction for a co-batching signal (it can only ever ADD coverage, never hide a
    divergence; coverage gates GREEN, it never forces RED)."""
    lo = a0 if a0 >= b0 else b0
    hi = a1 if a1 <= b1 else b1
    return lo <= hi


def compute_cobatch_coverage(
    arm_map: dict,
    *,
    min_coverage_fraction: float = MIN_COBATCH_COVERAGE_FRACTION,
) -> dict:
    """Per-id co-batching coverage from the request-interval contract. Pure; never raises.

    For each id, ``was_co_batched[id]`` is True iff there exists ANOTHER id whose
    ``[dispatch_ts, complete_ts]`` interval OVERLAPS this id's interval. This
    request-interval overlap among concurrently in-flight ids is the AUTHORITATIVE
    co-batching signal (the slot-poller peak is only a corroborating FLOOR, evaluated
    separately in :func:`decide_status`). ``coverage_fraction`` = (#ids co-batched) /
    (#ids).

    Returns::

        {
          "checked": bool,              # were usable per-id intervals present at all?
          "n_ids": int,                 # ids in the map
          "n_with_interval": int,       # ids carrying BOTH usable timestamps
          "n_missing_interval": int,    # ids missing/uncertain a timestamp (fail-closed)
          "missing_interval_ids": [...],# sorted; the ids that broke attribution
          "n_co_batched": int,          # ids with >=1 overlapping OTHER id
          "co_batched_ids": [...],      # sorted
          "coverage_fraction": float,   # n_co_batched / max(1, n_ids)
          "attribution_certain": bool,  # True iff EVERY id had a usable interval
          "required_co_batched": int,   # ceil(frac*n_ids), floored at MIN_COBATCH_COVERAGE_IDS
          "min_coverage_fraction": float,
          "cobatch_coverage_ok": bool,  # the gate verdict (see below)
        }

    ``checked`` is False (the gate is INERT) ONLY when NO id carries a ``dispatch_ts``
    key at all -- i.e. the map predates the interval contract or is a hand-built unit
    fixture. The driver's real maps always carry the key (a transport-failure row
    carries ``dispatch_ts=None``), so they are always ``checked``.

    FAIL CLOSED (``cobatch_coverage_ok=False``) whenever the gate is checked AND any of:
      * ``attribution_certain`` is False (>=1 id missing/uncertain a timestamp), OR
      * ``n_ids < MIN_COBATCH_COVERAGE_IDS`` (a lone id can never co-batch), OR
      * ``n_co_batched < required_co_batched`` (too few ids actually co-batched).
    When ``checked`` is False the verdict is a benign True (the gate is not enforced),
    mirroring the completion-floor's "absent counts -> not enforced" convention so every
    legacy caller / pure-report test is preserved byte-for-byte.
    """
    arm_map = arm_map if isinstance(arm_map, dict) else {}
    ids = sorted(arm_map)
    n_ids = len(ids)

    # Is the interval contract even present? (Inert on legacy / hand-built maps.)
    any_dispatch_key = any(
        isinstance(arm_map.get(tid), dict)
        and COBATCH_DISPATCH_TS_FIELD in arm_map[tid]
        for tid in ids
    )

    intervals: dict = {}
    missing_interval_ids: list = []
    for tid in ids:
        r = arm_map.get(tid)
        d = _ts_float(r.get(COBATCH_DISPATCH_TS_FIELD)) if isinstance(r, dict) else None
        c = _ts_float(r.get(COBATCH_COMPLETE_TS_FIELD)) if isinstance(r, dict) else None
        if d is None or c is None:
            missing_interval_ids.append(tid)
            continue
        # Guard against an inverted interval (complete before dispatch): normalize so the
        # overlap test is well-defined; a degenerate point interval is still valid.
        intervals[tid] = (d, c) if c >= d else (c, d)

    co_batched: set = set()
    interval_ids = list(intervals)
    for i in range(len(interval_ids)):
        ai0, ai1 = intervals[interval_ids[i]]
        for j in range(i + 1, len(interval_ids)):
            bj0, bj1 = intervals[interval_ids[j]]
            if _interval_overlaps(ai0, ai1, bj0, bj1):
                co_batched.add(interval_ids[i])
                co_batched.add(interval_ids[j])

    n_co_batched = len(co_batched)
    coverage_fraction = (n_co_batched / n_ids) if n_ids else 0.0
    try:
        frac = float(min_coverage_fraction)
    except (TypeError, ValueError):
        frac = MIN_COBATCH_COVERAGE_FRACTION
    # Scale the absolute requirement with N, floored at MIN_COBATCH_COVERAGE_IDS so a
    # 2-id run still needs both co-batched (never a vacuous "0 required").
    required_co_batched = (
        max(MIN_COBATCH_COVERAGE_IDS, int(math.ceil(frac * n_ids))) if n_ids else 0
    )
    attribution_certain = (n_ids > 0) and (not missing_interval_ids)

    checked = bool(any_dispatch_key)
    if not checked:
        cobatch_coverage_ok = True   # gate inert (legacy / unit fixtures) -- not enforced
    else:
        cobatch_coverage_ok = (
            attribution_certain
            and n_ids >= MIN_COBATCH_COVERAGE_IDS
            and n_co_batched >= required_co_batched
        )

    return {
        "checked": checked,
        "n_ids": n_ids,
        "n_with_interval": len(intervals),
        "n_missing_interval": len(missing_interval_ids),
        "missing_interval_ids": missing_interval_ids,
        "n_co_batched": n_co_batched,
        "co_batched_ids": sorted(co_batched),
        "coverage_fraction": coverage_fraction,
        "attribution_certain": attribution_certain,
        "required_co_batched": required_co_batched,
        "min_coverage_fraction": frac,
        "cobatch_coverage_ok": cobatch_coverage_ok,
    }


# ---------------------------------------------------------------------------
# Per-id logit summary + drift (the OFF-BY-DEFAULT corroboration primitive).
# logprobs are VOLATILE (never an INVARIANT_FIELD): they corroborate a GREEN, they do
# NOT gate a score. These helpers are inert unless the driver plumbs a logprob summary
# AND the caller arms the gate with logit_drift_eps>0. Pure; deterministic.
# ---------------------------------------------------------------------------
def _logit_value(result: dict) -> float | None:
    """Best-available per-id scalar logit/logprob summary, or None if absent. Pure.

    Tries :data:`LOGIT_SUMMARY_FIELDS` in order and returns the first finite numeric. A
    map with no such field returns None for every id and the logit gate is inert by
    construction (a None-vs-None delta is 0.0 -> never AMBER)."""
    if not isinstance(result, dict):
        return None
    for fld in LOGIT_SUMMARY_FIELDS:
        v = _ts_float(result.get(fld))   # reuse the finite-float coercion
        if v is not None:
            return v
    return None


def compute_logit_drift_ids(map_x: dict, map_y: dict, eps: float) -> dict:
    """Per-id |logit_x - logit_y| drift summary between two arm maps. Pure; never raises.

    Returns ``{"armed": bool, "eps": float, "max_logit_delta": float,
    "logit_drift_ids": [sorted ids whose delta > eps], "per_id_delta": {id: delta}}``.
    ``armed`` is True iff eps>0 AND at least one SHARED id carries a logit summary on BOTH
    sides. When not armed the result is benign (empty drift ids, 0.0 max) so the logit gate
    is inert. Only ids that are SCORE-clean matter to the caller (a score divergence is
    already RED); this helper does not itself read scores -- decide_status intersects
    ``logit_drift_ids`` with the score-clean / token-only set."""
    map_x = map_x if isinstance(map_x, dict) else {}
    map_y = map_y if isinstance(map_y, dict) else {}
    try:
        e = float(eps)
    except (TypeError, ValueError):
        e = 0.0
    shared = sorted(set(map_x) & set(map_y))
    per_id_delta: dict = {}
    drift_ids: set = set()
    max_delta = 0.0
    any_pair = False
    for tid in shared:
        lx = _logit_value(map_x.get(tid) or {})
        ly = _logit_value(map_y.get(tid) or {})
        if lx is None or ly is None:
            continue
        any_pair = True
        delta = abs(lx - ly)
        per_id_delta[tid] = delta
        if delta > max_delta:
            max_delta = delta
        if e > 0.0 and delta > e:
            drift_ids.add(tid)
    armed = (e > 0.0) and any_pair
    return {
        "armed": armed,
        "eps": e,
        "max_logit_delta": max_delta,
        "logit_drift_ids": sorted(drift_ids),
        "per_id_delta": per_id_delta,
    }


# ---------------------------------------------------------------------------
# build_union_arm_c -- fold T stochastic ARM_C re-passes into ONE map for the gate.
# This closes the CRITICAL single-pass false-GREEN hole: continuous-batching
# divergence (#7052) is STOCHASTIC -- co-batch composition varies run-to-run with
# arrival timing -- so a divergence that surfaces only when prompts i,j land in the
# same forward pass can be ABSENT from one pass and PRESENT in the next. Running
# ARM_C once observes a single interleaving; the design mandates T concurrent
# re-passes with the AC gate evaluated on the UNION of divergent ids (any pass
# divergent => RED). This builder produces that union map.
# ---------------------------------------------------------------------------
def build_union_arm_c(map_a: dict, c_passes: list) -> dict:
    """Fold T per-pass ARM_C maps into ONE union map, biased toward divergence.

    For each test_id, the union representative is chosen so that the union is
    DIVERGENT vs ARM_A iff the id diverged from ARM_A on ANY of the T passes
    (any-pass-divergent => RED):

      * if the id is MISSING from any pass (lost/dropped under co-batching on that
        pass), the union drops it too (a lost id is a divergence -> only_x in the
        AC report);
      * else if the id diverges from ARM_A on >=1 pass, the union takes the FIRST
        such diverging pass's result (so compute_divergence_report(A, union) reports
        it divergent, with the real diverging field, not a synthetic one);
      * else (score-clean on every pass) the union takes the FIRST pass whose
        completion CONTENT (``content_sha``) differs from ARM_A's, falling back to
        pass 0 only when EVERY pass is byte-identical to ARM_A. This makes the
        token-only (content) union SYMMETRIC with the score union above: a token-only
        divergence (same score, different completion bytes -> AMBER per
        :func:`decide_status` Row 4) surfacing on ANY pass 1..T-1 is carried into the
        union representative, so ``compute_divergence_report(A, union).token_divergence_ids``
        is non-empty and the cell is correctly AMBER (``green_with_caveat``), not GREEN.
        (Otherwise the representative would be pinned to pass 0, so a token drift that did
        NOT land on pass 0 would be silently dropped -- an AMBER->GREEN downgrade on most
        stochastic continuous-batching interleavings, #7052. This rule can only make a
        verdict MORE conservative: GREEN -> green_with_caveat. It NEVER promotes a real
        score divergence -- that is already RED via the score union, which has priority.)

    ``map_a`` is ARM_A's {test_id: result}; ``c_passes`` is a list of ARM_C maps
    (already meta-stripped). Pure; never raises. With T==1 this returns pass 0
    verbatim (a list of one), so the single-pass behavior is preserved exactly when
    --reps drives one pass.
    """
    map_a = map_a if isinstance(map_a, dict) else {}
    passes = [p for p in (c_passes or []) if isinstance(p, dict)]
    if not passes:
        return {}
    if len(passes) == 1:
        return dict(passes[0])

    # The union's id-set is the INTERSECTION across passes: an id missing from any
    # pass is treated as lost (excluded here) so the AC report flags it via only_x.
    common_ids = set(passes[0])
    for p in passes[1:]:
        common_ids &= set(p)

    union: dict = {}
    for tid in common_ids:
        ra = map_a.get(tid) or {}
        chosen = None
        for p in passes:
            rc = p.get(tid) or {}
            if _invariant_fields_differ(ra, rc):
                chosen = rc  # first diverging pass wins (real field + values)
                break
        if chosen is None:
            # Score-clean on every pass. Pick the FIRST pass whose completion CONTENT
            # differs from ARM_A's so a token-only (AMBER) divergence on ANY pass -- not
            # just pass 0 -- survives into the union. Fall back to pass 0 only when no
            # pass drifted (all byte-identical to A): the representative is then pass 0
            # exactly as before, so a genuinely-clean id is unchanged.
            sha_a = content_sha(_result_content(ra))
            for p in passes:
                rc = p.get(tid) or {}
                if content_sha(_result_content(rc)) != sha_a:
                    chosen = rc  # first token-divergent pass wins (surfaces the AMBER)
                    break
            if chosen is None:
                chosen = passes[0].get(tid) or {}
        union[tid] = dict(chosen)
    return union


def fold_pass_divergence_counts(map_a: dict, c_passes: list) -> list:
    """Per-pass count of AC-divergent ids (id missing OR any INVARIANT_FIELD differs).

    Returns ``[n_divergent_pass0, n_divergent_pass1, ...]`` for the cert's audit
    trail (so a reviewer sees which pass surfaced the divergence). This counts SCORE
    (INVARIANT_FIELD) + missing-id divergences ONLY -- its semantics define the cert's
    ``per_pass_ac_divergent`` field and are deliberately UNCHANGED. The per-pass
    TOKEN-ONLY tally is a SEPARATE additive trail (see
    :func:`fold_pass_token_only_counts`). Pure; never raises.
    """
    map_a = map_a if isinstance(map_a, dict) else {}
    passes = [p for p in (c_passes or []) if isinstance(p, dict)]
    counts: list = []
    a_ids = set(map_a)
    for p in passes:
        n = 0
        p_ids = set(p)
        # ids present in A but missing this pass (or vice-versa) are divergent
        n += len(a_ids ^ p_ids)
        for tid in (a_ids & p_ids):
            if _invariant_fields_differ(map_a.get(tid) or {}, p.get(tid) or {}):
                n += 1
        counts.append(n)
    return counts


def fold_pass_token_only_counts(map_a: dict, c_passes: list) -> list:
    """Per-pass count of TOKEN-ONLY divergent ids (score-clean, completion CONTENT differs).

    Returns ``[n_token_only_pass0, n_token_only_pass1, ...]`` -- the AMBER companion to
    :func:`fold_pass_divergence_counts` (which counts SCORE/missing divergences). An id
    counts for a pass iff it is PRESENT and SCORE-clean (no INVARIANT_FIELD differs) on
    that pass BUT its ``content_sha`` differs from ARM_A's. This is the per-pass audit
    trail that lets a reviewer recover WHICH stochastic interleaving surfaced a token-only
    (#7052) divergence -- recoverable even though the union folds it into a single
    representative. Stamped on the cert as the additive ``per_pass_ac_token_only`` key;
    older readers ignore it and ``per_pass_ac_divergent`` keeps its prior meaning. Pure;
    never raises.
    """
    map_a = map_a if isinstance(map_a, dict) else {}
    passes = [p for p in (c_passes or []) if isinstance(p, dict)]
    counts: list = []
    a_ids = set(map_a)
    for p in passes:
        n = 0
        for tid in (a_ids & set(p)):
            ra = map_a.get(tid) or {}
            rc = p.get(tid) or {}
            # score-clean for this id on this pass, but the completion bytes differ.
            if (not _invariant_fields_differ(ra, rc)) and \
                    content_sha(_result_content(ra)) != content_sha(_result_content(rc)):
                n += 1
        counts.append(n)
    return counts


# ---------------------------------------------------------------------------
# compute_divergence_report -- the full (NON fail-fast) per-test_id profile.
# ---------------------------------------------------------------------------
def compute_divergence_report(
    map_x: dict,
    map_y: dict,
    pair_label: str,
    *,
    logit_drift_eps: float = DEFAULT_LOGIT_DRIFT_EPS,
) -> dict:
    """Full (NON fail-fast) per-test_id divergence profile between two arm maps.

    ``map_x`` / ``map_y`` are ``{test_id: result_dict}``. Unlike
    ``concurrent_dispatch.assert_score_invariant`` (which raises on the FIRST
    mismatch), this walks EVERY shared test_id and records EVERY divergent field, so
    a reviewer sees the complete severity profile (``by_family``, ``by_fill``,
    token-only-vs-score). Pure; never raises.

    A test_id is "divergent" if ANY field in ``INVARIANT_FIELDS`` differs between the
    two maps OR the test_id is present in only one map. ``token_identical`` records
    whether the two sides' completion content sha matches -- it is RECORDED, NOT a
    divergence trigger by itself (token-only divergence => AMBER, decided in
    :func:`decide_status`, not here).

    Returns a dict with this exact shape::

        {
          "pair": str,
          "n_compared": int,            # shared test_ids actually compared
          "n_divergent": int,           # shared test_ids with >=1 INVARIANT_FIELD diff
          "divergence_rate": float,     # n_divergent / max(1, n_compared)
          "only_x": [test_id, ...],     # ids missing from map_y (sorted)
          "only_y": [test_id, ...],     # ids missing from map_x (sorted)
          "divergent_ids": [test_id, ...],   # sorted
          "per_id": [                   # ONE entry per (divergent test_id, divergent field)
            {"test_id", "field", "x", "y",
             "x_content_sha", "y_content_sha", "token_identical"}, ...
          ],
          "n_token_only": int,          # shared+score-clean ids whose content sha differs
          "token_divergence_ids": [test_id, ...],  # sorted; score-equal, content-different
          "by_family": {fam: {"n": int, "divergent": int}},
          "by_fill":   {fill: int},     # divergent count per fill bucket
          "logit_drift": {              # corroboration summary (ADDITIVE; never gates here)
            "armed": bool, "eps": float, "max_logit_delta": float,
            "logit_drift_ids": [test_id, ...], "per_id_delta": {test_id: float},
          },
        }

    Notes:
      * ``only_x``/``only_y`` ids are counted as divergent (a lost/extra id is a
        divergence) but cannot populate ``per_id`` (no counterpart to diff a field
        against); they ARE listed in ``divergent_ids`` and counted in ``n_divergent``.
      * ``per_id`` emits one row PER divergent INVARIANT_FIELD, not one per id, so a
        test_id that diverges on both ``score`` and ``failure_mode`` yields two rows
        (full profile, not first-mismatch).
      * ``logit_drift`` is a forensic corroboration summary computed from optional
        per-id logprob fields (:data:`LOGIT_SUMMARY_FIELDS`). It is RECORDED, never a
        divergence trigger here (logprobs are VOLATILE). With the default
        ``logit_drift_eps == 0.0`` (or no logprob fields) ``logit_drift_ids`` is empty
        and ``armed`` is False. ``decide_status`` decides whether a drifted, SCORE-clean
        id becomes an AMBER subtype (only when eps>0).
    """
    map_x = map_x if isinstance(map_x, dict) else {}
    map_y = map_y if isinstance(map_y, dict) else {}

    ids_x = set(map_x)
    ids_y = set(map_y)
    only_x = sorted(ids_x - ids_y)
    only_y = sorted(ids_y - ids_x)
    shared = sorted(ids_x & ids_y)

    per_id: list[dict] = []
    divergent_ids: set = set(only_x) | set(only_y)
    token_divergence_ids: set = set()

    # by_family / by_fill aggregates are built over the SHARED ids using whichever
    # side carries the family/fill metadata (they are arm-invariant by construction;
    # prefer map_x, fall back to map_y).
    by_family: dict = {}
    by_fill: dict = {}

    for tid in shared:
        rx = map_x.get(tid) or {}
        ry = map_y.get(tid) or {}

        fam = _family_key(rx) if rx.get("family") is not None else _family_key(ry)
        fill = _fill_key(rx) if rx.get("fill_ratio") is not None else _fill_key(ry)
        fam_bucket = by_family.setdefault(fam, {"n": 0, "divergent": 0})
        fam_bucket["n"] += 1
        by_fill.setdefault(fill, 0)

        sha_x = content_sha(_result_content(rx))
        sha_y = content_sha(_result_content(ry))
        token_identical = (sha_x == sha_y)

        id_diverged = False
        for fld in INVARIANT_FIELDS:
            xv = rx.get(fld)
            yv = ry.get(fld)
            if xv != yv:
                id_diverged = True
                per_id.append({
                    "test_id": tid,
                    "field": fld,
                    "x": xv,
                    "y": yv,
                    "x_content_sha": sha_x,
                    "y_content_sha": sha_y,
                    "token_identical": token_identical,
                })

        if id_diverged:
            divergent_ids.add(tid)
            fam_bucket["divergent"] += 1
            by_fill[fill] = by_fill.get(fill, 0) + 1
        elif not token_identical:
            # Score-clean for this id, but the completion text differs -> token-only
            # divergence (the AMBER signal). Recorded here; classified in decide_status.
            token_divergence_ids.add(tid)

    n_compared = len(shared)
    n_divergent = len(divergent_ids)
    # Per-id logit-summary drift over the SHARED ids (ADDITIVE; default-off). Inert
    # (armed=False, empty drift ids) unless eps>0 AND both sides carry a logprob summary.
    logit_drift = compute_logit_drift_ids(map_x, map_y, logit_drift_eps)
    return {
        "pair": str(pair_label),
        "n_compared": n_compared,
        "n_divergent": n_divergent,
        "divergence_rate": (n_divergent / n_compared) if n_compared else 0.0,
        "only_x": only_x,
        "only_y": only_y,
        "divergent_ids": sorted(divergent_ids),
        "per_id": per_id,
        "n_token_only": len(token_divergence_ids),
        "token_divergence_ids": sorted(token_divergence_ids),
        "by_family": by_family,
        "by_fill": by_fill,
        "logit_drift": logit_drift,
    }


# ---------------------------------------------------------------------------
# decide_status -- the GREEN / RED / AMBER / UNVERIFIED rule (first match wins).
# ---------------------------------------------------------------------------
def decide_status(
    report_ac: dict,
    report_ab: dict,
    report_bc: dict,
    *,
    client_overlap_depth: int,
    server_busy_slots: int,
    n_ok_arm_a: int | None = None,
    n_ok_arm_c: int | None = None,
    min_ok_fraction: float = MIN_OK_FRACTION,
    arm_a_map: dict | None = None,
    arm_c_map: dict | None = None,
    cobatch_coverage: dict | None = None,
    min_coverage_fraction: float = MIN_COBATCH_COVERAGE_FRACTION,
    logit_drift_eps: float = DEFAULT_LOGIT_DRIFT_EPS,
) -> dict:
    """Apply the GREEN/RED/AMBER/UNVERIFIED rule over the three pairwise reports.

    Inputs are the three reports from :func:`compute_divergence_report`:
      * ``report_ac`` = (ARM_A, ARM_C)  -- **THE GATE** (sequential reference vs batched)
      * ``report_ab`` = (ARM_A, ARM_B)  -- slot-allocation isolation
      * ``report_bc`` = (ARM_B, ARM_C)  -- pure co-batching isolation
    plus the observed overlap (``client_overlap_depth`` from the request-interval
    intersection, ``server_busy_slots`` from the real ``/slots`` poll / mock
    ``/slots-debug``), and OPTIONALLY the count of genuinely-scored ('ok') trials in
    ARM_A / ARM_C (``n_ok_arm_a`` / ``n_ok_arm_c``) for the COMPLETION-FLOOR gate.

    COVERAGE (per-id co-batching attribution) -- OPT-IN, FAIL-CLOSED. The peak overlap
    scalars above prove only that ≥2 requests were co-resident at SOME instant; they do
    NOT prove WHICH ids shared a forward pass. Supply ``arm_a_map``/``arm_c_map`` (the
    per-arm ``{test_id: result}`` maps carrying the ``dispatch_ts``/``complete_ts``
    request-interval contract, ARM_C being the UNION map) -- or a precomputed
    ``cobatch_coverage`` dict from :func:`compute_cobatch_coverage` -- and a GREEN then
    ALSO requires real per-id co-batch coverage (>= ``min_coverage_fraction`` of ids
    actually co-batched, scaled with N), with the server peak only a corroborating floor
    (Row 2). When coverage is UNCERTAIN (any id missing/None ts) the gate FAILS CLOSED to
    ``green_unverified`` (never green). The gate is INERT when no maps are supplied OR the
    maps predate the ts contract (no ``dispatch_ts`` key on any id) -- so every legacy
    caller / pure-report unit test is preserved byte-for-byte.

    LOGIT DRIFT (corroboration) -- OPT-IN via ``logit_drift_eps > 0`` (default 0.0). When
    armed AND a score-clean / token-only id's per-id logit summary moved by more than eps,
    that id is surfaced as a first-class AMBER SUBTYPE ('token AND logit drift');
    token-only-but-logit-stable stays the weaker AMBER. Logprobs are VOLATILE -- this can
    only ever make a verdict MORE conservative (GREEN -> AMBER), NEVER RED, and NEVER
    promotes a real divergence.

    Returns::

        {"status", "divergence_class", "anomaly", "token_divergence_ids",
         "overlap_ok", "completion_floor_ok",
         "cobatch_coverage_ok", "cobatch_coverage_fraction", "coverage_checked",
         "logit_drift_ids", "reasons": [str, ...]}

    (The last five keys are ADDITIVE -- older readers ignore them; ``cobatch_coverage_ok``
    defaults True / ``coverage_checked`` False when the coverage gate is not engaged, and
    ``logit_drift_ids`` is [] unless the logit gate is armed.)

    Decision precedence (FIRST MATCH WINS) -- the truth table:

      1. AC.n_divergent > 0                       -> 'failed'  (RED)
            divergence_class = 'co_batching'            if AB clean
                             = 'slot_allocation(+co_batching)' if AB also dirty
      2. AC clean BUT not overlap_ok              -> 'green_unverified'
            (false-GREEN-by-non-execution: the arm never actually co-batched, so a
             "pass" proves nothing -- treated as RED for promotion)
      2b. AC clean, overlap_ok, BUT too few REAL completions ('ok' trials) in either
            ARM_A or ARM_C (< min_ok_fraction of n_compared)  -> 'green_unverified'
            (false-GREEN-by-all-failure: two arms that BOTH fail identically compare
             EQUAL on INVARIANT_FIELDS -- "matching nothing" must not read as
             "invariant something". Skipped when the ok-counts are not supplied.)
      2c. AC clean, overlap_ok, floor met, BUT coverage CHECKED and not cobatch_coverage_ok
            -> 'green_unverified'
            (false-GREEN-by-peak-only: the peak scalar said "≥2 co-resident somewhere" but
             per-id request intervals show too few ids ACTUALLY co-batched -- or the
             attribution is uncertain. Skipped when coverage is not engaged.)
      3. AC clean, overlap_ok, floor met, coverage met, BUT AB.n_divergent>0
            -> 'green_with_caveat'  anomaly = 'AC_agree_B_disagrees'
      4. AC clean, overlap_ok, floor met, coverage met, AB & BC clean, >=1 token-only
            divergence -> 'green_with_caveat'   token_divergence_ids = [...]
            (when the logit gate is armed and a token-only id ALSO drifted in logit, the
             anomaly is upgraded to 'token_and_logit_drift'.)
      5. AC clean, overlap_ok, floor met, coverage met, AB & BC clean, token-identical
            -> 'green'

    The completion-floor (Row 2b) and coverage (Row 2c) gates are the SAFE direction:
    when in doubt they over-report as ``green_unverified`` (non-promotable). Both are
    OPT-IN: when their inputs are absent the legacy truth table applies unchanged. The
    driver ALWAYS supplies the completion counts; it supplies the coverage maps once it
    opts in (the per-id ts contract is already stamped on its result dicts).

    Pure; no I/O. Tolerant of malformed/empty reports (treats missing counts as 0).
    """
    def _nd(rep: dict) -> int:
        try:
            return int((rep or {}).get("n_divergent", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _tok_ids(rep: dict) -> list:
        ids = (rep or {}).get("token_divergence_ids", []) or []
        return list(ids) if isinstance(ids, (list, tuple, set)) else []

    ac_div = _nd(report_ac)
    ab_div = _nd(report_ab)
    bc_div = _nd(report_bc)

    try:
        depth = int(client_overlap_depth)
    except (TypeError, ValueError):
        depth = 0
    try:
        busy = int(server_busy_slots)
    except (TypeError, ValueError):
        busy = 0

    overlap_ok = (depth >= MIN_CLIENT_OVERLAP_DEPTH) and (busy >= MIN_SERVER_BUSY_SLOTS)

    # Completion-floor evaluation (Row 2b). Only ENFORCED when both ok-counts are
    # supplied (the driver always supplies them; pure-report unit tests do not, so
    # they keep the legacy behavior). The floor is computed against the number of
    # SHARED test_ids actually compared in the AC gate (n_compared) -- that is the
    # universe the GREEN would certify. A required count of 0 (e.g. n_compared==0)
    # makes the floor trivially satisfiable, which is fine: the empty-AC case is
    # already demoted by the overlap gate (Row 2) since an empty run cannot co-batch.
    try:
        ac_n_compared = int((report_ac or {}).get("n_compared", 0) or 0)
    except (TypeError, ValueError):
        ac_n_compared = 0
    try:
        frac = float(min_ok_fraction)
    except (TypeError, ValueError):
        frac = MIN_OK_FRACTION
    required_ok = int(math.ceil(frac * ac_n_compared)) if ac_n_compared > 0 else 0
    floor_checked = (n_ok_arm_a is not None) and (n_ok_arm_c is not None)
    if floor_checked:
        try:
            n_ok_a = int(n_ok_arm_a)
        except (TypeError, ValueError):
            n_ok_a = 0
        try:
            n_ok_c = int(n_ok_arm_c)
        except (TypeError, ValueError):
            n_ok_c = 0
        completion_floor_ok = (n_ok_a >= required_ok) and (n_ok_c >= required_ok)
    else:
        n_ok_a = None
        n_ok_c = None
        completion_floor_ok = True   # not enforced when counts absent

    # ---- Co-batching COVERAGE evaluation (Row 2c). OPT-IN + FAIL-CLOSED. ----
    # Engaged when a precomputed coverage dict OR the per-arm maps are supplied. The maps
    # carry the per-id dispatch_ts/complete_ts contract; ARM_C should be the UNION map the
    # gate certifies. Computed coverage is INERT (checked=False) on maps that predate the
    # ts contract (no dispatch_ts key) so legacy callers / pure-report tests are unchanged.
    if isinstance(cobatch_coverage, dict):
        _cov = cobatch_coverage
    elif arm_c_map is not None:
        _cov = compute_cobatch_coverage(arm_c_map, min_coverage_fraction=min_coverage_fraction)
    else:
        _cov = None
    coverage_checked = bool(_cov.get("checked")) if isinstance(_cov, dict) else False
    if coverage_checked:
        cobatch_coverage_ok = bool(_cov.get("cobatch_coverage_ok", False))
        cobatch_coverage_fraction = _cov.get("coverage_fraction")
    else:
        cobatch_coverage_ok = True            # not enforced when coverage absent/unchecked
        cobatch_coverage_fraction = None

    # ---- Logit-drift corroboration (OFF unless logit_drift_eps>0). ----
    # The AC report already computed the per-id logit drift (when armed); read it back so
    # the verdict need not recompute. A token-only id that ALSO drifted in logit is the
    # stronger AMBER subtype. Inert (empty) under the default eps=0.0.
    _ac_logit = (report_ac or {}).get("logit_drift") if isinstance(report_ac, dict) else None
    if not (isinstance(_ac_logit, dict) and _ac_logit.get("armed")):
        # report_ac may have been built without eps (e.g. hand-wired); recompute from maps
        # if available and the caller armed the gate here.
        try:
            _eps = float(logit_drift_eps)
        except (TypeError, ValueError):
            _eps = 0.0
        if _eps > 0.0 and arm_a_map is not None and arm_c_map is not None:
            _ac_logit = compute_logit_drift_ids(arm_a_map, arm_c_map, _eps)
    logit_drift_ids = (
        list(_ac_logit.get("logit_drift_ids", [])) if isinstance(_ac_logit, dict) else []
    )
    logit_gate_armed = bool(_ac_logit.get("armed")) if isinstance(_ac_logit, dict) else False

    # Additive fields stamped on EVERY return block (older readers ignore them). Built once
    # so the legacy rows keep identical behaviour while carrying the new diagnostics.
    def _extra() -> dict:
        return {
            "completion_floor_ok": completion_floor_ok,
            "coverage_checked": coverage_checked,
            "cobatch_coverage_ok": cobatch_coverage_ok,
            "cobatch_coverage_fraction": cobatch_coverage_fraction,
            "logit_drift_ids": list(logit_drift_ids),
        }

    reasons: list[str] = []
    reasons.append(
        f"AC.n_divergent={ac_div}; AB.n_divergent={ab_div}; BC.n_divergent={bc_div}"
    )
    reasons.append(
        f"client_overlap_depth={depth}"
        f"(>= {MIN_CLIENT_OVERLAP_DEPTH}?{depth >= MIN_CLIENT_OVERLAP_DEPTH}); "
        f"server_busy_slots={busy}(>= {MIN_SERVER_BUSY_SLOTS}?{busy >= MIN_SERVER_BUSY_SLOTS}); "
        f"overlap_ok={overlap_ok}"
    )
    if floor_checked:
        reasons.append(
            f"completion_floor: n_ok_A={n_ok_a}, n_ok_C={n_ok_c}, "
            f"required>={required_ok} (>= {frac:.2f} of n_compared={ac_n_compared}); "
            f"completion_floor_ok={completion_floor_ok}"
        )
    if coverage_checked and isinstance(_cov, dict):
        reasons.append(
            f"cobatch_coverage: n_co_batched={_cov.get('n_co_batched')}/"
            f"{_cov.get('n_ids')} (fraction={_cov.get('coverage_fraction'):.3f}, "
            f"required>={_cov.get('required_co_batched')} @ "
            f">= {_cov.get('min_coverage_fraction'):.2f}); "
            f"attribution_certain={_cov.get('attribution_certain')}; "
            f"missing_interval={_cov.get('n_missing_interval')}; "
            f"server_peak_busy_slots={busy} (corroborating floor only); "
            f"cobatch_coverage_ok={cobatch_coverage_ok}"
        )
    if logit_gate_armed:
        reasons.append(
            f"logit_drift gate ARMED (eps={_ac_logit.get('eps')}): "
            f"max_delta={_ac_logit.get('max_logit_delta')}, "
            f"drift_ids={logit_drift_ids}"
        )

    # ---- Row 1: AC score-level divergence -> RED (highest precedence). ----
    if ac_div > 0:
        if ab_div > 0:
            divergence_class = DIVCLASS_SLOT_ALLOC
            reasons.append(
                "AC diverges AND AB diverges -> the N-slot server itself differs from "
                "the sequential reference (slot allocation), compounded by co-batching."
            )
        else:
            divergence_class = DIVCLASS_CO_BATCHING
            reasons.append(
                "AC diverges but AB is clean -> divergence is pure continuous-batching "
                "(co-batching), not slot allocation."
            )
        return {
            "status": STATUS_FAILED,
            "divergence_class": divergence_class,
            "anomaly": None,
            "token_divergence_ids": _tok_ids(report_ac),
            "overlap_ok": overlap_ok,
            "reasons": reasons,
            **_extra(),
        }

    # ---- Row 2: AC clean but no real co-batching observed -> UNVERIFIED. ----
    if not overlap_ok:
        reasons.append(
            "AC is score-clean BUT real co-batching was NOT observed "
            "(overlap below threshold). A pass with no overlap is not reference-grade "
            "(false-GREEN-by-non-execution) -> green_unverified."
        )
        return {
            "status": STATUS_GREEN_UNVERIFIED,
            "divergence_class": None,
            "anomaly": None,
            "token_divergence_ids": _tok_ids(report_ac),
            "overlap_ok": overlap_ok,
            "reasons": reasons,
            **_extra(),
        }

    # ---- Row 2b: AC clean + overlap_ok but too few REAL completions -> UNVERIFIED. ----
    # The structural twin of the canned-mock vacuity: two arms that BOTH fail
    # identically (all http_timeout / all empty under contention) compare EQUAL on
    # INVARIANT_FIELDS, so AC looks "clean" and overlap_ok can be satisfied by a
    # stuck-but-busy slot -- yet ZERO real completions were batched. Demote to
    # green_unverified (non-promotable). Only fires when the ok-counts were supplied.
    if floor_checked and not completion_floor_ok:
        reasons.append(
            "AC is score-clean and overlap is real, BUT too few trials genuinely "
            f"completed ('ok'): n_ok_A={n_ok_a}, n_ok_C={n_ok_c}, required>={required_ok} "
            f"(>= {frac:.2f} of {ac_n_compared}). Most trials are http_timeout/empty/"
            "arm_deadline failures, which compare EQUAL across arms but certify NOTHING "
            "about batched real outputs (false-GREEN-by-all-failure) -> green_unverified."
        )
        return {
            "status": STATUS_GREEN_UNVERIFIED,
            "divergence_class": None,
            "anomaly": None,
            "token_divergence_ids": _tok_ids(report_ac),
            "overlap_ok": overlap_ok,
            "reasons": reasons,
            **_extra(),
        }

    # ---- Row 2c: AC clean + overlap_ok + floor met, but per-id co-batch coverage
    # CHECKED and insufficient/uncertain -> UNVERIFIED. The peak scalar said "≥2
    # co-resident at some instant", but the request-interval attribution shows too few
    # ids ACTUALLY co-batched (or an id's interval is missing -> attribution uncertain).
    # FAIL CLOSED to green_unverified (non-promotable) -- the peak alone is not a per-cell
    # co-batching proof. Only fires when coverage was engaged (maps / precomputed dict
    # supplied AND the ts contract is present). The safe direction. ----
    if coverage_checked and not cobatch_coverage_ok:
        reasons.append(
            "AC is score-clean, overlap PEAK is real, and the completion-floor is met, "
            "BUT per-id request-interval coverage is insufficient/uncertain: only "
            f"{(_cov or {}).get('n_co_batched')} of {(_cov or {}).get('n_ids')} ids "
            f"actually co-batched (required >= {(_cov or {}).get('required_co_batched')}; "
            f"attribution_certain={(_cov or {}).get('attribution_certain')}). The server "
            "peak proves SOMETHING was co-resident, not WHICH ids shared a pass, so a "
            "GREEN earned on the peak alone could certify a cell that mostly ran serially "
            "(false-GREEN-by-peak-only) -> green_unverified."
        )
        return {
            "status": STATUS_GREEN_UNVERIFIED,
            "divergence_class": None,
            "anomaly": None,
            "token_divergence_ids": _tok_ids(report_ac),
            "overlap_ok": overlap_ok,
            "reasons": reasons,
            **_extra(),
        }

    # ---- Row 3: AC clean + overlap_ok but AB dirty -> AMBER (coincidental agreement). ----
    if ab_div > 0:
        reasons.append(
            "AC agrees and overlap is real, BUT AB diverges (the N-slot serial arm "
            "differs from the sequential reference). A and C agree only by coincidence "
            "-> green_with_caveat (operator sign-off)."
        )
        return {
            "status": STATUS_GREEN_WITH_CAVEAT,
            "divergence_class": None,
            "anomaly": ANOMALY_AC_AGREE_B_DISAGREES,
            "token_divergence_ids": _tok_ids(report_ac),
            "overlap_ok": overlap_ok,
            "reasons": reasons,
            **_extra(),
        }

    # ---- Row 4: AC clean + overlap_ok + AB/BC clean but token-only divergence -> AMBER. ----
    ac_token_ids = _tok_ids(report_ac)
    if ac_token_ids:
        # Logit subtype: when the logit gate is ARMED and a token-only id ALSO drifted in
        # its logit summary, the caveat is the STRONGER 'token_and_logit_drift' (tokens AND
        # logits moved). token-only-but-logit-stable keeps the plain (None-anomaly) AMBER.
        # Logit drift NEVER promotes RED -- it only sharpens the AMBER label.
        token_and_logit = sorted(set(ac_token_ids) & set(logit_drift_ids))
        _anom = ANOMALY_TOKEN_AND_LOGIT_DRIFT if (logit_gate_armed and token_and_logit) else None
        if _anom:
            reasons.append(
                "AC scores match and overlap is real, BUT the completion content differs "
                f"on {len(ac_token_ids)} id(s) AND the logit summary drifted (> eps) on "
                f"{len(token_and_logit)} of them ({token_and_logit}) -> green_with_caveat "
                "(token AND logit drift; stronger sign-off)."
            )
        else:
            reasons.append(
                "AC scores match and overlap is real, BUT the completion content differs "
                f"on {len(ac_token_ids)} id(s) (token-only divergence). Scores are stable "
                "but the bytes are not -> green_with_caveat (operator sign-off)."
                + (" Logits within eps (token-only, logit-stable)." if logit_gate_armed else "")
            )
        # NOTE: BC token-divergence with AB/BC score-clean is not required to gate; we
        # surface AC's token ids (the gate pair). If BC is dirty here it is covered by
        # the AC/AB rows above (BC can only diverge in score if AC or AB did).
        return {
            "status": STATUS_GREEN_WITH_CAVEAT,
            "divergence_class": None,
            "anomaly": _anom,
            "token_divergence_ids": ac_token_ids,
            "overlap_ok": overlap_ok,
            "reasons": reasons,
            **_extra(),
        }

    # ---- Row 5: the strongest pass -> GREEN (promotable when source=='live'). ----
    reasons.append(
        "AC score-identical, overlap real, completion-floor met, "
        + ("per-id co-batch coverage met, " if coverage_checked else "")
        + "AB/BC clean, content byte-identical -> green (promotable iff source=='live')."
    )
    return {
        "status": STATUS_GREEN,
        "divergence_class": None,
        "anomaly": None,
        "token_divergence_ids": [],
        "overlap_ok": overlap_ok,
        "reasons": reasons,
        **_extra(),
    }


# ---------------------------------------------------------------------------
# build_cert -- assemble the full cert PAYLOAD (superset of write_cert_artifact).
# ---------------------------------------------------------------------------
def build_cert(
    *,
    model: str,
    ctx: int,
    dispatch_n: int,
    decision: dict,
    report_ac: dict,
    report_ab: dict,
    report_bc: dict,
    overlap: dict,
    source: str,                     # MUST be 'live' for a promotable real run
    reps: int,
    kv_label: str = "q8_0",
    ts_utc: str | None = None,
    n_passes: int = 1,
    per_pass_ac_divergent: list | None = None,
    completion_counts: dict | None = None,
    gate: str = "abc_union",
    cobatch_coverage: dict | None = None,
    n_hot_trials: int | None = None,
    ctx_sweep: dict | None = None,
    per_pass_ac_token_only: list | None = None,
    dense_smoke_model: str | None = None,
) -> dict:
    """Assemble the full cert PAYLOAD dict (NOT written here -- see persist_cert).

    The returned dict is a SUPERSET of ``concurrent_dispatch.write_cert_artifact``'s
    payload: it carries the same ``CERT_REQUIRED_FIELDS`` (model, ctx, dispatch_n,
    status, source) plus the gate-specific keys (overlap, overlap_ok, gate,
    completion_floor, divergence_report, arm_b, divergence_class, anomaly,
    token_divergence_ids, n_passes, per_pass_ac_divergent). ``decision['status']`` maps
    to the cert ``status``; the ``STATUS_GREEN_*`` variants are written verbatim.
    ``n_passes`` is T (the number of concurrent ARM_C re-passes; the AC report is the
    UNION over them) and ``per_pass_ac_divergent`` is the per-pass AC SCORE-divergent-id
    count for the audit trail. ``per_pass_ac_token_only`` (optional, additive) is the
    per-pass TOKEN-ONLY-divergent-id count (the AMBER companion); it defaults to [] so the
    schema is unchanged when a caller does not supply it.

    ``completion_counts`` (optional) is ``{"a": count_ok_completions(map_a),
    "c": count_ok_completions(map_c)}`` (and optionally "b"); its n_ok / n_failure are
    stamped under ``completion_floor`` so a reviewer / consumer can see how many trials
    GENUINELY scored vs failed (the all-failure-vacuity audit). ``gate`` records WHICH
    producer minted the cert ('abc_union' = the rigorous A/B/C-union driver) so a
    consumer can require that gate before promoting. Both ``gate`` and a top-level
    ``overlap_ok`` mirror are written so the promotion gate can fold in the co-batching
    proof + producer identity without reaching into nested blocks.

    ``cobatch_coverage`` (optional) is the per-id co-batching attribution census from
    :func:`compute_cobatch_coverage`; its ``cobatch_coverage_ok`` (mirrored from the
    decision) is folded in by :func:`is_promotable`, which FAILS CLOSED only when the block
    is CHECKED and false (a benign unchecked block -- the current driver, the test fixtures --
    leaves promotion unchanged). ``n_hot_trials`` and ``ctx_sweep`` stamp the rule-of-three
    95% divergence-rate upper bound + an optional per-ctx sweep grid; both are diagnostics,
    never gates. All are ADDITIVE -- older readers ignore them.

    ``dense_smoke_model`` (optional) names a model alias that should be treated as the
    "dense smoke" baseline (the least divergence-prone case): when the tested ``model``
    equals it, the promotion_scope flag ``is_dense_smoke_model`` is True. Defaults to None
    (the flag is then False unless the model itself trips no MoE hint), keeping the
    mechanism without baking any specific model name into the generic core.

    ``cert_is_green`` (unchanged) only reads ``status`` and ``source``, so:
      * STATUS_GREEN (literal "green") + source=='live'  -> cert_is_green True
      * STATUS_GREEN_UNVERIFIED / STATUS_GREEN_WITH_CAVEAT -> NOT the literal "green"
        -> cert_is_green False (the desired "not auto-promotable" semantics)
    The extra keys are ignored by cert_is_green and by old readers.

    Pure (builds a dict only); ``ts_utc`` left as-is (None -> persist_cert stamps it).
    Tolerant of malformed ``decision``/reports/overlap (defaults applied).
    """
    decision = decision or {}
    status = decision.get("status", STATUS_FAILED)

    overlap = overlap or {}
    _overlap_ok = bool(overlap.get("overlap_ok", decision.get("overlap_ok", False)))
    overlap_block = {
        "client_max_overlap_depth": int(overlap.get("client_max_overlap_depth", 0) or 0),
        "server_peak_busy_slots": int(overlap.get("server_peak_busy_slots", 0) or 0),
        "overlap_ok": _overlap_ok,
    }

    # Completion-floor census (the all-failure-vacuity audit). Defaults to a benign
    # "0 known / floor not evaluated" block when counts are not supplied (legacy
    # callers / pure-report tests) so the schema is stable.
    completion_counts = completion_counts or {}
    _cc_a = completion_counts.get("a") or {}
    _cc_b = completion_counts.get("b") or {}
    _cc_c = completion_counts.get("c") or {}
    completion_floor = {
        "n_ok_a": int(_cc_a.get("n_ok", 0) or 0),
        "n_failure_a": int(_cc_a.get("n_failure", 0) or 0),
        "n_ok_b": int(_cc_b.get("n_ok", 0) or 0),
        "n_failure_b": int(_cc_b.get("n_failure", 0) or 0),
        "n_ok_c": int(_cc_c.get("n_ok", 0) or 0),
        "n_failure_c": int(_cc_c.get("n_failure", 0) or 0),
        "min_ok_fraction": float(MIN_OK_FRACTION),
        # True iff the decision evaluated the floor AND it passed. decide_status writes
        # completion_floor_ok=True when the floor was not enforced (counts absent), so a
        # cert from a legacy path reads floor_ok True but with zero n_ok stamped -- that
        # is fine: a legacy cert is gated by its weaker 'gate' tag instead.
        "completion_floor_ok": bool(decision.get("completion_floor_ok", True)),
    }

    ab_passed = (int((report_ab or {}).get("n_divergent", 0) or 0) == 0)
    bc_passed = (int((report_bc or {}).get("n_divergent", 0) or 0) == 0)

    # PROMOTION SCOPE (additive; ignored by cert_is_green). A GREEN here licenses ONLY
    # the EXACT (model, ctx, N) tested -- never another ctx, N, or model. Three reasons
    # this must be explicit on the artifact:
    #   * a dense model is the LEAST divergence-prone case. MoE models route experts per
    #     token, and batch composition changes which experts fire -- the textbook #7052
    #     regime. A dense GREEN says NOTHING about MoE expert-routing divergence.
    #   * continuous-batching divergence GROWS with per-slot KV width (high ctx) and
    #     slot count (high N). A low-ctx/low-N GREEN does not generalize UP to the
    #     divergence-maximizing cells.
    #   * the cost is asymmetric: a RED costs ~nothing (the safe default is already
    #     --parallel 1), a false GREEN corrupts downstream trust. So the safe default is
    #     sequential, and batched is opt-in PER-CELL behind its own live cert.
    _model_l = str(model).lower()
    _looks_moe = any(t in _model_l for t in _MOE_NAME_HINTS)
    _is_dense_smoke = (dense_smoke_model is not None) and (
        _model_l == str(dense_smoke_model).lower()
    )
    promotion_scope = {
        "valid_for": {"model": str(model), "ctx": int(ctx), "dispatch_n": int(dispatch_n)},
        "generalizes": False,
        "note": ("This cert licenses ONLY the exact (model, ctx, N) above. It does NOT "
                 "generalize across ctx, N, or model. A dense model is the least "
                 "divergence-prone case; a GREEN on one says NOTHING about MoE "
                 "expert-routing divergence or about higher-ctx / higher-N cells, where "
                 "#7052 divergence is most likely. Re-run the live gate per (model, ctx, "
                 "N) before promoting any batched cell; on ANY non-green keep that cell "
                 "--parallel 1."),
        "is_dense_smoke_model": bool(_is_dense_smoke),
        "tested_model_looks_moe": bool(_looks_moe),
    }

    # ---- cobatch_coverage block (additive; the per-id co-batching attribution audit +
    # the cert-level gate is_promotable folds in). PRECEDENCE: when a full coverage dict is
    # EXPLICITLY supplied it is the AUTHORITATIVE per-id census (so the cert block -- and
    # therefore is_promotable -- is a true backstop even if decide_status was bypassed /
    # computed without coverage). When no dict is supplied we fall back to the decision's
    # mirrored cobatch_coverage_ok / coverage_checked (the driver path where decide_status
    # already folded the coverage in). When NEITHER is present the block is a benign "not
    # enforced" (cobatch_coverage_ok True, coverage_checked False) so a strong-gate cert
    # minted WITHOUT coverage (the current driver, the test fixtures) stays promotable
    # exactly as before. ----
    _cov = cobatch_coverage if isinstance(cobatch_coverage, dict) else {}
    _cov_supplied = bool(_cov)
    if _cov_supplied:
        _cov_checked = bool(_cov.get("checked", False))
        _cov_ok = bool(_cov.get("cobatch_coverage_ok", True))
        _cov_fraction = _cov.get("coverage_fraction")
    else:
        _cov_checked = bool(decision.get("coverage_checked", False))
        _cov_ok = bool(decision.get("cobatch_coverage_ok", True))
        _cov_fraction = decision.get("cobatch_coverage_fraction")
    cobatch_coverage_block = {
        "coverage_checked": _cov_checked,
        "cobatch_coverage_ok": _cov_ok,
        "coverage_fraction": _cov_fraction,
        "n_ids": int(_cov.get("n_ids", 0) or 0),
        "n_co_batched": int(_cov.get("n_co_batched", 0) or 0),
        "n_missing_interval": int(_cov.get("n_missing_interval", 0) or 0),
        "attribution_certain": bool(_cov.get("attribution_certain", False)),
        "required_co_batched": int(_cov.get("required_co_batched", 0) or 0),
        "min_coverage_fraction": float(_cov.get("min_coverage_fraction",
                                                MIN_COBATCH_COVERAGE_FRACTION)),
        # The peak scalar is a CORROBORATING floor only (the primary signal is per-id
        # request-interval overlap above). Mirrored here for the audit trail.
        "server_peak_busy_slots_floor": int(overlap_block["server_peak_busy_slots"]),
    }

    # ---- sensitivity block (additive). Stamps how many HOT (genuinely-scored, 'ok')
    # trials underwrote a GREEN and the rule-of-three 95% upper bound on the true
    # divergence rate when ZERO divergences were observed (3 / n_hot_trials). A reviewer
    # MUST read a GREEN as "no divergence DETECTED at this (model,ctx,N,T,K)", per-cell --
    # it does NOT generalize and it is only as strong as the trial count behind it.
    # n_hot_trials defaults to the ARM_C 'ok' census (n_ok_c) when not supplied. ----
    if n_hot_trials is not None:
        try:
            _n_hot = max(0, int(n_hot_trials))
        except (TypeError, ValueError):
            _n_hot = 0
    else:
        _n_hot = int(_cc_c.get("n_ok", 0) or 0)
    _ac_div_for_rate = int((report_ac or {}).get("n_divergent", 0) or 0)
    rate_upper_bound_95 = (3.0 / _n_hot) if (_n_hot > 0 and _ac_div_for_rate == 0) else None
    sensitivity = {
        "n_hot_trials": _n_hot,
        "n_ac_divergent": _ac_div_for_rate,
        # rule-of-three: with 0 observed divergences in n_hot trials, the true rate is
        # <= 3/n_hot at ~95% confidence. None when divergences WERE seen (status would be
        # 'failed') or no hot trials underwrote the cell.
        "rate_upper_bound_95": rate_upper_bound_95,
        "interpretation": ("GREEN means 'no divergence DETECTED at this (model,ctx,N,T,K)', "
                           "per-cell; it does NOT generalize and is only as strong as "
                           "n_hot_trials (rule-of-three 95% upper bound 3/n_hot_trials)."),
    }

    # ---- ctx-sweep aggregation (additive; optional). A caller running the gate across
    # multiple ctx values can pass per-ctx divergence-rate cells; stamped verbatim so a
    # reviewer sees the full sweep. Each cell is per-(ctx) and still does NOT generalize
    # ACROSS the swept points (it is a grid of independent per-cell verdicts). ----
    ctx_sweep_block = ctx_sweep if isinstance(ctx_sweep, dict) else None

    # A human-readable "mismatch" summary string for the REQUIRED-fields cert (kept
    # in sync with write_cert_artifact's 'mismatch' field). None on a clean AC.
    ac_div = int((report_ac or {}).get("n_divergent", 0) or 0)
    if status == STATUS_FAILED:
        first = (report_ac or {}).get("per_id") or []
        head = first[0] if first else None
        if head:
            mismatch = (
                f"AC divergent on {ac_div} id(s); e.g. test_id={head.get('test_id')!r} "
                f"field={head.get('field')!r} A={head.get('x')!r} C={head.get('y')!r}"
            )
        else:
            only = ((report_ac or {}).get("only_x") or []) + ((report_ac or {}).get("only_y") or [])
            mismatch = f"AC id-set differs; {ac_div} divergent id(s); only={only[:5]}"
    else:
        mismatch = None

    cert = {
        # ---- CERT_REQUIRED_FIELDS (what cert_is_green / is_promotable read) ----
        "model": str(model),
        "ctx": int(ctx),
        "dispatch_n": int(dispatch_n),
        "status": str(status),
        "source": str(source),
        # ---- shared with write_cert_artifact ----
        "kv_label": str(kv_label),
        "reps": int(reps),
        "invariant_fields": list(INVARIANT_FIELDS),
        "mismatch": mismatch,
        "ts_utc": ts_utc,  # None here; persist_cert stamps a real UTC string.
        # ---- gate-specific diagnostics (additive; ignored by cert_is_green) ----
        "divergence_class": decision.get("divergence_class"),
        "anomaly": decision.get("anomaly"),
        "token_divergence_ids": list(decision.get("token_divergence_ids", []) or []),
        "overlap": overlap_block,
        # Top-level mirror of overlap.overlap_ok so the promotion gate can fold in the
        # co-batching proof WITHOUT reaching into the nested overlap block (a legacy cert
        # may have no overlap block at all; this lets the gate treat a missing/false
        # overlap_ok as NON-promotable, failing closed).
        "overlap_ok": _overlap_ok,
        # WHICH producer minted this cert. The rigorous A/B/C-union driver writes
        # 'abc_union'. Consumers that require the strong gate check this so the rigor
        # cannot be bypassed by a weaker same-filename writer.
        "gate": str(gate),
        # Completion-floor census (how many trials GENUINELY scored vs failed) so an
        # all-failure cell is auditable + the floor verdict is on the artifact.
        "completion_floor": completion_floor,
        "arm_b": {"ab_passed": bool(ab_passed), "bc_passed": bool(bc_passed)},
        # Per-(model,ctx,N) scope limitation -- a GREEN does NOT generalize. Additive +
        # ignored by cert_is_green; for the operator/consumer reviewer.
        "promotion_scope": promotion_scope,
        # T concurrent re-passes (ARM_C); the AC report above is the UNION over them
        # (any pass divergent => RED). per_pass_ac_divergent is the per-pass count of
        # AC-divergent ids for the audit trail (which interleaving surfaced it).
        "n_passes": int(n_passes or 1),
        "per_pass_ac_divergent": list(per_pass_ac_divergent or []),
        # The per-pass TOKEN-ONLY (score-clean, content-sha-divergent) audit trail -- the
        # AMBER companion to per_pass_ac_divergent. Lets a reviewer recover WHICH stochastic
        # interleaving surfaced a #7052 token-only divergence (the union folds it into one
        # representative). Defaults to [] when not supplied (legacy callers / pure-report
        # tests) so the schema is strictly additive.
        "per_pass_ac_token_only": list(per_pass_ac_token_only or []),
        # Per-id co-batching coverage audit + the cert-level gate verdict (is_promotable
        # folds this in, failing closed when CHECKED-and-false; a benign unchecked block
        # leaves promotion exactly as before). The server peak is a corroborating floor.
        "cobatch_coverage": cobatch_coverage_block,
        # How many HOT trials underwrote the verdict + the rule-of-three 95% upper bound
        # on the true divergence rate (a GREEN is "no divergence DETECTED at this cell").
        "sensitivity": sensitivity,
        # Optional per-ctx sweep aggregation (per-cell divergence-rate grid). None when
        # the caller ran a single cell.
        "ctx_sweep": ctx_sweep_block,
        "divergence_report": {
            "AC": report_ac or {},
            "AB": report_ab or {},
            "BC": report_bc or {},
        },
        "decision_reasons": list(decision.get("reasons", []) or []),
    }
    return cert


# ---------------------------------------------------------------------------
# persist_cert -- the ONLY I/O in this module (atomic; reuses write_cert_artifact).
# ---------------------------------------------------------------------------
def persist_cert(cert_dir: str, cert: dict) -> str:
    """Atomically write ``cert`` to ``{cert_dir}/cert_filename(model,ctx,N)``; return path.

    Strategy: first call ``concurrent_dispatch.write_cert_artifact`` to lay down a
    valid REQUIRED-fields cert at the canonical path (this guarantees the file is
    readable by ``load_cert`` and evaluable by ``cert_is_green`` even if the superset
    re-write were to fail), THEN atomically re-write the SAME path with the full
    superset payload using the identical tempfile+os.replace discipline. The ONLY I/O
    in this module.

    ``cert['ts_utc']`` is stamped here if None (build_cert leaves it None so the
    timestamp is the write time). Returns the path written.
    """
    import json
    import os
    import tempfile
    from datetime import datetime, timezone

    cert = dict(cert or {})
    if not cert.get("ts_utc"):
        cert["ts_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    model = cert.get("model")
    ctx = int(cert.get("ctx", 0) or 0)
    dispatch_n = int(cert.get("dispatch_n", 0) or 0)

    # Step 1: lay down a guaranteed-valid REQUIRED-fields cert (reuse the audited
    # writer). This makes the path exist + parse even if step 2 is interrupted.
    path = write_cert_artifact(
        cert_dir,
        model,
        ctx,
        dispatch_n,
        cert.get("status", STATUS_FAILED),
        source=cert.get("source", "mock"),
        kv_label=cert.get("kv_label", "q8_0"),
        n_sample=int(cert.get("reps", 0) or 0),
        mismatch=cert.get("mismatch"),
        invariant_fields=tuple(cert.get("invariant_fields", INVARIANT_FIELDS)),
        ts_utc=cert.get("ts_utc"),
    )

    # Step 2: atomically re-write the SAME file with the full superset payload, using
    # the same tempfile+os.replace discipline as write_cert_artifact so a concurrent
    # reader never sees a partial file.
    os.makedirs(cert_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cert_dir, prefix=".cert-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


# ---------------------------------------------------------------------------
# is_promotable -- the machine-readable promotion verdict.
# ---------------------------------------------------------------------------
def is_promotable(cert: dict, *, require_gate: str | None = "abc_union") -> bool:
    """True iff this cert may promote a batched cell (the promotion gate).

    Requires ALL of:
      * ``concurrent_dispatch.cert_is_green(cert, require_source='live')`` -- green + live,
      * ``overlap_ok`` is present AND true (the mandatory "real co-batching was
        observed" proof; read from the top-level mirror, falling back to the nested
        ``overlap.overlap_ok``). A cert lacking any overlap signal FAILS CLOSED -- a
        legacy cert with no overlap block is NON-promotable, never open.
      * ``completion_floor.completion_floor_ok`` is true (the all-failure-vacuity
        guard: a cell where too few trials genuinely scored does NOT promote). A cert
        with no completion_floor block is treated as floor-OK ONLY when it ALSO carries
        the strong gate tag -- a legacy cert without the block but from a weak gate is
        already rejected by ``require_gate`` below.
      * if ``require_gate`` is not None, ``gate`` equals it (default 'abc_union', the
        rigorous A/B/C-union driver) -- so a weaker producer's byte-compatible cert can
        NOT promote through the strong path. Pass ``require_gate=None`` to accept any
        producer (only for legacy/compat callers).
      * ``cobatch_coverage.cobatch_coverage_ok`` is true WHENEVER that block reports it was
        CHECKED (the per-id co-batching attribution gate). A block that was NOT checked
        (the current driver / legacy certs that predate per-id coverage) is tolerated as
        coverage-OK -- EXACTLY like the completion_floor's "absent -> not enforced" rule --
        so this is strictly additive and only makes a CHECKED-and-insufficient cert
        non-promotable. A cert that was checked but reports coverage_ok False (too few ids
        actually co-batched, or attribution uncertain) FAILS CLOSED here even if its peak
        overlap_ok is true. (peak alone is not per-cell proof.)

    ``green_unverified`` / ``green_with_caveat`` / ``failed`` all return False (the first
    two because their status string is not the literal "green"; the last because
    cert_is_green rejects it). Use THIS (not bare ``cert_is_green``) as the machine-readable
    verdict.
    """
    cert = cert or {}
    if not cert_is_green(cert, require_source="live"):
        return False
    # overlap_ok: prefer the top-level mirror; fall back to the nested block; FAIL CLOSED
    # if neither is present (a cert with no overlap signal must not promote).
    if "overlap_ok" in cert:
        overlap_ok = bool(cert.get("overlap_ok"))
    elif isinstance(cert.get("overlap"), dict) and "overlap_ok" in cert["overlap"]:
        overlap_ok = bool(cert["overlap"].get("overlap_ok"))
    else:
        overlap_ok = False
    if not overlap_ok:
        return False
    # producer gate: require the strong tag unless explicitly waived.
    if require_gate is not None and cert.get("gate") != require_gate:
        return False
    # completion floor: a present block must report floor_ok True; absence is tolerated
    # ONLY because the gate check above already rejected weak/legacy producers.
    cf = cert.get("completion_floor")
    if isinstance(cf, dict) and not bool(cf.get("completion_floor_ok", True)):
        return False
    # Per-id co-batching coverage: a block that was CHECKED must report coverage_ok
    # True; an unchecked / absent block is tolerated (mirrors completion_floor) so a
    # strong-gate cert minted WITHOUT coverage (current driver, fixtures) is unaffected.
    # The peak overlap_ok above proves SOMETHING co-batched; this proves enough of the
    # RIGHT ids did. FAIL CLOSED only on a checked-and-insufficient block.
    cov = cert.get("cobatch_coverage")
    if isinstance(cov, dict) and bool(cov.get("coverage_checked", False)) \
            and not bool(cov.get("cobatch_coverage_ok", True)):
        return False
    return True


# ---------------------------------------------------------------------------
# Convenience: drive all three pairwise reports + the decision from raw arm maps.
# Pure; lets the driver hand three maps + overlap and get back (reports, decision)
# without re-implementing the pairing order anywhere.
# ---------------------------------------------------------------------------
def diff_arms(
    arm_results: dict,
    *,
    client_overlap_depth: int,
    server_busy_slots: int,
    logit_drift_eps: float = DEFAULT_LOGIT_DRIFT_EPS,
) -> dict:
    """Diff the three arm maps and decide the status in one pure call.

    ``arm_results`` is ``{ARM_A: map_a, ARM_B: map_b, ARM_C: map_c}`` where each map
    is ``{test_id: result_dict}`` (ARM_C's map is the UNION over T concurrent
    re-passes: a test_id is divergent if it diverged on ANY pass -- the union is built
    by the driver before calling this). Computes the three pairwise reports
    (AC = the gate, AB = slot-allocation isolation, BC = co-batching isolation) and
    runs :func:`decide_status` over them.

    Returns::

        {"report_ac", "report_ab", "report_bc", "decision",
         "completion_counts", "cobatch_coverage"}

    The wrapper ALSO computes the per-id co-batching coverage from ARM_C's map and feeds
    it to :func:`decide_status`, so the SAME coverage gate the driver enforces also
    applies through this public seam (a peak-only / uncertain-attribution cell is demoted
    to green_unverified here too). On maps that predate the dispatch_ts/complete_ts
    contract (e.g. hand-built unit fixtures) the coverage is ``checked=False`` (inert), so
    the legacy report/decision behaviour is preserved byte-for-byte. ``logit_drift_eps``
    is forwarded (default 0.0 = off) for the corroboration.

    Pure; never raises (missing arms -> empty maps -> reports with n_compared=0,
    which decide_status treats as AC-clean -> green_unverified unless overlap_ok is
    forced, so a caller that loses an arm gets a non-promotable verdict, never a
    silent green).
    """
    arm_results = arm_results if isinstance(arm_results, dict) else {}
    map_a = arm_results.get(ARM_A) or {}
    map_b = arm_results.get(ARM_B) or {}
    map_c = arm_results.get(ARM_C) or {}

    report_ac = compute_divergence_report(map_a, map_c, f"{ARM_A}{ARM_C}",
                                          logit_drift_eps=logit_drift_eps)
    report_ab = compute_divergence_report(map_a, map_b, f"{ARM_A}{ARM_B}")
    report_bc = compute_divergence_report(map_b, map_c, f"{ARM_B}{ARM_C}")

    # Completion-floor census from the real arm maps (the all-failure-vacuity guard).
    cc_a = count_ok_completions(map_a)
    cc_c = count_ok_completions(map_c)

    # Per-id co-batching coverage from ARM_C's (union) map. Inert on ts-less maps.
    coverage = compute_cobatch_coverage(map_c)

    decision = decide_status(
        report_ac, report_ab, report_bc,
        client_overlap_depth=client_overlap_depth,
        server_busy_slots=server_busy_slots,
        n_ok_arm_a=cc_a["n_ok"],
        n_ok_arm_c=cc_c["n_ok"],
        arm_a_map=map_a,
        arm_c_map=map_c,
        cobatch_coverage=coverage,
        logit_drift_eps=logit_drift_eps,
    )
    return {
        "report_ac": report_ac,
        "report_ab": report_ab,
        "report_bc": report_bc,
        "decision": decision,
        "completion_counts": {"a": cc_a, "b": count_ok_completions(map_b), "c": cc_c},
        "cobatch_coverage": coverage,
    }

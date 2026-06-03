"""Pure truth-table tests for invariance_diff -- RED / UNVERIFIED / AMBER / GREEN.

The verdict logic is pure (no network / subprocess / threads / GPU): it consumes three
``{test_id: result_dict}`` arm maps and decides the status. These tests drive
``decide_status`` and the supporting report/union/cert builders directly from hand-built
maps, so the full GREEN/RED/AMBER/UNVERIFIED truth table is verified with zero I/O.

The single most important property: a SCORE divergence between the serial reference (ARM_A)
and the batched arm (ARM_C) must read as RED / non-promotable, while a fully-clean,
fully-overlapping, fully-scored run reads as GREEN. Everything in between (no overlap, too
few real completions, uncertain co-batch attribution, a control-arm anomaly, token-only
drift) must demote to a non-promotable verdict -- never silently promote.
"""
from __future__ import annotations

from batch_invariance import invariance_diff as idiff
from batch_invariance.invariance_diff import (
    STATUS_FAILED,
    STATUS_GREEN,
    STATUS_GREEN_UNVERIFIED,
    STATUS_GREEN_WITH_CAVEAT,
)

from .conftest import arm_map, result_row

# Overlap that comfortably clears the floor (depth/busy >= 2).
_OK_OVERLAP = dict(client_overlap_depth=4, server_busy_slots=4)


def _reports(map_a, map_c, map_b=None):
    """Build the three pairwise reports the verdict reads (AB defaults to A==B == clean)."""
    if map_b is None:
        map_b = map_a
    return (
        idiff.compute_divergence_report(map_a, map_c, "AC"),
        idiff.compute_divergence_report(map_a, map_b, "AB"),
        idiff.compute_divergence_report(map_b, map_c, "BC"),
    )


def _decide(map_a, map_c, map_b=None, **overlap):
    """Run the full pure pipeline (reports + floor census + coverage) -> decision dict."""
    if map_b is None:
        map_b = map_a
    rep_ac, rep_ab, rep_bc = _reports(map_a, map_c, map_b)
    cc_a = idiff.count_ok_completions(map_a)
    cc_c = idiff.count_ok_completions(map_c)
    cov = idiff.compute_cobatch_coverage(map_c)
    ov = dict(_OK_OVERLAP)
    ov.update(overlap)
    return idiff.decide_status(
        rep_ac, rep_ab, rep_bc,
        n_ok_arm_a=cc_a["n_ok"], n_ok_arm_c=cc_c["n_ok"],
        arm_a_map=map_a, arm_c_map=map_c, cobatch_coverage=cov,
        **ov,
    )


# ---------------------------------------------------------------------------
# Row 5 -- GREEN (the only promotable verdict).
# ---------------------------------------------------------------------------
def test_clean_overlapping_scored_run_is_green():
    a = arm_map(4)
    c = arm_map(4)  # byte-identical to A
    d = _decide(a, c)
    assert d["status"] == STATUS_GREEN
    assert d["overlap_ok"] is True
    assert d["completion_floor_ok"] is True
    assert d["coverage_checked"] is True
    assert d["cobatch_coverage_ok"] is True
    assert d["divergence_class"] is None
    assert d["anomaly"] is None


# ---------------------------------------------------------------------------
# Row 1 -- RED on a SCORE divergence (the headline property).
# ---------------------------------------------------------------------------
def test_score_divergence_is_red_co_batching():
    a = arm_map(4)
    c = arm_map(4)
    # Flip the SCORE on one id in ARM_C only -> A and C diverge, A and B stay equal.
    c["t1"] = result_row(score=0.0, passed=False)
    d = _decide(a, c)
    assert d["status"] == STATUS_FAILED
    # AB clean -> pure continuous-batching divergence.
    assert d["divergence_class"] == idiff.DIVCLASS_CO_BATCHING


def test_score_divergence_with_dirty_ab_is_slot_allocation():
    a = arm_map(4)
    b = arm_map(4)
    c = arm_map(4)
    c["t1"] = result_row(score=0.0, passed=False)   # AC diverges
    b["t2"] = result_row(failure_mode="http_timeout", score=0.0, passed=False)  # AB diverges too
    d = _decide(a, c, b)
    assert d["status"] == STATUS_FAILED
    assert d["divergence_class"] == idiff.DIVCLASS_SLOT_ALLOC


def test_failure_mode_divergence_is_red():
    # failure_mode is an INVARIANT_FIELD: a one-arm timeout must surface as RED, not pass.
    a = arm_map(3)
    c = arm_map(3)
    c["t0"] = result_row(failure_mode="http_timeout", score=0.0, passed=False, content="")
    d = _decide(a, c)
    assert d["status"] == STATUS_FAILED


def test_missing_id_in_arm_c_is_red():
    # A lost id (dropped under co-batching) is a divergence.
    a = arm_map(3)
    c = arm_map(3)
    del c["t2"]
    rep_ac, rep_ab, rep_bc = _reports(a, c)
    assert rep_ac["n_divergent"] >= 1
    assert "t2" in rep_ac["only_x"]


# ---------------------------------------------------------------------------
# Row 2 -- UNVERIFIED when no real co-batching was observed.
# ---------------------------------------------------------------------------
def test_clean_but_no_overlap_is_unverified():
    a = arm_map(4)
    c = arm_map(4)
    d = _decide(a, c, client_overlap_depth=1, server_busy_slots=1)
    assert d["status"] == STATUS_GREEN_UNVERIFIED
    assert d["overlap_ok"] is False


def test_overlap_needs_both_client_and_server():
    a = arm_map(4)
    c = arm_map(4)
    # client overlap high but server busy slots below threshold -> still not overlap_ok.
    d = _decide(a, c, client_overlap_depth=4, server_busy_slots=1)
    assert d["status"] == STATUS_GREEN_UNVERIFIED
    assert d["overlap_ok"] is False


# ---------------------------------------------------------------------------
# Row 2b -- UNVERIFIED on the completion floor (all-failure vacuity).
# ---------------------------------------------------------------------------
def test_all_failure_arms_are_unverified_not_green():
    # Two arms that BOTH fail identically compare EQUAL on INVARIANT_FIELDS -> n_divergent==0,
    # but ZERO trials genuinely scored. The floor must demote this "matching nothing" pass.
    a = arm_map(4, failure_mode="empty", score=0.0, passed=False, content="")
    c = arm_map(4, failure_mode="empty", score=0.0, passed=False, content="")
    rep_ac, _, _ = _reports(a, c)
    assert rep_ac["n_divergent"] == 0          # they look "clean"
    d = _decide(a, c)
    assert d["status"] == STATUS_GREEN_UNVERIFIED
    assert d["completion_floor_ok"] is False


def test_floor_is_inert_when_counts_absent():
    # When n_ok counts are NOT supplied the floor is not enforced (legacy/pure-report path).
    a = arm_map(4)
    c = arm_map(4)
    rep_ac, rep_ab, rep_bc = _reports(a, c)
    d = idiff.decide_status(rep_ac, rep_ab, rep_bc, **_OK_OVERLAP)  # no n_ok_* , no maps
    assert d["status"] == STATUS_GREEN
    assert d["completion_floor_ok"] is True     # benign default


# ---------------------------------------------------------------------------
# Row 2c -- UNVERIFIED on per-id co-batch coverage (peak-only / uncertain attribution).
# ---------------------------------------------------------------------------
def test_no_per_id_overlap_fails_coverage_closed():
    # Peak overlap is forced True, but per-id intervals are DISJOINT (no id actually
    # co-batched) -> coverage fails closed to unverified even though scores are clean.
    a = arm_map(4)
    c = arm_map(4, overlap=False)               # disjoint intervals
    d = _decide(a, c)
    assert d["status"] == STATUS_GREEN_UNVERIFIED
    assert d["coverage_checked"] is True
    assert d["cobatch_coverage_ok"] is False


def test_uncertain_attribution_missing_ts_fails_closed():
    # A single id missing its interval makes attribution uncertain -> fail closed.
    a = arm_map(4)
    c = arm_map(4)
    c["t2"] = result_row(dispatch_ts=None, complete_ts=None)
    cov = idiff.compute_cobatch_coverage(c)
    assert cov["checked"] is True
    assert cov["attribution_certain"] is False
    assert cov["cobatch_coverage_ok"] is False
    d = _decide(a, c)
    assert d["status"] == STATUS_GREEN_UNVERIFIED


def test_coverage_inert_on_legacy_maps_without_ts_contract():
    # Maps that carry NO dispatch_ts key at all (hand-built legacy fixtures) leave the
    # coverage gate INERT (checked=False) so the legacy truth table applies unchanged.
    a = {f"t{i}": {"score": 1.0, "passed": True, "expected_answer": "A",
                   "prompt_tokens_measured": 10, "failure_mode": "ok"} for i in range(3)}
    c = {k: dict(v) for k, v in a.items()}
    cov = idiff.compute_cobatch_coverage(c)
    assert cov["checked"] is False
    assert cov["cobatch_coverage_ok"] is True   # not enforced
    rep_ac, rep_ab, rep_bc = _reports(a, c)
    d = idiff.decide_status(
        rep_ac, rep_ab, rep_bc,
        n_ok_arm_a=3, n_ok_arm_c=3, arm_a_map=a, arm_c_map=c, **_OK_OVERLAP)
    assert d["status"] == STATUS_GREEN          # inert coverage -> still green


# ---------------------------------------------------------------------------
# Row 3 -- AMBER when the slot-control arm B disagrees with the reference.
# ---------------------------------------------------------------------------
def test_ab_anomaly_is_amber():
    a = arm_map(4)
    c = arm_map(4)                              # AC clean
    b = arm_map(4)
    b["t1"] = result_row(score=0.0, passed=False)   # AB diverges (B != reference)
    d = _decide(a, c, b)
    assert d["status"] == STATUS_GREEN_WITH_CAVEAT
    assert d["anomaly"] == idiff.ANOMALY_AC_AGREE_B_DISAGREES


# ---------------------------------------------------------------------------
# Row 4 -- AMBER on token-only divergence (same score, different completion bytes).
# ---------------------------------------------------------------------------
def test_token_only_divergence_is_amber():
    a = arm_map(4, content="hello world")
    c = arm_map(4, content="hello world")
    # Same INVARIANT_FIELDS, DIFFERENT completion content on one id -> token-only AMBER.
    c["t2"] = result_row(content="HELLO WORLD different bytes")
    rep_ac, _, _ = _reports(a, c)
    assert rep_ac["n_divergent"] == 0          # scores identical
    assert "t2" in rep_ac["token_divergence_ids"]
    d = _decide(a, c)
    assert d["status"] == STATUS_GREEN_WITH_CAVEAT
    assert "t2" in d["token_divergence_ids"]


def test_red_takes_precedence_over_token_only():
    # If an id diverges on BOTH score and content, RED (score) wins -- token-only never
    # downgrades a real score divergence.
    a = arm_map(3, content="x")
    c = arm_map(3, content="x")
    c["t0"] = result_row(score=0.0, passed=False, content="totally different")
    d = _decide(a, c)
    assert d["status"] == STATUS_FAILED


# ---------------------------------------------------------------------------
# Precedence: RED beats the overlap/floor/coverage demotions.
# ---------------------------------------------------------------------------
def test_score_divergence_is_red_even_without_overlap():
    a = arm_map(3)
    c = arm_map(3)
    c["t0"] = result_row(score=0.0, passed=False)
    d = _decide(a, c, client_overlap_depth=1, server_busy_slots=1)
    assert d["status"] == STATUS_FAILED         # Row 1 fires before Row 2


# ---------------------------------------------------------------------------
# content_sha -- the token diagnostic primitive.
# ---------------------------------------------------------------------------
def test_content_sha_none_equals_empty():
    assert idiff.content_sha(None) == idiff.content_sha("")


def test_content_sha_distinguishes_text():
    assert idiff.content_sha("a") != idiff.content_sha("b")
    assert idiff.content_sha("same") == idiff.content_sha("same")


# ---------------------------------------------------------------------------
# build_union_arm_c -- fold T passes, biased toward divergence (any pass divergent => RED).
# ---------------------------------------------------------------------------
def test_union_surfaces_divergence_present_on_only_one_pass():
    a = arm_map(3)
    clean_pass = arm_map(3)
    dirty_pass = arm_map(3)
    dirty_pass["t1"] = result_row(score=0.0, passed=False)
    # Divergence is on pass 1 only; the union must still carry it (stochastic #7052).
    union = idiff.build_union_arm_c(a, [clean_pass, dirty_pass])
    rep = idiff.compute_divergence_report(a, union, "AC")
    assert rep["n_divergent"] >= 1
    assert "t1" in rep["divergent_ids"]


def test_union_single_pass_is_passthrough():
    a = arm_map(3)
    c = arm_map(3)
    union = idiff.build_union_arm_c(a, [c])
    assert set(union) == set(c)


def test_fold_pass_divergence_counts_per_pass():
    a = arm_map(3)
    p0 = arm_map(3)                              # clean
    p1 = arm_map(3)
    p1["t0"] = result_row(score=0.0, passed=False)   # 1 divergent
    counts = idiff.fold_pass_divergence_counts(a, [p0, p1])
    assert counts == [0, 1]


def test_union_token_only_divergence_survives_into_union():
    a = arm_map(3, content="base")
    p0 = arm_map(3, content="base")             # byte-identical to A
    p1 = arm_map(3, content="base")
    p1["t2"] = result_row(content="DRIFTED bytes")   # token-only on pass 1
    union = idiff.build_union_arm_c(a, [p0, p1])
    rep = idiff.compute_divergence_report(a, union, "AC")
    assert "t2" in rep["token_divergence_ids"]


# ---------------------------------------------------------------------------
# diff_arms -- the one-call pure pipeline used by the driver seam.
# ---------------------------------------------------------------------------
def test_diff_arms_green_on_clean_maps():
    a = arm_map(4)
    out = idiff.diff_arms(
        {idiff.ARM_A: a, idiff.ARM_B: arm_map(4), idiff.ARM_C: arm_map(4)},
        client_overlap_depth=4, server_busy_slots=4)
    assert out["decision"]["status"] == STATUS_GREEN
    assert "report_ac" in out and "cobatch_coverage" in out


def test_diff_arms_red_on_score_divergence():
    a = arm_map(4)
    c = arm_map(4)
    c["t1"] = result_row(score=0.0, passed=False)
    out = idiff.diff_arms(
        {idiff.ARM_A: a, idiff.ARM_B: arm_map(4), idiff.ARM_C: c},
        client_overlap_depth=4, server_busy_slots=4)
    assert out["decision"]["status"] == STATUS_FAILED

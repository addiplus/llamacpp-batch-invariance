"""Cert assembly, persistence, and the promotion gate (pure; zero I/O except tmp files).

Covers the cert schema + the machine-readable promotion verdict:

* ``build_cert`` maps a decision + reports into the full superset payload.
* ``is_promotable`` requires green + ``source=='live'`` + real overlap + completion floor
  + the strong gate tag + (when checked) co-batch coverage -- and FAILS CLOSED on any gap.
* ``cert_is_green`` accepts ONLY the literal ``"green"`` with the required provenance, so a
  mock-sourced green and the two ``green_*`` variants are non-promotable by construction.
* ``persist_cert`` round-trips through ``load_cert`` atomically.
* ``assert_score_invariant`` raises on a score/ id-set mismatch.
"""
from __future__ import annotations

import pytest

from batch_invariance import concurrent_dispatch as cd
from batch_invariance import invariance_diff as idiff

from .conftest import arm_map, result_row

_OK_OVERLAP = dict(client_overlap_depth=4, server_busy_slots=4)


def _decision_and_reports(map_a, map_c, map_b=None):
    if map_b is None:
        map_b = map_a
    rep_ac = idiff.compute_divergence_report(map_a, map_c, "AC")
    rep_ab = idiff.compute_divergence_report(map_a, map_b, "AB")
    rep_bc = idiff.compute_divergence_report(map_b, map_c, "BC")
    cov = idiff.compute_cobatch_coverage(map_c)
    decision = idiff.decide_status(
        rep_ac, rep_ab, rep_bc,
        n_ok_arm_a=idiff.count_ok_completions(map_a)["n_ok"],
        n_ok_arm_c=idiff.count_ok_completions(map_c)["n_ok"],
        arm_a_map=map_a, arm_c_map=map_c, cobatch_coverage=cov, **_OK_OVERLAP)
    return decision, rep_ac, rep_ab, rep_bc, cov


def _build_green_cert(source="live", **overrides):
    a = arm_map(4)
    c = arm_map(4)
    decision, rep_ac, rep_ab, rep_bc, cov = _decision_and_reports(a, c)
    assert decision["status"] == idiff.STATUS_GREEN
    kwargs = dict(
        model="demo-model", ctx=2048, dispatch_n=4, decision=decision,
        report_ac=rep_ac, report_ab=rep_ab, report_bc=rep_bc,
        overlap={"client_max_overlap_depth": 4, "server_peak_busy_slots": 4,
                 "overlap_ok": True},
        source=source, reps=2,
        completion_counts={"a": idiff.count_ok_completions(a),
                           "c": idiff.count_ok_completions(c)},
        gate="abc_union", cobatch_coverage=cov,
    )
    kwargs.update(overrides)
    return idiff.build_cert(**kwargs)


# ---------------------------------------------------------------------------
# A clean live cert is promotable; the same cert from a mock is NOT.
# ---------------------------------------------------------------------------
def test_clean_live_cert_is_promotable():
    cert = _build_green_cert(source="live")
    assert cert["status"] == "green"
    assert cert["source"] == "live"
    assert idiff.is_promotable(cert) is True
    assert cd.cert_is_green(cert, require_source="live") is True


def test_mock_sourced_green_is_not_promotable():
    cert = _build_green_cert(source="mock")
    assert cert["status"] == "green"            # the literal green ...
    assert cd.cert_is_green(cert, require_source="live") is False   # ... but mock provenance
    assert idiff.is_promotable(cert) is False


def test_cert_is_green_accepts_mock_only_when_source_not_required():
    cert = _build_green_cert(source="mock")
    assert cd.cert_is_green(cert, require_source=None) is True
    assert cd.cert_is_green(cert, require_source="live") is False


# ---------------------------------------------------------------------------
# is_promotable FAILS CLOSED on every missing proof.
# ---------------------------------------------------------------------------
def test_promotion_fails_closed_without_overlap():
    cert = _build_green_cert(source="live")
    cert["overlap_ok"] = False
    cert["overlap"]["overlap_ok"] = False
    assert idiff.is_promotable(cert) is False


def test_promotion_fails_closed_on_wrong_gate():
    cert = _build_green_cert(source="live", gate="some_weak_gate")
    assert idiff.is_promotable(cert) is False           # strong gate required by default
    assert idiff.is_promotable(cert, require_gate=None) is True   # waiver path


def test_promotion_fails_closed_on_failed_floor():
    cert = _build_green_cert(source="live")
    cert["completion_floor"]["completion_floor_ok"] = False
    assert idiff.is_promotable(cert) is False


def test_promotion_fails_closed_on_checked_insufficient_coverage():
    cert = _build_green_cert(source="live")
    cert["cobatch_coverage"]["coverage_checked"] = True
    cert["cobatch_coverage"]["cobatch_coverage_ok"] = False
    assert idiff.is_promotable(cert) is False


def test_red_cert_is_never_promotable():
    a = arm_map(4)
    c = arm_map(4)
    c["t0"] = result_row(score=0.0, passed=False)
    decision, rep_ac, rep_ab, rep_bc, cov = _decision_and_reports(a, c)
    assert decision["status"] == idiff.STATUS_FAILED
    cert = idiff.build_cert(
        model="demo-model", ctx=2048, dispatch_n=4, decision=decision,
        report_ac=rep_ac, report_ab=rep_ab, report_bc=rep_bc,
        overlap={"client_max_overlap_depth": 4, "server_peak_busy_slots": 4,
                 "overlap_ok": True},
        source="live", reps=2, gate="abc_union", cobatch_coverage=cov)
    assert cert["status"] == "failed"
    assert cert["mismatch"]                     # human-readable mismatch summary present
    assert idiff.is_promotable(cert) is False


# ---------------------------------------------------------------------------
# promotion_scope + sensitivity (the honest-caveat metadata).
# ---------------------------------------------------------------------------
def test_cert_stamps_non_generalizing_scope():
    cert = _build_green_cert(source="live")
    scope = cert["promotion_scope"]
    assert scope["generalizes"] is False
    assert scope["valid_for"] == {"model": "demo-model", "ctx": 2048, "dispatch_n": 4}


def test_moe_hint_flag_on_scope():
    cert = _build_green_cert(source="live", model="demo-moe-a3b")
    assert cert["promotion_scope"]["tested_model_looks_moe"] is True
    dense = _build_green_cert(source="live", model="demo-dense")
    assert dense["promotion_scope"]["tested_model_looks_moe"] is False


def test_rule_of_three_bound_on_green():
    # With 0 observed divergences and N hot trials, the 95% upper bound is 3/N.
    cert = _build_green_cert(source="live", n_hot_trials=30)
    assert cert["sensitivity"]["rate_upper_bound_95"] == pytest.approx(3.0 / 30)


# ---------------------------------------------------------------------------
# persist_cert -> load_cert round-trip (the only I/O in the pure module).
# ---------------------------------------------------------------------------
def test_persist_then_load_round_trip(tmp_path):
    cert = _build_green_cert(source="live")
    path = idiff.persist_cert(str(tmp_path), cert)
    loaded = cd.load_cert(str(tmp_path), "demo-model", 2048, 4)
    assert loaded is not None
    assert loaded["status"] == "green"
    assert loaded["source"] == "live"
    assert loaded["ts_utc"]                     # stamped at write time
    assert idiff.is_promotable(loaded) is True
    assert path.endswith(cd.cert_filename("demo-model", 2048, 4))


def test_cert_filename_is_collision_free_per_cell():
    f1 = cd.cert_filename("m", 2048, 4)
    f2 = cd.cert_filename("m", 4096, 4)
    f3 = cd.cert_filename("m", 2048, 8)
    assert len({f1, f2, f3}) == 3
    # path separators in a model id are flattened (no nested dirs in the filename).
    assert "/" not in cd.cert_filename("a/b", 1, 1)


def test_load_cert_missing_returns_none(tmp_path):
    assert cd.load_cert(str(tmp_path), "absent", 1, 1) is None


# ---------------------------------------------------------------------------
# assert_score_invariant -- the fail-fast client-plumbing assertion.
# ---------------------------------------------------------------------------
def test_assert_score_invariant_passes_on_identical():
    a = arm_map(3)
    c = arm_map(3)
    idiff.assert_score_invariant(a, c)          # no raise


def test_assert_score_invariant_raises_on_score_diff():
    a = arm_map(3)
    c = arm_map(3)
    c["t1"] = result_row(score=0.0, passed=False)
    with pytest.raises(AssertionError):
        idiff.assert_score_invariant(a, c)


def test_assert_score_invariant_raises_on_idset_diff():
    a = arm_map(3)
    c = arm_map(3)
    del c["t2"]
    with pytest.raises(AssertionError):
        idiff.assert_score_invariant(a, c)

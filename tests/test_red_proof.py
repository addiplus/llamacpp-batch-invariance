"""THE RED-PROOF: the gate actually BITES offline (mock-driven, no GPU / no model).

These are the load-bearing integration tests. They drive the REAL A/B/C arm runner +
the REAL pure diff against the bundled in-process mock over loopback HTTP, using the
driver's documented test seam (``arm_base_url`` aims the arms at the mock; launch/teardown
are no-op fakes). The whole point is to prove -- without any real server -- that:

  * a DIVERGENT run (the mock's ``--score-divergence`` knob makes a co-batched completion
    score differently from the serial one) yields a RED, NON-promotable cert, with the
    completion-floor + co-batch-coverage gates PASSING (so the RED is a true positive, not
    an artifact of an empty or non-overlapping run); and

  * a CLEAN run yields green / green_unverified APPROPRIATELY -- a clean run scored against
    a ``source='mock'`` cert is non-promotable BY CONSTRUCTION (the provenance gate), and a
    clean run that never co-batched is demoted to green_unverified.

If the gate could not be driven RED here, the offline pass would be structurally vacuous --
which is exactly the failure mode this proof exists to rule out.
"""
from __future__ import annotations

from batch_invariance import concurrent_dispatch as cd
from batch_invariance import invariance_diff as idiff

# A work-set whose expected_answer EQUALS the prompt text. The mock's --score-divergence
# knob echoes the user prompt back as the completion when a request is co-batched; with
# expected==prompt the echoed (co-batched) completion scores 1.0 while the serial arm's
# canned "x x x" content scores 0.0 -> a GENUINE score/passed/failure_mode divergence (RED),
# not a token-only AMBER. This is the anti-vacuity RED driver.
_P1 = "the hidden needle is four two four two"
_P2 = "secret marker alpha bravo charlie"
_P3 = "remember the phrase delta echo foxtrot"

_RED_ROWS = [
    {"item": _P1, "expected_answer": _P1, "family": "retrieval", "fill": 0.1},
    {"item": _P2, "expected_answer": _P2, "family": "retrieval", "fill": 0.2},
    {"item": _P3, "expected_answer": _P3, "family": "retrieval", "fill": 0.3},
]

# A work-set whose expected_answer matches the HONEST canned completion, so a clean run is
# genuinely score-clean ('ok') in every arm. The mock's canned content is "x " repeated;
# with max_tokens small the serial completion is a short run of "x" tokens.
_CLEAN_ROWS = [
    {"item": "say x please", "expected_answer": "x x x x x x x x", "family": "exact", "fill": 0.1},
    {"item": "again say x", "expected_answer": "x x x x x x x x", "family": "exact", "fill": 0.2},
    {"item": "and once more", "expected_answer": "x x x x x x x x", "family": "exact", "fill": 0.3},
]


# ---------------------------------------------------------------------------
# THE RED PROOF -- a divergent mock run yields RED / non-promotable.
# ---------------------------------------------------------------------------
def test_score_divergence_yields_red_nonpromotable(mock_server, gate_runner, tmp_path,
                                                   make_workset_file):
    wpath = make_workset_file(_RED_ROWS)
    with mock_server(slots=4, serve_sleep=0.03, score_divergence=True) as base:
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="live", parallel=4, reps=2, gate_passes=6)

    # RED on a SCORE divergence ...
    assert cert["status"] == idiff.STATUS_FAILED
    assert cert["divergence_report"]["AC"]["n_divergent"] >= 1
    # ... and NON-promotable.
    assert idiff.is_promotable(cert) is False
    assert cd.cert_is_green(cert, require_source="live") is False


def test_red_is_a_true_positive_not_an_artifact(mock_server, gate_runner, tmp_path,
                                                make_workset_file):
    # The RED must be EARNED on a fully-overlapping, fully-scored run: the completion-floor
    # and co-batch-coverage gates must PASS, so the RED is a real co-batching divergence and
    # not an empty/non-overlapping artifact.
    wpath = make_workset_file(_RED_ROWS)
    with mock_server(slots=4, serve_sleep=0.03, score_divergence=True) as base:
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="live", parallel=4, reps=2, gate_passes=6)

    assert cert["status"] == idiff.STATUS_FAILED
    # overlap was genuinely observed (peak >= 2, client depth >= 2).
    assert cert["overlap"]["overlap_ok"] is True
    assert cert["overlap"]["server_peak_busy_slots"] >= 2
    # the completion floor PASSED (real completions occurred in both gate arms).
    assert cert["completion_floor"]["completion_floor_ok"] is True
    assert cert["completion_floor"]["n_ok_a"] >= 1
    assert cert["completion_floor"]["n_ok_c"] >= 1
    # A real co-batching-involved RED. The divergence is genuine continuous-batching drift,
    # but the EXACT sub-class is a function of concurrency timing on this LIVE mock run: the
    # N-slot serial arm (ARM_B) usually matches the reference -> DIVCLASS_CO_BATCHING, yet
    # under load ARM_B MAY ALSO drift, in which case the gate (deterministically, given those
    # inputs) reports DIVCLASS_SLOT_ALLOC == "slot_allocation(+co_batching)". Both are valid
    # true positives -- the "earned, not an artifact" property is proven above by
    # status=failed + overlap_ok + completion_floor_ok + AC.n_divergent >= 1, NOT by which
    # serial arm happened to drift. (The pure, deterministic CO_BATCHING-vs-SLOT_ALLOC split
    # is pinned exactly in test_invariance_diff.py against hand-built arm maps.)
    assert cert["divergence_class"] in (idiff.DIVCLASS_CO_BATCHING, idiff.DIVCLASS_SLOT_ALLOC)


def test_red_divergence_surfaces_across_passes(mock_server, gate_runner, tmp_path,
                                               make_workset_file):
    # The per-pass audit trail records the divergence on the concurrent passes (the union
    # over T passes is what gates; any pass divergent => RED).
    wpath = make_workset_file(_RED_ROWS)
    with mock_server(slots=4, serve_sleep=0.03, score_divergence=True) as base:
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="live", parallel=4, reps=2, gate_passes=6)
    assert cert["status"] == idiff.STATUS_FAILED
    assert cert["n_passes"] == 6
    assert sum(cert["per_pass_ac_divergent"]) >= 1


# ---------------------------------------------------------------------------
# CLEAN run -- green / green_unverified APPROPRIATELY.
# ---------------------------------------------------------------------------
def test_clean_run_is_green_status_but_mock_source_not_promotable(
        mock_server, gate_runner, tmp_path, make_workset_file):
    # An HONEST mock + a clean work-set: the gate reaches the literal "green" status (no
    # divergence, real overlap, floor + coverage met) -- but with source='mock' the cert is
    # NON-promotable BY CONSTRUCTION (a real GREEN must be earned live). This is the
    # "green appropriately" half of the proof.
    wpath = make_workset_file(_CLEAN_ROWS)
    with mock_server(slots=4, serve_sleep=0.03) as base:   # all divergence knobs OFF
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="mock", parallel=4, reps=2, gate_passes=4)

    assert cert["status"] == idiff.STATUS_GREEN            # the literal green ...
    assert cert["divergence_report"]["AC"]["n_divergent"] == 0
    assert cert["overlap"]["overlap_ok"] is True
    assert cert["completion_floor"]["completion_floor_ok"] is True
    # ... yet NON-promotable because the provenance is mock.
    assert cert["source"] == "mock"
    assert idiff.is_promotable(cert) is False
    assert cd.cert_is_green(cert, require_source="live") is False


def test_clean_live_run_is_promotable(mock_server, gate_runner, tmp_path, make_workset_file):
    # The SAME clean run with source='live' (the seam lets us stamp it) IS promotable -- the
    # only difference from the mock cert above is the provenance. This proves the gate does
    # mint a promotable GREEN when (and only when) the run is clean AND live-sourced.
    wpath = make_workset_file(_CLEAN_ROWS)
    with mock_server(slots=4, serve_sleep=0.03) as base:
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="live", parallel=4, reps=2, gate_passes=4)

    assert cert["status"] == idiff.STATUS_GREEN
    assert cert["source"] == "live"
    assert idiff.is_promotable(cert) is True


def test_all_failure_run_is_demoted_by_completion_floor(
        mock_server, gate_runner, tmp_path, make_workset_file):
    # The mock returns 200 + EMPTY content on every (co-batched) request, so every arm is
    # all-failure ('empty') and the arms compare EQUAL on INVARIANT_FIELDS ("matching
    # nothing"). The completion floor must demote this to green_unverified DESPITE overlap.
    wpath = make_workset_file(_CLEAN_ROWS)
    with mock_server(slots=4, serve_sleep=0.03, empty_completions=True) as base:
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="live", parallel=4, reps=2, gate_passes=4)

    assert cert["status"] == idiff.STATUS_GREEN_UNVERIFIED
    assert cert["completion_floor"]["completion_floor_ok"] is False
    assert idiff.is_promotable(cert) is False


# ---------------------------------------------------------------------------
# The cert persists + reloads with the verdict intact (end-to-end artifact).
# ---------------------------------------------------------------------------
def test_red_cert_persists_and_reloads(mock_server, gate_runner, tmp_path, make_workset_file):
    wpath = make_workset_file(_RED_ROWS)
    with mock_server(slots=4, serve_sleep=0.03, score_divergence=True) as base:
        cert = gate_runner(base, wpath, tmp_path,
                           cert_source="live", parallel=4, reps=2, gate_passes=6)
    cert_dir = tmp_path / "certs"
    path = idiff.persist_cert(str(cert_dir), cert)
    assert path
    reloaded = cd.load_cert(str(cert_dir), cert["model"], cert["ctx"], cert["dispatch_n"])
    assert reloaded is not None
    assert reloaded["status"] == idiff.STATUS_FAILED
    assert idiff.is_promotable(reloaded) is False

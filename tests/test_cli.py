"""The subcommand CLI -- the README quickstart commands, end to end (no GPU / no model).

These tests cover the four ``batch-invariance`` subcommands exactly as the README
quickstart documents them, so each README command copy-pastes and works:

  * ``run-mock``            -> a GREEN, source=mock, NON-promotable cert.
  * ``run-mock --batch-divergence`` -> the SAME gate goes RED (failed / non-promotable).
  * ``run-live --dry-run``  -> exits 0 and launches NOTHING (placeholder server-bin/model).
  * ``verify-cert PATH``    -> the bundled phi-4 RED cert reports failed/RED + non-promotable,
                               and exits 0 (a validly-parsed cert is a successful verify).
  * ``plan-n``              -> prints a per-N footprint plan and exits 0.

Plus the dispatch contract: ``main([])`` (no subcommand) prints help + exits non-zero;
``--help`` lists the four subcommands; ``run-mock`` writes the cert under <out-dir>.

Everything runs through the package's first-class mock-driving core (no real server), so
the suite stays GPU-free.
"""
from __future__ import annotations

import json
import os

import pytest

from batch_invariance import cli
from batch_invariance import invariance_diff as idiff
from batch_invariance.live_invariance import (
    EXIT_ERROR,
    EXIT_PROMOTABLE,
)

# The repository root (this file is <repo>/tests/test_cli.py); the bundled proof cert lives
# at <repo>/examples/certs/demo_dense_live_cert.json. Resolved from __file__ so the test is
# independent of the pytest invocation CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO_CERT = os.path.join(_REPO_ROOT, "examples", "certs", "demo_dense_live_cert.json")


# ---------------------------------------------------------------------------
# Dispatch contract: --help lists the 4 subcommands; bare invocation exits non-zero.
# ---------------------------------------------------------------------------
def test_help_lists_four_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("run-mock", "run-live", "verify-cert", "plan-n"):
        assert name in out


def test_each_subcommand_has_help(capsys):
    for name in ("run-mock", "run-live", "verify-cert", "plan-n"):
        with pytest.raises(SystemExit) as exc:
            cli.main([name, "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert name in out


def test_bare_invocation_prints_help_and_exits_nonzero(capsys):
    # No subcommand -> help to stderr + a NON-zero usage code (must not crash).
    rc = cli.main([])
    assert rc != 0
    err = capsys.readouterr().err
    assert "run-mock" in err and "verify-cert" in err


# ---------------------------------------------------------------------------
# run-mock (default) -> GREEN, source=mock, NON-promotable.
# ---------------------------------------------------------------------------
def test_run_mock_default_writes_green_mock_nonpromotable(tmp_path, capsys):
    out_dir = tmp_path / "certs"
    rc = cli.main(["run-mock", "--out-dir", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GREEN" in out

    # The cert lands under <out-dir>/dispatch-cert/ and is green / mock / non-promotable.
    cert_dir = out_dir / "dispatch-cert"
    files = list(cert_dir.glob("*.json"))
    assert len(files) == 1
    cert = json.loads(files[0].read_text(encoding="utf-8"))
    assert cert["status"] == idiff.STATUS_GREEN
    assert cert["source"] == "mock"
    assert idiff.is_promotable(cert) is False


def test_run_mock_default_core_is_green_and_nonpromotable(tmp_path):
    # The first-class core (what examples/run_gate.py also calls) returns the same verdict.
    cert = cli.run_mock_gate(batch_divergence=False, out_dir=str(tmp_path))
    assert cert["status"] == idiff.STATUS_GREEN
    assert cert["source"] == "mock"
    assert cert["overlap"]["overlap_ok"] is True
    assert idiff.is_promotable(cert) is False


# ---------------------------------------------------------------------------
# run-mock --batch-divergence -> RED / failed / NON-promotable.
# ---------------------------------------------------------------------------
def test_run_mock_batch_divergence_writes_red(tmp_path, capsys):
    out_dir = tmp_path / "certs"
    rc = cli.main(["run-mock", "--out-dir", str(out_dir), "--batch-divergence"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RED" in out

    cert_dir = out_dir / "dispatch-cert"
    files = list(cert_dir.glob("*.json"))
    assert len(files) == 1
    cert = json.loads(files[0].read_text(encoding="utf-8"))
    assert cert["status"] == idiff.STATUS_FAILED
    assert cert["divergence_report"]["AC"]["n_divergent"] >= 1
    assert idiff.is_promotable(cert) is False


def test_run_mock_batch_divergence_core_is_red(tmp_path):
    cert = cli.run_mock_gate(batch_divergence=True, out_dir=str(tmp_path))
    assert cert["status"] == idiff.STATUS_FAILED
    # the RED is a true positive: overlap real + completion floor met (not an empty artifact).
    assert cert["overlap"]["overlap_ok"] is True
    assert cert["completion_floor"]["completion_floor_ok"] is True
    assert idiff.is_promotable(cert) is False


# ---------------------------------------------------------------------------
# run-live --dry-run -> exit 0, launches nothing (placeholder server-bin/model).
# ---------------------------------------------------------------------------
def test_run_live_dry_run_exits_zero_and_launches_nothing(tmp_path, capsys):
    out_dir = tmp_path / "certs"
    rc = cli.main([
        "run-live",
        "--server-bin", "/nonexistent/llama-server",
        "--model", "/nonexistent/model.gguf",
        "--workset", str(_workset_file(tmp_path)),
        "--ctx", "8192", "--parallel", "4",
        "--cert-source", "live",
        "--out-dir", str(out_dir),
        "--dry-run",
    ])
    # dry-run is a clean preview (live_invariance returns EXIT_PROMOTABLE==0 for it).
    assert rc == EXIT_PROMOTABLE
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "NO server process launched" in out
    # Nothing was written: no cert dir / no server log.
    assert not (out_dir / "dispatch-cert").exists() or \
        not list((out_dir / "dispatch-cert").glob("*.json"))


def test_run_live_missing_inputs_is_clean_error(tmp_path, capsys):
    # No server-bin/model and no profile -> a clean usage error, not a crash.
    rc = cli.main(["run-live", "--ctx", "8192", "--out-dir", str(tmp_path)])
    assert rc == EXIT_ERROR
    err = capsys.readouterr().err
    assert "missing required input" in err


# ---------------------------------------------------------------------------
# verify-cert -> the bundled phi-4 RED cert: failed / RED / non-promotable, exit 0.
# ---------------------------------------------------------------------------
def test_verify_cert_demo_red_is_failed_and_nonpromotable(capsys):
    assert os.path.isfile(_DEMO_CERT), f"bundled cert missing: {_DEMO_CERT}"
    rc = cli.main(["verify-cert", _DEMO_CERT])
    # A validly-parsed cert (even a RED) is a successful verification -> exit 0.
    assert rc == 0
    out = capsys.readouterr().out
    assert "status        = failed" in out
    assert "is_promotable = False" in out
    assert "NOT promotable" in out


def test_verify_cert_malformed_is_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    rc = cli.main(["verify-cert", str(bad)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "not valid cert JSON" in err


def test_verify_cert_missing_file_is_nonzero(tmp_path, capsys):
    rc = cli.main(["verify-cert", str(tmp_path / "does_not_exist.json")])
    assert rc != 0
    err = capsys.readouterr().err
    assert "no such file" in err


def test_verify_cert_promotable_green_exits_zero(tmp_path, capsys):
    # A clean live run minted via the seam (cert_source='live') is promotable; verify-cert
    # must report PROMOTABLE and exit 0.
    out_dir = tmp_path / "live"
    # Drive a clean live mock run through the core, but stamp source=live by hand-asserting
    # via a built green cert: reuse the conftest-style clean path through run_mock_gate is
    # mock-sourced, so build a promotable cert directly from the pure helpers instead.
    from tests.conftest import arm_map
    a = arm_map(4)
    c = arm_map(4)
    rep_ac = idiff.compute_divergence_report(a, c, "AC")
    rep_ab = idiff.compute_divergence_report(a, a, "AB")
    rep_bc = idiff.compute_divergence_report(a, c, "BC")
    cov = idiff.compute_cobatch_coverage(c)
    decision = idiff.decide_status(
        rep_ac, rep_ab, rep_bc, client_overlap_depth=4, server_busy_slots=4,
        n_ok_arm_a=idiff.count_ok_completions(a)["n_ok"],
        n_ok_arm_c=idiff.count_ok_completions(c)["n_ok"],
        arm_a_map=a, arm_c_map=c, cobatch_coverage=cov)
    cert = idiff.build_cert(
        model="demo-model", ctx=2048, dispatch_n=4, decision=decision,
        report_ac=rep_ac, report_ab=rep_ab, report_bc=rep_bc,
        overlap={"client_max_overlap_depth": 4, "server_peak_busy_slots": 4, "overlap_ok": True},
        source="live", reps=2, gate="abc_union", cobatch_coverage=cov,
        completion_counts={"a": idiff.count_ok_completions(a),
                           "c": idiff.count_ok_completions(c)})
    assert idiff.is_promotable(cert) is True
    path = idiff.persist_cert(str(out_dir), cert)
    rc = cli.main(["verify-cert", path])
    assert rc == 0
    out = capsys.readouterr().out
    assert "is_promotable = True" in out
    assert "PROMOTABLE" in out


# ---------------------------------------------------------------------------
# plan-n -> prints a per-N plan and exits 0.
# ---------------------------------------------------------------------------
def test_plan_n_prints_plan(capsys):
    rc = cli.main([
        "plan-n",
        "--model-alias", "demo-model",
        "--ctx", "8192", "--parallel", "4",
        "--gguf-gb", "9.0",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "plan-n" in out
    assert "projected" in out
    assert "NOTHING launched" in out
    # the grid covers N=1..parallel.
    assert "8192" in out          # server_ctx at N=1
    assert "32768" in out         # server_ctx at N=4 (8192*4)


def test_plan_n_without_weight_size_is_clean_error(capsys):
    # No --gguf-gb and a nonexistent --model -> a clear error, not a crash.
    rc = cli.main(["plan-n", "--ctx", "8192", "--parallel", "4",
                   "--model", "/nonexistent/model.gguf"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "cannot size the model weights" in err


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _workset_file(tmp_path) -> str:
    rows = [
        {"item": "say x please", "expected_answer": "x x x x", "family": "exact", "fill": 0.1},
        {"item": "again say x", "expected_answer": "x x x x", "family": "exact", "fill": 0.2},
    ]
    p = tmp_path / "ws.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return str(p)

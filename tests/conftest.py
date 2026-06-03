"""Shared pytest fixtures for the batch_invariance test suite (stdlib + pytest only).

Two families of helpers live here:

* PURE-logic helpers (``arm_map`` / ``result_row``) build the ``{test_id: result_dict}``
  maps the diff/verdict functions consume, so a test can construct an exact divergence
  scenario without any server. These are the inputs ``invariance_diff`` was designed
  around -- the whole point of the pure/impure seam is that the verdict logic is testable
  from hand-built maps with zero I/O.

* MOCK-driven helpers (``mock_server`` / ``run_gate_against_mock``) drive the REAL arm
  runner + diff against the bundled in-process mock server over loopback HTTP -- NO GPU,
  NO model, NO real ``llama-server``. ``run_gate_against_mock`` uses the driver's
  documented test seam: ``arm_base_url`` points the arms at the mock, and the
  popen/kill/mem/ready dependencies are replaced by no-op fakes so the launch/teardown
  path is exercised without spawning anything. This is how the RED-proof tests make the
  gate go RED offline while the completion-floor / coverage / provenance gates pass.

Everything here is Python standard library only.
"""
from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator

import pytest

from batch_invariance import invariance_diff as idiff
from batch_invariance import mock_llama_server as mock
from batch_invariance.live_invariance import LiveInvarianceDriver


# ---------------------------------------------------------------------------
# A fake Popen handle: the driver only ever reads ``.pid`` and calls ``.wait``.
# ---------------------------------------------------------------------------
class _FakeProc:
    """Stand-in for ``subprocess.Popen`` -- exposes only what the driver touches."""

    def __init__(self, pid: int = 424242) -> None:
        self.pid = int(pid)

    def wait(self, timeout=None):  # noqa: ANN001 - mirrors Popen.wait
        return 0


def _fake_popen_factory(pid: int = 424242):
    """Return a callable usable as the driver's ``popen=`` injectable (never spawns)."""

    def _popen(*_args, **_kwargs):
        return _FakeProc(pid)

    return _popen


# ---------------------------------------------------------------------------
# PURE-logic factories: build {test_id: result_dict} arm maps by hand.
# ---------------------------------------------------------------------------
def result_row(
    *,
    score: float = 1.0,
    passed: bool = True,
    expected_answer="A",
    prompt_tokens_measured: int = 10,
    failure_mode: str = "ok",
    content: str = "the answer",
    dispatch_ts: float | None = 0.0,
    complete_ts: float | None = 1.0,
    family: str = "fam",
    fill_ratio: float = 0.1,
    **extra,
) -> dict:
    """One result dict carrying every INVARIANT_FIELD + the co-batch interval contract.

    Defaults describe a genuinely-scored ('ok'), score=1.0 trial whose
    [dispatch_ts, complete_ts] interval is [0,1] (so two such rows overlap and count as
    co-batched). Override any field to construct a divergence / failure / missing-interval
    scenario. ``**extra`` lets a test stamp diagnostic-only keys (e.g. a logit summary).
    """
    row = {
        "score": score,
        "passed": passed,
        "expected_answer": expected_answer,
        "prompt_tokens_measured": prompt_tokens_measured,
        "failure_mode": failure_mode,
        "content": content,
        "response_first_200": content[:200],
        "family": family,
        "fill_ratio": fill_ratio,
        "dispatch_ts": dispatch_ts,
        "complete_ts": complete_ts,
    }
    row.update(extra)
    return row


def arm_map(n: int = 3, *, overlap: bool = True, **row_kwargs) -> dict:
    """Build an ``{test_id: result_dict}`` map of ``n`` genuinely-scored rows.

    When ``overlap`` is True every row shares the [0,1] interval, so all ids count as
    co-batched (coverage satisfied). When False each row gets a disjoint interval
    ([2i, 2i+1]) so NO id co-batches (coverage fails) -- the knob the coverage tests use.
    Row content/score/etc. is uniform unless ``row_kwargs`` overrides it; per-id tweaks
    are applied by the caller after construction.
    """
    out: dict = {}
    for i in range(n):
        if overlap:
            ds, cs = 0.0, 1.0
        else:
            ds, cs = float(2 * i), float(2 * i + 1)
        out[f"t{i}"] = result_row(dispatch_ts=ds, complete_ts=cs, **row_kwargs)
    return out


@pytest.fixture
def make_result_row():
    """Fixture exposing :func:`result_row` to tests."""
    return result_row


@pytest.fixture
def make_arm_map():
    """Fixture exposing :func:`arm_map` to tests."""
    return arm_map


# ---------------------------------------------------------------------------
# MOCK-driven harness: start the in-process mock over loopback (no GPU / no model).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def mock_server_cm(*, slots: int = 4, serve_sleep: float = 0.02, **knobs) -> Iterator[str]:
    """Start the bundled mock on an ephemeral loopback port; yield its base URL.

    ``**knobs`` are forwarded to :func:`mock_llama_server.serve` (e.g.
    ``score_divergence=True`` / ``empty_completions=True`` / ``batch_divergence=True``).
    The server is shut down cleanly on exit, draining any wedged slot first so no daemon
    thread leaks into a sibling test.
    """
    httpd = mock.serve(0, slots=slots, serve_sleep=serve_sleep, ready_log=False, **knobs)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        with contextlib.suppress(Exception):
            httpd.stop_hangs()
        with contextlib.suppress(Exception):
            httpd.shutdown()
        with contextlib.suppress(Exception):
            t.join(timeout=5)


@pytest.fixture
def mock_server():
    """Fixture handing back the :func:`mock_server_cm` context manager."""
    return mock_server_cm


def write_workset(tmp_path, rows: list[dict]) -> str:
    """Write a work-set JSON file under ``tmp_path`` and return its path (str)."""
    p = tmp_path / "workset.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return str(p)


@pytest.fixture
def make_workset_file(tmp_path):
    """Fixture: write a work-set JSON and return its path (bound to this test's tmp_path)."""

    def _make(rows: list[dict]) -> str:
        return write_workset(tmp_path, rows)

    return _make


def run_gate_against_mock(
    base_url: str,
    workset_path: str,
    out_dir,
    *,
    cert_source: str = "live",
    parallel: int = 4,
    reps: int = 2,
    gate_passes: int = 6,
    ctx: int = 2048,
    n_predict: int = 8,
    **driver_kwargs,
) -> dict:
    """Run the REAL A/B/C gate against the mock at ``base_url``; return the cert dict.

    Uses the driver's documented TEST SEAM: ``arm_base_url`` aims the arms at the mock and
    the launch/teardown dependencies are no-op fakes, so the genuine
    ``run_arm`` -> ``score_one`` -> ``invariance_diff`` path runs with NO real server. A
    fixed ``gguf_gb`` makes the footprint guard inert (no real file to size). ``cert_source``
    defaults to ``"live"`` so a clean run can reach the literal promotable check; pass
    ``"mock"`` to assert the provenance gate keeps it non-promotable.
    """
    drv = LiveInvarianceDriver(
        server_bin="/nonexistent/server-bin",
        model_path="model.gguf",
        model_alias="demo-model",
        port=18888,
        ctx=ctx,
        parallel=parallel,
        n_predict=n_predict,
        out_dir=str(out_dir),
        cert_source=cert_source,
        reps=reps,
        gate_passes=gate_passes,
        workset_path=workset_path,
        popen=_fake_popen_factory(),
        kill=lambda _pid, _sig: None,
        mem_reader=lambda: None,           # off-Linux / unreadable -> permissive
        wait_ready=lambda _port, _timeout, host="127.0.0.1": 0,
        is_port_free=lambda _port, host="127.0.0.1": True,
        # grace_s=0: teardown signals a FAKE pid (no real child), so the SIGTERM->SIGKILL
        # grace sleep is pure dead time in tests. Zero it so the mock-driven gate runs in
        # ~1s instead of ~10s. (The real CLI keeps the default 5s grace for a real server.)
        grace_s=0.0,
        arm_base_url=base_url,
        gguf_gb=1.0,
        **driver_kwargs,
    )
    return drv.run()


@pytest.fixture
def gate_runner():
    """Fixture exposing :func:`run_gate_against_mock` to tests."""
    return run_gate_against_mock


# Re-export the pure status constants so tests can import them off conftest if desired.
STATUS_GREEN = idiff.STATUS_GREEN
STATUS_FAILED = idiff.STATUS_FAILED
STATUS_GREEN_UNVERIFIED = idiff.STATUS_GREEN_UNVERIFIED
STATUS_GREEN_WITH_CAVEAT = idiff.STATUS_GREEN_WITH_CAVEAT

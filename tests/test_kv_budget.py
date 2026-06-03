"""KV-cache + footprint math (pure) and the live-mem down-clamp monotonicity.

``kv_gb`` is linear in ctx * parallel * kv_factor; ``get_kv_budget_gb`` derives the usable
budget from a live MemAvailable reading (or a static pool) minus reservation/margin/floor,
clamped non-negative. ``gguf_size_gb`` stats a real file. ``is_port_free`` is a bind probe.
All but ``gguf_size_gb`` / ``is_port_free`` are deterministic functions of their arguments.
"""
from __future__ import annotations

import socket

import pytest

from batch_invariance import kv_budget


# ---------------------------------------------------------------------------
# kv_factor / kv_gb -- linearity + the quant scaling.
# ---------------------------------------------------------------------------
def test_kv_factor_known_quants():
    assert kv_budget.kv_factor("q8_0") == 1.0
    assert kv_budget.kv_factor("f16") == 2.0
    assert kv_budget.kv_factor("q4_0") == 0.5


def test_kv_factor_unknown_falls_back_to_one():
    # An unrecognised quant over- (never under-) estimates -> 1.0 (q8_0-equivalent).
    assert kv_budget.kv_factor("totally-made-up") == 1.0


def test_kv_gb_at_reference_ctx_is_anchor():
    # At full reference ctx, parallel=1, q8_0 -> the 13 GB anchor.
    assert kv_budget.kv_gb(kv_budget.KV_REF_CTX, 1, "q8_0") == pytest.approx(13.0)


def test_kv_gb_linear_in_parallel():
    one = kv_budget.kv_gb(8192, 1, "q8_0")
    four = kv_budget.kv_gb(8192, 4, "q8_0")
    assert four == pytest.approx(4 * one)


def test_kv_gb_linear_in_ctx():
    small = kv_budget.kv_gb(8192, 1, "q8_0")
    big = kv_budget.kv_gb(16384, 1, "q8_0")
    assert big == pytest.approx(2 * small)


def test_kv_gb_f16_doubles_q8():
    q8 = kv_budget.kv_gb(8192, 2, "q8_0")
    f16 = kv_budget.kv_gb(8192, 2, "f16")
    assert f16 == pytest.approx(2 * q8)


def test_job_mem_gb_sums_weights_kv_overhead():
    gguf = 3.0
    total = kv_budget.job_mem_gb(gguf, 8192, 2, "q8_0")
    expected = gguf + kv_budget.kv_gb(8192, 2, "q8_0") + kv_budget.PER_SERVER_OVERHEAD_GB
    assert total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_kv_budget_gb -- live vs static source, clamps, monotonicity.
# ---------------------------------------------------------------------------
def test_live_budget_subtracts_reservation_margin_floor():
    # 32 GiB free = 32768 MiB; budget = 32 - reserved - margin - floor.
    b = kv_budget.get_kv_budget_gb(32768, reserved_gb=4.0, margin_gb=2.0, hard_floor_gb=8.0)
    assert b == pytest.approx(32.0 - 4.0 - 2.0 - 8.0)


def test_static_pool_ignores_live_floor():
    b = kv_budget.get_kv_budget_gb(None, total_gb=40.0, reserved_gb=4.0, margin_gb=2.0)
    assert b == pytest.approx(40.0 - 4.0 - 2.0)


def test_budget_clamped_non_negative():
    # A tiny pool that the floor exceeds -> 0.0, never negative.
    b = kv_budget.get_kv_budget_gb(1024, hard_floor_gb=8.0)
    assert b == 0.0


def test_budget_zero_when_no_source():
    assert kv_budget.get_kv_budget_gb(None) == 0.0
    assert kv_budget.get_kv_budget_gb(0) == 0.0


def test_budget_monotonic_in_available_memory():
    low = kv_budget.get_kv_budget_gb(16384)
    high = kv_budget.get_kv_budget_gb(65536)
    assert high > low


def test_more_reservation_never_increases_budget():
    base = kv_budget.get_kv_budget_gb(32768, reserved_gb=0.0)
    more = kv_budget.get_kv_budget_gb(32768, reserved_gb=10.0)
    assert more <= base


# ---------------------------------------------------------------------------
# gguf_size_gb -- file size only; loud on absence.
# ---------------------------------------------------------------------------
def test_gguf_size_reads_real_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * (2 * 1024 * 1024))     # 2 MiB
    gb = kv_budget.gguf_size_gb(str(f))
    assert gb == pytest.approx(2 / 1024, rel=1e-6)


def test_gguf_size_missing_raises():
    with pytest.raises(FileNotFoundError):
        kv_budget.gguf_size_gb("/no/such/model.gguf")


def test_gguf_size_empty_path_raises():
    with pytest.raises(FileNotFoundError):
        kv_budget.gguf_size_gb("")


# ---------------------------------------------------------------------------
# is_port_free -- a bind probe (the only socket call); reads, never connects/kills.
# ---------------------------------------------------------------------------
def test_free_port_reports_free():
    # Find an ephemeral port, close it, then assert the probe sees it free.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert kv_budget.is_port_free(port) is True


def test_bound_port_reports_not_free():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert kv_budget.is_port_free(port) is False
    finally:
        s.close()

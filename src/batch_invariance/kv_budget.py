#!/usr/bin/env python3
"""kv_budget.py -- KV-cache projection + memory-footprint math (PURE, stdlib-only).

This module is the small, referentially-transparent core that answers "how much
memory does an ``N``-slot llama.cpp server at context ``C`` need, and what slot
count fits a given budget?" It carries NONE of the multi-server packing, port
allocation, job scheduling, or cross-process claim machinery -- only the KV/byte
math and the two memory primitives a single-experiment live driver needs:

  * :func:`kv_gb`        -- KV-cache footprint of one server (GB), linear in
                            ``ctx * parallel * kv_factor``.
  * :func:`job_mem_gb`   -- total co-residency weight of one server
                            (weights + KV + a per-server compute pad).
  * :func:`get_kv_budget_gb` -- the usable memory budget for a server, derived from
                            a live ``MemAvailable`` reading (or a static pool size)
                            minus a reservation, a safety margin, and a hard floor.
  * :func:`gguf_size_gb` -- a model's on-disk size in GB (file size only).
  * :func:`is_port_free` -- a race-free "is this TCP port bindable?" probe (the only
                            socket call; used by a live driver's own-port precheck).

WHY PURE: every function except :func:`is_port_free` is a deterministic function of
its arguments (no subprocess, no network, no GPU, no filesystem except the explicit
``gguf_size_gb`` stat), so the KV math and budget derivation are instantly
unit-testable with zero files and zero models. :func:`is_port_free` only READS
availability via ``socket.bind`` -- it never connects to, writes to, or terminates
anything. The companion modules layer the impure edge (HTTP, the experiment server
child process) on top of these primitives.

The KV constants below are generic llama.cpp facts (KV scales linearly with total
KV tokens = ctx * parallel and with the quant's bytes-per-element), not a policy of
any particular machine -- the per-box budget knobs (pool size, reservation, margin)
are all parameters of :func:`get_kv_budget_gb`, defaulted permissively.
"""
from __future__ import annotations

import os
import socket

# ---------------------------------------------------------------------------
# Generic context / footprint constants. These are llama.cpp facts + conservative
# defaults, NOT host policy: every per-box knob (total pool, reservation, margin) is
# a parameter of get_kv_budget_gb() and defaults to a permissive value.
# ---------------------------------------------------------------------------
HARD_CTX_CAP = 131072            # a sane global ceiling on a single server's ctx
PER_SERVER_OVERHEAD_GB = 1.5     # compute buffers + framework context pad per server
HARD_FLOOR_GB = 8.0              # keep this much MemAvailable free as defense-in-depth

# KV at FULL 131072 ctx, q8_0 K+V, a single slot is ~13 GB per server (measured on a
# 30B-class model; the figure is dominated by ctx, not by the weights, so it is a
# reasonable generic reference). KV scales linearly off this anchor.
KV_GB_AT_FULL_CTX = 13.0
KV_REF_CTX = 131072

# Bytes-per-element of a KV quant relative to q8_0 (= 1.0). f16 is 2x; the q4/q5
# variants are fractions. Used to scale the q8_0 reference figure to other quants.
_KV_FACTOR = {
    "q8_0": 1.0,
    "f16": 2.0,
    "q4_0": 0.5,
    "q4_1": 0.5,
    "q5_0": 0.625,
    "q5_1": 0.625,
}


def kv_factor(kv: str = "q8_0") -> float:
    """Bytes-per-element multiplier of a KV quant relative to q8_0 (=1.0).

    An unrecognised label falls back to 1.0 (q8_0-equivalent) rather than raising,
    so a typo'd quant over- (never under-) estimates the footprint.
    """
    return _KV_FACTOR.get(str(kv).lower(), 1.0)


def kv_gb(ctx: int, parallel: int = 1, kv: str = "q8_0") -> float:
    """KV-cache footprint in GB.

    ``kv_gb = 13 * (ctx / 131072) * parallel * kv_factor(kv)``

    The ``* parallel`` term is what makes this correct for an ``N``-slot server:
    with ``--ctx-size = N * C`` each of the ``N`` slots gets the full ``C`` of KV, so
    total KV = ``N * kv_gb(C, 1)`` = ``kv_gb(C, N)``. Worked (parallel=1, q8_0): an
    8B-ctx server ~= 1.6 GB; a full-131072-ctx server ~= 13 GB.
    """
    return (KV_GB_AT_FULL_CTX
            * (float(ctx) / float(KV_REF_CTX))
            * float(parallel)
            * kv_factor(kv))


def job_mem_gb(gguf_gb: float, ctx: int, parallel: int = 1, kv: str = "q8_0",
               overhead_gb: float = PER_SERVER_OVERHEAD_GB) -> float:
    """Total memory weight of one server = weights + KV(ctx, parallel) + compute pad."""
    return float(gguf_gb) + kv_gb(ctx, parallel, kv) + float(overhead_gb)


def get_kv_budget_gb(
    mem_available_mib: float | None = None,
    *,
    total_gb: float | None = None,
    reserved_gb: float = 0.0,
    margin_gb: float = 0.0,
    hard_floor_gb: float = HARD_FLOOR_GB,
) -> float:
    """Usable memory budget (GB) for one experiment server.

    The single budget-derivation entry point a caller uses to turn "how much memory
    is there?" into "how much may this server use?" Two mutually-exclusive sources:

      * LIVE (preferred): pass ``mem_available_mib`` -- the value a caller read from
        the OS (e.g. ``/proc/meminfo`` MemAvailable, in MiB). This module does NOT
        read it itself (that keeps it free of subprocess/host coupling); the caller
        injects the number. The budget is then::

            mem_available_mib / 1024 - reserved_gb - margin_gb - hard_floor_gb

      * STATIC (fallback): pass ``total_gb`` -- a known total pool size in GB. The
        budget is::

            total_gb - reserved_gb - margin_gb

        (the live floor does not apply to a static pool; reservation + margin do).

    ``reserved_gb`` carves out memory another process already owns (e.g. a baseline
    server you must not starve); ``margin_gb`` is a safety cushion; ``hard_floor_gb``
    keeps that much MemAvailable genuinely free under the live path. All three default
    to a permissive posture (reservation/margin 0; floor :data:`HARD_FLOOR_GB`), so a
    bare ``get_kv_budget_gb(mem)`` simply returns the free memory minus the floor.

    Returns a NON-NEGATIVE budget (clamped at 0.0) so a downstream slot-count solver
    never sees a negative figure. If neither ``mem_available_mib`` (> 0) nor
    ``total_gb`` is usable, returns 0.0 (a caller treats that as "fall back to the
    serial single-slot path", never a 0-slot launch).
    """
    avail = None
    if mem_available_mib is not None:
        try:
            avail = float(mem_available_mib)
        except (TypeError, ValueError):
            avail = None
    if avail is not None and avail > 0:
        budget = (avail / 1024.0) - float(reserved_gb) - float(margin_gb) - float(hard_floor_gb)
        return max(0.0, budget)
    if total_gb is not None:
        try:
            budget = float(total_gb) - float(reserved_gb) - float(margin_gb)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, budget)
    return 0.0


def gguf_size_gb(gguf_path: str) -> float:
    """On-disk size of a model file in GB (``os.path.getsize`` only -- no lookup table).

    Raises ``FileNotFoundError`` when the path is empty or absent, so a footprint
    guard fails loudly instead of silently sizing an unknown model to 0 GB. When the
    file is remote/unavailable, a caller should supply the projected size by another
    route (e.g. a ``--gguf-gb`` override) rather than relying on this.
    """
    if not gguf_path:
        raise FileNotFoundError("gguf_size_gb: empty path")
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"gguf_size_gb: no such file {gguf_path!r}")
    return os.path.getsize(gguf_path) / (1024.0 ** 3)


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Race-free free-port check via ``socket.bind``.

    Tries to bind ``(host, port)`` WITHOUT ``SO_REUSEADDR``; returns True iff the bind
    succeeds (the port is free). This only READS availability -- it never connects to,
    sends to, or terminates whatever might hold the port. ``SO_REUSEADDR`` is left OFF
    so the probe reflects what a real (non-reuse) server bind would see, rather than
    falsely reporting a live port as free.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()

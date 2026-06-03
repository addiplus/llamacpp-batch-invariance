#!/usr/bin/env python3
"""http_harness.py -- the IMPURE HTTP edge for the live batch-invariance driver.

This module is the thin, generic transport layer the live driver dispatches through:
an OpenAI-compatible HTTP client (``/v1/chat/completions``), a ``/tokenize`` token
counter for prompt calibration, a readiness probe (``/v1/models`` with the
"loading model" gotcha handled), a live-memory floor reader, and PID-scoped process
teardown helpers. It deliberately carries NO prompt set, NO scorer, and NO model
registry -- those live behind the pluggable ``workset`` / ``scorer_api`` seams. Pure
Python standard library only (``json``/``os``/``signal``/``socket``/``subprocess``/
``time``/``urllib``); no third-party runtime dependency.

The readiness probe (:func:`wait_for_ready`) is the strict one: a 200 that still says
"loading model", or a 200 whose body carries no model list, is NOT ready. This avoids
the classic race where a server accepts the socket before the model is resident.

The teardown helpers (:func:`cleanup_pids`, :func:`_os_kill`, :data:`_SPAWNED_PIDS`)
are PID-SCOPED: only processes this library explicitly spawned are ever signalled, and
only by PID -- never by name/pattern match. SIGTERM is sent first, then SIGKILL after a
grace period (SIGKILL falls back to SIGTERM on platforms without it). This is the safe
teardown the live driver reuses so a launched experiment server is always reaped without
risking any other long-running process on the host.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


# ============================================================
# OpenAI-compatible HTTP helpers
# ============================================================
def http_post_json(url: str, body: dict, timeout: int = 120) -> dict:
    """POST a JSON ``body`` to ``url`` and return the decoded JSON response.

    Raises the underlying urllib/OSError on transport failure (the caller -- the live
    driver's per-request scorer path -- catches it and records a FAILURE trial so one
    wedged request cannot void an arm). Pure I/O: no global state, no import-time
    side effects."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get(url: str, timeout: int = 5) -> dict | None:
    """GET ``url`` and return decoded JSON, or None on any transport/parse failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError):
        return None


class TokenizeError(RuntimeError):
    """Raised by :func:`tokenize_count` when the ``/tokenize`` endpoint is unreachable
    or returns an unparseable / unexpected-shape response. Callers catch this and fall
    back to estimate-based sizing (stamping the trial so the saturation check still
    fires) rather than guessing a token count."""


def tokenize_count(base_url: str, text: str, timeout: int = 30) -> int:
    """Return the MEASURED token count of ``text`` from a server's ``/tokenize``.

    POSTs ``{"content": text}`` to ``{base_url}/tokenize`` and reads the count from
    whatever shape the server returns. llama.cpp returns ``{"tokens": [ids...]}``; some
    builds/wrappers instead return ``{"count": N}`` or ``{"n_tokens": N}``. We are
    tolerant of all three (in that priority). Pure I/O: no import-time side effects, no
    global state.

    Raises :class:`TokenizeError` on any transport failure (URLError/Timeout/Connection),
    a body that does not parse as JSON, or a response whose shape carries no recognizable
    token count -- so the caller can fall back deterministically."""
    url = base_url.rstrip("/") + "/tokenize"
    body = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise TokenizeError(f"/tokenize transport error: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise TokenizeError(f"/tokenize returned non-JSON body: {exc}") from exc
    if isinstance(payload, dict):
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        for key in ("count", "n_tokens", "num_tokens"):
            val = payload.get(key)
            if isinstance(val, bool):
                continue  # bool is an int subclass; never a real token count
            if isinstance(val, int):
                return val
            if isinstance(val, float) and val.is_integer():
                return int(val)
    raise TokenizeError(
        f"/tokenize response shape unrecognized (no tokens/count/n_tokens): "
        f"{str(payload)[:200]!r}"
    )


# ============================================================
# Readiness probe (the strict one: 503/"loading model" is NOT ready)
# ============================================================
def is_ready_response(status_code: int, body_text: str) -> bool:
    """True only when the server returns a real model list.

    A non-200, a body that still says "loading model", or a 200 whose JSON carries no
    ``data``/``models`` list all read as NOT ready -- this is what makes the probe robust
    against a server that accepts the socket before the model is resident."""
    if status_code != 200:
        return False
    if "loading model" in (body_text or "").lower():
        return False
    try:
        data = json.loads(body_text)
    except (ValueError, TypeError):
        return False
    return bool(data.get("data") or data.get("models"))


def _probe_ready(port: int, host: str = "127.0.0.1", timeout: int = 5) -> bool:
    """Single readiness probe against ``/v1/models``, honoring the loading-model gotcha."""
    url = f"http://{host}:{int(port)}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return is_ready_response(resp.getcode(), resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        return is_ready_response(e.code, body)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def wait_for_ready(port: int, ready_timeout: int = 180, host: str = "127.0.0.1") -> int:
    """Poll ``/v1/models`` until :func:`is_ready_response` is True or timeout.

    Returns the integer seconds waited. Raises :class:`TimeoutError` if the server never
    becomes ready within ``ready_timeout`` seconds. This is the readiness contract the
    live driver awaits after launching its experiment server."""
    for i in range(1, int(ready_timeout) + 1):
        if _probe_ready(port, host=host):
            return i
        time.sleep(1)
    raise TimeoutError(f"server :{port} never became ready in {ready_timeout}s")


# ============================================================
# Live-memory floor reader (the precheck's BLOCKING guard input)
# ============================================================
def read_mem_available_mib() -> float | None:
    """Free memory in MiB from ``/proc/meminfo`` MemAvailable, else ``free -m``.

    Returns None when neither source is available (e.g. off Linux) -- the live driver
    treats a None / <=0 reading as a permissive no-op for the memory floor (the static
    footprint cap still applies), so the verifier is portable to a CPU-only CI box."""
    try:
        text = Path("/proc/meminfo").read_text()
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if line.lower().startswith("mem:"):
                parts = line.split()
                return float(parts[-1])
    except Exception:
        pass
    return None


# ============================================================
# PID-scoped process teardown (NEVER pattern-kills)
# ============================================================
# SIGKILL is absent on some platforms; fall back to SIGTERM there.
_SIGTERM = signal.SIGTERM
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)

# Tracks the exact PIDs THIS library spawned so any atexit/signal teardown only ever
# touches these -- never an arbitrary process on the host.
_SPAWNED_PIDS: set[int] = set()


def _os_kill(pid: int, sig) -> None:
    """Send ``sig`` to ``pid`` via ``os.kill`` (the injectable seam for tests)."""
    os.kill(pid, sig)


def cleanup_pids(pids, kill, grace: float = 5.0) -> None:
    """Terminate ONLY the given ``pids`` (SIGTERM, then SIGKILL after ``grace`` seconds).

    NEVER pattern-kills and NEVER touches a PID not in ``pids``. ``kill`` is injected
    (defaults to :func:`_os_kill`) so the teardown path is unit-testable without signing
    real processes. A PID that has already exited (ProcessLookupError/OSError) is skipped
    silently -- teardown is idempotent."""
    for pid in list(pids):
        try:
            kill(pid, _SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    if pids:
        time.sleep(grace)
    for pid in list(pids):
        try:
            kill(pid, _SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def nvidia_smi_snapshot() -> dict:
    """Capture GPU state via ``nvidia-smi`` (diagnostics only; fail-soft).

    Returns a dict of utilization/power/temperature/memory fields, or ``{"error": ...}``
    on any failure. This is purely informational telemetry -- it is NEVER on the verdict
    path, so its absence (no GPU, no driver, CI) does not affect any cert."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_util_pct": float(parts[0]) if parts[0] and parts[0] != "[N/A]" else None,
            "power_w": float(parts[1]) if parts[1] and parts[1] != "[N/A]" else None,
            "temp_c": float(parts[2]) if parts[2] and parts[2] != "[N/A]" else None,
            "memory_used_mib": float(parts[3]) if parts[3] and parts[3] != "[N/A]" else None,
        }
    except Exception as e:
        return {"error": str(e)}

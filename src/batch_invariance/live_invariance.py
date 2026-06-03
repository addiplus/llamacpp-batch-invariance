#!/usr/bin/env python3
"""live_invariance.py -- the LIVE batch-invariance DRIVER for llama.cpp-style servers.

WHAT THIS IS. Continuous batching (``--parallel N`` + overlapping requests) merges
concurrent HTTP requests into one forward pass. This driver answers, against a REAL
OpenAI-compatible server: **does batched concurrent dispatch produce the SAME scored
outputs as running those same requests one at a time?** If IDENTICAL, that cell's
batched outputs are safe to trust; if DIVERGENT, the cell stays serial-only. The
deliverable is ONE artifact per ``(model, ctx, N)``: a ``source=='live'`` cert that
``concurrent_dispatch.cert_is_green(cert, require_source='live')`` accepts, written ONLY
when the serial and concurrent arms are score-identical AND the concurrent arm provably
co-batched (real overlap observed).

WHY A GREEN MUST BE EARNED LIVE. The null hypothesis is that batching DOES change
outputs: batched matmul / RMSNorm / attention reductions can reorder even at temp=0
(llama.cpp #7052 observed 5-8 unique completions for one prompt at 8 slots / temp=0 on
H100/A100-class hardware; PR #16016 deterministic mode is OFF by default; Thinking
Machines, "Defeating Nondeterminism in LLM Inference"). An offline mock returns content
independent of batch composition, so an offline check is structurally vacuous -- GREEN
must therefore be EARNED against a real server, and the test suite proves the gate can go
RED on a divergent mock.

THE DRIVER'S JOB (this file): own at most ONE short-lived experiment server child
process, launch it, run Arms A/B/C against it over real HTTP, collect per-prompt
completion text + score, hand the per-arm result maps to the PURE diff
(``invariance_diff``), write the cert, print GREEN/RED, and ALWAYS tear down the server
it launched via try/finally. It re-reads real ``/proc/meminfo`` and ABORTS if launching
would be unsafe, holds the experiment footprint under a configured cap, and never binds
or addresses a user-supplied "untouchable" baseline port.

SAFETY DOCTRINE (load-bearing, generic): teardown is PID-SCOPED -- only the process this
driver spawned is ever signalled, and only by PID, never by name/pattern match. It is
idempotent and wired to atexit + SIGINT + SIGTERM + every finally. The footprint cap and
the live-memory floor refuse to launch rather than risk an out-of-memory hang on a shared
unified-memory host. Never send SIGKILL to a long-running server process out of band; the
PID-scoped graceful teardown (SIGTERM then SIGKILL after a grace period) is the only path.

CONTRACT (the seam): the driver builds per-arm result maps ``{test_id: result_dict}`` and
hands them to ``invariance_diff`` (PURE -- no network/subprocess/threads/GPU). The pure
diff never touches the network and never knows what an "arm" launched; it only compares
maps. That split is what makes the gate logic unit-testable with zero GPU. This module is
the impure driver; bring your own server binary, work-set (``--workset`` JSON), and scorer
(``--scorer pkg.mod:fn``).
"""
from __future__ import annotations

import argparse
import atexit
import inspect
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import concurrent_dispatch as cd
from . import invariance_diff as idiff
from . import kv_budget
from .http_harness import (
    _SPAWNED_PIDS,
    _os_kill,
    cleanup_pids,
    http_post_json,
    read_mem_available_mib,
    tokenize_count,
    wait_for_ready,
)
from .invariance_diff import ARM_A, ARM_B, ARM_C
from .scorer_api import DEFAULT_SCORER, classify_transport_error, resolve_scorer
from .workset import build_body, load_workset

# ---------------------------------------------------------------------------
# Defaults. Ports are CLI-parameterized -- a generic user picks them. The ONLY port
# this driver refuses to touch is the user-supplied baseline (a server they already
# have running) plus any extra ports they explicitly mark untouchable.
# ---------------------------------------------------------------------------
DEFAULT_EXPERIMENT_PORT = 8192   # the port the experiment server binds AND arms dispatch to
DEFAULT_BASELINE_PORT = 8192     # the user's already-running baseline server (never touched)

# The standard llama.cpp-style server flags. Overridable per-launch via --server-flag.
DEFAULT_SERVER_FLAGS = (
    "-ngl", "999", "--flash-attn", "on", "--no-mmap",
    "-b", "2048", "-ub", "2048",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--threads", "10", "--threads-batch", "10", "--jinja",
)
DEFAULT_REQ_TIMEOUT_S = 120
DEFAULT_READY_TIMEOUT_S = 180
DEFAULT_GRACE_S = 5.0
DEFAULT_ARM_DEADLINE_S = 900

# T (ARM_C concurrent re-passes the AC gate unions over) is DECOUPLED from --reps. The
# CLI default is 8 passes (detection power against stochastic co-batch divergence -- more
# passes sample more interleavings, and any pass divergent => RED). Programmatic
# construction (the test seam) leaves gate_passes=None and falls back to max(1, reps),
# preserving the legacy coupling so the existing suite stays green.
DEFAULT_GATE_PASSES = 8

# HARD experiment-footprint cap (GB). The experiment server can run CONCURRENT with a
# user's baseline on a shared (possibly unified-memory) pool; a mis-sized launch is the
# out-of-memory failure mode this verifier guards against. Enforced in precheck() (S0,
# before the live-mem floor) so an obvious ``--ctx 32768 --parallel 8`` is REFUSED, not
# merely warned. Overridable only via the explicit --max-footprint-gb flag.
MAX_FOOTPRINT_GB = 15.0

# The default model id stamped into the cert/seeds when --model-alias is omitted. Kept
# stable so the work-set is reproducible regardless of the on-disk gguf path.
DEFAULT_MODEL_ID = "model"

# F0 empty-retry: the BLANK-completion failure modes a retry may re-attempt. Both label a
# blank/instant-EOS completion: ``empty`` (no usable text, completion_tokens>0) and
# ``premature_eos`` (instant EOS, completion_tokens==0). Retrying ONLY these (and only when
# --empty-retries>0, default 0) is the SAFE direction: it can convert a blank FAILURE into a
# real completion (more comparison data) but can NEVER hide a real score divergence.
_RETRIABLE_EMPTY_MODES = frozenset({"empty", "premature_eos"})

# Exit codes. Pinned: 0 ONLY when is_promotable(cert) is True.
EXIT_PROMOTABLE = 0      # status=='green' AND source=='live' AND overlap_ok
EXIT_ERROR = 1           # launch/health/timeout/guard before a verdict
EXIT_REFUSED = 2         # missing ack flag
EXIT_NOT_PROMOTABLE = 3  # failed | green_unverified | green_with_caveat
EXIT_SIGINT = 130
EXIT_SIGTERM = 143


class GuardError(RuntimeError):
    """A safety guard (S0-S7) refused BEFORE any server launch (precheck failure)."""


# ---------------------------------------------------------------------------
# N-derivation / footprint math (PURE). Thin wrappers over kv_budget's raw KV
# primitives, co-located with their only consumer (this driver). KV scales LINEARLY
# with total KV tokens = ctx * N and with the quant's bytes-per-element; an N-slot
# server must launch with ``-c = ctx * N`` so every slot gets the full context.
# ---------------------------------------------------------------------------
def _server_ctx_for(ctx: int, dispatch_n: int) -> int:
    """The ``-c`` an N-slot server must launch with so each slot gets the full ctx.

    ``ctx * max(1, N)``. With N=1 this is a no-op (== ctx)."""
    return int(ctx) * max(1, int(dispatch_n))


def _kv_total_gb(ctx: int, dispatch_n: int, kv: str = "q8_0") -> float:
    """Total KV footprint of an N-slot server at ctx (== kv_budget.kv_gb(ctx, N, kv))."""
    return kv_budget.kv_gb(int(ctx), int(dispatch_n), kv)


def _clamp_n_to_live_mem(n_star: int, ctx: int, gguf_gb: float,
                         mem_available_mib: float, *, kv: str = "q8_0") -> int:
    """Clamp ``n_star`` DOWN to the largest N that fits LIVE free memory + a floor.

    Returns the largest ``n`` in ``[1, n_star]`` such that
    ``gguf_gb + _kv_total_gb(ctx, n, kv) + PER_SERVER_OVERHEAD_GB + HARD_FLOOR_GB
    <= mem_available_mib / 1024``. Defense-in-depth: strictly DOWN-only (never raises N).
    ``mem_available_mib <= 0`` (off-Linux / unreadable) returns ``n_star`` unchanged
    (permissive). If even N=1 will not fit, returns 1 (the caller's hard backstop)."""
    n_star = max(1, int(n_star))
    avail_gb = float(mem_available_mib) / 1024.0
    if avail_gb <= 0:
        return n_star
    fixed = (float(gguf_gb) + float(kv_budget.PER_SERVER_OVERHEAD_GB)
             + float(kv_budget.HARD_FLOOR_GB))
    for n in range(n_star, 0, -1):
        if fixed + _kv_total_gb(ctx, n, kv) <= avail_gb + 1e-9:
            return n
    return 1


def _score_one_shot(base_url: str, item: dict, *, dispatch_n: int, n_predict: int,
                    req_timeout: int, n_probs: int, scorer: Callable) -> dict:
    """Fire ONE HTTP request and score it (no retry). The single-shot core of score_one.

    Carries EVERY INVARIANT_FIELD plus VOLATILE/diagnostic keys. On transport failure
    records a FAILURE trial (never raises): the pluggable ``scorer`` is consulted ONLY on a
    real 200 body; the no-response path is classified by
    :func:`scorer_api.classify_transport_error` so a timeout/oom that hits one arm shows as
    divergence (``failure_mode`` is an INVARIANT_FIELD)."""
    body = build_body(item, dispatch_n=dispatch_n, n_predict=n_predict, n_probs=n_probs)
    url = _join_url(base_url, "/v1/chat/completions")
    t0 = time.monotonic()
    timed_out = False
    exc: BaseException | None = None
    resp: dict = {}
    try:
        resp = http_post_json(url, body, timeout=int(req_timeout))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        exc = e
        # urllib raises socket.timeout (a TimeoutError/OSError subclass) on read timeout.
        timed_out = "timed out" in str(e).lower() or "timeout" in str(e).lower()
    t1 = time.monotonic()

    expected = item.get("expected_answer")
    if exc is not None:
        content = ""
        score = 0.0
        passed = False
        # Transport-error path: classify without calling the scorer (no body to score).
        failure_mode = classify_transport_error(exc)
        completion_tokens = None
        reason = str(exc)
    else:
        choices = resp.get("choices") or [{}]
        content = ((choices[0] or {}).get("message", {}) or {}).get("content", "") or ""
        completion_tokens = (resp.get("usage") or {}).get("completion_tokens")
        # Pluggable scorer: (score, passed, failure_mode). The generic gate's only
        # semantic dependency on failure_mode is the literal "ok" (the completion floor)
        # and the empty-retry set {"empty","premature_eos"}.
        score, passed, failure_mode = scorer(content, expected, item=item)
        score = float(score)
        passed = bool(passed)
        failure_mode = str(failure_mode)
        reason = ""

    return {
        "test_id": item["test_id"],
        "family": item.get("family"),
        # ---- INVARIANT_FIELDS (must match across arms) ----
        "score": score,
        "passed": passed,
        "expected_answer": expected,
        "prompt_tokens_measured": item.get("prompt_tokens_measured"),
        "failure_mode": failure_mode,
        # ---- VOLATILE / diagnostic (ignored by assert_score_invariant) ----
        "content": content,
        "response_first_200": content[:200].replace("\n", " "),
        "completion_tokens": completion_tokens,
        "wall_ms": (t1 - t0) * 1000.0,
        "fill_ratio": item.get("fill_ratio"),
        "request_start_monotonic": t0,
        "response_end_monotonic": t1,
        # ---- per-id co-batch coverage contract (driver <-> invariance_diff) ----
        # dispatch_ts / complete_ts are the AUTHORITATIVE per-id request interval the diff
        # folds into was_co_batched[id]: two ids are co-batched iff their
        # [dispatch_ts, complete_ts] intervals OVERLAP. Captured as time.monotonic() floats
        # at the HTTP SEND (t0) and RESPONSE RECEIPT (t1). A transport-failure trial still
        # carries its t0/t1 (the request WAS dispatched, the interval is real). These are
        # VOLATILE/diagnostic, so they never gate a score comparison; the diff's coverage
        # gate FAILS CLOSED (-> green_unverified, never green) when they are missing/None.
        "dispatch_ts": t0,
        "complete_ts": t1,
        "timed_out": timed_out,
        "reason": reason,
    }


def score_one(base_url: str, item: dict, *, dispatch_n: int, n_predict: int,
              req_timeout: int = DEFAULT_REQ_TIMEOUT_S,
              n_probs: int = 0,
              empty_retries: int = 0,
              scorer: Callable = DEFAULT_SCORER,
              retry_prompt_fn: Callable[[dict, int], dict] | None = None) -> dict:
    """Fire ONE request against an ALREADY-RUNNING server; return a result_dict.

    Carries EVERY INVARIANT_FIELD plus VOLATILE/diagnostic keys. On transport failure
    records a FAILURE trial instead of raising, so one wedged request cannot void an arm --
    and because failure_mode is an INVARIANT_FIELD, a timeout that hits ONE arm only
    correctly shows as divergence.

    F0 empty-retry (ADDITIVE, default OFF): when ``empty_retries > 0`` AND a completion is
    a BLANK-completion failure (``failure_mode in {"empty","premature_eos"}``), re-fire up
    to that many times. Each retry re-mints the SAME ``test_id`` (identity preserved -- A
    and C still compare on the same key); ``retry_prompt_fn(item, attempt)`` may re-place
    the prompt (when supplied; absent => re-fire the same item verbatim). This is the SAFE
    direction: a retry can only convert a BLANK FAILURE into a real completion; it CANNOT
    hide a real divergence (the same policy runs in every arm, identity preserved). With the
    default ``empty_retries == 0`` this is byte-identical to the single-shot path."""
    result = _score_one_shot(base_url, item, dispatch_n=dispatch_n, n_predict=n_predict,
                             req_timeout=req_timeout, n_probs=n_probs, scorer=scorer)
    attempts = 0
    max_retries = max(0, int(empty_retries))
    cur_item = item
    while (max_retries > 0 and attempts < max_retries
           and result.get("failure_mode") in _RETRIABLE_EMPTY_MODES):
        attempts += 1
        if retry_prompt_fn is not None:
            try:
                cur_item = retry_prompt_fn(cur_item, attempts)
            except Exception:
                cur_item = item  # any failure -> re-fire verbatim (still safe)
        retry_res = _score_one_shot(base_url, cur_item, dispatch_n=dispatch_n,
                                    n_predict=n_predict, req_timeout=req_timeout,
                                    n_probs=n_probs, scorer=scorer)
        # Keep the ORIGINAL test_id so the arm map key never drifts under retry.
        retry_res["test_id"] = item["test_id"]
        result = retry_res
        if result.get("failure_mode") not in _RETRIABLE_EMPTY_MODES:
            break
    result["retry_attempts"] = int(attempts)
    return result


class SlotPoller:
    """Background poller of the server's overlap telemetry during ARM_C.

    A llama.cpp-style server exposes ``GET /slots`` (a LIST of per-slot objects); an
    offline mock may instead expose ``GET /slots-debug`` (max_observed_concurrency). The
    poller consults BOTH and keeps the MAX observation. ``.peak_busy_slots`` is the max
    concurrently-PROCESSING slots observed (real: slots flagged by ``is_processing`` -- the
    current schema -- OR the legacy ``state != idle`` enum; mock: max_observed_concurrency).
    On any HTTP error the poll is skipped (peak unchanged). It only ever targets ``base_url``."""

    def __init__(self, base_url: str, interval: float = 0.03) -> None:
        self._base = base_url.rstrip("/")
        self._interval = float(interval)
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> None:
        # Consult BOTH endpoints and keep the MAX. /slots is the real server's
        # INSTANTANEOUS per-slot view; /slots-debug is a mock's CUMULATIVE peak. Reading
        # both and taking the max fixes the under-count when every /slots poll happens to
        # land on an idle instant even though requests DID co-batch (a cumulative counter
        # still proves it). This is the SAFE direction: neither endpoint can FABRICATE
        # overlap (both are real observations), so the max can only move overlap toward the
        # TRUE peak -- it never invents co-batching. On a real server (no /slots-debug) that
        # endpoint errors and is skipped, so the real path reads /slots alone.
        for path in ("/slots", "/slots-debug"):
            try:
                with urllib.request.urlopen(self._join(path), timeout=2) as r:
                    payload = json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    OSError, ValueError):
                continue
            busy = self._busy_from_payload(payload)
            if busy is not None and busy > self._peak:
                self._peak = int(busy)

    @staticmethod
    def _busy_from_payload(payload) -> int | None:
        # Real server GET /slots: a LIST of slot dicts. The current schema keys each slot
        # on the boolean ``is_processing`` and has NO ``state`` key; older builds used a
        # numeric/string ``state`` enum (0/'idle' == free). We count a slot busy under
        # EITHER schema (forward+back-compat) so the overlap proof is not dead on a modern
        # server. The legacy branch is unchanged, so a mock and any state-enum server still
        # behave as before; we only ADD the is_processing/id_task recognition.
        if isinstance(payload, list):
            busy = 0
            for slot in payload:
                if not isinstance(slot, dict):
                    continue
                # Modern schema: is_processing is a real bool (True == busy).
                proc = slot.get("is_processing")
                if proc is True:
                    busy += 1
                    continue
                # Defensive: a non-(-1) id_task also means the slot is working.
                idt = slot.get("id_task", slot.get("id_slot_task"))
                if isinstance(idt, int) and not isinstance(idt, bool) and idt != -1:
                    busy += 1
                    continue
                # Legacy schema: 0/'0'/'idle'/None is free; anything else is busy.
                state = slot.get("state")
                if state is not None and state not in (0, "0", "idle"):
                    busy += 1
            return busy
        # Mock /slots-debug (or a server exposing a summary dict).
        if isinstance(payload, dict):
            for key in ("max_observed_concurrency", "in_flight", "processing"):
                val = payload.get(key)
                if isinstance(val, bool):
                    continue
                if isinstance(val, int):
                    return val
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self._interval)
        # one final sample so a short arm still records a peak
        self._poll_once()

    def _join(self, path: str) -> str:
        return self._base + (path if path.startswith("/") else "/" + path)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="slot-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def peak_busy_slots(self) -> int:
        return int(self._peak)


def _client_max_overlap_depth(results: list[dict]) -> int:
    """Max number of request [start, end) intervals that intersect at any instant.

    Computed from each result's monotonic start/end via a sweep-line over endpoints:
    +1 at a start, -1 at an end; the running max is the client-observed concurrency. A
    serial arm yields 1; a genuinely co-batched arm yields >=2. Pure (no I/O)."""
    events: list[tuple[float, int]] = []
    for r in results:
        s = r.get("request_start_monotonic")
        e = r.get("response_end_monotonic")
        if s is None or e is None:
            continue
        if e < s:
            e = s
        events.append((float(s), 1))
        events.append((float(e), -1))
    if not events:
        return 0
    # Sort so that at an identical timestamp, ENDS (-1) are processed BEFORE STARTS (+1):
    # two requests that merely touch at an instant are not counted as overlap.
    events.sort(key=lambda ev: (ev[0], ev[1]))
    depth = 0
    peak = 0
    for _, delta in events:
        depth += delta
        if depth > peak:
            peak = depth
    return int(peak)


def run_arm(base_url: str, workset: list[dict], *, arm: str, parallel: int,
            concurrent: bool, n_predict: int,
            req_timeout: int = DEFAULT_REQ_TIMEOUT_S,
            arm_deadline: float = DEFAULT_ARM_DEADLINE_S,
            slots_poller: SlotPoller | None = None,
            n_probs: int = 0,
            empty_retries: int = 0,
            scorer: Callable = DEFAULT_SCORER,
            retry_prompt_fn: Callable[[dict, int], dict] | None = None) -> dict:
    """Run one arm against an ALREADY-RUNNING server; return {test_id: result_dict}.

      ARM_A: concurrent=False, parallel=1, dispatch_n stamped into body = 1 (the serial
             ground truth).
      ARM_B: concurrent=False, parallel=N, dispatch_n stamped into body = 1 (serial
             dispatch at the N-slot server; the slot-allocation control).
      ARM_C: concurrent=True,  parallel=N, dispatch_n stamped into body = N (concurrent
             dispatch; build_body adds the concurrent-hardening keys).

    SERIAL: a simple for-loop, one request at a time. CONCURRENT: a barrier-released worker
    pool of ``parallel`` workers so >=2 requests are genuinely in flight; workers are joined
    with a FINITE timeout (arm_deadline), NEVER an unbounded join. If slots_poller is
    provided (ARM_C only) it is started before dispatch + stopped after. The per-arm meta
    (overlap depths) is attached under the reserved key ``"_arm_meta"`` so the result MAP
    stays {test_id: result_dict} for the pure diff.

    F0 empty-retry (ADDITIVE, default OFF): ``empty_retries``/``retry_prompt_fn`` are
    threaded UNCHANGED into every score_one call so the SAME retry policy applies to EVERY
    arm identically (preserving A/C symmetry). Default 0 == no retry."""
    dispatch_n = int(parallel) if arm == ARM_C else 1
    results_by_id: dict[str, dict] = {}
    lock = threading.Lock()
    deadline = time.monotonic() + float(arm_deadline)

    if slots_poller is not None:
        slots_poller.start()
    try:
        if not concurrent:
            # SERIAL -- one request at a time. Stop if the arm deadline passes (a wedged
            # slot is recorded as a transport-failure trial by score_one's own per-request
            # timeout; the deadline is the whole-arm backstop).
            for item in workset:
                if time.monotonic() > deadline:
                    results_by_id[item["test_id"]] = _deadline_trial(item, n_probs)
                    continue
                r = score_one(base_url, item, dispatch_n=dispatch_n,
                              n_predict=n_predict, req_timeout=req_timeout,
                              n_probs=n_probs, empty_retries=empty_retries,
                              scorer=scorer, retry_prompt_fn=retry_prompt_fn)
                results_by_id[r["test_id"]] = r
        else:
            # CONCURRENT -- barrier-released pool of `parallel` workers so >=2 requests are
            # genuinely co-resident.
            n_workers = max(2, int(parallel))
            n_workers = min(n_workers, max(1, len(workset)))
            barrier = threading.Barrier(n_workers) if n_workers > 1 else None
            queue = list(workset)
            qlock = threading.Lock()

            def _next() -> dict | None:
                with qlock:
                    return queue.pop(0) if queue else None

            def _worker() -> None:
                if barrier is not None:
                    try:
                        barrier.wait(timeout=min(60.0, float(arm_deadline)))
                    except threading.BrokenBarrierError:
                        pass
                while True:
                    if time.monotonic() > deadline:
                        # drain remaining as deadline trials so the map stays complete
                        while True:
                            leftover = _next()
                            if leftover is None:
                                break
                            with lock:
                                results_by_id[leftover["test_id"]] = _deadline_trial(
                                    leftover, n_probs)
                        return
                    item = _next()
                    if item is None:
                        return
                    r = score_one(base_url, item, dispatch_n=dispatch_n,
                                  n_predict=n_predict, req_timeout=req_timeout,
                                  n_probs=n_probs, empty_retries=empty_retries,
                                  scorer=scorer, retry_prompt_fn=retry_prompt_fn)
                    with lock:
                        results_by_id[r["test_id"]] = r

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_worker) for _ in range(n_workers)]
                # finite, bounded join -- never unbounded
                for f in futures:
                    try:
                        f.result(timeout=float(arm_deadline) + req_timeout + 30)
                    except Exception:
                        pass
    finally:
        if slots_poller is not None:
            slots_poller.stop()

    # Attach per-arm overlap meta WITHOUT polluting the test_id keyspace the pure diff
    # consumes. The driver strips this before diffing.
    client_depth = _client_max_overlap_depth(list(results_by_id.values()))
    results_by_id["_arm_meta"] = {
        "arm": arm,
        "parallel": int(parallel),
        "concurrent": bool(concurrent),
        "dispatch_n": dispatch_n,
        "client_max_overlap_depth": int(client_depth),
        "server_peak_busy_slots": (int(slots_poller.peak_busy_slots)
                                   if slots_poller is not None else 0),
        "n_results": len(results_by_id) - 1,
    }
    return results_by_id


def _deadline_trial(item: dict, n_probs: int) -> dict:
    """A synthetic 'arm_deadline' failure trial (kept INVARIANT-complete)."""
    return {
        "test_id": item["test_id"],
        "family": item.get("family"),
        "score": 0.0,
        "passed": False,
        "expected_answer": item.get("expected_answer"),
        "prompt_tokens_measured": item.get("prompt_tokens_measured"),
        "failure_mode": "arm_deadline",
        "content": "",
        "response_first_200": "",
        "completion_tokens": None,
        "wall_ms": None,
        "fill_ratio": item.get("fill_ratio"),
        "request_start_monotonic": None,
        "response_end_monotonic": None,
        # A deadline/drained trial NEVER dispatched an HTTP request, so its request interval
        # is UNKNOWN. dispatch_ts/complete_ts are None so the diff's was_co_batched
        # derivation treats this id's co-batching as uncertain and the coverage gate FAILS
        # CLOSED (a cell containing such trials cannot reach the required coverage to certify
        # -> green_unverified, never green).
        "dispatch_ts": None,
        "complete_ts": None,
        "timed_out": True,
        "reason": "arm_deadline exceeded",
    }


def _strip_meta(arm_map: dict) -> dict:
    """Return the {test_id: result} map WITHOUT the reserved _arm_meta key."""
    return {k: v for k, v in arm_map.items() if k != "_arm_meta"}


def _accepted_kwargs(func, candidate: dict) -> dict:
    """Return the subset of ``candidate`` kwargs that ``func`` actually accepts.

    Used to thread additive knobs (``logit_drift_eps`` / ``ctx_sweep``) into the PURE diff
    ONLY when its signature declares them, so this driver stays correct regardless of the
    diff's version and NEVER raises a TypeError by passing an unknown kwarg. A function
    declaring **kwargs is treated as accepting everything. Pure; on any introspection
    failure returns {} (the call falls back to the diff's exact behavior)."""
    if not candidate:
        return {}
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidate)
    return {k: v for k, v in candidate.items() if k in params}


# ---------------------------------------------------------------------------
# URL discipline. Every dispatch URL goes through _join_url, which refuses any port the
# caller marked untouchable (the baseline server + explicit extras). The untouchable set
# is computed per-driver from --baseline-port / --untouchable-port, NOT a hard-coded band.
# ---------------------------------------------------------------------------
def _join_url(base_url: str, path: str, untouchable_ports: frozenset[int] | None = None) -> str:
    """Join an arbitrary base_url with a path; refuse an untouchable port if given.

    ``untouchable_ports`` is the per-run guard set (baseline + extras). When None (the
    default for the low-level score path, where the base_url is already the validated
    experiment/mock URL) no port check is applied. The driver passes its computed set at
    the seam that builds the arm base URL."""
    base = base_url.rstrip("/")
    if untouchable_ports:
        for bad in untouchable_ports:
            assert f":{int(bad)}" not in base, \
                f"refuse untouchable port in base_url {base!r}"
    return base + (path if path.startswith("/") else "/" + path)


def build_server_argv(server_bin: str, model_path: str, port: int,
                      server_ctx: int, parallel: int,
                      server_flags: tuple[str, ...] = DEFAULT_SERVER_FLAGS,
                      host: str = "127.0.0.1") -> list[str]:
    """Standard llama.cpp-style server argv with explicit --parallel and -c=server_ctx.

    ``server_ctx`` is ``kv_budget.server_ctx_for(ctx, parallel)`` for arms B/C (= ctx*N)
    and == ctx for arm A (parallel=1). Binds ``--host 127.0.0.1`` (loopback; the experiment
    is on-box)."""
    return [str(server_bin), "-m", str(model_path),
            "--host", str(host), "--port", str(int(port)),
            *server_flags, "-c", str(int(server_ctx)),
            "--parallel", str(int(parallel))]


class LiveInvarianceDriver:
    """Owns the experiment server lifecycle and the A/B/C run. ONE instance per ctx run.

    Reuses ``http_harness.{wait_for_ready, cleanup_pids, _os_kill, _SPAWNED_PIDS}`` for
    readiness + PID-scoped teardown, ``kv_budget.{is_port_free, gguf_size_gb, kv_total_gb,
    server_ctx_for, clamp_n_to_live_mem, PER_SERVER_OVERHEAD_GB, HARD_FLOOR_GB}`` for the
    footprint guard, and the pluggable ``workset`` / ``scorer`` seams for prompts + scoring.
    Dependencies (popen/kill/mem_reader/wait_ready/is_port_free) are INJECTABLE so the whole
    flow is unit-testable against a mock with NO real launch."""

    def __init__(self, *, server_bin: str, model_path: str, model_alias: str,
                 port: int, ctx: int, parallel: int, n_predict: int,
                 out_dir: str, cert_source: str, reps: int,
                 seed_base: int = 1234,
                 baseline_port: int = DEFAULT_BASELINE_PORT,
                 untouchable_ports: tuple[int, ...] | None = None,
                 untouchable_pids: tuple[int, ...] | None = None,
                 server_flags: tuple[str, ...] = DEFAULT_SERVER_FLAGS,
                 req_timeout: int = DEFAULT_REQ_TIMEOUT_S,
                 ready_timeout: int = DEFAULT_READY_TIMEOUT_S,
                 grace_s: float = DEFAULT_GRACE_S,
                 arm_deadline: float = DEFAULT_ARM_DEADLINE_S,
                 n_probs: int = 0,
                 kv_label: str = "q8_0",
                 workset_path: str | None = None,
                 scorer: Callable = DEFAULT_SCORER,
                 invariant_fields: tuple[str, ...] | None = None,
                 max_footprint_gb: float = MAX_FOOTPRINT_GB,
                 gguf_gb: float | None = None,
                 # --- detection-power T, DECOUPLED from reps (additive, safe default) ---
                 # gate_passes is the number of ARM_C concurrent re-passes the AC gate unions
                 # over (T). DEFAULT None => fall back to max(1, reps) so existing
                 # programmatic callers/tests keep the legacy behavior exactly. Raising T can
                 # only make a verdict MORE conservative (more interleavings; any pass => RED).
                 gate_passes: int | None = None,
                 # --- SWA-aware needle placement + empty-retry (additive, safe default) ---
                 # position_strategy 'fixed' is the default. 'adaptive' only changes placement
                 # when SWA is DETECTED, and only ever moves the needle EARLIER (inside the
                 # window) so an SWA model completes instead of empty -- it never hides a
                 # divergence (A and C use the SAME placement + test_id).
                 position_strategy: str = "fixed",
                 empty_retries: int = 0,
                 swa_window: int | None = None,
                 # --- ctx-sweep + logit-drift-eps plumbing (consumed by the diff) ---
                 # Both DEFAULT OFF. They are threaded to the PURE diff
                 # (decide_status/build_cert) ONLY when its signature accepts them
                 # (introspected at call time). Neither can make a verdict MORE lenient.
                 logit_drift_eps: float = 0.0,
                 ctx_sweep: bool = False,
                 # injectables for tests (default to the real primitives):
                 popen=subprocess.Popen,
                 kill=_os_kill,
                 mem_reader=read_mem_available_mib,
                 wait_ready=wait_for_ready,
                 is_port_free=None,
                 # test seam: when set, arms dispatch against this base_url instead of
                 # http://127.0.0.1:port (so a mock on an ephemeral port can drive the REAL
                 # run_arm/score_one path with NO real launch).
                 arm_base_url: str | None = None,
                 use_slot_poller: bool = True) -> None:
        self.server_bin = str(server_bin)
        self.model_path = str(model_path)
        self.model_alias = str(model_alias)
        self.port = int(port)
        self.ctx = int(ctx)
        self.parallel = int(parallel)
        self.n_predict = int(n_predict)
        self.out_dir = str(out_dir)
        self.cert_source = str(cert_source)
        self.reps = int(reps)
        self.seed_base = int(seed_base)
        self.baseline_port = int(baseline_port)
        # The per-run guard set: the baseline server plus any explicit extras. Computed
        # ONCE here so every dispatch/launch/precheck site reads the SAME set.
        extras = tuple(int(p) for p in (untouchable_ports or ()))
        self.untouchable_ports = frozenset({int(self.baseline_port), *extras})
        self.untouchable_pids = frozenset(int(p) for p in (untouchable_pids or ()))
        self.server_flags = tuple(server_flags)
        self.req_timeout = int(req_timeout)
        self.ready_timeout = int(ready_timeout)
        self.grace_s = float(grace_s)
        self.arm_deadline = float(arm_deadline)
        self.n_probs = int(n_probs)
        self.kv_label = str(kv_label)
        self.workset_path = workset_path
        self.scorer = scorer if callable(scorer) else DEFAULT_SCORER
        self.invariant_fields = (tuple(invariant_fields)
                                 if invariant_fields else tuple(cd.INVARIANT_FIELDS))
        self.max_footprint_gb = float(max_footprint_gb)
        self._gguf_gb_override = (float(gguf_gb) if gguf_gb is not None else None)
        # Store T as-given (may be None). effective_gate_passes() resolves the None-fallback
        # to max(1, reps) so the coupling break is transparent to callers.
        self.gate_passes = None if gate_passes is None else int(gate_passes)
        self.position_strategy = str(position_strategy or "fixed")
        self.empty_retries = max(0, int(empty_retries))
        self.swa_window = None if swa_window is None else int(swa_window)
        self.logit_drift_eps = float(logit_drift_eps or 0.0)
        self.ctx_sweep = bool(ctx_sweep)
        # SWA detection result (populated lazily). None == "not yet probed".
        self._swa_window_detected: int | None = None
        self._popen = popen
        self._kill = kill
        self._mem_reader = mem_reader
        self._wait_ready = wait_ready
        self._is_port_free = is_port_free if is_port_free is not None else kv_budget.is_port_free
        self._arm_base_url = arm_base_url
        self._use_slot_poller = bool(use_slot_poller)
        self._proc = None
        self._log_dir = Path(self.out_dir)

    # ------------------------------------------------------------------ guards
    def _gguf_gb(self) -> float:
        if self._gguf_gb_override is not None:
            return self._gguf_gb_override
        try:
            return kv_budget.gguf_size_gb(self.model_path)
        except (KeyError, OSError) as exc:
            # absent file + no override: refuse to under-estimate. The footprint guard
            # treats this as unknown and the caller should pass --gguf-gb for a remote file.
            raise GuardError(
                f"cannot size model gguf {self.model_path!r} (file absent); pass "
                f"--gguf-gb to provide the projected weight size for the footprint guard"
            ) from exc

    def _projected_gb(self, n: int) -> float:
        return (self._gguf_gb()
                + _kv_total_gb(self.ctx, int(n), self.kv_label)
                + kv_budget.PER_SERVER_OVERHEAD_GB)

    def precheck(self) -> None:
        """Run S0-S7 in order; raise GuardError on any violation BEFORE launch.

        On failure the caller writes a 'failed'-status cert (source=cert_source) and exits
        non-zero WITHOUT having launched a server."""
        # S0 -- HARD experiment-footprint cap. Enforce the configured cap BEFORE the live-mem
        # floor: the floor alone would admit a large server on a box with ample free RAM even
        # when running concurrent with the user's baseline. The footprint is the N-slot B/C
        # server (the larger of the two we ever launch). This is a model+KV projection, not a
        # live-mem read, so the cap always applies.
        proj = self._projected_gb(self.parallel)
        if self.max_footprint_gb > 0 and proj > self.max_footprint_gb + 1e-9:
            raise GuardError(
                f"S0: projected experiment footprint {proj:.2f}GB exceeds the "
                f"{self.max_footprint_gb:.2f}GB cap (ctx={self.ctx}, N={self.parallel}, "
                f"server_ctx=ctx*N={_server_ctx_for(self.ctx, self.parallel)}). "
                f"The experiment can run concurrent with your baseline server; lower --ctx "
                f"or --parallel (or raise --max-footprint-gb for an intentional larger run).")
        # S1 -- untouchable port. The experiment port must not be the baseline / an extra.
        if self.port in self.untouchable_ports:
            raise GuardError(f"S1: refuse untouchable port {self.port} "
                             f"(untouchable set: {sorted(self.untouchable_ports)})")
        # S3 -- own-port-free. If something already holds our port we REFUSE (we only ever
        # kill what WE spawned; never adopt/kill an occupant). Skipped when arms run against
        # an injected mock base_url (no real launch).
        if self._arm_base_url is None:
            if not self._is_port_free(self.port):
                raise GuardError(f"S3: port {self.port} is already bound; refusing "
                                 f"(will not adopt/kill an occupant)")
        # S4 -- live-mem floor (BLOCKING). avail<=0 (off-Linux/unreadable) is a permissive
        # no-op. The footprint is sized for the LARGER of the two servers we ever launch.
        avail = self._mem_reader()
        try:
            avail_f = float(avail) if avail is not None else 0.0
        except (TypeError, ValueError):
            avail_f = 0.0
        if avail_f > 0:
            projected = self._projected_gb(self.parallel)
            need = projected + kv_budget.HARD_FLOOR_GB
            if (avail_f / 1024.0) < need:
                raise GuardError(
                    f"S4: live MemAvailable {avail_f/1024.0:.2f}GB < required "
                    f"{need:.2f}GB (projected {projected:.2f}GB + floor "
                    f"{kv_budget.HARD_FLOOR_GB:.2f}GB); refusing to launch")
        # S7 -- live-N clamp (down-only; permissive on avail<=0). Defense-in-depth: never
        # raises N, only lowers it to fit ACTUAL free mem.
        clamped = _clamp_n_to_live_mem(
            self.parallel, self.ctx, self._gguf_gb(), avail_f, kv=self.kv_label)
        if clamped < self.parallel:
            # A clamp means our requested N does not fit live mem even after S4 -- we treat
            # that as a guard refusal rather than silently shrinking the cell, because a
            # different N changes the cert identity (cert_filename keys on N).
            raise GuardError(
                f"S7: requested parallel={self.parallel} clamps to {clamped} under live "
                f"mem; refusing (would change cert identity). Re-run with "
                f"--parallel {clamped}.")

    # --------------------------------------------------------------- lifecycle
    def launch(self, parallel: int, server_ctx: int) -> int:
        """Popen the experiment server (start_new_session=True); return its pid.

        Registers the pid in the spawned-PID set BEFORE any line that can raise, so teardown
        (which snapshots that set) can always reap it. Logs to {out_dir}/server-{port}.log."""
        argv = build_server_argv(self.server_bin, self.model_path, self.port,
                                 int(server_ctx), int(parallel),
                                 server_flags=self.server_flags)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"server-{self.port}.log"
        proc = self._popen(
            argv,
            stdout=log_path.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Register BEFORE storing/returning so a crash between Popen and the next line still
        # leaves the pid reapable by teardown.
        _SPAWNED_PIDS.add(proc.pid)
        self._proc = proc
        return proc.pid

    def await_health(self) -> int:
        """wait_ready(port, ready_timeout); raise on TimeoutError -> teardown."""
        return int(self._wait_ready(self.port, self.ready_timeout))

    def teardown(self) -> None:
        """IDEMPOTENT, PID-scoped teardown. Safe to call >1x (finally + signal handler).

        Snapshots the spawned pid, ASSERTS it is disjoint from the untouchable-pid set (S2),
        cleanup_pids(pids, kill, grace) (SIGTERM -> grace -> SIGKILL), proc.wait(10), discards
        the pid, clears self._proc. NEVER pattern-kills; NEVER touches an untouchable pid."""
        proc = self._proc
        if proc is None:
            return
        pid = getattr(proc, "pid", None)
        if pid is None:
            self._proc = None
            return
        pids = [pid]
        # S2 -- paranoia: refuse to ever pass an untouchable pid to the killer.
        assert set(pids).isdisjoint(self.untouchable_pids), \
            f"S2: refuse to kill untouchable pid(s) {set(pids) & self.untouchable_pids}"
        safe = [p for p in pids if p not in self.untouchable_pids]
        cleanup_pids(safe, self._kill, grace=self.grace_s)
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        _SPAWNED_PIDS.discard(pid)
        self._proc = None

    # ----------------------------------------------------------------- the run
    def _arm_url(self) -> str:
        """The base_url arms dispatch against: injected mock (tests) or the experiment port."""
        if self._arm_base_url is not None:
            return self._arm_base_url
        return f"http://127.0.0.1:{int(self.port)}"

    # ------------------------------------------------------------- detection T
    def effective_gate_passes(self) -> int:
        """Resolve T = number of ARM_C concurrent re-passes the AC gate unions over.

        When ``gate_passes`` was supplied (the CLI default is 8) use it; otherwise FALL BACK
        to ``max(1, int(reps))`` -- the legacy coupling -- so every existing programmatic
        caller/test (which constructs with ``reps`` only) keeps the prior behavior. Always
        >= 1. Raising T can only make the verdict MORE conservative."""
        if self.gate_passes is not None:
            return max(1, int(self.gate_passes))
        return max(1, int(self.reps))

    # ------------------------------------------------------- SWA detection (/props)
    def _probe_props_n_swa(self, base_url: str) -> int | None:
        """Probe GET /props and return default_generation_settings.n_swa (or None).

        A llama.cpp-style server exposes the sliding-window attention size under
        ``/props -> default_generation_settings.n_swa`` (0/absent == dense, no SWA). This
        probe is PURELY READ-ONLY and FAILS SOFT: any transport error, a 404 (a mock with no
        /props route), malformed JSON, or a missing/non-int key all return None. It NEVER
        raises and NEVER affects a verdict directly -- it only informs the adaptive PLACEMENT
        strategy, which itself can only move the needle earlier (the safe direction)."""
        url = _join_url(base_url, "/props")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        dgs = payload.get("default_generation_settings")
        candidates = []
        if isinstance(dgs, dict):
            candidates.append(dgs.get("n_swa"))
        candidates.append(payload.get("n_swa"))  # fallback shape
        for val in candidates:
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                return int(val)
        return None

    def _resolve_swa_window(self, base_url: str) -> int:
        """Resolve the effective SWA window n_swa (0 == no SWA / dense). Memoized.

        Precedence: explicit ``--swa-window`` override (incl. 0 to force dense) > /props
        probe > 0. FAIL-CLOSED toward 'no SWA' (0) so that on any uncertainty the adaptive
        strategy becomes a NO-OP and placement stays at the fixed position -- uncertainty
        NEVER moves the needle. Result is cached for the run."""
        if self._swa_window_detected is not None:
            return self._swa_window_detected
        if self.swa_window is not None:
            win = max(0, int(self.swa_window))
        else:
            probed = self._probe_props_n_swa(base_url)
            win = int(probed) if (probed is not None and probed > 0) else 0
        self._swa_window_detected = win
        return win

    def _swa_active(self, n_swa: int) -> bool:
        """True iff a sliding window is genuinely narrower than the context (0 < n_swa < ctx).
        A window >= ctx is effectively dense, so adaptive is a no-op."""
        return 0 < int(n_swa) < int(self.ctx)

    def _resolve_position_for(self, base_position: float, n_swa: int,
                              jitter_rng: random.Random | None = None) -> float:
        """Resolve the needle position for the active strategy (PURE given its inputs).

        - 'fixed'   : return ``base_position`` unchanged (the default).
        - 'jitter'  : a small SEEDED micro-jitter (+/-0.02) around base_position (clamped to
                      (0,1)); used by the empty-retry path to resample placement.
        - 'adaptive': when SWA is ACTIVE (0 < n_swa < ctx), place the needle at the START
                      (just inside the window) so an SWA model does not instant-EOS to empty;
                      when SWA is absent it is a NO-OP (returns base_position). The adaptive
                      move is EARLIER-only -- it never hides a divergence (A and C share the
                      SAME resolved position + test_id)."""
        strat = self.position_strategy
        if strat == "adaptive":
            if self._swa_active(int(n_swa)):
                # needle at the very start so it lands inside the sliding window
                return max(1e-6, min(0.02, (float(n_swa) / max(1.0, float(self.ctx))) / 2.0))
            return float(base_position)  # SWA absent => no-op
        if strat == "jitter":
            rng = jitter_rng or random.Random(0)
            delta = (rng.random() - 0.5) * 0.04  # +/-0.02
            return max(1e-6, min(1.0 - 1e-6, float(base_position) + delta))
        return float(base_position)  # 'fixed' (default): unchanged

    def _make_workset(self) -> list[dict]:
        """Build the per-trial work-set via the pluggable ``--workset`` JSON loader.

        A /tokenize closure against the SAME base_url makes prompt_tokens_measured land on
        the real per-slot ctx; on any /tokenize failure the loader falls back to estimate
        sizing. The closure is built against the arm base_url (experiment server or mock).
        ONE work-set is built and shared by every arm (so test_ids stay aligned and any
        placement choice is symmetric across A/B/C).

        The generic ``load_workset`` expands fixed JSON row templates and does not itself
        synthesize fillers, so the adaptive/jitter ``--position-strategy`` only influences
        placement when the loader in use accepts a ``position_resolver`` (a profile may
        provide one). We thread the resolver in DEFENSIVELY -- only when the strategy is
        non-default AND ``load_workset`` declares the parameter -- so the call is correct
        against the generic loader (which ignores placement) and any richer one alike."""
        base = self._arm_url()

        def _counter(text: str) -> int:
            return tokenize_count(base, text, timeout=30)

        kwargs = dict(
            ctx=self.ctx,
            reps=self.reps,
            n_predict=self.n_predict,
            n_probs=self.n_probs,
            model_id=self.model_alias,
            token_counter=_counter,
        )

        # Resolve a position_resolver ONLY when the strategy is non-default. For
        # 'adaptive'/'jitter' resolve the SWA window once (probe /props or the --swa-window
        # override; fail-closed to 0 == no SWA), then hand the loader a pure resolver that
        # ALL arms share (placement stays symmetric across A/B/C). Passed only if accepted.
        if self.position_strategy != "fixed":
            n_swa = self._resolve_swa_window(base)

            def position_resolver(base_position, fill_ratio):  # noqa: ANN001 (local)
                return self._resolve_position_for(float(base_position), int(n_swa))

            kwargs.update(_accepted_kwargs(
                load_workset, {"position_resolver": position_resolver}))

        return load_workset(self.workset_path, **kwargs)

    def _retry_prompt_fn(self):
        """Build the per-request empty-retry re-fire closure (or None when disabled).

        Returns a callable ``(item, attempt) -> item'`` used by score_one ONLY when a
        completion is 'empty'/'premature_eos' and --empty-retries > 0. In the generic core
        the closure re-fires the SAME item VERBATIM (identity preserved -- the original
        test_id is kept, so A and C still compare on the same key). A profile that synthesizes
        fresh fillers can supply its own re-placement via the work-set; the generic safe
        default is a verbatim re-fire (it can only convert a blank failure into a real
        completion, never hide a divergence). Returns None when empty_retries == 0."""
        if self.empty_retries <= 0:
            return None

        def _fn(item: dict, attempt: int) -> dict:
            # Generic safe default: re-fire the SAME item (verbatim). score_one re-stamps the
            # original test_id defensively, so the arm-map key never drifts under retry.
            return dict(item)

        return _fn

    def run(self) -> dict:
        """The full A->(B,C)->diff flow. Returns the cert dict (from build_cert).

        Fewest launches: for ARM_A the cleanest faithful reference is a SEPARATE
        ``--parallel 1`` server (server_ctx = ctx) launched + measured + torn down BEFORE the
        N-slot server starts (sequential -> never co-resident -> footprint stays small). Then
        launch ONCE with ``--parallel N`` (server_ctx = server_ctx_for(ctx, N)); run ARM_C
        (concurrent, +SlotPoller) T times and ARM_B (serial) against it. Order: launch(1,ctx)
        -> await_health -> ARM_A (+ ARM_A self-identity control) -> teardown ->
        launch(N, ctx*N) -> await_health -> ARM_C x T (each +SlotPoller) -> ARM_B -> teardown
        -> diff. The AC gate runs on the UNION of the T ARM_C passes (any pass divergent =>
        RED; batch divergence is STOCHASTIC, so one pass can miss a divergence another
        surfaces). Wrap launch..diff in try/finally with teardown() in finally.

        TEST SEAM: when arm_base_url is injected, the arms dispatch against the mock and
        launch/await/teardown are driven through the injected popen/wait_ready/kill (the fakes
        no-op), so the REAL run_arm/score_one/diff path is exercised with NO GPU."""
        base_url = self._arm_url()
        workset = self._make_workset()
        retry_fn = self._retry_prompt_fn()

        # ---- ARM_A: separate --parallel 1 server (the serial ground truth) ----
        arm_a_map: dict = {}
        arm_a_map_2: dict = {}
        try:
            self.launch(parallel=1, server_ctx=self.ctx)
            self.await_health()
            arm_a_map = run_arm(base_url, workset, arm=ARM_A, parallel=1,
                                concurrent=False, n_predict=self.n_predict,
                                req_timeout=self.req_timeout,
                                arm_deadline=self.arm_deadline, n_probs=self.n_probs,
                                empty_retries=self.empty_retries, scorer=self.scorer,
                                retry_prompt_fn=retry_fn)
            # ARM_A self-identity control: run twice; the two maps MUST be byte-identical on
            # INVARIANT_FIELDS, else the HARNESS is nondeterministic.
            arm_a_map_2 = run_arm(base_url, workset, arm=ARM_A, parallel=1,
                                  concurrent=False, n_predict=self.n_predict,
                                  req_timeout=self.req_timeout,
                                  arm_deadline=self.arm_deadline, n_probs=self.n_probs,
                                  empty_retries=self.empty_retries, scorer=self.scorer,
                                  retry_prompt_fn=retry_fn)
        finally:
            self.teardown()

        self_identity_ok = self._maps_invariant_equal(_strip_meta(arm_a_map),
                                                      _strip_meta(arm_a_map_2))

        # ---- ARM_C (concurrent, T re-passes) + ARM_B (serial) on the N-slot server ----
        # Anti-false-GREEN: batch divergence is STOCHASTIC (co-batch composition varies
        # run-to-run with arrival timing), so ARM_C is run T times against the SAME N-slot
        # server and the AC gate is evaluated on the UNION of divergent ids (any pass
        # divergent => RED). A single pass observes ONE interleaving and could miss a
        # divergence that only surfaces when prompts i,j co-batch.
        arm_b_map: dict = {}
        arm_c_passes: list[dict] = []
        peak_busy_across_passes = 0
        max_client_overlap = 0
        server_ctx_n = _server_ctx_for(self.ctx, self.parallel)
        n_passes = self.effective_gate_passes()
        # Per-pass arrival jitter is SEEDED off seed_base so each pass samples a different
        # co-batch interleaving while staying fully reproducible. It perturbs only ARRIVAL
        # TIMING (which requests co-batch), NEVER a score/verdict.
        pass_jitter_rng = random.Random(f"pass-jitter|{int(self.seed_base)}|{int(self.ctx)}")
        try:
            self.launch(parallel=self.parallel, server_ctx=server_ctx_n)
            self.await_health()
            for _pass in range(n_passes):
                # Seeded micro-delay BEFORE each concurrent pass so the worker barrier
                # releases into a different interleaving each pass (bounded, deterministic;
                # skipped on the first pass).
                if _pass > 0 and self._use_slot_poller:
                    time.sleep(pass_jitter_rng.uniform(0.0, 0.01))
                # A FRESH poller per pass so server_peak_busy_slots is the per-pass peak; we
                # fold the MAX across passes below (overlap proof holds if ANY pass co-batched).
                poller = SlotPoller(base_url) if self._use_slot_poller else None
                raw_c = run_arm(base_url, workset, arm=ARM_C, parallel=self.parallel,
                                concurrent=True, n_predict=self.n_predict,
                                req_timeout=self.req_timeout,
                                arm_deadline=self.arm_deadline,
                                slots_poller=poller, n_probs=self.n_probs,
                                empty_retries=self.empty_retries, scorer=self.scorer,
                                retry_prompt_fn=retry_fn)
                c_meta = raw_c.get("_arm_meta", {})
                peak_busy_across_passes = max(
                    peak_busy_across_passes,
                    int(c_meta.get("server_peak_busy_slots", 0) or 0))
                max_client_overlap = max(
                    max_client_overlap,
                    int(c_meta.get("client_max_overlap_depth", 0) or 0))
                arm_c_passes.append(_strip_meta(raw_c))
            arm_b_map = run_arm(base_url, workset, arm=ARM_B, parallel=self.parallel,
                                concurrent=False, n_predict=self.n_predict,
                                req_timeout=self.req_timeout,
                                arm_deadline=self.arm_deadline, n_probs=self.n_probs,
                                empty_retries=self.empty_retries, scorer=self.scorer,
                                retry_prompt_fn=retry_fn)
        finally:
            self.teardown()

        return self._diff_and_build(
            arm_a_map, arm_b_map, arm_c_passes, self_identity_ok,
            server_busy=peak_busy_across_passes,
            client_overlap=max_client_overlap,
            n_passes=n_passes)

    def _maps_invariant_equal(self, map_x: dict, map_y: dict) -> bool:
        """True iff two arm maps match on test_id set AND all configured invariant_fields."""
        if set(map_x) != set(map_y):
            return False
        for tid in map_x:
            a = map_x[tid]
            b = map_y[tid]
            for fld in self.invariant_fields:
                if a.get(fld) != b.get(fld):
                    return False
        return True

    def _diff_and_build(self, arm_a_map: dict, arm_b_map: dict, arm_c_passes,
                        self_identity_ok: bool, *,
                        server_busy: int = 0, client_overlap: int = 0,
                        n_passes: int = 1) -> dict:
        """Hand the per-arm maps to the PURE diff and assemble the cert (no I/O).

        ``arm_c_passes`` is the LIST of T per-pass ARM_C maps (already _strip_meta'd by
        run()); the AC gate is evaluated on the UNION over those passes (any pass divergent
        => RED). ``server_busy``/``client_overlap`` are the MAX observed across all passes
        (overlap proven if ANY pass co-batched). ``n_passes`` (== T) and the per-pass
        AC-divergence counts are stamped on the cert. Tolerant of a bare dict (single-pass
        legacy callers) for ``arm_c_passes``."""
        a = _strip_meta(arm_a_map)
        b = _strip_meta(arm_b_map)
        if isinstance(arm_c_passes, dict):
            c_passes = [_strip_meta(arm_c_passes)]
        else:
            c_passes = [p for p in (arm_c_passes or []) if isinstance(p, dict)]
        # The gate's ARM_C map is the UNION over the T concurrent re-passes.
        c = idiff.build_union_arm_c(a, c_passes)
        per_pass_ac_divergent = idiff.fold_pass_divergence_counts(a, c_passes)
        # The per-pass TOKEN-ONLY (score-clean, content-divergent) audit trail -- the AMBER
        # companion to per_pass_ac_divergent.
        per_pass_ac_token_only = idiff.fold_pass_token_only_counts(a, c_passes)

        report_ac = idiff.compute_divergence_report(a, c, "AC")
        report_ab = idiff.compute_divergence_report(a, b, "AB")
        report_bc = idiff.compute_divergence_report(b, c, "BC")

        client_overlap = int(client_overlap or 0)
        server_busy = int(server_busy or 0)

        # Completion-floor census (the all-failure vacuity). Count genuinely-scored ('ok')
        # trials in BOTH gate arms so decide_status can REFUSE a GREEN on a cell where too
        # few real completions occurred (an all-failure arm compares EQUAL on
        # INVARIANT_FIELDS and would otherwise look "clean"). The ARM_C census is over the
        # UNION map. ARM_B counted for the audit trail only.
        cc_a = idiff.count_ok_completions(a)
        cc_b = idiff.count_ok_completions(b)
        cc_c = idiff.count_ok_completions(c)

        # Per-id co-batching COVERAGE census over the UNION ARM_C map (the representative the
        # gate certifies). The server peak scalars prove only that >=2 requests were
        # co-resident at SOME instant; they do NOT prove WHICH ids shared a forward pass. The
        # union map carries the real dispatch_ts/complete_ts request-interval contract on
        # every row, so this is the AUTHORITATIVE per-id signal. FORWARDED to BOTH
        # decide_status (the per-id coverage gate) and build_cert (the is_promotable
        # backstop) -- without it a stuck-but-busy peak alone could mint a PROMOTABLE green.
        cobatch_coverage = idiff.compute_cobatch_coverage(c)

        # ctx-sweep + logit-drift-eps are consumed by the PURE diff. Thread them in
        # DEFENSIVELY -- only the kwargs decide_status actually declares are passed (a diff
        # that has not added them is called exactly as before). These can only make the
        # verdict more conservative.
        _diff_knobs = {"logit_drift_eps": float(self.logit_drift_eps),
                       "ctx_sweep": bool(self.ctx_sweep)}
        decision = idiff.decide_status(
            report_ac, report_ab, report_bc,
            client_overlap_depth=client_overlap,
            server_busy_slots=server_busy,
            n_ok_arm_a=cc_a["n_ok"],
            n_ok_arm_c=cc_c["n_ok"],
            arm_a_map=a,
            arm_c_map=c,
            cobatch_coverage=cobatch_coverage,
            **_accepted_kwargs(idiff.decide_status, _diff_knobs),
        )

        # Harness null control: if ARM_A != ARM_A on INVARIANT_FIELDS the harness is
        # nondeterministic -- force a failed verdict (the gate cannot certify anything).
        if not self_identity_ok:
            decision = dict(decision)
            decision["status"] = idiff.STATUS_FAILED
            reasons = list(decision.get("reasons") or [])
            reasons.append("ARM_A self-identity control FAILED: harness nondeterministic")
            decision["reasons"] = reasons

        overlap = {
            "client_max_overlap_depth": client_overlap,
            "server_peak_busy_slots": server_busy,
            "overlap_ok": bool(decision.get("overlap_ok")),
        }
        _cert_knobs = dict(_diff_knobs)
        cert = idiff.build_cert(
            model=self.model_alias,
            ctx=self.ctx,
            dispatch_n=self.parallel,
            decision=decision,
            report_ac=report_ac,
            report_ab=report_ab,
            report_bc=report_bc,
            overlap=overlap,
            source=self.cert_source,
            reps=self.reps,
            kv_label=self.kv_label,
            n_passes=int(n_passes),
            per_pass_ac_divergent=per_pass_ac_divergent,
            per_pass_ac_token_only=per_pass_ac_token_only,
            completion_counts={"a": cc_a, "b": cc_b, "c": cc_c},
            gate="abc_union",   # the rigorous A/B/C-union driver (the STRONG gate)
            cobatch_coverage=cobatch_coverage,
            **_accepted_kwargs(idiff.build_cert, _cert_knobs),
        )
        return cert


# ---------------------------------------------------------------------------
# Teardown wiring -- all exit paths funnel through one idempotent teardown. A module-level
# registry holds the LIVE driver so the atexit/signal handlers can reach it without a
# global driver reference leak across runs.
# ---------------------------------------------------------------------------
_ACTIVE_DRIVER: LiveInvarianceDriver | None = None


def _teardown_active() -> None:
    drv = _ACTIVE_DRIVER
    if drv is not None:
        try:
            drv.teardown()
        except Exception:
            pass


def _install_signal_teardown() -> None:
    """Register atexit + SIGINT/SIGTERM -> teardown. Guarded for non-main-thread test
    contexts (signal.signal raises ValueError off the main thread)."""
    atexit.register(_teardown_active)

    def _sigint(signum, frame):
        _teardown_active()
        sys.exit(EXIT_SIGINT)

    def _sigterm(signum, frame):
        _teardown_active()
        sys.exit(EXIT_SIGTERM)

    try:
        signal.signal(signal.SIGINT, _sigint)
    except (ValueError, OSError):
        pass
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Programmatic run entry (the CLI in cli.py builds the parser + calls run_main).
# ---------------------------------------------------------------------------
ACK_FLAG = "--i-understand-this-launches-a-real-server"


def run_main(args: argparse.Namespace) -> int:
    """Build a driver from parsed ``args`` and execute the run; return an EXIT_* code.

    This is the orchestration body the CLI calls after parsing. It keeps the port-safety
    FIRST refusal (refuse if experiment_port is in the untouchable set), the --dry-run plan
    preview, the precheck -> failed-cert path, the ack gate, driver.run(), persist_cert,
    is_promotable, and the EXIT_* contract."""
    global _ACTIVE_DRIVER
    out_dir = args.out_dir
    cert_dir = os.path.join(out_dir, "dispatch-cert")

    # Resolve the scorer (pluggable). Default is the builtin exact-match scorer.
    scorer = DEFAULT_SCORER
    spec = getattr(args, "scorer", None)
    if spec:
        scorer = resolve_scorer(spec)

    # Resolve the invariant-field tuple (CLI override or the generic default).
    inv_fields = None
    raw_inv = getattr(args, "invariant_fields", None)
    if raw_inv:
        inv_fields = tuple(f.strip() for f in str(raw_inv).split(",") if f.strip())

    extras = tuple(getattr(args, "untouchable_port", None) or ())
    untouchable_pids = tuple(getattr(args, "untouchable_pid", None) or ())
    baseline_port = int(getattr(args, "baseline_port", DEFAULT_BASELINE_PORT))
    experiment_port = int(getattr(args, "experiment_port", DEFAULT_EXPERIMENT_PORT))
    untouchable_set = frozenset({baseline_port, *(int(p) for p in extras)})

    # PORT SAFETY -- refuse an untouchable port FIRST, before constructing the driver and
    # before the --dry-run preview. This is the FAIL-CLOSED direction: the experiment can
    # never even PREVIEW a plan that addresses the baseline/untouchable port.
    if experiment_port in untouchable_set:
        print(f"REFUSING: --experiment-port {experiment_port} is in the untouchable set "
              f"{sorted(untouchable_set)} (it would collide with your baseline server). "
              f"Choose a free experiment port. No plan printed, nothing launched.",
              file=sys.stderr)
        return EXIT_ERROR

    server_flags = (tuple(getattr(args, "server_flag", None))
                    if getattr(args, "server_flag", None) else DEFAULT_SERVER_FLAGS)

    driver = LiveInvarianceDriver(
        server_bin=args.server_bin, model_path=args.model, model_alias=args.model_alias,
        port=experiment_port, ctx=args.ctx, parallel=args.parallel,
        n_predict=args.n_predict, out_dir=out_dir, cert_source=args.cert_source,
        reps=args.reps, seed_base=args.seed,
        baseline_port=baseline_port, untouchable_ports=extras,
        untouchable_pids=untouchable_pids, server_flags=server_flags,
        req_timeout=args.req_timeout, ready_timeout=args.ready_timeout,
        arm_deadline=args.arm_deadline, n_probs=args.n_probs, kv_label=args.kv,
        workset_path=args.workset, scorer=scorer, invariant_fields=inv_fields,
        max_footprint_gb=args.max_footprint_gb,
        gguf_gb=getattr(args, "gguf_gb", None),
        gate_passes=args.gate_passes, position_strategy=args.position_strategy,
        empty_retries=args.empty_retries, swa_window=args.swa_window,
        logit_drift_eps=args.logit_drift_eps, ctx_sweep=args.ctx_sweep,
    )
    _ACTIVE_DRIVER = driver
    _install_signal_teardown()

    # --dry-run is a pure preview: print the plan + footprint math and launch NOTHING. Run
    # it BEFORE precheck so an operator can preview an over-cap config and SEE the projection
    # that precheck S0 would refuse.
    if args.dry_run:
        try:
            projected = driver._projected_gb(args.parallel)
        except GuardError as ge:
            print(f"DRY RUN: cannot project footprint: {ge}", file=sys.stderr)
            return EXIT_ERROR
        avail = driver._mem_reader()
        try:
            avail_gb = (float(avail) / 1024.0) if avail else 0.0
        except (TypeError, ValueError):
            avail_gb = 0.0
        residual = ((avail_gb - (projected + kv_budget.HARD_FLOOR_GB))
                    if avail_gb > 0 else float("inf"))
        _print_plan(args, projected, residual, experiment_port, untouchable_set)
        return EXIT_PROMOTABLE  # dry-run is a clean preview, not a verdict

    # precheck (S0-S7) BEFORE any launch. On guard failure, write a failed cert and exit
    # non-zero WITHOUT having launched a server.
    try:
        driver.precheck()
    except GuardError as ge:
        cert = idiff.build_cert(
            model=args.model_alias, ctx=args.ctx, dispatch_n=args.parallel,
            decision={"status": idiff.STATUS_FAILED, "divergence_class": None,
                      "anomaly": None, "token_divergence_ids": [],
                      "overlap_ok": False, "reasons": [f"precheck guard: {ge}"]},
            report_ac={}, report_ab={}, report_bc={},
            overlap={"client_max_overlap_depth": 0, "server_peak_busy_slots": 0,
                     "overlap_ok": False},
            source=args.cert_source, reps=args.reps, kv_label=args.kv)
        path = idiff.persist_cert(cert_dir, cert)
        print(f"GUARD REFUSED (S-guard): {ge}")
        print(f"cert(failed) -> {path}")
        return EXIT_ERROR

    # A REAL launch requires the explicit ack flag when stamping a live cert.
    if args.cert_source == "live" and not args.ack:
        print(f"REFUSING to launch a real server without {ACK_FLAG}.\n"
              f"Run with --dry-run to preview, or pass the ack flag.", file=sys.stderr)
        return EXIT_REFUSED

    # Run the driver. teardown is guaranteed by run()'s own try/finally AND the atexit/signal
    # wiring; a launch/health error surfaces as EXIT_ERROR.
    try:
        cert = driver.run()
    except (TimeoutError, OSError, RuntimeError) as e:
        print(f"ERROR before a verdict: {e}", file=sys.stderr)
        return EXIT_ERROR

    path = idiff.persist_cert(cert_dir, cert)
    promotable = idiff.is_promotable(cert)
    status = cert.get("status")
    ac = (cert.get("divergence_report") or {}).get("AC") or {}
    print("=" * 72)
    print(f"live batch-invariance verdict: model={cert.get('model')} ctx={cert.get('ctx')} "
          f"N={cert.get('dispatch_n')} source={cert.get('source')}")
    print(f"  status={status}  AC.n_divergent={ac.get('n_divergent')}  "
          f"overlap_ok={(cert.get('overlap') or {}).get('overlap_ok')}")
    print(f"  cert -> {path}")
    if promotable:
        print("  VERDICT: GREEN (promotable -- batched outputs match serial for this cell)")
        return EXIT_PROMOTABLE
    if status == idiff.STATUS_FAILED:
        print("  VERDICT: RED (score divergence or harness fault -- batched is NOT safe)")
    elif status == idiff.STATUS_GREEN_UNVERIFIED:
        print("  VERDICT: GREEN_UNVERIFIED (no real co-batching observed -- not promotable)")
    elif status == idiff.STATUS_GREEN_WITH_CAVEAT:
        print("  VERDICT: GREEN_WITH_CAVEAT (token-only drift or B anomaly -- sign-off)")
    else:
        print(f"  VERDICT: NOT PROMOTABLE (status={status}, source={cert.get('source')})")
    return EXIT_NOT_PROMOTABLE


def _print_plan(args, projected_gb: float, residual_margin: float,
                experiment_port: int, untouchable_set: frozenset[int]) -> None:
    server_ctx_n = _server_ctx_for(int(args.ctx), int(args.parallel))
    kv = _kv_total_gb(int(args.ctx), int(args.parallel), args.kv)
    print("=== live batch-invariance - DRY RUN (launches NOTHING) ===")
    print(f"model={args.model_alias} gguf={args.model}")
    print(f"ctx={args.ctx}  dispatch_n(N)={args.parallel}  "
          f"server_ctx=ctx*N={server_ctx_n}  n_predict={args.n_predict}")
    cap = float(getattr(args, "max_footprint_gb", MAX_FOOTPRINT_GB))
    print(f"KV(ctx,N)={kv:.2f}GB + weights + overhead -> projected {projected_gb:.2f}GB "
          f"(<= {cap:.2f}GB cap [ENFORCED in precheck S0]: "
          f"{'OK' if projected_gb <= cap + 1e-9 else 'EXCEEDS -> precheck WILL REFUSE'})")
    print(f"residual_margin_vs_floor={residual_margin:.2f}GB")
    safe = int(experiment_port) not in untouchable_set
    print(f"experiment_port={experiment_port}  untouchable={'ok' if safe else 'REFUSE'}  "
          f"(untouchable set: {sorted(untouchable_set)})")
    print(f"arms: A(--parallel 1, ctx={args.ctx}) ; B/C(--parallel {args.parallel}, "
          f"ctx*N={server_ctx_n})")
    cert_path = os.path.join(
        args.out_dir, "dispatch-cert",
        cd.cert_filename(args.model_alias, args.ctx, args.parallel))
    print(f"cert -> {cert_path}")
    print("NO server process launched (dry-run).")

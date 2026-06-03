#!/usr/bin/env python3
"""mock_llama_server.py -- a tiny stdlib HTTP stand-in for llama-server (OFFLINE).

PURPOSE: let the concurrent-dispatch plumbing be exercised end-to-end with NO GPU,
NO models, and NO real llama-server. It mimics just enough of llama-server's
OpenAI-compatible HTTP surface that a readiness probe, the dispatch path, and the
batched-dispatch worker pool all work against it:

  GET  /v1/models            -> {"object":"list","data":[{"id":"local"}]}
  GET  /health               -> 200 {"status":"ok"}
  GET  /props                -> {"default_generation_settings":{"n_ctx":C,"n_swa":S,...},
                                 "total_slots":N}  (SWA probe; n_swa 0 == dense)
  GET  /slots                -> [ {"id_slot":i,"is_processing":bool,"id_task":t}, ... ]
                                 (per-id coverage; modern llama-server LIST shape)
  GET  /slots-debug          -> {"max_observed_concurrency":N, "in_flight":M,
                                 "served":K}  (overlap probe)
  POST /tokenize             -> {"tokens":[...]}  (len ~ words in "content")
  POST /v1/chat/completions  -> canned {choices, usage, timings} so a client parses

It can also simulate the real-server quirks the harness guards against:
  --loading-secs N : reply 503 {"error":"Loading model"} for the first N seconds
                     (the "Loading model" readiness gotcha).
  --stuck-slot     : accept a chat request then hang forever (slot-stuck sim, the
                     llama.cpp #20906 hung-slot case). Use with a client-side timeout.
  --slots N        : serve up to N chat requests CONCURRENTLY (a small per-request
                     sleep so overlap is observable); the (N+1)th request QUEUES on a
                     Condition until a slot frees, mirroring llama.cpp's "over-fire ->
                     queue" behavior. Tracks the peak observed concurrency for the
                     /slots-debug overlap assert.
  --stuck-after K  : #20906 repro -- serve K chat requests normally, then HANG the next
                     slot forever (one stuck slot among healthy ones). The realistic
                     single-wedged-slot case. Use with a client-side timeout.

This is the RED-PROOF fixture: the ``--batch-divergence`` / ``--score-divergence`` /
``--empty-completions`` / ``--swa-empty-beyond-window`` knobs make the completion
depend on batch composition (or on the sliding-window / completion-floor failure
modes), so the invariance gate can be driven to a RED / UNVERIFIED verdict OFFLINE --
proving the gate actually BITES -- without a GPU. With every knob OFF (the default) the
mock is HONEST: content is independent of batch composition, so an offline pass is
structurally vacuous and is NEVER promotable (a real GREEN must be earned live).

USAGE (offline tests / CI wiring only -- this is a test stand-in, never a substitute
for a real-server live run): ``python mock_llama_server.py --port 8201 [--slots 8]``.
``python mock_llama_server.py --demo`` runs a self-contained, zero-arg demonstration
(serial vs concurrent, honest vs ``--batch-divergence``) and exits. Binds 127.0.0.1
(loopback) by default so it can never be mistaken for a public server, and refuses to
bind the configured forbidden port (default 8080) so a misconfigured run cannot collide
with an already-running server on that port.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The mock refuses to ever bind this port. Defaults to 8080 (the conventional default
# llama-server port, a likely collision target); overridable via --forbidden-port so a
# caller's own already-running server port can be protected instead.
DEFAULT_FORBIDDEN_PORT = 8080


class _NoReuseHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that binds WITHOUT SO_REUSEADDR, mirroring llama-server.

    The stdlib default is allow_reuse_address=True (SO_REUSEADDR), under which TWO
    servers can bind the SAME port on Linux -- which would silently MASK a port
    collision in any end-to-end test. Real llama-server does NOT set SO_REUSEADDR, so
    a second bind on a live port fails with EADDRINUSE. We mirror that so the mock is
    a faithful collision detector (a 2nd bind on the same port raises), making the
    actual-port-collision class of bug observable in mock e2e tests, not hidden.

    ThreadingHTTPServer already serves each request on its own thread (daemon), so
    the --slots N concurrency model below is exercised exactly as real concurrent
    HTTP would exercise it.
    """
    allow_reuse_address = False
    daemon_threads = True


class _SlotPool:
    """Bounded concurrency gate that mimics llama.cpp's N-slot continuous batching.

    Up to ``max_slots`` chat requests are admitted concurrently; the (N+1)th BLOCKS
    on the condition variable until a slot frees (over-fire -> queue, never dropped).
    Records the PEAK observed in-flight count so a test can ASSERT genuine N-way
    overlap (guarding the false-pass where invariance holds only because nothing
    actually ran concurrently). Also counts total served so --stuck-after can fire
    the (K+1)th request.

    ``max_slots <= 0`` disables the gate entirely (unbounded), preserving the
    legacy single-knob behavior for callers that never pass --slots.
    """

    def __init__(self, max_slots: int) -> None:
        self.max_slots = int(max_slots)
        self._cv = threading.Condition()
        self._in_flight = 0
        self._max_observed = 0
        self._served = 0          # count of requests that have ACQUIRED a slot
        self._enabled = self.max_slots > 0
        # Per-slot occupancy for the LIST-shaped GET /slots view (per-id coverage).
        # Maps a 0-based slot index -> the id_task currently processing in it (a
        # distinct positive int per in-flight request, == its serve_index). A slot
        # absent from this dict is IDLE (id_task -1, is_processing False). This is the
        # ground truth the modern-llama-server /slots list renders from, so a poller's
        # is_processing/id_task list branch sees genuine per-slot ids (not just an
        # aggregate peak). ADDITIVE: snapshot() is unchanged.
        self._slot_task: dict[int, int] = {}

    def acquire(self) -> tuple:
        """Block until a slot is free; return (serve_index, in_flight_at_acquire).

        ``serve_index`` is this request's 1-based serve number (lets --stuck-after
        hang the (K+1)th). ``in_flight_at_acquire`` is the slot occupancy INCLUDING
        this request at the moment it was admitted; a value > 1 means this request
        was genuinely co-resident with >=1 peer (i.e. part of a real batch). The
        batch-divergence knob uses it to make content depend on batch composition,
        the way real continuous batching CAN (llama.cpp #7052) -- so an offline RED
        abort can be exercised WITHOUT monkeypatching the gate.

        ALSO assigns this request the lowest free slot index and records its id_task
        (== serve_index) so the LIST-shaped GET /slots view reflects which slots are
        genuinely busy with which distinct task. Returns
        (serve_index, in_flight_at_acquire) unchanged -- the slot index is internal.
        """
        with self._cv:
            if self._enabled:
                while self._in_flight >= self.max_slots:
                    self._cv.wait()
            self._in_flight += 1
            self._served += 1
            if self._in_flight > self._max_observed:
                self._max_observed = self._in_flight
            # Claim the lowest free slot index for the /slots list view. When the
            # gate is disabled (unbounded), index by (served-1) so distinct
            # concurrent requests still get distinct slot ids.
            idx = self._lowest_free_slot_index()
            self._slot_task[idx] = self._served
            return self._served, self._in_flight

    def _lowest_free_slot_index(self) -> int:
        # Lowest non-negative integer not currently occupied. Bounded by max_slots
        # when the gate is enabled (admission guarantees a free index exists); when
        # unbounded, grows as needed so concurrent ids never collide.
        i = 0
        while i in self._slot_task:
            i += 1
        return i

    def release(self) -> None:
        with self._cv:
            self._in_flight -= 1
            # Free the slot whose id_task is the OLDEST still-held one we own. We do
            # not know which request is calling release (the handler does not thread
            # its slot index back), but FIFO-by-task-id is sufficient for the list
            # view's purpose: the count of busy slots and their distinct ids stay
            # correct. Pop the lowest id_task currently held.
            if self._slot_task:
                oldest_idx = min(self._slot_task, key=lambda k: self._slot_task[k])
                self._slot_task.pop(oldest_idx, None)
            self._cv.notify()

    def slots_list(self) -> list:
        """Render the CURRENT occupancy as a llama.cpp-master-style /slots LIST.

        One dict per slot. For a bounded pool we emit exactly ``max_slots`` slots
        (idle ones included, the real-server shape); for an unbounded pool we emit
        one slot per currently-in-flight request (no fixed bound exists). Each slot
        dict carries at least ``id_slot`` (int), ``is_processing`` (bool), and
        ``id_task`` (int; -1 when idle) -- exactly the keys a modern-schema slot
        poller recognises. ADDITIVE: this is a NEW view; snapshot() (the /slots-debug
        dict) is untouched.
        """
        with self._cv:
            busy_by_idx = dict(self._slot_task)
        if self._enabled:
            n = self.max_slots
            indices = list(range(n))
        else:
            # Unbounded: surface every busy slot index plus index 0 so an idle
            # unbounded pool still returns a non-empty, well-formed list.
            indices = sorted(set(busy_by_idx) | {0})
        out = []
        for idx in indices:
            task = busy_by_idx.get(idx)
            processing = task is not None
            out.append({
                "id_slot": int(idx),
                "is_processing": bool(processing),
                "id_task": int(task) if processing else -1,
            })
        return out

    def snapshot(self) -> dict:
        with self._cv:
            return {
                "max_slots": self.max_slots,
                "in_flight": self._in_flight,
                "max_observed_concurrency": self._max_observed,
                "served": self._served,
            }


def make_handler(started_at: float, loading_secs: float, stuck_slot: bool,
                 decode_tok_s: float, slot_pool: _SlotPool,
                 stuck_after: int = 0, serve_sleep: float = 0.05,
                 hang_release: threading.Event | None = None,
                 batch_divergence: bool = False,
                 stochastic_divergence_p: float = 0.0,
                 score_divergence: bool = False,
                 empty_completions: bool = False,
                 n_ctx: int = 4096,
                 n_swa: int = 0,
                 swa_empty_beyond_window: bool = False,
                 slots_view_override: list | None = None):
    """Build a request handler closed over the server's behavior knobs.

    ``hang_release`` (test-hygiene): an Event the #20906 ``_hang_forever`` slot
    WAITS on instead of an uninterruptible ``time.sleep(3600)``. ``httpd.shutdown()``
    stops ``accept()`` but does NOT terminate handler threads already wedged in a
    stuck slot -- under the real test that LEAKS daemon ``process_request_thread``s
    that survive shutdown and contaminate sibling tests (the observed flaky
    Path(None) TypeError in the 16-thread reentrancy storm). Setting this event
    (via ``httpd.stop_hangs()``) lets every wedged handler return at shutdown so the
    process is left thread-clean. Defaults to a never-set event => the legacy
    hang-forever behavior is preserved for callers that do not drain.

    ``stochastic_divergence_p`` (ADDITIVE, default 0.0 == OFF): a more faithful model
    of llama.cpp #7052 than the deterministic ``batch_divergence`` prefix. Real
    continuous-batching nondeterminism is STOCHASTIC -- a co-batched request diverges
    only SOME of the time, depending on which peers share its forward pass. When this
    is > 0 AND the request was co-batched (``in_flight_at_acquire > 1``), the mock
    prepends ``"BATCHED "`` with probability p, keyed off a per-request hash of
    (prompt, served-index) so a given request's outcome is REPRODUCIBLE within a run
    yet VARIES request-to-request and pass-to-pass. This lets a test prove the gate's
    UNION-over-T-passes catches a divergence that surfaces on only some passes (the
    single-pass false-pass the T=3 union closes), and that the RED rate tracks p.
    p<=0 is a no-op => the existing default-off byte-identity holds.

    ``score_divergence`` (ADDITIVE, default False == OFF): the GENUINE score-flip
    model -- strictly more faithful to #7052 than the ``batch_divergence`` prefix,
    which (with a correctly score-invariant gate) is only a token-only AMBER
    perturbation. When ON and the request was co-batched (in_flight>1), the mock
    ECHOES the user prompt back as the completion. For retrieval/exact/summary tasks
    the prompt CONTAINS the expected answer, so the scorer returns score=1.0 /
    passed=True -- whereas the serial arm's canned ``"x x x"`` scores 0.0 / False.
    That flips score, passed, and failure_mode (all real INVARIANT_FIELDS) => a
    genuine SCORE divergence the gate MUST catch as RED (not AMBER). This is the
    anti-vacuity RED driver: it proves the gate detects a materially differently-SCORED
    completion under batching, not merely byte drift of an equally-wrong answer.
    Default OFF => the honest mock is untouched.

    ``empty_completions`` (ADDITIVE, default False == OFF): the COMPLETION-FLOOR test
    driver. The slot is acquired + ``serve_sleep`` elapses (so co-batching is
    GENUINELY observed on /slots-debug and the client intervals overlap), but the
    completion ``content`` is the EMPTY string -- modelling "server returns 200 + empty
    content under batch pressure" (the realistic all-failure regime when the experiment
    server is starved). Empty content scores 0.0 with ``failure_mode='empty'`` (NOT
    'ok'), so an arm of these is all-failure: it compares EQUAL across arms on
    INVARIANT_FIELDS (looks "clean") yet ZERO trials genuinely scored. This lets a test
    prove the completion-floor gate demotes such a "matching nothing" pass to an
    unverified verdict DESPITE overlap being satisfied. Default OFF => the honest mock
    is untouched.

    ``n_ctx`` / ``n_swa`` (ADDITIVE): the values reported by the GET /props endpoint
    under ``default_generation_settings`` so an offline test can probe the server's
    effective context window AND its sliding-window-attention span exactly the way a
    real driver does (llama.cpp /props -> default_generation_settings.n_swa / .n_ctx).
    ``n_swa`` defaults to 0 == DENSE attention (no SWA); a test sets it to e.g. 512 to
    simulate a SWA build. These are pure telemetry: they do NOT change completion
    content unless ``swa_empty_beyond_window`` is also ON.

    ``swa_empty_beyond_window`` (ADDITIVE, default False == OFF): the SWA-EMPTY-BUG
    driver. Reproduces the real llama.cpp sliding-window failure where a needle that
    sits BEYOND the attention window is invisible to the model, so the completion is
    instant-EOS / empty. When ON **and** ``n_swa > 0`` **and** the request's prompt
    token-proxy (its word count, the same proxy /tokenize uses) EXCEEDS ``n_swa`` (the
    needle is past the window), the mock returns 200 with EMPTY content AND
    ``completion_tokens == 0`` -- so the scorer classifies it ``premature_eos`` (instant
    EOS), distinct from the unconditional ``empty_completions`` knob (which keeps
    completion_tokens non-zero -> ``empty``). When n_swa == 0 (dense) the needle is
    ALWAYS in window so this knob is a no-op regardless of prompt length: that is the
    "SWA off" half of the on/off a scorer test drives. Default OFF => honest mock
    untouched. A test can also force a single request past the window with a body
    ``"prompt_token_proxy"`` override (see _completion_payload) when it does not want to
    pad real text.

    ``slots_view_override`` (ADDITIVE, default None == live view): when provided, GET
    /slots returns THIS exact list verbatim instead of the live _SlotPool occupancy.
    This is the DETERMINISTIC per-id seam: a test supplies a fixed list of per-slot
    dicts to simulate (a) genuine concurrency (several ``is_processing:true`` with
    distinct ``id_task``) or (b) a STUCK-but-busy slot (an ``is_processing:true`` that
    never completes) WITHOUT any timing race. None => the honest live view from
    slot_pool.slots_list(). Never affects /slots-debug.
    """
    _release = hang_release if hang_release is not None else threading.Event()
    try:
        _stoch_p = float(stochastic_divergence_p)
    except (TypeError, ValueError):
        _stoch_p = 0.0
    _score_div = bool(score_divergence)
    _empty = bool(empty_completions)
    try:
        _n_ctx = int(n_ctx)
    except (TypeError, ValueError):
        _n_ctx = 4096
    try:
        _n_swa = int(n_swa)
    except (TypeError, ValueError):
        _n_swa = 0
    _swa_empty = bool(swa_empty_beyond_window)
    _slots_override = (list(slots_view_override)
                       if slots_view_override is not None else None)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence per-request stderr spam in tests
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _is_loading(self) -> bool:
            return (time.time() - started_at) < loading_secs

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                if self._is_loading():
                    # 503 with the exact phrase the readiness probe rejects
                    self._send(503, {"error": "Loading model"})
                    return
                self._send(200, {"object": "list", "data": [{"id": "local"}]})
                return
            if self.path.startswith("/health"):
                if self._is_loading():
                    self._send(503, {"status": "loading model"})
                    return
                self._send(200, {"status": "ok"})
                return
            if self.path.startswith("/props"):
                # SWA probe: real llama-server exposes the effective generation
                # settings (incl. n_ctx and the sliding-window span n_swa) here. A
                # test reads default_generation_settings.n_swa to decide DENSE (0/absent)
                # vs SWA (e.g. 512). Shape mirrors llama.cpp /props.
                self._send(200, {
                    "default_generation_settings": {
                        "n_ctx": _n_ctx,
                        "n_swa": _n_swa,
                        "n_predict": -1,
                        "model": "local",
                    },
                    "total_slots": slot_pool.max_slots,
                })
                return
            # /slots-debug MUST be checked BEFORE /slots: both share the "/slots"
            # prefix, and the legacy dict probe is the longer, more specific path.
            if self.path.startswith("/slots-debug"):
                # Overlap probe: report the peak concurrency the slot pool has observed
                # so a test can assert genuine N-way batching.
                self._send(200, slot_pool.snapshot())
                return
            if self.path.startswith("/slots"):
                # Per-id coverage: the modern llama-server GET /slots LIST. Each entry
                # has id_slot / is_processing / id_task so a poller's list branch can
                # see WHICH distinct tasks co-reside (genuine concurrency) and detect a
                # STUCK-but-busy slot. Returns the deterministic override when supplied,
                # else the live _SlotPool occupancy.
                if _slots_override is not None:
                    self._send(200, _slots_override)
                else:
                    self._send(200, slot_pool.slots_list())
                return
            self._send(404, {"error": f"no route {self.path}"})

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw or b"{}")
            except (ValueError, TypeError):
                return {}

        def _hang_forever(self) -> None:
            # Simulate llama.cpp #20906: the slot hangs and never replies. WAIT on
            # the release event (not an uninterruptible sleep) so a clean shutdown
            # (httpd.stop_hangs()) lets this handler thread return instead of leaking
            # as a daemon that outlives the server and pollutes sibling tests. With
            # the event never set this blocks effectively forever (the real wedge);
            # the client's own timeout is what raises in the worker, exactly as
            # before -- the only change is shutdown-time drainability.
            _release.wait()

        def do_POST(self):
            body = self._read_body()
            if self.path.startswith("/tokenize"):
                content = ""
                if isinstance(body, dict):
                    content = body.get("content") or ""
                ntok = max(1, len(str(content).split()))
                self._send(200, {"tokens": list(range(ntok))})
                return
            if self.path.startswith("/v1/chat/completions") or self.path.startswith("/completion"):
                # Global stuck-slot: EVERY request hangs (legacy single knob).
                if stuck_slot:
                    self._hang_forever()
                # Slot pool: acquire a slot (over-fire QUEUES), track overlap. The
                # serve_index lets --stuck-after hang exactly the (K+1)th request
                # while the first K (and any concurrent peers) complete normally.
                # in_flight_now > 1 means this request was admitted alongside >=1
                # peer (a genuine batch) -- the batch-divergence knob keys off it.
                serve_index, in_flight_now = slot_pool.acquire()
                try:
                    if stuck_after > 0 and serve_index > stuck_after:
                        # one stuck slot among healthy ones -- the realistic #20906.
                        self._hang_forever()
                    # small sleep so concurrent requests genuinely OVERLAP (otherwise
                    # a fast handler could serialize them and hide the batching).
                    if serve_sleep > 0:
                        time.sleep(serve_sleep)
                    self._send(200, self._completion_payload(
                        body, in_flight_at_acquire=in_flight_now,
                        serve_index=serve_index))
                finally:
                    slot_pool.release()
                return
            self._send(404, {"error": f"no route {self.path}"})

        def _completion_payload(self, body: dict,
                                in_flight_at_acquire: int = 1,
                                serve_index: int = 0) -> dict:
            msgs = body.get("messages") if isinstance(body, dict) else None
            prompt_chars = 0
            if msgs:
                prompt_chars = sum(len(str(m.get("content", ""))) for m in msgs)
            max_toks = int(body.get("max_tokens") or 16) if isinstance(body, dict) else 16
            completion_tokens = max(1, min(max_toks, 64))
            gen_ms = (completion_tokens / max(decode_tok_s, 1e-6)) * 1000.0
            # Prompt token-proxy = word count across all message content (the SAME
            # proxy the /tokenize endpoint uses), or an explicit body override so a
            # test can drive "needle beyond window" without padding real text.
            prompt_token_proxy = 0
            if isinstance(body, dict) and body.get("prompt_token_proxy") is not None:
                try:
                    prompt_token_proxy = int(body.get("prompt_token_proxy"))
                except (TypeError, ValueError):
                    prompt_token_proxy = 0
            elif msgs:
                prompt_token_proxy = sum(
                    len(str(m.get("content", "")).split()) for m in msgs)
            # SWA-EMPTY-BUG: when the knob is ON, attention is sliding-window
            # (n_swa>0), and the needle sits BEYOND the window (prompt token-proxy >
            # n_swa), the needle is invisible -> instant-EOS / empty completion. This
            # zeroes completion_tokens (-> scorer 'premature_eos'), distinct from the
            # unconditional empty_completions knob (-> 'empty'). n_swa==0 (dense) or a
            # prompt within the window leaves this OFF, so it is the "SWA off" half of
            # the on/off a scorer test drives. Highest precedence: it models a
            # transport-level instant-EOS that overrides any content-shaping knob.
            swa_empty = (_swa_empty and _n_swa > 0
                         and prompt_token_proxy > _n_swa)
            # Default (all knobs OFF): content is INDEPENDENT of batch composition (the
            # honest mock -- it canNOT prove server-side batch-invariance, only client
            # plumbing). batch_divergence ON makes the content depend on whether this
            # request was admitted in a real batch (>1 in flight), simulating the
            # llama.cpp #7052 nondeterminism so an OFFLINE cert can actually ABORT on a
            # seq-vs-conc divergence without monkeypatching the gate.
            content = "x " * completion_tokens
            co_batched = int(in_flight_at_acquire) > 1
            if swa_empty:
                # Instant-EOS: empty content AND zero completion tokens. finish_reason
                # 'stop' (not 'length') mirrors a model that emitted EOS immediately
                # because it could not see the needle past the sliding window.
                content = ""
                completion_tokens = 0
                gen_ms = 0.0
            elif _empty:
                # COMPLETION-FLOOR driver: a genuinely-co-batched request that
                # nonetheless returns EMPTY content (200 + ""), modelling the starved
                # experiment server. Applies to ALL requests (serial + concurrent) so
                # both the serial and concurrent arms are all-failure and compare EQUAL
                # on INVARIANT_FIELDS -- the "matching nothing" the floor must reject.
                # Takes precedence over the divergence knobs. completion_tokens is left
                # non-zero so the scorer classifies this 'empty' (not 'premature_eos');
                # either way it is a non-'ok' failure that trips the floor.
                content = ""
            elif _score_div and co_batched:
                # GENUINE score flip: echo the user prompt back. For retrieval/exact/
                # summary tasks the prompt contains the answer, so the scorer PASSES
                # this (1.0) while the serial arm's canned "x x x" FAILS (0.0) => a real
                # score/passed/failure_mode divergence (RED), not byte drift.
                echoed = ""
                if msgs:
                    user_msgs = [m for m in msgs
                                 if isinstance(m, dict) and m.get("role") == "user"]
                    src = user_msgs[-1] if user_msgs else (msgs[-1] if msgs else {})
                    echoed = str(src.get("content", "")) if isinstance(src, dict) else ""
                content = echoed or content
            elif batch_divergence and co_batched:
                content = "BATCHED " + content
            elif _stoch_p > 0.0 and co_batched:
                # Stochastic #7052 model: diverge with probability p, keyed off a
                # per-request hash so the outcome is reproducible within a run yet
                # varies request-to-request AND pass-to-pass (serve_index advances
                # every served request, so the same prompt re-fired on a later pass
                # hashes differently). Uses stdlib hashing only (no new import beyond
                # the module's existing ones) -- additive + deterministic.
                import hashlib as _hl
                msg_txt = ""
                if msgs:
                    msg_txt = "".join(str(m.get("content", "")) for m in msgs)
                h = _hl.sha256(f"{msg_txt}|{int(serve_index)}".encode()).digest()
                # map the first 4 bytes to [0,1)
                frac = int.from_bytes(h[:4], "big") / float(1 << 32)
                if frac < _stoch_p:
                    content = "BATCHED " + content
            finish_reason = "stop" if (swa_empty or completion_tokens == 0) else "length"
            payload = {
                "id": "mock-cmpl",
                "object": "chat.completion",
                "model": "local",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": max(1, prompt_chars // 2),
                    "completion_tokens": completion_tokens,
                    "total_tokens": max(1, prompt_chars // 2) + completion_tokens,
                },
                "timings": {
                    "predicted_per_second": decode_tok_s,
                    "predicted_ms": gen_ms,
                    "predicted_n": completion_tokens,
                    "prompt_per_second": decode_tok_s * 10.0,
                },
            }
            # Echo the request's cache_prompt flag so a test can assert the concurrent
            # body sent cache_prompt:false (a named llama.cpp nondeterminism source).
            # Absent on a body that never set it (the serial path), so default-off
            # byte-identity of the request is testable.
            if isinstance(body, dict) and "cache_prompt" in body:
                payload["echo_cache_prompt"] = body.get("cache_prompt")
            # Echo a requested n_probs (the optional logit-drift probe) so an offline
            # test can assert the canary body asked for logprobs. Also return a canned
            # `logprobs` block (content INDEPENDENT of batch composition, like the rest
            # of the honest mock) so a response parser has a logprobs shape to read;
            # this is forensic only and never feeds the score / INVARIANT_FIELDS. Absent
            # unless the body requested n_probs, so default-off byte-identity holds.
            if isinstance(body, dict) and body.get("n_probs"):
                k = int(body.get("n_probs") or 0)
                payload["echo_n_probs"] = k
                payload["choices"][0]["logprobs"] = {
                    "content": [
                        {"token": tok, "logprob": 0.0,
                         "top_logprobs": [{"token": tok, "logprob": 0.0}
                                          for _ in range(max(1, k))]}
                        for tok in content.split()[:completion_tokens]
                    ]
                }
            return payload

    return Handler


def serve(port: int, host: str = "127.0.0.1", loading_secs: float = 0.0,
          stuck_slot: bool = False, decode_tok_s: float = 50.0,
          ready_log: bool = True, slots: int = 0, stuck_after: int = 0,
          serve_sleep: float = 0.05,
          batch_divergence: bool = False,
          stochastic_divergence_p: float = 0.0,
          score_divergence: bool = False,
          empty_completions: bool = False,
          n_ctx: int = 4096,
          n_swa: int = 0,
          swa_empty_beyond_window: bool = False,
          slots_view_override: list | None = None,
          forbidden_port: int = DEFAULT_FORBIDDEN_PORT) -> ThreadingHTTPServer:
    """Construct + return a started-ready ThreadingHTTPServer (caller runs serve_forever).

    Refuses to bind ``forbidden_port`` (default 8080) so even a misconfigured run can't
    collide with an already-running server on that port. Returns the server so tests can
    grab .server_port (works with port=0 for an OS-assigned ephemeral port) and shut it
    down cleanly.

    ``slots``: max concurrent chat requests; the (N+1)th queues. 0 (the default)
    disables the gate -> legacy unbounded behavior. ``stuck_after`` hangs the (K+1)th
    served chat request (one wedged slot among healthy ones). ``serve_sleep`` is the
    small per-request delay that makes concurrent requests genuinely overlap so the
    /slots-debug peak is observable.

    ``stochastic_divergence_p`` (ADDITIVE, default 0.0 == OFF): probability that a
    CO-BATCHED request (in_flight>1) diverges (``"BATCHED "`` prefix), keyed off a
    per-request hash so the outcome is reproducible-within-run yet varies pass-to-pass
    -- a faithful #7052 model that lets a test prove the gate's UNION-over-T-passes
    catches a divergence present on only some passes. 0.0 leaves the honest mock
    untouched.

    ``score_divergence`` (ADDITIVE, default False == OFF): the GENUINE score-flip model
    -- when a request is co-batched the mock echoes the user prompt (which for
    retrieval/exact/summary tasks contains the answer) so the scorer PASSES it while the
    serial arm FAILS the canned content => a real SCORE divergence (RED). The
    anti-vacuity RED driver. Default OFF => honest mock untouched.

    ``empty_completions`` (ADDITIVE, default False == OFF): the COMPLETION-FLOOR driver
    -- every request acquires a slot + sleeps (overlap is genuinely observed) but returns
    200 + EMPTY content, so an arm is all-failure ('empty') yet co-batched. Lets a test
    prove the completion-floor gate rejects a "matching nothing" all-failure pass even
    when overlap holds. Default OFF => honest mock untouched.

    ``n_ctx`` / ``n_swa`` (ADDITIVE): values reported by the GET /props endpoint
    (default_generation_settings) for the SWA probe. ``n_swa`` defaults to 0 == DENSE
    (no sliding window); a test sets 512 to simulate a SWA build. Pure telemetry unless
    ``swa_empty_beyond_window`` is also ON.

    ``swa_empty_beyond_window`` (ADDITIVE, default False == OFF): the SWA-EMPTY-BUG
    driver. When ON AND n_swa>0 AND a request's prompt token-proxy exceeds n_swa (the
    needle is past the sliding window), the completion is instant-EOS / empty with
    completion_tokens==0 (-> scorer 'premature_eos'), the real SWA-needle-invisible
    failure. n_swa==0 (dense) or an in-window prompt is a no-op. Default OFF => honest
    mock untouched.

    ``slots_view_override`` (ADDITIVE, default None == live view): a fixed list of
    per-slot dicts that GET /slots returns verbatim, the DETERMINISTIC per-id seam for
    simulating genuine concurrency (several is_processing:true with distinct id_task) or
    a STUCK-but-busy slot WITHOUT a timing race. None => the live _SlotPool occupancy.
    Never affects /slots-debug.

    The slot pool is attached as ``httpd.slot_pool`` so a test can read its snapshot
    directly without an HTTP round-trip if it prefers.
    """
    if int(port) == int(forbidden_port):
        raise ValueError(
            f"mock refuses to bind the forbidden port :{int(forbidden_port)}")
    slot_pool = _SlotPool(slots)
    # Release event for the #20906 hang slot so wedged handler threads can be drained
    # at shutdown (test hygiene -- prevents leaked daemons contaminating sibling tests).
    hang_release = threading.Event()
    handler = make_handler(time.time(), loading_secs, stuck_slot, decode_tok_s,
                           slot_pool, stuck_after=int(stuck_after),
                           serve_sleep=float(serve_sleep), hang_release=hang_release,
                           batch_divergence=bool(batch_divergence),
                           stochastic_divergence_p=float(stochastic_divergence_p),
                           score_divergence=bool(score_divergence),
                           empty_completions=bool(empty_completions),
                           n_ctx=int(n_ctx), n_swa=int(n_swa),
                           swa_empty_beyond_window=bool(swa_empty_beyond_window),
                           slots_view_override=slots_view_override)
    # NO SO_REUSEADDR (see _NoReuseHTTPServer) so a 2nd bind on a live port FAILS,
    # exactly like real llama-server -- the mock can then surface port collisions.
    httpd = _NoReuseHTTPServer((host, int(port)), handler)
    httpd.slot_pool = slot_pool  # expose for direct in-process inspection
    httpd._hang_release = hang_release

    def _stop_hangs() -> None:
        """Release every wedged #20906 slot so its handler thread returns.

        Call BEFORE/with httpd.shutdown() in tests that used stuck_slot/stuck_after
        so no daemon process_request_thread leaks past the test (the leak causes a
        flaky Path(None) TypeError in later thread-storm tests). No-op if nothing is
        wedged. Safe to call multiple times.
        """
        hang_release.set()
    httpd.stop_hangs = _stop_hangs
    if ready_log:
        # emit a llama-server-ish startup line so log scrapers/humans see liveness
        print(f"mock-llama-server listening on http://{host}:{httpd.server_port} "
              f"(loading_secs={loading_secs} stuck_slot={stuck_slot} "
              f"slots={slots} stuck_after={stuck_after})", flush=True)
    return httpd


def _run_demo() -> int:
    """Self-contained, zero-GPU demonstration of the RED-proof fixture.

    Starts the mock on an ephemeral loopback port with --slots 2, fires the SAME
    request both ways, and prints the contrast that the invariance gate keys on:

      1) HONEST mock (default): a serial vs a co-batched completion are IDENTICAL --
         the offline pass is structurally vacuous (NEVER promotable; a real GREEN must
         be earned against a real server).
      2) --batch-divergence ON: the co-batched completion gains a "BATCHED " prefix
         while the serial one does not -- exactly the seq-vs-conc divergence the gate
         is built to catch (a RED on a real run).

    Uses only stdlib (urllib) and shuts the servers down cleanly. Returns 0.
    """
    import urllib.request

    def _post(url: str, payload: dict, timeout: float = 5.0) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _content(resp: dict) -> str:
        return (((resp.get("choices") or [{}])[0] or {}).get("message", {})
                or {}).get("content", "")

    def _fire_both(httpd) -> tuple:
        """Return (serial_content, max_cobatched_content) against a running mock."""
        host, port = httpd.server_address[0], httpd.server_port
        url = f"http://{host}:{port}/v1/chat/completions"
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "the answer is 42"}],
            "max_tokens": 4, "temperature": 0.0, "seed": 1,
        }
        # Serial: one request alone (in_flight == 1, never co-batched).
        serial = _content(_post(url, payload))
        # Concurrent: fire several at once so at least one is admitted co-batched.
        results: list = []
        lock = threading.Lock()

        def _worker():
            try:
                c = _content(_post(url, payload))
            except Exception as exc:  # pragma: no cover - demo best-effort
                c = f"<error: {exc}>"
            with lock:
                results.append(c)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # The co-batched contrast is whichever concurrent result differs most from
        # serial; for the demo we just surface the set of distinct concurrent outputs.
        return serial, results

    print("=== mock_llama_server --demo : offline batch-invariance RED proof ===")
    print("(stdlib only, no GPU, no model, loopback only)\n")

    # 1) HONEST mock -- serial and concurrent completions are identical.
    honest = serve(0, slots=2, serve_sleep=0.05, ready_log=False)
    try:
        threading.Thread(target=honest.serve_forever, daemon=True).start()
        serial, conc = _fire_both(honest)
        distinct = sorted(set(conc))
        print("[1] HONEST mock (default, all divergence knobs OFF):")
        print(f"      serial completion      = {serial!r}")
        print(f"      concurrent completions = {distinct!r}")
        identical = all(c == serial for c in conc)
        print(f"      -> serial == concurrent ? {identical}  "
              f"(an offline pass here is VACUOUS and never promotable)\n")
    finally:
        honest.stop_hangs()
        honest.shutdown()

    # 2) --batch-divergence ON -- the co-batched completion diverges.
    dirty = serve(0, slots=2, serve_sleep=0.05, batch_divergence=True, ready_log=False)
    try:
        threading.Thread(target=dirty.serve_forever, daemon=True).start()
        serial, conc = _fire_both(dirty)
        distinct = sorted(set(conc))
        diverged = any(c != serial for c in conc)
        print("[2] --batch-divergence ON (co-batched content gains a 'BATCHED ' prefix):")
        print(f"      serial completion      = {serial!r}")
        print(f"      concurrent completions = {distinct!r}")
        print(f"      -> serial != some concurrent ? {diverged}  "
              f"(this is the seq-vs-conc divergence the gate catches as RED)\n")
    finally:
        dirty.stop_hangs()
        dirty.shutdown()

    print("Done. The gate compares serial vs concurrent per test_id: case [1] is a "
          "vacuous (non-promotable) pass; case [2] is a true-positive RED.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mock llama-server (offline plumbing only)")
    ap.add_argument("--demo", action="store_true",
                    help="run a self-contained, zero-arg RED-proof demonstration "
                         "(serial vs concurrent, honest vs --batch-divergence) and exit. "
                         "Needs no --port, no GPU, no model.")
    ap.add_argument("--port", type=int, default=None,
                    help="TCP port to bind (required unless --demo). Use 0 for an "
                         "OS-assigned ephemeral port.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--forbidden-port", type=int, default=DEFAULT_FORBIDDEN_PORT,
                    help="a port the mock refuses to bind (default 8080); set this to "
                         "your own already-running server's port to protect it.")
    ap.add_argument("--loading-secs", type=float, default=0.0,
                    help="reply 503 'Loading model' for the first N seconds")
    ap.add_argument("--stuck-slot", action="store_true",
                    help="hang on EVERY chat completion (global slot-stuck #20906 sim)")
    ap.add_argument("--slots", type=int, default=0,
                    help="serve up to N chat requests concurrently; the (N+1)th queues "
                         "(over-fire). 0 = unbounded (legacy).")
    ap.add_argument("--stuck-after", type=int, default=0,
                    help="hang the (K+1)th served chat request (one wedged slot among "
                         "healthy ones -- realistic #20906 repro).")
    ap.add_argument("--serve-sleep", type=float, default=0.05,
                    help="per-request serve delay (s) so concurrent requests overlap "
                         "observably for the /slots-debug peak (default 0.05).")
    ap.add_argument("--batch-divergence", action="store_true",
                    help="DIAGNOSTIC: make content depend on batch composition "
                         "(prepend 'BATCHED ' when >1 slot in flight), simulating the "
                         "llama.cpp #7052 server-side nondeterminism so an offline cert "
                         "can ABORT on seq-vs-conc divergence. Default OFF (the honest "
                         "mock: content independent of batching).")
    ap.add_argument("--stochastic-divergence-p", type=float, default=0.0,
                    help="DIAGNOSTIC: probability a CO-BATCHED request diverges "
                         "('BATCHED ' prefix), keyed per-request so it varies "
                         "pass-to-pass (faithful #7052 stochastic model for the "
                         "union-over-T-passes gate test). Default 0.0 (OFF).")
    ap.add_argument("--score-divergence", action="store_true",
                    help="DIAGNOSTIC: on a CO-BATCHED request echo the user prompt "
                         "(which contains the answer for retrieval/exact tasks) so the "
                         "scorer PASSES it while the serial arm FAILS => a GENUINE score "
                         "divergence (RED), the anti-vacuity driver. Default OFF.")
    ap.add_argument("--empty-completions", action="store_true",
                    help="DIAGNOSTIC: return 200 + EMPTY content on every (co-batched) "
                         "request so an arm is all-failure ('empty') yet overlapping -- "
                         "the completion-floor gate test driver. Default OFF.")
    ap.add_argument("--n-ctx", type=int, default=4096,
                    help="effective context window reported by GET /props "
                         "(default_generation_settings.n_ctx). Default 4096.")
    ap.add_argument("--n-swa", type=int, default=0,
                    help="sliding-window-attention span reported by GET /props "
                         "(default_generation_settings.n_swa). 0 = DENSE (no SWA, "
                         "default); e.g. 512 simulates a SWA build for the probe.")
    ap.add_argument("--swa-empty-beyond-window", action="store_true",
                    help="DIAGNOSTIC: with --n-swa>0, return instant-EOS / EMPTY "
                         "(completion_tokens=0 -> 'premature_eos') when a request's "
                         "prompt token-proxy exceeds n_swa (needle past the window) -- "
                         "the SWA-empty-bug driver. Default OFF.")
    ap.add_argument("--decode-tok-s", type=float, default=50.0)
    args = ap.parse_args(argv)
    if args.demo:
        return _run_demo()
    if args.port is None:
        ap.error("--port is required (or pass --demo for the zero-arg demonstration)")
    httpd = serve(args.port, args.host, args.loading_secs, args.stuck_slot,
                  args.decode_tok_s, slots=args.slots, stuck_after=args.stuck_after,
                  serve_sleep=args.serve_sleep,
                  batch_divergence=args.batch_divergence,
                  stochastic_divergence_p=args.stochastic_divergence_p,
                  score_divergence=args.score_divergence,
                  empty_completions=args.empty_completions,
                  n_ctx=args.n_ctx, n_swa=args.n_swa,
                  swa_empty_beyond_window=args.swa_empty_beyond_window,
                  forbidden_port=args.forbidden_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

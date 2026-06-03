"""The offline mock server honours its endpoints + divergence knobs (stdlib HTTP only).

The mock is the RED-proof fixture: with every knob OFF it is HONEST (completion independent
of batch composition), and the divergence knobs make the completion depend on batch
composition so the gate can be driven RED / UNVERIFIED offline. These tests pin the
endpoint contract (/v1/models, /props, /slots, /tokenize, /v1/chat/completions) and the
knob behaviours via direct loopback HTTP -- no GPU, no model.
"""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from batch_invariance import mock_llama_server as mock


def _get(base: str, path: str, timeout: float = 5.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.getcode(), json.loads(r.read().decode("utf-8"))


def _post(base: str, path: str, payload: dict, timeout: float = 5.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.getcode(), json.loads(r.read().decode("utf-8"))


def _chat(base: str, prompt: str, *, max_tokens: int = 4):
    return _post(base, "/v1/chat/completions", {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 1,
    })


def _content(resp: dict) -> str:
    return (((resp.get("choices") or [{}])[0] or {}).get("message", {}) or {}).get("content", "")


# ---------------------------------------------------------------------------
# Endpoint contract.
# ---------------------------------------------------------------------------
def test_models_endpoint_ready(mock_server):
    with mock_server() as base:
        code, body = _get(base, "/v1/models")
        assert code == 200
        assert body["data"][0]["id"] == "local"


def test_props_reports_ctx_and_swa(mock_server):
    with mock_server(n_ctx=8192, n_swa=512) as base:
        code, body = _get(base, "/props")
        dgs = body["default_generation_settings"]
        assert dgs["n_ctx"] == 8192
        assert dgs["n_swa"] == 512


def test_props_dense_by_default(mock_server):
    with mock_server() as base:
        _, body = _get(base, "/props")
        assert body["default_generation_settings"]["n_swa"] == 0


def test_tokenize_counts_words(mock_server):
    with mock_server() as base:
        code, body = _post(base, "/tokenize", {"content": "one two three four"})
        assert code == 200
        assert len(body["tokens"]) == 4


def test_slots_list_shape(mock_server):
    with mock_server(slots=3) as base:
        code, body = _get(base, "/slots")
        assert code == 200
        assert isinstance(body, list)
        assert len(body) == 3
        for slot in body:
            assert "id_slot" in slot and "is_processing" in slot and "id_task" in slot


def test_slots_debug_is_a_summary_dict(mock_server):
    with mock_server(slots=2) as base:
        code, body = _get(base, "/slots-debug")
        assert code == 200
        assert "max_observed_concurrency" in body and "in_flight" in body


def test_unknown_route_is_404(mock_server):
    with mock_server() as base:
        try:
            urllib.request.urlopen(base + "/nope", timeout=5)
            raised = False
        except urllib.error.HTTPError as e:
            raised = (e.code == 404)
        assert raised


# ---------------------------------------------------------------------------
# Honest mock -- content independent of batch composition.
# ---------------------------------------------------------------------------
def test_honest_completion_is_canned(mock_server):
    with mock_server() as base:
        _, resp = _chat(base, "the answer is 42", max_tokens=4)
        content = _content(resp)
        # Default canned content is "x " repeated -- never echoes / never "BATCHED".
        assert "BATCHED" not in content
        assert set(content.split()) <= {"x"}


# ---------------------------------------------------------------------------
# Divergence knobs (serial path, in_flight==1 -> knobs that key on co-batching are off).
# ---------------------------------------------------------------------------
def test_score_divergence_serial_request_is_not_echoed(mock_server):
    # A lone (serial) request is never co-batched, so score_divergence does NOT echo it.
    with mock_server(score_divergence=True) as base:
        _, resp = _chat(base, "secret answer alpha", max_tokens=4)
        assert "secret answer alpha" not in _content(resp)


def test_empty_completions_returns_blank(mock_server):
    # empty_completions applies to ALL requests (serial + concurrent).
    with mock_server(empty_completions=True) as base:
        _, resp = _chat(base, "anything", max_tokens=4)
        assert _content(resp) == ""


def test_swa_empty_beyond_window_when_needle_past_window(mock_server):
    # n_swa>0 + prompt token-proxy past the window -> instant-EOS empty, completion_tokens 0.
    with mock_server(n_swa=3, swa_empty_beyond_window=True) as base:
        _, resp = _post(base, "/v1/chat/completions", {
            "model": "local",
            "messages": [{"role": "user", "content": "a b c d e f g h"}],  # 8 words > 3
            "max_tokens": 8, "temperature": 0.0, "seed": 1,
        })
        assert _content(resp) == ""
        assert resp["usage"]["completion_tokens"] == 0
        assert resp["choices"][0]["finish_reason"] == "stop"


def test_swa_in_window_is_normal(mock_server):
    # Needle within the window -> normal canned content even with the knob on.
    with mock_server(n_swa=100, swa_empty_beyond_window=True) as base:
        _, resp = _chat(base, "short", max_tokens=4)
        assert _content(resp) != ""


# ---------------------------------------------------------------------------
# Co-batching: concurrent fire genuinely overlaps + score_divergence echoes the prompt.
# ---------------------------------------------------------------------------
def test_concurrent_fire_observes_overlap_and_echo(mock_server):
    needle = "needle token zulu"
    with mock_server(slots=4, serve_sleep=0.05, score_divergence=True) as base:
        # serial baseline: never co-batched -> not echoed.
        _, serial_resp = _chat(base, needle, max_tokens=4)
        assert needle not in _content(serial_resp)

        # fire several at once so at least one is admitted co-batched -> echoed.
        results: list[str] = []
        lock = threading.Lock()

        def worker():
            try:
                _, r = _chat(base, needle, max_tokens=4)
                c = _content(r)
            except Exception as exc:               # pragma: no cover - best effort
                c = f"<err {exc}>"
            with lock:
                results.append(c)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # At least one concurrent completion echoed the prompt (was co-batched).
        assert any(needle in c for c in results)
        # And /slots-debug recorded a peak >= 2 (genuine N-way overlap).
        _, dbg = _get(base, "/slots-debug")
        assert dbg["max_observed_concurrency"] >= 2


# ---------------------------------------------------------------------------
# Refuses the forbidden port (collision guard).
# ---------------------------------------------------------------------------
def test_mock_refuses_forbidden_port():
    with pytest.raises(ValueError):
        mock.serve(8080, forbidden_port=8080, ready_log=False)

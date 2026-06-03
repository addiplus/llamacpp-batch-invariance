"""Work-set determinism + body construction (pure stdlib).

A work-set is built ONCE and shared by every arm, so the diff compares arms per test_id.
Determinism is load-bearing: the same row + rep must mint the SAME seed and test_id across
runs and platforms (stdlib BLAKE2b, not the salted built-in hash), and two same-(family,
fill) rows must still mint DISTINCT keys via the position ladder. ``build_body`` adds the
concurrent-hardening keys ONLY for N>1 so the serial and concurrent request bodies do not
silently diverge.
"""
from __future__ import annotations

import json

import pytest

from batch_invariance import workset


# ---------------------------------------------------------------------------
# seed_from_tuple / mint_test_id -- determinism + distinctness.
# ---------------------------------------------------------------------------
def test_seed_is_deterministic():
    s1 = workset.seed_from_tuple("m", 2048, "fam", 0.5, 0)
    s2 = workset.seed_from_tuple("m", 2048, "fam", 0.5, 0)
    assert s1 == s2


def test_seed_is_nonneg_63bit():
    s = workset.seed_from_tuple("m", 2048, "fam", 0.5, 7)
    assert 0 <= s <= (1 << 63) - 1


def test_seed_varies_with_rep_and_position():
    base = workset.seed_from_tuple("m", 2048, "fam", 0.5, 0)
    assert workset.seed_from_tuple("m", 2048, "fam", 0.5, 1) != base
    assert workset.seed_from_tuple("m", 2048, "fam", 0.6, 0) != base


def test_test_id_excludes_dispatch_n_so_arms_align():
    # The minted id keys on (model, ctx, family, position, rep, fill) -- NOT the slot count,
    # so the serial (A) and concurrent (C) maps share the same keys.
    tid = workset.mint_test_id("m", 2048, "fam", 0.5, 0, 0.1)
    assert tid == "m|2048|fam|0.500000|0|fill0.10"


def test_test_id_distinct_per_rep():
    a = workset.mint_test_id("m", 2048, "fam", 0.5, 0, 0.1)
    b = workset.mint_test_id("m", 2048, "fam", 0.5, 1, 0.1)
    assert a != b


# ---------------------------------------------------------------------------
# build_workset -- expansion, the position ladder, reps, the token counter.
# ---------------------------------------------------------------------------
def test_build_expands_rows_times_reps():
    rows = [{"item": "q1", "expected_answer": "a1"},
            {"item": "q2", "expected_answer": "a2"}]
    items = workset.build_workset(rows, 2048, reps=3)
    assert len(items) == 2 * 3


def test_same_seed_yields_identical_worksets():
    rows = [{"item": "q1", "expected_answer": "a1", "family": "f", "fill": 0.2}]
    a = workset.build_workset(rows, 2048, reps=2, model_id="m")
    b = workset.build_workset(rows, 2048, reps=2, model_id="m")
    assert a == b                                # fully deterministic


def test_distinct_positions_for_same_family_rows():
    # Two rows in the same family/fill still get distinct positions (the ladder) -> distinct
    # test_ids and seeds, so they never collide in the arm map.
    rows = [{"item": "q1", "expected_answer": "a", "family": "f", "fill": 0.1},
            {"item": "q2", "expected_answer": "a", "family": "f", "fill": 0.1}]
    items = workset.build_workset(rows, 2048, reps=1)
    positions = {it["position"] for it in items}
    test_ids = {it["test_id"] for it in items}
    assert len(positions) == 2
    assert len(test_ids) == 2


def test_every_item_carries_the_required_keys():
    rows = [{"item": "q1", "expected_answer": "a1", "family": "f", "fill": 0.3,
             "system": "be terse"}]
    (it,) = workset.build_workset(rows, 4096, reps=1, model_id="m", n_probs=2)
    for key in ("test_id", "family", "position", "rep", "fill_ratio", "seed",
                "prompt_text", "system", "expected_answer", "max_tokens",
                "prompt_tokens_measured", "n_probs"):
        assert key in it
    assert it["family"] == "f"
    assert it["fill_ratio"] == 0.3
    assert it["system"] == "be terse"
    assert it["n_probs"] == 2
    assert it["prompt_tokens_measured"] is None   # no counter supplied


def test_token_counter_stamps_measured_tokens_once():
    rows = [{"item": "one two three", "expected_answer": "a"}]
    calls = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    items = workset.build_workset(rows, 2048, reps=3, token_counter=counter)
    assert all(it["prompt_tokens_measured"] == 3 for it in items)
    # Measured ONCE per row (a property of the prompt), reused across the 3 reps.
    assert len(calls) == 1


def test_token_counter_failure_is_non_fatal():
    rows = [{"item": "q", "expected_answer": "a"}]

    def bad_counter(_text: str) -> int:
        raise RuntimeError("tokenize down")

    (it,) = workset.build_workset(rows, 2048, reps=1, token_counter=bad_counter)
    assert it["prompt_tokens_measured"] is None   # left None, no raise


# ---------------------------------------------------------------------------
# Input shapes + aliases + validation.
# ---------------------------------------------------------------------------
def test_bare_string_list_is_accepted():
    items = workset.build_workset(["q1", "q2"], 2048, reps=1)
    assert len(items) == 2
    assert items[0]["expected_answer"] == ""


def test_prompt_text_alias_is_accepted():
    rows = [{"prompt_text": "hi", "expected_answer": "a"}]
    (it,) = workset.build_workset(rows, 2048, reps=1)
    assert it["prompt_text"] == "hi"


def test_fill_ratio_alias_is_accepted():
    rows = [{"item": "q", "expected_answer": "a", "fill_ratio": 0.7}]
    (it,) = workset.build_workset(rows, 2048, reps=1)
    assert it["fill_ratio"] == 0.7


def test_missing_prompt_raises():
    with pytest.raises(ValueError):
        workset.build_workset([{"expected_answer": "a"}], 2048, reps=1)


def test_non_list_top_level_raises():
    with pytest.raises(ValueError):
        workset.build_workset({"item": "q"}, 2048, reps=1)


def test_empty_list_raises():
    with pytest.raises(ValueError):
        workset.build_workset([], 2048, reps=1)


# ---------------------------------------------------------------------------
# build_body -- concurrent-hardening keys ONLY for N>1.
# ---------------------------------------------------------------------------
def test_serial_body_has_no_concurrent_keys():
    item = {"prompt_text": "hi", "seed": 7}
    body = workset.build_body(item, dispatch_n=1, n_predict=16)
    assert body["temperature"] == 0.0
    assert body["seed"] == 7
    assert body["max_tokens"] == 16
    assert "cache_prompt" not in body
    assert "top_k" not in body


def test_concurrent_body_adds_hardening_keys():
    item = {"prompt_text": "hi", "seed": 7}
    body = workset.build_body(item, dispatch_n=4, n_predict=16)
    assert body["cache_prompt"] is False
    assert body["top_k"] == 1


def test_system_message_only_when_present():
    with_sys = workset.build_body({"prompt_text": "u", "system": "s", "seed": 1},
                                  dispatch_n=1, n_predict=8)
    assert with_sys["messages"][0] == {"role": "system", "content": "s"}
    no_sys = workset.build_body({"prompt_text": "u", "seed": 1}, dispatch_n=1, n_predict=8)
    assert all(m["role"] != "system" for m in no_sys["messages"])


def test_n_probs_block_only_when_positive():
    body = workset.build_body({"prompt_text": "u", "seed": 1}, dispatch_n=1,
                              n_predict=8, n_probs=3)
    assert body["n_probs"] == 3 and body["logprobs"] is True and body["top_logprobs"] == 3
    body0 = workset.build_body({"prompt_text": "u", "seed": 1}, dispatch_n=1,
                               n_predict=8, n_probs=0)
    assert "n_probs" not in body0


# ---------------------------------------------------------------------------
# load_workset -- file I/O wrapper + error surfaces.
# ---------------------------------------------------------------------------
def test_load_workset_from_file(tmp_path):
    rows = [{"item": "q1", "expected_answer": "a1", "family": "f"}]
    p = tmp_path / "ws.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    items = workset.load_workset(str(p), ctx=2048, reps=2)
    assert len(items) == 2


def test_load_workset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        workset.load_workset("/no/such/ws.json", ctx=2048, reps=1)


def test_load_workset_bad_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        workset.load_workset(str(p), ctx=2048, reps=1)

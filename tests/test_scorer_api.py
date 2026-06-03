"""The pluggable scorer contract: default exact-match scorer, resolver, error classifier.

All pure stdlib. The scorer turns one completion into ``(score, passed, failure_mode)``;
those three feed the gate directly (score/passed are INVARIANT_FIELDS, failure_mode drives
the completion floor). The transport classifier covers the no-response path so a one-arm
timeout/oom reads as a divergence rather than a silent pass.
"""
from __future__ import annotations

import pytest

from batch_invariance import scorer_api


# ---------------------------------------------------------------------------
# default_scorer -- exact-match baseline (strict: leniency is the only dangerous way).
# ---------------------------------------------------------------------------
def test_exact_match_scores_one():
    score, passed, mode = scorer_api.default_scorer("42", "42")
    assert score == 1.0 and passed is True and mode == "ok"


def test_mismatch_scores_zero_but_is_ok():
    # "ok" means "a real completion was scored", not "the answer was correct".
    score, passed, mode = scorer_api.default_scorer("43", "42")
    assert score == 0.0 and passed is False and mode == "ok"


def test_whitespace_is_stripped_both_sides():
    score, passed, mode = scorer_api.default_scorer("  42 \n", "42")
    assert score == 1.0 and passed is True


def test_empty_completion_is_empty_failure_mode():
    score, passed, mode = scorer_api.default_scorer("   ", "42")
    assert score == 0.0 and passed is False and mode == "empty"


def test_none_completion_is_empty():
    score, passed, mode = scorer_api.default_scorer(None, "42")
    assert mode == "empty" and score == 0.0


def test_none_expected_matches_empty_completion_text():
    # str(None).strip() vs ""; an empty completion is 'empty' regardless.
    score, passed, mode = scorer_api.default_scorer("", None)
    assert mode == "empty"


def test_item_kwarg_is_accepted_and_ignored():
    score, passed, mode = scorer_api.default_scorer("x", "x", item={"family": "f"})
    assert score == 1.0


# ---------------------------------------------------------------------------
# resolve_scorer -- pkg.mod:fn resolution + the default passthrough.
# ---------------------------------------------------------------------------
def test_resolve_none_returns_default():
    assert scorer_api.resolve_scorer(None) is scorer_api.DEFAULT_SCORER
    assert scorer_api.resolve_scorer("") is scorer_api.DEFAULT_SCORER


def test_resolve_real_callable():
    fn = scorer_api.resolve_scorer("batch_invariance.scorer_api:default_scorer")
    assert fn is scorer_api.default_scorer


def test_resolve_missing_colon_raises():
    with pytest.raises(ValueError):
        scorer_api.resolve_scorer("batch_invariance.scorer_api.default_scorer")


def test_resolve_blank_halves_raise():
    with pytest.raises(ValueError):
        scorer_api.resolve_scorer("  :  ")


def test_resolve_bad_module_raises():
    with pytest.raises(ModuleNotFoundError):
        scorer_api.resolve_scorer("no_such_module_xyz:fn")


def test_resolve_missing_attr_raises():
    with pytest.raises(AttributeError):
        scorer_api.resolve_scorer("batch_invariance.scorer_api:does_not_exist")


def test_resolve_non_callable_raises():
    with pytest.raises(TypeError):
        scorer_api.resolve_scorer("batch_invariance.scorer_api:OK_FAILURE_MODE")


# ---------------------------------------------------------------------------
# classify_transport_error -- OOM > timeout > other (all non-'ok').
# ---------------------------------------------------------------------------
def test_classify_oom_precedence():
    assert scorer_api.classify_transport_error(RuntimeError("CUDA error: out of memory")) == "oom"
    assert scorer_api.classify_transport_error(Exception("process killed")) == "oom"


def test_classify_timeout():
    assert scorer_api.classify_transport_error(TimeoutError("request timed out")) == "http_timeout"


def test_classify_other_and_none():
    assert scorer_api.classify_transport_error(ValueError("connection refused")) == "error_other"
    assert scorer_api.classify_transport_error(None) == "error_other"


def test_all_transport_labels_are_non_ok():
    # Every transport label must be != "ok" so a one-arm transport failure is a divergence.
    for exc in (RuntimeError("oom"), TimeoutError("timeout"), ValueError("x"), None):
        assert scorer_api.classify_transport_error(exc) != scorer_api.OK_FAILURE_MODE


def test_reserved_labels_match_invariance_diff():
    from batch_invariance import invariance_diff as idiff
    assert scorer_api.OK_FAILURE_MODE == idiff.OK_FAILURE_MODE

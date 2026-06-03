#!/usr/bin/env python3
"""spark_scorer.py -- bind the agentic-coding ``quality_scorer`` into the generic
``scorer_api`` contract (PURE, stdlib-only).

The invariance gate plugs in a scorer via ``--scorer pkg.mod:fn`` and calls it with
the generic 3-tuple contract from :mod:`batch_invariance.scorer_api`::

    def scorer(response_text: str, expected_answer, *, item: Optional[dict] = None
              ) -> tuple[float, bool, str]:
        '''Return (score, passed, failure_mode).'''

This module is the thin adapter that maps the per-family ``quality_scorer`` (exact /
numeric / summary / multi-key / format / function-call / code-edit scoring, plus the
no-LLM failure-mode classifier) onto that contract. Point the driver at it with::

    --scorer batch_invariance.profiles.spark.spark_scorer:score

The adapter is deliberately small: it reads the work-set item's ``family`` (the only
extra signal ``quality_scorer`` needs beyond the response text + gold answer), scores
the completion, derives a canonical ``failure_mode``, and returns the 3-tuple the gate
compares across arms. It is pure stdlib (only the sibling ``quality_scorer``); it adds
no runtime dependency and is fully unit-testable offline.

RESERVED ``failure_mode`` LABELS (see :mod:`batch_invariance.scorer_api`): ``"ok"`` is
the ONLY label the completion floor counts as a genuinely-scored trial;
``"empty"`` / ``"premature_eos"`` mark a blank / instant-EOS completion (the
empty-retry trigger). ``quality_scorer.classify_failure_mode`` emits exactly those
labels (plus ``runaway_repetition`` / ``reasoning_leak`` / ``no_valid_json`` /
``parse_fail`` / ``instruction_fade`` / ``oom`` / ``http_timeout`` / ``error_other``),
so the gate's floor + retry behaviour is honoured out of the box.

WHY ``family`` MATTERS: ``quality_scorer.score_response`` dispatches on the family
(e.g. ``E2`` parses a JSON object, ``E4`` checks AST-equivalence, ``A2`` is exact
match). The generic work-set carries ``family`` on every item (it is a free-form
field the gate also reads for its per-family aggregates), so the adapter pulls it from
``item["family"]``. When no item / family is supplied (the contract allows ``item`` to
be ``None``), the adapter falls back to an exact-match score so it never raises on a
bare ``score(text, expected)`` call -- a scored result is always returned.

NOTE on ``extracted_answer``: the generic gate's ``INVARIANT_FIELDS`` is the 5-tuple
``(score, passed, expected_answer, prompt_tokens_measured, failure_mode)`` -- it does
NOT include ``extracted_answer``. The 3-tuple contract here matches that. A deployment
that wants ``extracted_answer`` re-added to the compared fields (via
``--invariant-fields``) would own its own normalization of the scorer's raw extracted
text; this adapter stays at the clean 3-tuple and does not thread that field.
"""
from __future__ import annotations

from . import quality_scorer

# Reserved labels re-exported from the generic contract so a caller importing this
# adapter does not have to reach into two modules. ``OK_FAILURE_MODE`` is the literal
# the completion floor counts; it equals ``quality_scorer``'s ``"ok"`` label.
OK_FAILURE_MODE = "ok"
EMPTY_FAILURE_MODE = "empty"
PREMATURE_EOS_FAILURE_MODE = "premature_eos"

# The families ``quality_scorer.score_response`` knows how to grade. Used only to pick
# the exact-match fallback for an unknown/absent family (never to reject input).
_KNOWN_FAMILIES = frozenset(
    {"A1", "A2", "A3", "A4", "B1", "B2", "C1", "D1", "E1", "E2", "E3", "E4", "E5"}
)


def _exact_fallback(response_text: str, expected_answer) -> tuple[float, bool, str]:
    """Exact-match 3-tuple used when no family is available (``item`` is ``None`` or
    carries no recognised ``family``). Strict by construction -- the gate's only
    dangerous direction is leniency, so an over-strict fallback is safe.

    ``score`` is 1.0 iff the stripped completion equals the stripped gold answer;
    ``passed`` is ``score >= 0.5``; ``failure_mode`` is ``"ok"`` for any non-blank
    completion (it was genuinely scored, right or wrong) else ``"empty"`` (counts
    against the completion floor). Never raises.
    """
    text = "" if response_text is None else str(response_text)
    stripped = text.strip()
    expected = "" if expected_answer is None else str(expected_answer).strip()
    score = 1.0 if stripped == expected else 0.0
    passed = score >= 0.5
    failure_mode = OK_FAILURE_MODE if stripped else EMPTY_FAILURE_MODE
    return score, passed, failure_mode


def score(response_text: str, expected_answer, *,
          item: dict | None = None) -> tuple[float, bool, str]:
    """Score one completion with the agentic-coding ``quality_scorer`` (3-tuple contract).

    The ``--scorer batch_invariance.profiles.spark.spark_scorer:score`` entry point.

    Steps:
      1. Read the family from ``item["family"]`` (the work-set carries it on every
         item). If ``item`` is ``None`` or names no recognised family, fall back to a
         strict exact-match score (never raise).
      2. Delegate to :func:`quality_scorer.score_response` (family, expected, text) for
         the numeric ``score`` + ``passed`` + the per-family ``reason`` text.
      3. Derive the canonical ``failure_mode`` via
         :func:`quality_scorer.classify_failure_mode` (the no-LLM classifier:
         oom/timeout > empty/premature_eos > reasoning_leak > runaway_repetition >
         no_valid_json/parse_fail > instruction_fade > ok). A blank completion is
         routed to ``"empty"`` / ``"premature_eos"`` -- NEVER silently to ``"ok"`` --
         so the gate's completion floor and the driver's empty-retry trigger fire
         correctly.

    Returns ``(score: float, passed: bool, failure_mode: str)``. Pure + deterministic;
    a malformed item / unknown family degrades to the exact-match fallback rather than
    raising, so a single odd item can never abort an arm.
    """
    text = "" if response_text is None else str(response_text)

    family = None
    if item is not None:
        fam = item.get("family")
        if fam is not None:
            family = str(fam)

    if family not in _KNOWN_FAMILIES:
        # No usable family signal -> strict exact-match (still a real, scoreable
        # verdict). This covers a bare score(text, expected) call and any work-set row
        # whose family is free-form / outside the agentic-coding set.
        return _exact_fallback(text, expected_answer)

    # quality_scorer.score_response raises ValueError only for an UNKNOWN family; we
    # have already gated on _KNOWN_FAMILIES, so this branch scores cleanly. It DOES
    # coerce/validate the expected-answer shape per family (e.g. D1/E2/E3 want a dict,
    # E1 a list) -- a JSON-loaded work-set preserves those shapes, so they round-trip.
    result = quality_scorer.score_response(family, expected_answer, text)
    score_val = float(result["score"])
    passed = bool(result["passed"])
    reason = str(result.get("reason", ""))

    failure_mode = quality_scorer.classify_failure_mode(
        family=family,
        score=score_val,
        response_text=text,
        reason=reason,
    )
    return score_val, passed, str(failure_mode)


# The default callable name a deployment wires in via ``--scorer ...:score``.
__all__ = [
    "score",
    "OK_FAILURE_MODE",
    "EMPTY_FAILURE_MODE",
    "PREMATURE_EOS_FAILURE_MODE",
]

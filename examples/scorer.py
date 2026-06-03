#!/usr/bin/env python3
"""examples/scorer.py -- a tiny custom scorer for the batch-invariance gate.

Demonstrates the pluggable scorer seam. A scorer is any callable matching::

    def scorer(response_text, expected_answer, *, item=None) -> (score, passed, failure_mode)

The gate compares ``score`` / ``passed`` / ``failure_mode`` across the serial and batched
arms per test_id, so a scorer just has to be a deterministic function of the completion.

This one is a needle-recall scorer: it PASSES when the model's completion CONTAINS the
expected needle text (case-insensitive substring), which is what a model does when it
correctly recalls the planted phrase. Used by ``examples/run.sh`` against the mock's
``--score-divergence`` knob: a co-batched request echoes the prompt (which contains the
needle) and so PASSES, while the serial arm's canned content does NOT -- a genuine score
divergence the gate catches as RED.

Wire it in with ``--scorer examples.scorer:needle_recall_scorer`` (or, since the gate
resolves ``pkg.mod:fn`` via importlib, any importable dotted path).
"""
from __future__ import annotations

# Reserved failure-mode labels (see batch_invariance.scorer_api). "ok" is the ONLY label
# the completion floor counts as a genuinely-scored trial; an empty completion is "empty".
OK = "ok"
EMPTY = "empty"


def needle_recall_scorer(response_text, expected_answer, *, item=None):
    """Return (score, passed, failure_mode) for a needle-recall completion.

    * ``score``        -- 1.0 if the (case-insensitively) stripped expected needle text is a
                          substring of the completion, else 0.0. An empty expected answer is
                          treated as "always present" (score 1.0) so a degenerate row never
                          spuriously fails.
    * ``passed``       -- ``score >= 0.5`` (True only on a recall hit).
    * ``failure_mode`` -- "ok" when the completion has any non-whitespace text (it was
                          genuinely scored, hit or miss), else "empty".
    """
    text = "" if response_text is None else str(response_text)
    stripped = text.strip()
    needle = "" if expected_answer is None else str(expected_answer).strip()

    if not needle:
        score = 1.0
    else:
        score = 1.0 if needle.lower() in stripped.lower() else 0.0
    passed = score >= 0.5
    failure_mode = OK if stripped else EMPTY
    return score, passed, failure_mode


# A module-level default name the gate's `--scorer examples.scorer:scorer` could also use.
scorer = needle_recall_scorer

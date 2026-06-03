#!/usr/bin/env python3
"""scorer_api.py -- the pluggable scorer contract (PURE, stdlib-only).

The invariance gate compares a fixed set of SCORE-bearing fields across arms (see
``concurrent_dispatch.INVARIANT_FIELDS``). Two of those fields -- ``score`` /
``passed`` -- and the completion-floor census field ``failure_mode`` come from a
SCORER: a callable that turns one server completion into a numeric verdict. This
module defines that callable's contract, a default exact-match implementation, a
``pkg.mod:fn`` resolver so a caller can plug in its own task scorer, and a tiny
transport-error classifier for the no-response path.

THE CONTRACT (the seam the live driver plugs into)::

    def scorer(response_text: str, expected_answer, *, item: Optional[dict] = None
              ) -> tuple[float, bool, str]:
        '''Return (score, passed, failure_mode).'''

  * ``response_text`` is the model's completion text (already extracted from the
    server's JSON ``choices[0].message.content``); ``""`` for a transport failure.
  * ``expected_answer`` is the work-set item's gold answer (free-form -- a string,
    number, or whatever the task uses).
  * ``item`` (keyword-only, optional) is the full work-set item dict, so a scorer that
    needs the family / fill_ratio / prompt can read them; a simple scorer ignores it.
  * The return triple feeds the gate directly: ``score`` (float) and ``passed`` (bool)
    are INVARIANT_FIELDS; ``failure_mode`` (str) is too, AND drives the completion
    floor (only ``"ok"`` trials count as genuinely scored).

RESERVED ``failure_mode`` LABELS (a custom scorer SHOULD emit these to opt into the
gate's floor + retry behaviour; any other string is treated as a generic failure):

  * ``"ok"``             -- a genuinely-scored trial (the ONLY label the completion
                            floor counts; == ``invariance_diff.OK_FAILURE_MODE``).
  * ``"empty"``          -- a blank completion with completion_tokens > 0.
  * ``"premature_eos"``  -- an instant-EOS / zero-token completion (the
                            sliding-window-needle-invisible signature).

``"empty"`` and ``"premature_eos"`` are the two labels an optional empty-retry path
may re-attempt (a retry can only turn a blank failure into real data, never hide a
divergence). The transport classifier below adds ``"oom"`` / ``"http_timeout"`` /
``"error_other"`` for the no-response case; those are NON-``"ok"`` so a timeout that
hits one arm correctly reads as a divergence (failure_mode is an INVARIANT_FIELD).

The whole module is pure stdlib (only ``importlib`` for the resolver), so it adds no
runtime dependency and is fully unit-testable offline.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

# A scorer returns (score, passed, failure_mode). Kept as a module-level alias so a
# caller can annotate its own scorer against the shared contract type.
Scorer = Callable[..., tuple[float, bool, str]]

# The reserved labels (documented above). Exposed as constants so a custom scorer can
# import them instead of hard-coding the strings.
OK_FAILURE_MODE = "ok"
EMPTY_FAILURE_MODE = "empty"
PREMATURE_EOS_FAILURE_MODE = "premature_eos"
# The blank-completion labels an optional retry path may re-attempt.
RETRIABLE_EMPTY_MODES = frozenset({EMPTY_FAILURE_MODE, PREMATURE_EOS_FAILURE_MODE})


def default_scorer(response_text: str, expected_answer, *,
                   item: dict | None = None) -> tuple[float, bool, str]:
    """The built-in exact-match baseline scorer (used when ``--scorer`` is omitted).

    Semantics (deliberately strict -- the gate's only dangerous direction is leniency,
    so a baseline scorer that over-reports a difference is safe; one that silently
    treats different answers as equal is not):

      * ``score``        -- 1.0 if ``response_text.strip() == str(expected_answer).strip()``
                            else 0.0 (exact string match after stripping whitespace).
      * ``passed``       -- ``score >= 0.5`` (True only on the exact-match 1.0).
      * ``failure_mode`` -- ``"ok"`` when the completion has any non-whitespace text
                            (it was genuinely scored, pass or fail), else ``"empty"``
                            (a blank completion -- counts against the completion floor).

    Note ``failure_mode`` is ``"ok"`` even for a wrong-but-non-empty answer: "ok" means
    "this trial produced a real, scoreable completion," not "the answer was correct."
    A correct/incorrect split lives in ``score`` / ``passed``; the floor cares only that
    SOMETHING was scored. ``item`` is accepted for contract symmetry and ignored.
    """
    text = "" if response_text is None else str(response_text)
    stripped = text.strip()
    expected = "" if expected_answer is None else str(expected_answer).strip()
    score = 1.0 if stripped == expected else 0.0
    passed = score >= 0.5
    failure_mode = OK_FAILURE_MODE if stripped else EMPTY_FAILURE_MODE
    return score, passed, failure_mode


# The default callable a caller wires in when no ``--scorer`` is supplied.
DEFAULT_SCORER: Scorer = default_scorer


def resolve_scorer(spec: str | None) -> Scorer:
    """Resolve a ``"pkg.mod:fn"`` scorer spec to its callable; default if ``spec`` is empty.

    ``spec`` is ``"<dotted.module.path>:<attribute>"`` (e.g. ``"myproj.scoring:score"``).
    The module is imported via ``importlib.import_module`` and the attribute fetched off
    it. ``None`` / empty string returns :data:`DEFAULT_SCORER` (the exact-match baseline),
    so ``--scorer`` is optional end-to-end. The resolver does NOT call the scorer; it
    only loads it (a caller invokes it per completion).

    Raises ``ValueError`` on a malformed spec (missing ``":"``), ``ModuleNotFoundError``
    if the module cannot be imported, ``AttributeError`` if the attribute is absent, and
    ``TypeError`` if the resolved object is not callable -- all loud, at startup, before
    any work runs.
    """
    if not spec:
        return DEFAULT_SCORER
    if ":" not in spec:
        raise ValueError(
            f"scorer spec {spec!r} must be 'pkg.module:function' (missing ':')")
    mod_path, _, attr = spec.partition(":")
    mod_path = mod_path.strip()
    attr = attr.strip()
    if not mod_path or not attr:
        raise ValueError(
            f"scorer spec {spec!r} must name both a module and an attribute")
    module = import_module(mod_path)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"scorer spec {spec!r} resolved to a non-callable {type(fn)!r}")
    return fn


def classify_transport_error(exc: BaseException | None) -> str:
    """Map a transport-layer exception to a ``failure_mode`` for the no-response path.

    When a request never yields a 200 body (the server timed out, refused the
    connection, or OOM'd), there is no completion to score -- but the trial must still
    record a NON-``"ok"`` ``failure_mode`` so the gate sees a real difference if the
    same id succeeded in another arm. This is the generic subset of a richer error
    classifier; precedence is OOM > timeout > everything-else:

      * ``"oom"``          -- the error text mentions out-of-memory / CUDA OOM / 'killed'.
      * ``"http_timeout"`` -- the error text mentions 'timed out' / 'timeout'.
      * ``"error_other"``  -- any other transport failure (or ``None``).

    A scorer is NOT consulted on this path (there is no body); the driver records the
    returned label directly. All three are non-``"ok"`` so a one-arm transport failure
    surfaces as a divergence rather than a silent pass.
    """
    if exc is None:
        return "error_other"
    text = str(exc).lower()
    if ("out of memory" in text or "oom" in text or "cuda error" in text
            or "killed" in text):
        return "oom"
    if "timed out" in text or "timeout" in text:
        return "http_timeout"
    return "error_other"

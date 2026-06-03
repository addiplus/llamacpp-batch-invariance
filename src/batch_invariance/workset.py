#!/usr/bin/env python3
"""workset.py -- pluggable work-set loader (PURE, stdlib-only).

A "work-set" is the deterministic list of items the three arms fire (serial arm A,
serial-on-N-slots control arm B, concurrent arm C). The gate compares arms PER
``test_id``, so the same work-set is built ONCE and shared by every arm; this module
turns a JSON file into that shared list of item dicts. It carries no model-, task-,
or corpus-specific knowledge -- every prompt and gold answer comes from the JSON, so
the engine stays task-agnostic (a caller brings its own prompts + its own scorer).

THE INPUT JSON (``--workset <path>``): a top-level list of row templates, each::

    {
      "item":            "<the user prompt text>",      # required (alias: "prompt_text")
      "expected_answer": "<the gold answer>",            # required
      "family":          "<free-form group label>",      # optional, default "default"
      "fill":            0.30,                       # optional, default 0.0 (alias: "fill_ratio")
      "system":          "<optional system prompt>"      # optional
    }

A bare string list (``["q1", "q2", ...]``) is also accepted -- each string becomes an
item with an empty expected_answer -- so the smallest possible work-set is trivial to
write. The keys ``family`` / ``fill`` exist only because the gate's per-family and
per-fill aggregates read them back off each item; they are free-form (no registry
validation) in this generic core.

THE OUTPUT (one dict per (row x rep)), carrying EXACTLY what an arm needs to fire one
request and score it identically across arms::

    {test_id, family, position, rep, fill_ratio, seed, prompt_text, system,
     expected_answer, max_tokens, prompt_tokens_measured, n_probs}

DETERMINISM: each row gets a distinct ``position`` on an evenly-spread ladder
(``(row_idx + 1) / (n_rows + 1)``) so two same-(family, fill) rows still mint distinct
``test_id`` / ``seed`` keys; ``seed`` is derived by a stdlib BLAKE2b hash of
``(model_id, ctx, family, position, rep)`` (NO third-party RNG). ``prompt_tokens_measured``
is filled by an OPTIONAL injected ``token_counter`` (a ``/tokenize`` closure the driver
passes) so the value -- an INVARIANT_FIELD -- is stamped ONCE per item and reused for
every arm (it is a property of the prompt, not of the dispatch); ``None`` when no
counter is supplied. The module is pure (no network) unless that injected counter
itself does I/O.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

# A stable model id stamped into the seed/test_id derivation so the work-set is
# reproducible regardless of the on-disk model path. Overridable per-call via
# ``model_id=`` (e.g. so two models' worksets get distinct seeds); a caller that does
# not care leaves the default.
DEFAULT_MODEL_ID = "model"

# Bound the per-item seed to a positive 63-bit int so it round-trips cleanly through
# JSON and an OpenAI-compatible ``seed`` body field.
_SEED_MASK = (1 << 63) - 1


def seed_from_tuple(model_id: str, ctx: int, family: str,
                    position: float, rep: int) -> int:
    """Deterministic non-negative seed from the trial identity tuple (stdlib hash).

    Keys on ``(model_id, ctx, family, position, rep)`` -- the same tuple
    :func:`mint_test_id` keys on minus the fill -- via BLAKE2b (stdlib, stable across
    runs and platforms, unlike the salted built-in ``hash()``). ``position`` is
    formatted to 6 decimals so a re-derivation from the minted item is byte-stable.
    The result is masked to a positive 63-bit int.
    """
    key = f"{model_id}|{int(ctx)}|{family}|{float(position):.6f}|{int(rep)}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


def mint_test_id(model_id: str, ctx: int, family: str, position: float,
                 rep: int, fill_ratio: float) -> str:
    """Deterministic per-(model, ctx, family, position, rep, fill) key, IDENTICAL across arms.

    Format: ``f"{model}|{ctx}|{family}|{position:.6f}|{rep}|fill{fill_ratio:.2f}"``.
    The dispatch slot count is intentionally EXCLUDED so arm A (serial) and arm C
    (concurrent) maps align on the same keys -- the whole point of the seam is that the
    pure diff compares two ``{test_id: result}`` maps built from one work-set.
    """
    return (f"{model_id}|{int(ctx)}|{family}|{float(position):.6f}"
            f"|{int(rep)}|fill{float(fill_ratio):.2f}")


def _row_field(row: dict, *names: str, default=None):
    """First present-and-non-None value among ``names`` in ``row`` (key aliasing)."""
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _normalize_rows(raw) -> list[dict]:
    """Coerce the parsed JSON into a list of row-template dicts (or raise ValueError).

    Accepts either a top-level list of dict rows or a top-level list of bare strings
    (each string -> a prompt with an empty expected answer). Any other shape, or a row
    missing a prompt, is a loud ValueError so a malformed work-set fails at load time.
    """
    if not isinstance(raw, list):
        raise ValueError(
            "workset JSON must be a top-level list of row templates "
            f"(got {type(raw).__name__})")
    rows: list[dict] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            rows.append({"item": entry, "expected_answer": ""})
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"workset row {i} must be an object or a string (got "
                f"{type(entry).__name__})")
        prompt = _row_field(entry, "item", "prompt_text", "prompt")
        if prompt is None:
            raise ValueError(
                f"workset row {i} is missing a prompt (expected key 'item' or "
                "'prompt_text')")
        rows.append(entry)
    if not rows:
        raise ValueError("workset JSON is an empty list (no rows to run)")
    return rows


def build_body(item: dict, *, dispatch_n: int, n_predict: int,
               n_probs: int = 0) -> dict:
    """The canonical ``/v1/chat/completions`` body for one work-set item.

    BASE (ALL arms) -- a deterministic greedy request::

        {"model": "local",
         "messages": [{"role": "system", "content": item["system"]}?,
                      {"role": "user",   "content": item["prompt_text"]}],
         "max_tokens": n_predict, "stream": False,
         "temperature": 0.0, "seed": item["seed"]}

    CONCURRENT HARDENING (added ONLY when ``dispatch_n > 1``): ``cache_prompt: False``
    + ``top_k: 1``. The serial arms (A, B) dispatch with ``dispatch_n == 1``, so their
    body MUST NOT carry these keys -- leniency there would silently make the arm-A and
    arm-C bodies differ and confound the gate. The system message is omitted entirely
    when the item has no ``system``. The optional ``n_probs > 0`` block (``n_probs`` +
    ``logprobs`` + ``top_logprobs``) is forensic only -- logprobs are volatile and never
    feed the score fields.
    """
    messages = []
    system = item.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": item["prompt_text"]})
    body: dict = {
        "model": "local",
        "messages": messages,
        "max_tokens": int(n_predict),
        "stream": False,
        "temperature": 0.0,
        "seed": int(item["seed"]),
    }
    if int(dispatch_n) > 1:
        body["cache_prompt"] = False
        body["top_k"] = 1
    if int(n_probs) > 0:
        body["n_probs"] = int(n_probs)
        body["logprobs"] = True
        body["top_logprobs"] = int(n_probs)
    return body


def build_workset(
    rows: list[dict],
    ctx: int,
    *,
    reps: int,
    n_probs: int = 0,
    model_id: str = DEFAULT_MODEL_ID,
    token_counter: Callable[[str], int] | None = None,
) -> list[dict]:
    """Expand row templates across ``reps`` into the shared list of item dicts (PURE).

    The in-memory half of :func:`load_workset` (no file I/O), exposed separately so a
    caller can build a work-set from rows it already holds. Each row is placed on the
    even position ladder, then repeated ``reps`` times; every (row, rep) mints a
    distinct ``test_id`` + ``seed``. ``token_counter`` (optional) stamps
    ``prompt_tokens_measured`` from the prompt text; ``None`` leaves it ``None``.
    """
    norm = _normalize_rows(rows)
    n_rows = len(norm)
    items: list[dict] = []
    for row_idx, row in enumerate(norm):
        family = str(_row_field(row, "family", default="default"))
        fill_ratio = float(_row_field(row, "fill", "fill_ratio", default=0.0))
        prompt_text = str(_row_field(row, "item", "prompt_text", "prompt"))
        system = _row_field(row, "system")
        expected_answer = _row_field(row, "expected_answer", default="")
        position = round((row_idx + 1) / (n_rows + 1), 6)
        # Token count is a property of the prompt, not the dispatch -- measure ONCE per
        # row and reuse for every rep + every arm (prompt_tokens_measured is an
        # INVARIANT_FIELD, so it MUST be identical across arms). A counter failure is
        # non-fatal: leave it None (the gate then simply does not compare on it).
        prompt_tokens_measured = None
        if token_counter is not None:
            try:
                prompt_tokens_measured = int(token_counter(prompt_text))
            except Exception:
                prompt_tokens_measured = None
        for rep in range(int(reps)):
            seed = seed_from_tuple(model_id, int(ctx), family, position, rep)
            test_id = mint_test_id(model_id, int(ctx), family, position, rep, fill_ratio)
            items.append({
                "test_id": test_id,
                "family": family,
                "position": position,
                "rep": rep,
                "fill_ratio": fill_ratio,
                "seed": int(seed),
                "prompt_text": prompt_text,
                "system": system,
                "expected_answer": expected_answer,
                "max_tokens": None,  # the caller fills this via build_body's n_predict
                "prompt_tokens_measured": prompt_tokens_measured,
                "n_probs": int(n_probs),
            })
    return items


def load_workset(
    path: str,
    *,
    ctx: int,
    reps: int,
    n_predict: int = 0,
    n_probs: int = 0,
    model_id: str = DEFAULT_MODEL_ID,
    token_counter: Callable[[str], int] | None = None,
) -> list[dict]:
    """Load a work-set JSON file and expand it into the shared list of item dicts.

    Reads ``path`` (UTF-8 JSON; see the module docstring for the row-template shape),
    then delegates to :func:`build_workset`. Returns the list of item dicts every arm
    iterates. ``n_predict`` is accepted for signature symmetry with the driver's call
    site (the per-request ``max_tokens`` is applied later by :func:`build_body`, not
    baked into the item); it does not change the items here. Raises ``FileNotFoundError``
    if the path is absent, ``ValueError`` on malformed JSON or a bad row shape.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"workset file not found: {path!r}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"workset file {path!r} is not valid JSON: {exc}") from exc
    return build_workset(
        raw, ctx, reps=reps, n_probs=n_probs, model_id=model_id,
        token_counter=token_counter)

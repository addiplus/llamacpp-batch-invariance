#!/usr/bin/env python3
"""spark_workset.py -- map the agentic-coding prompt families onto the generic
work-set loader (PURE, stdlib-only).

The invariance engine fires a deterministic list of items at three arms (serial A,
serial-on-N-slots control B, concurrent C) and compares them per ``test_id``. The
generic :mod:`batch_invariance.workset` turns a JSON file of *row templates* into that
shared list; this module is the thin adapter that builds those row templates from the
agentic-coding :mod:`prompt_generator` families, so this profile is runnable
end-to-end without the caller hand-writing prompts.

TWO WAYS TO USE IT:

  1. Materialize a ``--workset`` JSON file the driver consumes, then run the gate::

        from batch_invariance.profiles.spark import spark_workset
        spark_workset.write_workset_json("spark.json", corner="balanced", ctx=8192)
        # then: batch-invariance run-live --workset spark.json --scorer \\
        #         batch_invariance.profiles.spark.spark_scorer:score ...

  2. Build the row templates (or the fully-expanded item dicts) in-memory and hand them
     to the generic loader yourself (:func:`build_rows` / :func:`build_items`).

WHAT A ROW TEMPLATE IS: the generic loader's input is a list of dicts shaped
``{prompt_text, expected_answer, family, fill, system}`` (see
:mod:`batch_invariance.workset`). :func:`prompt_generator.generate_prompt` already
returns ``prompt_text`` / ``expected_answer`` / ``system_prompt`` / ``family`` for a
given ``(family, seed, target_tokens, position)``; this adapter calls it once per
(family, fill) corner cell and reshapes the result into that template dict. The gold
answer for the structured families (D1 summary, E1 multi-key list, E2/E3 JSON) is a
dict/list and survives the JSON round-trip unchanged, so the matching
:mod:`spark_scorer` grades it correctly on reload.

SEED NOTE: the prompt *content* is fixed deterministically by
``prompt_generator.seed_from_tuple(model_id, ctx, family, position, rep)`` at build
time. The generic loader independently derives the per-item ``seed`` body field (the
OpenAI ``seed``) via its own stdlib hash; the two are decoupled by design -- the prompt
text is already baked into ``prompt_text``, and the body ``seed`` only pins the
server's sampler. Both are fully deterministic, so the work-set is reproducible.

CORPUS: ``generate_prompt`` needs a ``corpus_dir`` for haystack filler but falls back to
bundled in-module text when the directory is absent, so this adapter passes a
caller-chosen (possibly non-existent) path and stays stdlib-only with no data files
required. Supply a real ``corpus_dir`` to use your own filler.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import prompt_generator

# The agentic-coding families this profile samples (a subset of prompt_generator's
# FAMILIES, chosen to span recall, multi-hop math, retrieval, format-adherence,
# function-calling, and code-edit). Each is graded by the companion scorer adapter.
SPARK_FAMILIES: tuple[str, ...] = (
    "A2",  # exact-match secret recall
    "A4",  # multi-step arithmetic chain
    "B1",  # short arithmetic chain
    "C1",  # code-symbol return value
    "D1",  # 3-fact summary (dict, partial credit)
    "E1",  # RULER multi-key retrieval (ordered list)
    "E2",  # IFEval-style JSON format adherence
    "E3",  # function-call JSON tool-call
    "E4",  # code-edit / AST-equivalence
    "E5",  # semantic-hop code-symbol (perfect-retrieval control)
)

# Fill-ratio corners: how full of haystack each prompt is, as a fraction of ``ctx``.
# "balanced" walks low->high fill; "short" stays near-empty (recall is easy, isolates
# pure batching divergence); "long" stays near-full (the regime where divergence is
# most likely). The product (family x fill) is the set of cells the work-set covers.
SAMPLE_CORNERS: dict[str, tuple[float, ...]] = {
    "balanced": (0.10, 0.35, 0.60, 0.85),
    "short": (0.05, 0.15),
    "long": (0.75, 0.90, 0.97),
}

# Suggested concurrency (``--parallel N``) caps per corner: a fuller prompt costs more
# KV per slot, so the long corner caps N lower. These are HINTS for the operator's
# ``--parallel`` choice, NOT enforced here (the gate runs whatever N you pass).
N_CAPS: dict[str, int] = {
    "balanced": 8,
    "short": 16,
    "long": 4,
}

# Default model id stamped into the prompt-content seed derivation. Decoupled from the
# on-disk model path so the same work-set rebuilds identically regardless of where the
# GGUF lives; override per-call if two models should get distinct prompt content.
DEFAULT_MODEL_ID = "model"


def _target_tokens(ctx: int, fill: float) -> int:
    """Token budget for one prompt = ``round(ctx * fill)`` (>=1).

    ``fill`` is the fraction of the server context the haystack+question should
    occupy; the calibrator (in ``prompt_generator``) sizes the filler to hit it.
    """
    return max(1, int(round(float(ctx) * float(fill))))


def build_rows(
    corner: str = "balanced",
    *,
    ctx: int,
    families: tuple[str, ...] | None = None,
    corpus_dir: str = "./corpus",
    model_id: str = DEFAULT_MODEL_ID,
    position: float = 0.5,
) -> list[dict]:
    """Build the generic row-template list for one sampling ``corner`` (PURE).

    For each (family, fill) cell in ``SAMPLE_CORNERS[corner]`` x ``families`` (default
    :data:`SPARK_FAMILIES`), call :func:`prompt_generator.generate_prompt` and reshape
    its output into the row template the generic loader consumes::

        {"family": <fam>, "fill": <fill>, "prompt_text": <text>,
         "expected_answer": <gold>, "system": <system prompt>}

    The result is a plain list of dicts -- JSON-serialisable (:func:`write_workset_json`)
    or feedable straight to :func:`batch_invariance.workset.build_workset`. Determinism:
    the prompt content is seeded from ``(model_id, ctx, family, position, rep=0)`` via
    ``prompt_generator.seed_from_tuple``, so the same arguments always rebuild the same
    rows. ``position`` defaults to mid-haystack (0.5); pass another value to move the
    needle. Raises ``KeyError`` for an unknown ``corner`` and ``ValueError`` (from
    ``generate_prompt``) for an unknown family.
    """
    fills = SAMPLE_CORNERS[corner]
    fam_set = families if families is not None else SPARK_FAMILIES
    rows: list[dict] = []
    for family in fam_set:
        for fill in fills:
            target = _target_tokens(ctx, fill)
            seed = prompt_generator.seed_from_tuple(
                model_id, int(ctx), family, float(position), 0
            )
            spec = prompt_generator.generate_prompt(
                family=family,
                seed=seed,
                target_tokens=target,
                position=float(position),
                corpus_dir=Path(corpus_dir),
            )
            rows.append({
                "family": family,
                "fill": float(fill),
                "prompt_text": spec["prompt_text"],
                "expected_answer": spec["expected_answer"],
                "system": spec["system_prompt"],
            })
    return rows


def build_items(
    corner: str = "balanced",
    *,
    ctx: int,
    reps: int = 1,
    families: tuple[str, ...] | None = None,
    corpus_dir: str = "./corpus",
    model_id: str = DEFAULT_MODEL_ID,
    position: float = 0.5,
    n_probs: int = 0,
) -> list[dict]:
    """Build the fully-expanded item dicts (rows x ``reps``) the arms iterate (PURE).

    Convenience that chains :func:`build_rows` into
    :func:`batch_invariance.workset.build_workset`, so a caller holding no JSON file can
    get the same shared item list the driver would. Each item carries the keys an arm
    needs (``test_id``/``family``/``position``/``rep``/``fill_ratio``/``seed``/
    ``prompt_text``/``system``/``expected_answer``/``prompt_tokens_measured``/
    ``n_probs``). ``prompt_tokens_measured`` is left ``None`` here (no token counter is
    injected offline); the live driver fills it from its ``/tokenize`` closure when it
    builds the work-set itself.

    Imported lazily so importing this profile does not pull the generic work-set module
    until items are actually requested.
    """
    # ``spark`` is ``batch_invariance.profiles.spark``; the generic work-set module is
    # the grandparent package's ``workset``. Import it absolutely (lazily) to avoid a
    # brittle relative-dot level count.
    from batch_invariance import workset as _generic_workset
    rows = build_rows(
        corner, ctx=ctx, families=families, corpus_dir=corpus_dir,
        model_id=model_id, position=position,
    )
    return _generic_workset.build_workset(
        rows, int(ctx), reps=int(reps), n_probs=int(n_probs), model_id=model_id,
    )


def write_workset_json(
    path: str,
    corner: str = "balanced",
    *,
    ctx: int,
    families: tuple[str, ...] | None = None,
    corpus_dir: str = "./corpus",
    model_id: str = DEFAULT_MODEL_ID,
    position: float = 0.5,
) -> str:
    """Build a sampling corner's row templates and write them as a ``--workset`` JSON file.

    Materializes :func:`build_rows` to ``path`` (UTF-8, 2-space indented, no BOM) -- the
    exact row-template list shape the generic loader's ``--workset`` flag reads. Returns
    ``path``. After this, run the gate with
    ``--workset <path> --scorer batch_invariance.profiles.spark.spark_scorer:score``.

    The file is plain data (prompts + gold answers); the gate re-derives per-item seeds
    and (live) token counts on load, so the JSON is portable and reproducible.
    """
    rows = build_rows(
        corner, ctx=ctx, families=families, corpus_dir=corpus_dir,
        model_id=model_id, position=position,
    )
    Path(path).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


__all__ = [
    "SPARK_FAMILIES",
    "SAMPLE_CORNERS",
    "N_CAPS",
    "DEFAULT_MODEL_ID",
    "build_rows",
    "build_items",
    "write_workset_json",
]

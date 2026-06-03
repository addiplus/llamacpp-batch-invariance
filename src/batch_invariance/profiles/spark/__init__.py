"""batch_invariance.profiles.spark -- the reference measurement profile.

An agentic-coding work-set + code-output scorer for the batch-invariance engine. It
supplies the two model-/task-specific seams the otherwise task-agnostic core needs:

  * a deterministic prompt set -- recall, multi-step arithmetic, retrieval,
    format-adherence, function-calling, and code-edit families
    (:mod:`prompt_generator`), reshaped onto the generic work-set loader by
    :mod:`spark_workset`;
  * a scorer -- per-family exact / numeric / summary / multi-key / format /
    function-call / AST-code-edit grading + a no-LLM failure-mode classifier
    (:mod:`quality_scorer`), bound to :mod:`batch_invariance.scorer_api`'s 3-tuple
    contract by :mod:`spark_scorer`.

Plus two DATA files: ``models.example.toml`` (an ``alias -> gguf + server-flags``
registry, templated with ``{MODELS_DIR}``) and ``kv_profiles.toml`` (the KV-quant
byte/footprint reference for :mod:`batch_invariance.kv_budget`).

WIRING IT IN (no code needed -- both are CLI seams)::

    # 1. materialize a work-set JSON from the agentic-coding families
    python -c "from batch_invariance.profiles.spark import spark_workset as w; \\
               w.write_workset_json('spark.json', corner='balanced', ctx=8192)"

    # 2. run the live gate with this profile's scorer plugged in
    batch-invariance run-live \\
        --workset spark.json \\
        --scorer batch_invariance.profiles.spark.spark_scorer:score \\
        --server-bin /path/llama-server --model /path/model.gguf \\
        --ctx 8192 --parallel 4 --cert-source live --out-dir ./certs

The submodules are imported lazily (:pep:`562`) so importing this profile does not pull
in the prompt builder / scorer until a symbol is accessed.
"""
from __future__ import annotations

import importlib

# (attribute name) -> (submodule, symbol). Resolved on first access via __getattr__.
_LAZY: dict[str, tuple[str, str]] = {
    # scorer seam (the --scorer entry point + its reserved labels)
    "score": ("spark_scorer", "score"),
    "OK_FAILURE_MODE": ("spark_scorer", "OK_FAILURE_MODE"),
    "EMPTY_FAILURE_MODE": ("spark_scorer", "EMPTY_FAILURE_MODE"),
    "PREMATURE_EOS_FAILURE_MODE": ("spark_scorer", "PREMATURE_EOS_FAILURE_MODE"),
    # work-set seam (build/materialize the agentic-coding work-set)
    "build_rows": ("spark_workset", "build_rows"),
    "build_items": ("spark_workset", "build_items"),
    "write_workset_json": ("spark_workset", "write_workset_json"),
    "SPARK_FAMILIES": ("spark_workset", "SPARK_FAMILIES"),
    "SAMPLE_CORNERS": ("spark_workset", "SAMPLE_CORNERS"),
    "N_CAPS": ("spark_workset", "N_CAPS"),
    # raw building blocks (the prompt generator + scorer modules, if needed directly)
    "generate_prompt": ("prompt_generator", "generate_prompt"),
    "list_families": ("prompt_generator", "list_families"),
    "score_response": ("quality_scorer", "score_response"),
    "classify_failure_mode": ("quality_scorer", "classify_failure_mode"),
}


def __getattr__(name: str):
    """Lazily import a profile symbol on first access (:pep:`562`)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = target
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, symbol)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = sorted(_LAZY)

"""batch_invariance.profiles -- optional, swappable measurement plug-ins.

A *profile* bundles the model-/task-specific knowledge the otherwise task-agnostic
engine needs to score a real work-set: a deterministic prompt set, a scorer that
implements :mod:`batch_invariance.scorer_api`'s 3-tuple contract, and the config data
(model registry, KV table) to launch a server. The engine core stays model-agnostic --
profiles are the ONLY place task knowledge lives, plugged in via ``--workset`` /
``--scorer``.

The reference profile is :mod:`batch_invariance.profiles.spark` (an agentic-coding
work-set + code-output scorer). This package is a namespace only; it imports nothing at
load time so adding a profile never pulls in another's prompts or scorer.
"""
from __future__ import annotations

__all__: list[str] = []

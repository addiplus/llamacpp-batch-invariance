# Hardening levers (F0 to F5): why the verdict is hard to fool

This document is the **anti-vacuity history**, reframed generically. The point of a
batch-invariance verifier is to *catch* divergence; the failure mode that makes one
worthless is the opposite, quietly minting a **false GREEN** on a cell where it should
have flagged a problem. Each lever below closed one specific way a naive "run it twice and
compare" check could be fooled into a false GREEN.

The unifying design principle: **leniency is the only dangerous direction.** Every gate is
arranged so that when it is uncertain it *over*-reports (demotes a GREEN to `failed` or
`green_unverified`), never *under*-reports. A RED or AMBER can cost you a needless
investigation; a false GREEN corrupts whatever downstream system trusts the cert. So all
five levers fail toward "not promotable."

---

## F0: exact-match score gate

**The hole it closes:** a *tolerant* comparison. If you compare two runs with any
"close enough" tolerance, a real divergence that happens to land inside the tolerance band
is silently swallowed, the original sin of a vacuous gate.

**The lever.** The gate compares a fixed, explicit set of **invariant fields** for *exact*
equality:

```
score, passed, extracted_answer, expected_answer, prompt_tokens_measured, failure_mode
```

If any of these differs between two arms for a shared `test_id`, that id is divergent.
Full stop. Two consequences:

- **Tolerance is banned as a gate.** Exact match can only ever *over*-report (flag a
  difference that may not matter), never *under*-report (hide a real one). That asymmetry
  is the whole safety argument.
- **Volatile fields never gate.** Timing, throughput, and host fields are explicitly
  *volatile* and ignored. Token-level completion content (`content_sha`) and any logprob
  deltas are recorded as **diagnostics**: they can *raise* a caveat (see F4) but never by
  themselves trip a RED.

Generic reframing: pick the smallest set of fields that actually define "the same answer
for your task," compare them exactly, and treat everything else as advisory.

---

## F1: overlap floor (real co-residency)

**The hole it closes:** a "pass" with no co-batching. The arm under test *looks*
concurrent, but if the requests never actually shared a forward pass, then "serial and
concurrent agree" proves nothing: the concurrent path was never exercised. This is the
single most dangerous false GREEN: false-GREEN-by-non-execution.

**The lever.** A GREEN additionally requires *observed* overlap from two independent
signals:

- **client-side:** the peak depth of overlapping request intervals must be ≥ 2, and
- **server-side:** the peak busy-slot count (from the server's slot poll) must be ≥ 2.

Below either threshold the verdict is demoted to `green_unverified` (non-promotable).

Generic reframing, **needle placement / co-residency:** it is not enough to *send* N
requests; you must *prove* that ≥2 of them were genuinely in flight in the same forward
pass. If your harness cannot observe co-residency, it cannot earn a GREEN.

---

## F2: completion floor (a genuinely-scored sample)

**The hole it closes:** all-failure arms that match each other. Because `failure_mode` is
one of the invariant fields, two arms in which *every* trial fails *identically* (every
request times out, or returns empty under contention) compare EQUAL, zero divergences,
and look "clean." But zero real completions were ever batched. "Matching nothing" must not
read as "invariant something": false-GREEN-by-all-failure.

**The lever.** A GREEN requires that at least **80%** of the compared trials *genuinely
scored* (`failure_mode == "ok"`) in **both** the serial arm (A) and the concurrent arm
(C). The cert records the per-arm `n_ok` / `n_failure` census. Below the floor →
`green_unverified`.

Generic reframing, **detection passes / sample integrity:** count how many of your
comparisons are *real* scored outputs versus failures that compare equal by accident, and
refuse to certify on a sample that is mostly failures. A pass built on timeouts certifies
nothing about batched real outputs.

---

## F3: per-id co-batch coverage (which corners actually co-batched)

**The hole it closes:** peak-only attribution. F1's peak proves "≥2 requests were
co-resident at *some* instant in *some* pass." It does **not** prove *which* ids shared a
forward pass. A GREEN earned on the peak alone could certify a cell where most ids actually
ran alone and only a couple ever overlapped: false-GREEN-by-peak-only.

**The lever.** Using each id's `[dispatch_ts, complete_ts]` request interval, an id is
counted as *co-batched* iff some **other** id's interval overlaps its own. A GREEN
additionally requires ≥ **80%** of ids to have *actually* co-batched (the absolute count
scales with N, floored at 2). Critically, this gate is **fail-closed**: if *any* id is
missing a usable request interval, attribution is uncertain and the gate is treated as not
satisfied → `green_unverified`. The peak scalar from F1 is retained only as a
*corroborating* floor.

Generic reframing, **sample size / corner coverage:** prove that *enough of the right
cases* actually exercised the concurrent path, per-id, and fail closed whenever you cannot
attribute a case with certainty. Coverage you cannot prove does not count.

---

## F4: logit-drift corroboration (opt-in)

**The hole it closes:** none, by itself. F4 is a *sharpening* lever, not a new gate. It
makes an existing AMBER more informative without ever loosening anything.

**The lever.** When *armed* (a positive drift epsilon **and** the driver actually plumbs a
per-id logprob summary), an id that is **score-clean** but **token-only divergent** (same
score, different completion bytes) and whose logprob summary *also* moved by more than the
epsilon is upgraded from a plain AMBER to a stronger AMBER subtype (`token_and_logit_drift`).
A token-only id whose logits stayed within epsilon keeps the weaker AMBER.

Because logprobs are **volatile**, this lever can only ever make a verdict *more*
conservative, GREEN → AMBER, and **never** RED, and **never** promote. It is **off by
default** (epsilon = 0), in which case it is completely inert and the behavior is exactly
as if F4 did not exist.

Generic reframing, **logit drift + sensitivity:** a secondary, volatile signal can be
used to *corroborate* a suspicion and raise a caveat, but must never be allowed to gate a
score or to relax a verdict. Corroboration only tightens.

---

## F5: provenance + scope lock

**The hole it closes:** two distinct ways a GREEN could "escape":

1. **Wrong provenance.** A pass from the in-process mock (whose completions are
   independent of batch composition by construction, so it can *only* ever pass) must not
   be promotable. Neither should the two soft-green variants.
2. **Wrong scope.** A GREEN earned on one cheap, divergence-resistant cell must not be
   read as a license for *other* cells, especially the high-ctx, high-N, MoE cells where
   batching divergence is *most* likely.

**The lever.**

- **Provenance.** Only the literal `status == "green"` with `source == "live"` is
  promotable. Mock greens and the two `green_*` variants are non-promotable *by
  construction* (their status string is not the literal `green`), with no override flag in
  normal use. The producer tag `gate` records that the rigorous A/B/C-union driver minted
  the cert, and the machine-readable `is_promotable` fails closed if the producer, overlap,
  completion floor, or per-id coverage is missing or insufficient.
- **Scope.** The cert stamps `promotion_scope` with `valid_for = {model, ctx, N}` and
  `generalizes = false`, plus the rule-of-three sensitivity bound (`3 / n_hot_trials` when
  zero divergences were seen). A GREEN is, in-band, a per-cell *negative result* scoped to
  exactly `(model, ctx, N, temp, KV)`.

Generic reframing, **per-(case, cell) coverage + sensitivity:** make "where did this
result come from?" and "what exactly does it license?" first-class, machine-checkable
fields, so a green can never silently travel to a cell (or a provenance) it never earned.

---

## How the levers compose (the decision order)

The verdict applies the levers in a fixed precedence, **first match wins**:

1. **Score divergence (F0)** → `failed` (RED). Highest precedence; nothing below can rescue it.
2. **No real overlap (F1)** → `green_unverified`.
3. **Completion floor not met (F2)** → `green_unverified`.
4. **Coverage checked and insufficient/uncertain (F3)** → `green_unverified`.
5. **Control arm B disagrees** → `green_with_caveat` (AMBER, `AC_agree_B_disagrees`).
6. **Token-only divergence** (optionally sharpened by F4) → `green_with_caveat` (AMBER).
7. **Everything clean** → `green` (promotable iff `source == "live"`, F5).

Read top-to-bottom, that ordering is the anti-vacuity argument in one list: a RED is never
downgraded, and a GREEN is only reached after every way of faking one has been ruled out.

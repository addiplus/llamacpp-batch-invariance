# llamacpp-batch-invariance

A **per-cell batch-invariance verifier** for OpenAI-compatible local LLM servers
(llama.cpp *continuous batching* and work-alikes).

It answers exactly one question, per `(model, ctx, N)` cell, against a **real**
server:

> Does firing `N` requests **concurrently** at a `--parallel N` server, so the
> server merges them into one forward pass (llama.cpp continuous batching), produce
> the **same scored outputs** as running those same requests **one at a time**?

If identical, that cell's batched outputs are safe to trust and the tool emits a
GREEN cert. If not, it emits a RED cert and the cell stays "serial-only." The verdict
is a JSON **cert**, one file per cell, signed-by-construction: a mock-sourced pass and
the two "soft green" variants are non-promotable *by construction*, with no override.

<!-- Badge placeholders. This is a CORRECTNESS gate, not a speed tool, so there are
     deliberately NO throughput/benchmark badges. -->
![CI](https://img.shields.io/badge/CI-pytest%20%2B%20ruff-informational)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.10--3.13-blue)

---

## What it is, and what it is not

**It IS:**

- A *verifier* that produces a signed-by-construction **cert** per cell:
  GREEN / RED / AMBER / UNVERIFIED (the four verdicts; see
  [Output contract](#output-contract--verdict-semantics)).
- A *pluggable harness*: bring your own server binary, your own work-set, and your
  own scorer (subclass `scorer_api.Scorer`).
- A tool that launches at most **one** short-lived experiment server, PID-scoped and
  torn down via `atexit`+signals, and never touches any other server you are running.

**This is a single-cell invariance probe, NOT an orchestrator**. It does not route, schedule, autoscale, or decide *how* to batch your traffic; it only tells you, after the fact, whether one specific batched cell was output-identical to serial.

**It IS NOT a determinism *fix*.** It does not make llama.cpp deterministic (llama.cpp
PR #16016 adds an opt-in deterministic mode, OFF by default). This tool *detects*
non-determinism under batching; it does not remove it.

---

## Why batching can change outputs (the motivation)

Continuous batching merges concurrent requests into a single forward pass. The batched
matmul / RMSNorm / attention reductions can then run in a different reduction order than
they would for a lone request, and floating-point non-associativity means the resulting
logits, and therefore the sampled tokens, can differ **even at `temp=0`**. Reported
priors:

- llama.cpp issue **#7052**: 8 slots, `temp=0`, one prompt → 5 to 8 *unique* completions.
- llama.cpp PR **#16016**: a deterministic mode exists, but is **OFF by default**.
- Thinking Machines, *"Defeating Nondeterminism in LLM Inference."*

The consequence for this tool: the **null hypothesis is "batching DOES change
outputs."** A GREEN is therefore something you have to *earn* against a real server.
It is never assumed, and the verdict logic is deliberately built so the only way it can
err is to over-report (flag a divergence that does not matter), never to silently pass a
real one.

---

## Install

```bash
# 0.1.0 is GitHub-only (not yet on PyPI) -- install straight from the repo:
pip install "git+https://github.com/addiplus/llamacpp-batch-invariance.git"
# or, from a clone (contributors):
pip install -e ".[dev]"          # adds pytest + ruff only
```

Requires Python ≥ 3.10. **Zero runtime dependencies**: the package imports only the
Python standard library at runtime. That is a deliberate feature: the install is
auditable and airgap-friendly, and the verdict logic has no third-party surface that
could drift under you.

You supply the server binary (e.g. `llama-server`) and the model GGUFs yourself; this
package never downloads or bundles a model.

---

## Quickstart (generic, any OpenAI-compatible server)

**1. 30-second mock demo (no GPU, no model).** An in-process mock server with a
divergence knob lets you watch the gate flip without any hardware:

```bash
batch-invariance run-mock --out-dir ./certs                 # writes a GREEN cert
batch-invariance run-mock --out-dir ./certs --batch-divergence   # same gate, now RED
```

The mock can never mint a *promotable* cert: its certs carry `source=mock`, which the
promotion gate rejects by construction (see below). It exists only to prove the gate
can go RED and to run CI with zero GPU.

**2. One real cell against your own server:**

```bash
batch-invariance run-live \
  --server-bin /path/to/llama-server \
  --model /path/to/model.gguf \
  --workset examples/workset.json \
  --ctx 8192 --parallel 4 \
  --cert-source live \
  --out-dir ./certs
# -> writes ./certs/<model>__ctx8192__N4.json
# --workset is required for the explicit form (point it at your task's work-set, or use
# --profile below to supply one). --scorer is optional and defaults to exact-match;
# pass your own with --scorer pkg.mod:fn (see "Bring your own task" below).
```

The driver owns the experiment server's whole lifecycle: it spawns one server, runs the
three arms over real HTTP, hands the per-arm result maps to the pure diff, writes the
cert atomically, prints GREEN / RED, and tears the server down (PID-scoped) on exit.

**3. Bring your own task.** Implement a scorer for your own work by subclassing
`scorer_api.Scorer` (`extract_answer`, `classify_failure_mode`, `score`) and pass it in.
The engine stays model- and task-agnostic behind that interface; everything
domain-specific lives in your scorer.

---

## Quickstart (reference coding profile, agentic-coding work-set)

The repository ships one reference measurement **profile**, `spark`, under
`src/batch_invariance/profiles/spark/` (the only place model/task-specific knowledge
lives):

- a **deterministic agentic-coding work-set**: prompt families `A2 / A4 / B1 / C1 /
  D1 / E1-E5` generated at fixed fill ratios, so the same seed always produces the same
  serial and concurrent worksets, and
- a **code-output scorer** that extracts the answer, classifies the failure mode, and
  scores each completion.

Copy the example model registry to a real one (mapping each alias to a gguf path plus
server flags, and optionally a `server_bin`), then run a single cell against it. With
`--profile`, the engine fills the gguf path, server flags, scorer, and work-set from the
profile; supply `--server-bin` either in `models.toml` (the `[defaults].server_bin` key)
or on the command line, and `--models-dir` (or the `MODELS_DIR` env var) to expand the
`{MODELS_DIR}` placeholder in the gguf paths:

```bash
cp src/batch_invariance/profiles/spark/models.example.toml \
   src/batch_invariance/profiles/spark/models.toml
# edit models.toml: set each alias's gguf, your server_bin, and your flags, then:
batch-invariance run-live --profile spark \
  --model-alias <your-alias> --models-dir /path/to/models \
  --server-bin /path/to/llama-server \
  --ctx 8192 --parallel 4 --out-dir ./certs
```

**Operator note.** The live gate can be *written* on any machine (including a CPU-only
CI box) but is *run by the user on the GPU host*. Before it dispatches anything it
re-reads the host's real `/proc/meminfo`, refuses to start if memory is unsafe, holds
its footprint under a configured cap, and never binds or addresses any reserved port you
hand it. Teardown is PID-scoped: the driver only ever kills the one server PID it spawned
(tracked from spawn), and never pattern-kills a long-running server process by name.

---

## Output contract / verdict semantics

The cert is **one JSON file per cell**, named `{model}__ctx{C}__N{N}.json`, written
**atomically** (tempfile + `os.replace`, so a concurrent reader never sees a partial
file). The field-by-field schema and the full verdict truth table are in
[`docs/output-contract.md`](docs/output-contract.md). The five rows:

| Verdict | `status` | Promotable? | Meaning |
|---|---|---|---|
| **RED** | `failed` | no | A↔C scores diverge. `divergence_class` = `co_batching` (A↔B clean → pure continuous-batching) or `slot_allocation(+co_batching)` (A↔B also dirty). |
| **UNVERIFIED** | `green_unverified` | no | A↔C *looked* clean but the run did not earn it: no real co-batching overlap (F1), too few genuinely-scored trials (F2), or per-id co-batch coverage uncertain/too low (F3). Treated as RED for promotion. |
| **AMBER** | `green_with_caveat` | no (sign-off) | A↔C scores clean + overlap + floor + coverage all met, but either control arm B disagrees (`anomaly=AC_agree_B_disagrees`) or there is token-only / logit drift with scores identical. |
| **GREEN** | `green` | **yes** | A↔C token-identical, overlap real, floor met, coverage met, control arms clean. **Only** the literal `green` with `source=='live'` is promotable. |

**The three arms, stated plainly:**

- **A** = `--parallel 1`, serial dispatch, the ground truth.
- **B** = `--parallel N`, serial dispatch, the slot-allocation control (isolates "did
  merely *opening* N slots change anything?" from "did *co-batching* change anything?").
- **C** = `--parallel N`, concurrent dispatch, the thing under test. C is the **union**
  over `T` repeated concurrent passes: an id is divergent if it diverged on **any** pass
  (continuous-batching divergence is stochastic, so one pass is not enough).

**Provenance gate (critical).** `cert_is_green(cert, require_source='live')` returns True
**only** for `status=='green'` **and** `source=='live'`. A mock-sourced green and both
`green_*` variants are non-promotable by construction, and there is no flag to override this
in normal use. The producer tag `gate=abc_union` records that the rigorous A/B/C-union
driver minted the cert; the machine-readable promotion verdict (`is_promotable`) folds in
overlap, the completion floor, per-id coverage, and that producer tag.

---

## HONEST CAVEATS (read this before trusting a GREEN)

- **GREEN is a per-cell *negative* result, scoped to `(model, ctx, N, temp, KV)`.** It
  means *"no divergence was DETECTED at this exact cell,"* **not** *"this model is
  batch-invariant."* It does **not** generalize across ctx, N, model, KV-quant, or
  sampler. Re-run the live gate per cell before trusting any new cell; on **any**
  non-green verdict, keep that cell `--parallel 1`.

- **Rule of three.** With `n_hot_trials` genuinely-scored comparisons and **zero**
  divergences observed, the 95% upper bound on the true divergence rate is ≈
  `3 / n_hot_trials`. A GREEN is only as strong as its sample, and the cert records both
  `n_hot_trials` and this bound. **Absence of evidence is not evidence of absence.**

- **RED and AMBER are trustworthy; GREEN is the weak side.** The gate uses **exact**
  score match (never a tolerance). Exact match can only ever *over*-report: it can
  trigger a needless investigation, but it can never silently promote a real divergence.
  So a RED or AMBER is a real signal; a GREEN is a *failure to detect*, bounded by the
  sample. Tolerance is banned as a gate precisely because it would convert a real
  divergence into a false GREEN.

- **A dense-model GREEN says nothing about MoE routing.** Continuous-batching divergence
  is *most* likely exactly where a dense low-ctx/low-N GREEN tells you the least: in
  Mixture-of-Experts models (where batch composition changes which experts fire) and at
  high ctx / high N (#7052). The cert stamps this scope warning in-band
  (`promotion_scope.generalizes=false`); a green can never silently travel to another
  cell.

- **The bundled live RED is the proof the gate actually bites. See below.** A verifier
  that only ever rubber-stamps is worthless; the shipped cert demonstrates the tool
  catching a *real* divergence under conditions where it could have falsely passed.

---

## Live-validated: the bundled phi-4 (Q4_K_M) RED

The repository bundles one real cert as conclusive proof that the gate detects a genuine
divergence rather than rubber-stamping: [`examples/certs/demo_dense_live_cert.json`](examples/certs/demo_dense_live_cert.json),
from a **phi-4 (Q4_K_M)** dense-model run on a single GPU host.

It is a **conclusive RED** (`source=live`, `status=failed`,
`divergence_class=co_batching`), and, crucially, the RED was earned on a run where the
two hardest anti-vacuity gates **passed**:

- **completion floor passed**: `completion_floor.completion_floor_ok=true`, with **48/48
  trials genuinely scored** in *every* arm (A, B, and C). The RED is not the artifact of
  an empty or all-failure run; every compared trial really produced a scored completion.
- **co-batch coverage passed**: `cobatch_coverage.cobatch_coverage_ok=true`, with
  **46/48 ids provably co-batched** (coverage fraction 0.958) and attribution certain
  (zero ids missing a request interval). The RED is not the artifact of a run where
  prompts never actually shared a forward pass.

The divergence itself: control arm **A↔B is clean** (opening N slots changed nothing),
while **A↔C diverges on one id**: serial produced one completion, concurrent
co-batching produced a different one with the same score-bearing setup. A↔B clean + A↔C
dirty ⇒ the cause is **pure continuous-batching co-batching**, not slot allocation.

That is the whole story in one artifact: a *real* divergence, caught on a fully
overlapping, fully scored run, under conditions where a naive "compare two runs" check
would have happily minted a GREEN.

---

## The five hardening levers (F0 to F5)

A brief, generic restatement of *why* the verdict is hard to fool. Each lever closed a
distinct way a naive "compare two runs" check could mint a false GREEN. Full detail in
[`docs/hardening.md`](docs/hardening.md).

- **F0: exact-match score gate.** Compare a fixed `INVARIANT_FIELDS` set
  (`score, passed, extracted_answer, expected_answer, prompt_tokens_measured,
  failure_mode`) for exact equality. Tolerance is *banned* as a gate (it turns a real
  divergence into a false GREEN). Timing/throughput/host fields are `VOLATILE_FIELDS` and
  ignored; token content (`content_sha`) and logprob deltas are diagnostics, never gates.

- **F1: overlap floor (needle placement / co-residency).** A "pass" with no observed
  co-batching proves nothing. Require *real* overlap: client request-interval overlap
  depth ≥ 2 **and** server busy-slots ≥ 2. Fail → `green_unverified`.

- **F2: completion floor (detection passes / genuinely-scored sample).** Two arms that
  *both fail identically* compare EQUAL on `INVARIANT_FIELDS`, so "matching nothing" must
  not read as "invariant something." Require ≥ 80% of compared trials to be genuinely
  scored (`failure_mode=='ok'`) in **both** A and C. Fail → `green_unverified`.

- **F3: per-id co-batch coverage (sample size / corner attribution).** Peak overlap only
  proves "≥2 co-resident at *some* instant," not *which* ids shared a forward pass. Using
  per-id request intervals (`dispatch_ts` / `complete_ts`), require ≥ a coverage fraction
  of ids to have *actually* co-batched; uncertain attribution **fails closed** to
  `green_unverified`.

- **F4: logit-drift corroboration (logit drift + sensitivity; opt-in).** When armed, a
  score-clean but token-only-divergent id whose per-id logprob summary *also* moved is
  upgraded to a stronger AMBER. Logprobs are volatile → this can only make a verdict
  *more* conservative (GREEN → AMBER), never RED, never promote.

- **F5: provenance + scope lock (per-(id,cell) coverage + sensitivity).** Only
  `status=='green'` with `source=='live'` is promotable; mock greens and the two `green_*`
  variants are non-promotable by construction. The cert stamps `promotion_scope` (valid
  only for the exact `(model, ctx, N)`; `generalizes=false`) and the rule-of-three
  sensitivity bound, so a green can never silently travel to another cell.

---

## How it works (the seam)

The code is split into a **pure core** and an **impure driver**, and that split is the
reason the verdict logic is unit-testable with zero GPU:

```
   impure driver                       pure core
  (live_invariance)                 (invariance_diff)
  ----------------                  -----------------
  owns ONE server PID
  runs Arms A / B / C   --maps-->   diff_arms(A,B,C)
  over real HTTP          {test_id:  -> reports + decide_status
  stamps intervals         result}   -> GREEN/RED/AMBER/UNVERIFIED
  writes cert atomically  <--cert--   build_cert (pure dict)
```

The driver builds per-arm `{test_id: result}` maps and hands them to the pure diff,
which never touches the network, a subprocess, threads, or a GPU. The pure diff is fully
unit-tested, including **RED-on-divergence**, not merely identity, so a regression in
the verdict logic is caught on CPU-only CI.

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest                     # all GPU-free, via the in-process mock server
```

CI runs on Python 3.10-3.13 (`ruff` + `pytest`), GPU-free.

---

## Citations

- llama.cpp issue **#7052**: 8 slots, `temp=0` → 5 to 8 unique completions for one prompt.
- llama.cpp PR **#16016**: deterministic mode exists but is **OFF by default**.
- Thinking Machines, *"Defeating Nondeterminism in LLM Inference."*

(The `NOTICE` file mirrors these as attributions.)

---

## License

Apache-2.0. See [`LICENSE`](LICENSE). Third-party priors are credited in
[`NOTICE`](NOTICE).

# Output contract: the cert JSON, field by field

Every run writes exactly one **cert** per `(model, ctx, N)` cell: a single JSON file,
named `{model}__ctx{C}__N{N}.json`, written **atomically** (a temp file in the same
directory, then `os.replace`, so a concurrent reader never observes a partial file) and
serialized with `sort_keys=True` (stable byte output across runs).

The contract is **additive**: new diagnostic keys may be added over time; an older reader
that only knows the required fields keeps working. Only a small, explicit subset is read
by the promotion gate; everything else is a human-/reviewer-facing diagnostic.

---

## 1. Required fields (read by the promotion gate)

These five fields are the load-bearing contract. `cert_is_green()` reads only `status`
and `source`; `is_promotable()` additionally reads `overlap_ok`, `completion_floor`,
`cobatch_coverage`, and `gate` (below).

| Field | Type | Meaning |
|---|---|---|
| `model` | string | The model identifier for this cell. |
| `ctx` | int | The server context length (`--ctx-size`) for this cell. |
| `dispatch_n` | int | `N`, the `--parallel` slot count / concurrent request count under test. |
| `status` | string | The verdict: one of `green`, `failed`, `green_unverified`, `green_with_caveat`. See §2. |
| `source` | string | `live` (a real server) or `mock` (the in-process mock). **Only `live` is ever promotable.** |

### `status` vocabulary

| `status` | Verdict | Promotable | Set when |
|---|---|---|---|
| `failed` | **RED** | no | A↔C scores diverge (≥1 id differs on an invariant field, or an id is present in only one arm). |
| `green_unverified` | **UNVERIFIED** | no | A↔C looked clean but the run did not earn it (overlap / completion-floor / coverage gate failed). Treated as RED for promotion. |
| `green_with_caveat` | **AMBER** | no (sign-off) | A↔C scores clean + all floors met, but control arm B disagrees, or there is token-only / logit drift with scores identical. |
| `green` | **GREEN** | **yes** (iff `source==live`) | A↔C token-identical, overlap real, completion floor met, coverage met, control arms clean. |

Only the **literal** string `green` is accepted by `cert_is_green`. The two `green_*`
strings are intentionally *not* the literal `green`, so they are non-promotable by
construction with no special-casing.

---

## 2. Provenance / producer fields

| Field | Type | Meaning |
|---|---|---|
| `gate` | string | Which producer minted the cert. `abc_union` = the rigorous A/B/C-union driver (the strong gate `is_promotable` requires by default). |
| `overlap_ok` | bool | Top-level mirror of `overlap.overlap_ok`. Was *real* co-batching observed? `is_promotable` fails closed if this is absent or false. |
| `kv_label` | string | KV-cache quantization label for the cell (e.g. `q8_0`, `q4_0`, `f16`). |
| `reps` | int | Repetitions per trial inside one pass. |
| `n_passes` | int | `T`, the number of concurrent ARM_C re-passes folded into the union. |
| `invariant_fields` | string[] | The exact field set compared for equality (echoed for audit). |
| `ts_utc` | string | UTC write time, `YYYY-MM-DDTHH:MM:SSZ`. |
| `mismatch` | string \| null | Human-readable summary of the first divergence (null on a clean AC). |

---

## 3. The three arms and the divergence report

The driver runs three arms and produces three pairwise reports. Each arm is a
`{test_id: result}` map; the reports diff those maps.

- **A**: `--parallel 1`, serial, the ground truth.
- **B**: `--parallel N`, serial, slot-allocation control.
- **C**: `--parallel N`, concurrent, the thing under test (the **union** over `T` passes).

| Field | Type | Meaning |
|---|---|---|
| `divergence_class` | string \| null | On RED: `co_batching` (A↔B clean → pure continuous-batching) or `slot_allocation(+co_batching)` (A↔B also dirty). Null otherwise. |
| `anomaly` | string \| null | AMBER subtype: `AC_agree_B_disagrees` or `token_and_logit_drift`. Null otherwise. |
| `token_divergence_ids` | string[] | Ids where scores match but completion bytes differ (the token-only / AMBER signal). |
| `arm_b` | object | `{ab_passed: bool, bc_passed: bool}`, control-arm summary. |
| `per_pass_ac_divergent` | int[] | Per-pass count of AC score/missing-divergent ids (audit trail: which interleaving surfaced it). |
| `per_pass_ac_token_only` | int[] | Per-pass count of AC token-only-divergent ids (the AMBER companion trail). |
| `decision_reasons` | string[] | The ordered, human-readable reasons the verdict fired. |
| `divergence_report` | object | `{AC, AB, BC}`, the three pairwise reports (next table). |

### `divergence_report.{AC,AB,BC}`: one pairwise report

| Field | Type | Meaning |
|---|---|---|
| `pair` | string | `"AC"`, `"AB"`, or `"BC"`. |
| `n_compared` | int | Shared `test_id`s actually compared. |
| `n_divergent` | int | Shared ids with ≥1 invariant-field difference (+ ids present in only one arm). |
| `divergence_rate` | float | `n_divergent / max(1, n_compared)`. |
| `divergent_ids` | string[] | Sorted ids that diverged. |
| `only_x` / `only_y` | string[] | Ids present in only one of the two maps (a lost/extra id is a divergence). |
| `n_token_only` | int | Shared + score-clean ids whose completion bytes differ. |
| `token_divergence_ids` | string[] | Sorted; score-equal but content-different. |
| `per_id` | object[] | One row per (divergent id, divergent field): `{test_id, field, x, y, x_content_sha, y_content_sha, token_identical}`. |
| `by_family` | object | `{family: {n, divergent}}` aggregate. |
| `by_fill` | object | `{fill_bucket: divergent_count}` aggregate. |
| `logit_drift` | object | F4 summary: `{armed, eps, max_logit_delta, logit_drift_ids, per_id_delta}` (inert unless armed). |

---

## 4. The anti-vacuity gate blocks (F1 to F5)

These blocks record *why* a GREEN was (or was not) earned. `is_promotable` folds in the
ones marked **gates**; the rest are diagnostics.

### `overlap` (F1, gate)

| Field | Type | Meaning |
|---|---|---|
| `client_max_overlap_depth` | int | Peak count of client request intervals in flight at once. |
| `server_peak_busy_slots` | int | Peak busy slots seen on the server's slot poll. |
| `overlap_ok` | bool | True iff depth ≥ 2 **and** busy-slots ≥ 2. A pass without real overlap → `green_unverified`. |

### `completion_floor` (F2, gate)

| Field | Type | Meaning |
|---|---|---|
| `n_ok_a` / `n_ok_b` / `n_ok_c` | int | Genuinely-scored (`failure_mode=='ok'`) trial count per arm. |
| `n_failure_a` / `n_failure_b` / `n_failure_c` | int | Failure (timeout/empty/error) trial count per arm. |
| `min_ok_fraction` | float | Required fraction (0.80). |
| `completion_floor_ok` | bool | True iff ≥ `min_ok_fraction` of compared trials genuinely scored in **both** A and C. |

> Why this gate exists: `invariant_fields` includes `failure_mode`, so two arms that both
> fail identically compare EQUAL, so "matching nothing" must not read as "invariant
> something." Below the floor → `green_unverified`.

### `cobatch_coverage` (F3, gate, fail-closed)

| Field | Type | Meaning |
|---|---|---|
| `coverage_checked` | bool | Was per-id attribution evaluated? (False on maps predating the interval contract → gate inert.) |
| `cobatch_coverage_ok` | bool | The gate verdict (see below). |
| `coverage_fraction` | float | `n_co_batched / n_ids`. |
| `n_ids` | int | Ids in the cell. |
| `n_co_batched` | int | Ids that provably co-batched (≥1 overlapping other-id request interval). |
| `n_missing_interval` | int | Ids missing a usable `dispatch_ts`/`complete_ts` (each one makes attribution uncertain). |
| `attribution_certain` | bool | True iff **every** id had a usable request interval. |
| `required_co_batched` | int | `ceil(min_coverage_fraction × n_ids)`, floored at 2. |
| `min_coverage_fraction` | float | Required fraction (0.80). |
| `server_peak_busy_slots_floor` | int | The peak scalar, mirrored as a *corroborating* floor only. |

> `cobatch_coverage_ok` is True only when `attribution_certain` **and** `n_ids ≥ 2`
> **and** `n_co_batched ≥ required_co_batched`. Uncertain attribution **fails closed** to
> `green_unverified`. The peak proves *something* was co-resident, not *which* ids shared
> a forward pass.

### `sensitivity` (F5, diagnostic)

| Field | Type | Meaning |
|---|---|---|
| `n_hot_trials` | int | Genuinely-scored comparisons underwriting the verdict. |
| `n_ac_divergent` | int | AC-divergent ids observed. |
| `rate_upper_bound_95` | float \| null | Rule of three: `3 / n_hot_trials` when 0 divergences were seen; null otherwise. |
| `interpretation` | string | In-band reminder that GREEN is per-cell "not detected," bounded by `n_hot_trials`. |

### `promotion_scope` (F5, diagnostic)

| Field | Type | Meaning |
|---|---|---|
| `valid_for` | object | `{model, ctx, dispatch_n}`, the **only** cell this cert licenses. |
| `generalizes` | bool | Always `false`. A GREEN does not travel across ctx, N, model, KV, or sampler. |
| `note` | string | In-band warning that a dense-model / low-ctx / low-N GREEN says nothing about MoE routing or higher-ctx/N cells. |
| `tested_model_looks_moe` | bool | Heuristic flag on the cell's model id. |

`ctx_sweep` (object \| null) optionally carries a per-ctx grid of independent per-cell
verdicts; it never implies generalization across the swept points.

---

## 5. The promotion verdict (machine-readable)

`is_promotable(cert, require_gate="abc_union")` returns True iff **all** of:

1. `cert_is_green(cert, require_source="live")`: `status=="green"` **and** `source=="live"`.
2. `overlap_ok` is present **and** true (else fail closed).
3. `gate == "abc_union"` (the strong producer; pass `require_gate=None` to waive).
4. `completion_floor.completion_floor_ok` is true (a present-and-false block fails closed).
5. `cobatch_coverage.cobatch_coverage_ok` is true **whenever** `coverage_checked` is true
   (an unchecked/absent block is tolerated, mirroring the floor's "absent → not enforced").

`green_unverified`, `green_with_caveat`, and `failed` all return False. Use this, not
bare `cert_is_green`, as the gate before promoting any batched cell.

---

## 6. Worked example: the bundled live RED

[`../examples/certs/demo_dense_live_cert.json`](../examples/certs/demo_dense_live_cert.json) is a real
cert (a phi-4 / Q4_K_M dense-model live run). The load-bearing fields:

```jsonc
{
  "source": "live",
  "status": "failed",               // RED
  "divergence_class": "co_batching", // A↔B clean, A↔C dirty -> pure continuous-batching
  "ctx": 8192,
  "dispatch_n": 4,
  "n_passes": 8,
  "overlap":            { "overlap_ok": true, "client_max_overlap_depth": 4, "server_peak_busy_slots": 4 },
  "completion_floor":   { "completion_floor_ok": true, "n_ok_a": 48, "n_ok_b": 48, "n_ok_c": 48 },
  "cobatch_coverage":   { "cobatch_coverage_ok": true, "n_co_batched": 46, "n_ids": 48,
                          "coverage_fraction": 0.958, "attribution_certain": true,
                          "n_missing_interval": 0 },
  "arm_b":              { "ab_passed": true, "bc_passed": false }
}
```

Reading it: `status=failed` with `divergence_class=co_batching` is a RED caused by pure
continuous-batching. It is **not** a vacuous RED: `completion_floor_ok=true` (48/48
scored in every arm) and `cobatch_coverage_ok=true` (46/48 ids provably co-batched,
attribution certain) say the run was fully scored and fully overlapping. `arm_b.ab_passed`
true + `bc_passed` false confirms the cause is co-batching, not slot allocation. This is
the gate detecting a real divergence on a run that could otherwise have passed.

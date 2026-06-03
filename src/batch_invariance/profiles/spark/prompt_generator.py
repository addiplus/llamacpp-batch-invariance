"""Deterministic prompt generator for NoLiMa/RULER-grade coherence-cliff benchmarks.

Public API:
  - generate_prompt(family, seed, target_tokens, position, corpus_dir) -> dict
  - list_families() -> list[str]
  - estimate_tokens(text) -> int
  - seed_from_tuple(model_id, ctx_tier, family, position, rep) -> int

All randomness goes through random.Random(seed). The same (seed, family, position,
target_tokens, corpus contents) tuple always produces a byte-identical prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
from collections.abc import Callable
from pathlib import Path

# Cheap upper-bound chars/token guess used ONLY to size the filler reservoir and
# the corpus-tiling target before the binary search runs. NOT load-bearing for
# correctness: the binary search measures the REAL tokenizer and self-corrects to
# whatever it reports, so an over-generous reservoir merely gives bisection more
# headroom. ~4.0 deliberately over-estimates chars/token (a typical BPE tokenizer
# is ~3.7-4) so the reservoir is never short.
REAL_CHARS_PER_TOK_GUESS = 4.0
CALIBRATION_TOL = 0.03      # default measured-vs-target tolerance (fraction)
CALIBRATION_MAX_ITER = 12   # ~4096x char-resolution; ample for monotone bisection

# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------

FAMILIES: tuple[str, ...] = (
    "A1", "A2", "A3", "A4", "B1", "B2", "C1", "D1",
    "E1", "E2", "E3", "E4", "E5",
)


def list_families() -> list[str]:
    """Return the supported family identifiers."""
    return list(FAMILIES)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Crude token estimate combining char/4 + word count for robustness."""
    if not text:
        return 0
    return len(text) // 4 + len(text.split())


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def seed_from_tuple(
    model_id: str, ctx_tier: int, family: str, position: float, rep: int
) -> int:
    """Stable SHA-256 of the formatted tuple mod 2**31. Order-sensitive."""
    key = f"{model_id}|{ctx_tier}|{family}|{position:.6f}|{rep}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


# ---------------------------------------------------------------------------
# Fallback corpus (used when corpus_dir files are missing)
# ---------------------------------------------------------------------------

_FALLBACK_GUTENBERG = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, "
    "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
    "incredulity, it was the season of Light, it was the season of Darkness, it was "
    "the spring of hope, it was the winter of despair, we had everything before us, "
    "we had nothing before us. In those days the road from London to Dover ran past "
    "small farms and quiet villages, where shepherds tended their flocks and millers "
    "ground the morning grain. The mail coach rattled along through the mist while "
    "passengers held tight to their cloaks and watched for highwaymen at every bend. "
    "Long after the lamps were extinguished and the inns were closed, travellers told "
    "stories of strange lights upon the moor and of voices calling from the marshes. "
    "Children listened by the hearth while the fire crackled and the wind moaned in "
    "the chimney, and the old grandmother knit silently in her chair. "
)

_FALLBACK_CC_NEWS = (
    "Researchers announced today a new approach to evaluating long-context language "
    "models, citing the need for benchmarks resistant to memorization. The team "
    "described a procedure in which synthetic facts are inserted at controlled depths "
    "throughout otherwise neutral filler text. Industry analysts welcomed the report, "
    "noting that current evaluation suites tend to overstate model capabilities on "
    "tasks the systems have indirectly seen during pretraining. Several open-source "
    "groups have begun releasing similar tools, and a working group at the local "
    "university is expected to publish a comparative study before the end of the year. "
    "Meanwhile, a separate filing from a regional regulator outlined disclosure rules "
    "for automated decision systems used in housing applications, with implementation "
    "deadlines staggered across the next three quarters. "
)

_FALLBACK_PYTHON = (
    "def parse_config(path):\n"
    "    with open(path) as fh:\n"
    "        raw = fh.read()\n"
    "    lines = [line.strip() for line in raw.splitlines() if line.strip()]\n"
    "    config = {}\n"
    "    for line in lines:\n"
    "        if '=' in line:\n"
    "            key, value = line.split('=', 1)\n"
    "            config[key.strip()] = value.strip()\n"
    "    return config\n"
    "\n"
    "def render_template(template, context):\n"
    "    for key, value in context.items():\n"
    "        template = template.replace('{' + key + '}', str(value))\n"
    "    return template\n"
    "\n"
    "def chunked(iterable, size):\n"
    "    buf = []\n"
    "    for item in iterable:\n"
    "        buf.append(item)\n"
    "        if len(buf) == size:\n"
    "            yield buf\n"
    "            buf = []\n"
    "    if buf:\n"
    "        yield buf\n"
    "\n"
)


def _load_corpus_with_sources(
    corpus_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load corpus files, returning (corpus, sources). sources[key] is "real"
    when the on-disk file existed with non-blank content, else "fallback"."""
    specs = {
        "gutenberg": ("gutenberg.txt", _FALLBACK_GUTENBERG),
        "cc_news": ("cc_news.txt", _FALLBACK_CC_NEWS),
        "python": ("python_repo.txt", _FALLBACK_PYTHON),
    }
    out: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key, (fname, fallback) in specs.items():
        path = corpus_dir / fname
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    out[key], sources[key] = text, "real"
                else:
                    out[key], sources[key] = fallback, "fallback"
            else:
                out[key], sources[key] = fallback, "fallback"
        except OSError:
            out[key], sources[key] = fallback, "fallback"
    return out, sources


def _load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Load corpus files, falling back to bundled constants when absent."""
    out, _sources = _load_corpus_with_sources(corpus_dir)
    return out


# ---------------------------------------------------------------------------
# Filler assembly + needle insertion
# ---------------------------------------------------------------------------

def _expand_to_target(rng: random.Random, source_text: str, target_tokens: int) -> str:
    """Tile or trim source_text deterministically to approximately target_tokens.

    Bug fix 2026-05-28: original implementation returned source as-is when source
    already exceeded target — producing 577K-token prompts from a 2.4MB corpus
    against an 8K context server. Now samples a contiguous window when source
    is too big, snapped to sentence boundaries.
    """
    source_tokens = estimate_tokens(source_text)
    if source_tokens >= target_tokens:
        target_chars = max(64, int(target_tokens * 3))
        if len(source_text) <= target_chars:
            return source_text
        max_start = max(0, len(source_text) - target_chars)
        start = rng.randint(0, max_start)
        chunk = source_text[start:start + target_chars]
        first_period = max(chunk.find('. '), chunk.find('! '), chunk.find('? '))
        if 0 < first_period < len(chunk) // 4:
            chunk = chunk[first_period + 1:].lstrip()
        last_period = max(chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? '))
        if last_period > len(chunk) // 2:
            chunk = chunk[:last_period + 1]
        return chunk
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", source_text) if c.strip()]
    if not chunks:
        chunks = [source_text]
    pieces: list[str] = []
    running_tokens = 0
    shuffled = chunks[:]
    rng.shuffle(shuffled)
    idx = 0
    while running_tokens < target_tokens:
        piece = shuffled[idx % len(shuffled)]
        pieces.append(piece)
        running_tokens += estimate_tokens(piece) + 1
        idx += 1
        if idx > 1_000_000:
            break
    return " ".join(pieces)


def _build_whitespace_filler(target_tokens: int) -> str:
    """Pure-whitespace pad of ~target_tokens (blank lines). Zero distractor signal.
    No sentence punctuation, so _insert_at_position takes its char-cut branch and
    a multi-line needle is concatenated whole (needle-survival invariant holds)."""
    n_lines = max(1, target_tokens)
    return "\n".join("" for _ in range(n_lines))


def _build_masked_repeat_filler(target_tokens: int) -> str:
    """Repetition-only pad: one neutral comment line tiled to ~target_tokens. The
    line intentionally has NO sentence-ending '.!?' so it never splits a multi-line
    needle in _insert_at_position (needle-survival invariant holds)."""
    line = "# filler line, intentionally repeated"
    per_line = max(1, estimate_tokens(line))
    n = max(1, target_tokens // per_line)
    return "\n".join(line for _ in range(n))


# ---------------------------------------------------------------------------
# Needle-placement strategy (F0 PLACEMENT). Strategy-driven WITHOUT changing the
# default: "fixed" reproduces today's behaviour byte-for-byte.
# ---------------------------------------------------------------------------

# The three placement strategies. "fixed" is TODAY's default (verbatim position).
POSITION_STRATEGIES: tuple[str, ...] = ("fixed", "jitter", "adaptive")

# Default seeded jitter half-width (fraction of the [0,1] position axis). The
# needle position is perturbed by +/- this amount via random.Random(seed), then
# re-clamped to [0,1]. Small enough to keep the needle in the same broad region
# (start/mid/end) while breaking the exact-boundary determinism that can drop a
# needle into an SWA dead-zone on every rep. Tunable by callers.
POSITION_JITTER_HALF_WIDTH = 0.08

# An adaptive needle is pinned this close to the START of the haystack when an SWA
# (sliding-window attention) window is in play. Near-zero rather than exactly 0.0
# so the existing sentence-boundary walk still splices BEFORE the first sentence
# (an exact 0.0 also lands at the front, but a tiny positive value is robust to a
# degenerate single-sentence char-cut that would put nothing before the needle).
ADAPTIVE_SWA_START_POSITION = 0.0


def _resolve_position(
    position: float,
    *,
    strategy: str = "fixed",
    seed: int | None = None,
    swa_window: int | None = None,
    target_tokens: int | None = None,
) -> float:
    """Map a requested (position, strategy) onto the EFFECTIVE position in [0,1].

    Pure + fully seeded — same (position, strategy, seed, swa_window, target_tokens)
    tuple always returns the same float. No system-clock reads; all randomness goes
    through random.Random(seed).

    Strategies (F0 PLACEMENT):
      "fixed"    -> return the clamped requested position VERBATIM. This is the
                    DEFAULT and is byte-identical to the pre-strategy code path.
      "jitter"   -> perturb the clamped position by a seeded offset in
                    [-POSITION_JITTER_HALF_WIDTH, +POSITION_JITTER_HALF_WIDTH] and
                    re-clamp to [0,1]. Lets the driver RETRY-on-empty with a
                    DIFFERENT seeded position (pass a different seed -> a different,
                    still-deterministic offset). seed=None -> no jitter (returns the
                    clamped position, so an unseeded jitter is inert, never random).
      "adaptive" -> if an SWA window is in play, pin the needle near the START
                    (ADAPTIVE_SWA_START_POSITION) so it lands INSIDE a
                    sliding-window model's attended span. SWA is considered "in
                    play" only when swa_window is a positive int AND it is smaller
                    than target_tokens (the prompt is longer than the window, so the
                    tail would evict an early-but-not-start needle). When SWA is
                    ABSENT (swa_window is None / non-positive / >= target_tokens —
                    e.g. a DENSE full-attention model), adaptive is a NO-OP and
                    returns the clamped requested position VERBATIM (byte-identical
                    to "fixed"), so a dense vehicle is unaffected.

    An unknown strategy falls through to the fixed (verbatim) behaviour — never
    raises (mirrors generate_prompt's unknown-filler_variant tolerance).
    """
    clamped = max(0.0, min(1.0, float(position)))
    if strategy == "jitter":
        if seed is None:
            return clamped
        # Dedicated RNG so jitter NEVER perturbs the family-builder RNG stream
        # (determinism contract for the prompt body is independent of placement).
        # The RNG seed is a namespaced SHA-256 of the int seed (Python 3.14's
        # random.Random rejects a tuple seed), so jitter is decorrelated from the
        # family builder's random.Random(seed) yet still fully deterministic.
        digest = hashlib.sha256(f"position-jitter|{int(seed)}".encode()).digest()
        jrng = random.Random(int.from_bytes(digest[:8], "big"))
        offset = jrng.uniform(-POSITION_JITTER_HALF_WIDTH, POSITION_JITTER_HALF_WIDTH)
        return max(0.0, min(1.0, clamped + offset))
    if strategy == "adaptive":
        swa_in_play = (
            swa_window is not None
            and int(swa_window) > 0
            and target_tokens is not None
            and int(swa_window) < int(target_tokens)
        )
        if swa_in_play:
            return max(0.0, min(1.0, ADAPTIVE_SWA_START_POSITION))
        # SWA absent (dense model, or window >= prompt): NO-OP -> verbatim.
        return clamped
    # "fixed" and any unknown strategy: verbatim (today's behaviour).
    return clamped


def _insert_at_position(
    filler: str,
    needle: str,
    position: float,
    *,
    strategy: str = "fixed",
    seed: int | None = None,
    swa_window: int | None = None,
    target_tokens: int | None = None,
) -> str:
    """Insert needle into filler near the chunk boundary closest to position.

    F0 PLACEMENT: placement is now strategy-driven. The optional keyword args
    (strategy/seed/swa_window/target_tokens) select the EFFECTIVE position via
    _resolve_position; ALL default to today's behaviour, so a call that passes only
    (filler, needle, position) is byte-identical to the pre-strategy code.
    """
    if not filler:
        return needle
    position = _resolve_position(
        position, strategy=strategy, seed=seed,
        swa_window=swa_window, target_tokens=target_tokens,
    )
    clamped = max(0.0, min(1.0, position))
    sentences = re.split(r"(?<=[.!?])\s+", filler)
    if len(sentences) <= 1:
        # Fall back to char-level cut.
        cut = int(len(filler) * clamped)
        return filler[:cut].rstrip() + " " + needle + " " + filler[cut:].lstrip()
    # Walk sentences accumulating tokens; pick boundary closest to target.
    total_tokens = sum(estimate_tokens(s) for s in sentences) or 1
    target = total_tokens * clamped
    running = 0
    insert_idx = 0
    best_diff = float("inf")
    for i, s in enumerate(sentences):
        running += estimate_tokens(s)
        diff = abs(running - target)
        if diff < best_diff:
            best_diff = diff
            insert_idx = i + 1
    insert_idx = max(0, min(len(sentences), insert_idx))
    before = " ".join(sentences[:insert_idx]).strip()
    after = " ".join(sentences[insert_idx:]).strip()
    parts = [p for p in (before, needle, after) if p]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Measured-token calibration (binary search against an injected token counter)
# ---------------------------------------------------------------------------

def _build_reservoir(
    rng: random.Random,
    base_text: str,
    target_tokens: int,
    filler_variant: str,
) -> str:
    """Build an oversized, deterministic filler reservoir of at least
    ceil(target_tokens * REAL_CHARS_PER_TOK_GUESS) chars.

    coherent      -> tile _expand_to_target output (seeded sentence shuffle).
    whitespace    -> tile the blank-line builder.
    masked_repeat -> tile the neutral-comment builder.

    CORPUS-SHORTER-THAN-TARGET is handled here: if one pass of the builder is
    shorter than the needed char reservoir, we keep appending fresh builder
    passes (each re-seeded-shuffled for coherent via the SAME rng, so the tile
    sequence is fully determined by (seed, target)) until the reservoir is long
    enough. Tiling count is a pure function of (seed, target, corpus bytes), so
    the result stays byte-reproducible.
    """
    needed_chars = int(math.ceil(max(1, target_tokens) * REAL_CHARS_PER_TOK_GUESS))
    # One generous builder pass. _expand_to_target's own budget is in (its)
    # estimate_tokens units; we deliberately over-ask so a single pass usually
    # already exceeds needed_chars, but we still loop to be safe for tiny corpora.
    overshoot_tokens = max(1, int(target_tokens * REAL_CHARS_PER_TOK_GUESS))
    pieces: list[str] = []
    total = 0
    guard = 0
    while total < needed_chars:
        if filler_variant == "whitespace":
            chunk = _build_whitespace_filler(overshoot_tokens)
        elif filler_variant == "masked_repeat":
            chunk = _build_masked_repeat_filler(overshoot_tokens)
        else:
            chunk = _expand_to_target(rng, base_text, overshoot_tokens)
        if not chunk:
            chunk = base_text or _FALLBACK_GUTENBERG
        pieces.append(chunk)
        total += len(chunk) + 1
        guard += 1
        if guard > 100_000:        # pathological corpus guard (never expected)
            break
    return "\n".join(pieces)


def _calibrate_filler(
    rng: random.Random,
    base_text: str,
    needle: str,
    assemble: Callable[[str], str],
    target_tokens: int,
    token_counter: Callable[[str], int],
    position: float,
    filler_variant: str = "coherent",
    tol: float = CALIBRATION_TOL,
    max_iter: int = CALIBRATION_MAX_ITER,
    ctx_cap: int | None = None,
) -> tuple[str, int, bool]:
    """Binary-search a CHAR cut-length over a deterministic filler reservoir so
    the MEASURED token count of the FULLY ASSEMBLED prompt hits target_tokens
    within `tol`.

    `assemble(filler)` -> the full prompt_text (haystack + question scaffold +
    needle); the counter is measured on THAT, never on filler alone, so the
    needle / question / system overhead is inside the budget.

    Monotonicity: more filler chars -> a longer prompt -> >= tokens (the counter
    is monotone non-decreasing in input length for any real tokenizer), so
    bisection converges. Returns (filler_string, measured_tokens, capped) for the
    closest candidate seen (so a slightly-short reservoir still yields the best
    answer rather than raising). `capped` is True iff a ctx_cap was supplied AND
    the tolerance band around the aimed-at target crosses the ceiling
    (eff_target + abs_tol > ctx_cap) — i.e. the uncapped optimum could itself
    land above ctx_cap, so the ceiling is the binding constraint. (We do NOT key
    `capped` on "some probe exceeded the cap": the reservoir is deliberately
    oversized, so its full-length top probe exceeds ctx_cap on EVERY call and
    would make the flag meaningless. The tolerance-band test fires only for the
    genuine near-ctx corner.)

    ctx_cap (extreme-corner ctx overshoot guard): when set, the
    binary search treats it as a HARD ceiling the emitted prompt MUST NOT exceed.
    Background: a `tol`-band lands the calibrated prompt at up to target*(1+tol)
    tokens; with target ~= 0.98*ctx and a DENSER-than-the-reservoir-guess real
    tokenizer (a model could be ~4.3 chars/tok), target*(1+tol)
    can slip just OVER the server's hard `-c` cap and trip HTTP 400
    exceed_context. With ctx_cap given we (a) clamp the search target to
    min(target_tokens, ctx_cap) so we aim at-or-below the cap, and (b) NEVER
    return a candidate whose measured tokens exceed ctx_cap — among capped probes
    we keep the LARGEST measured count <= ctx_cap (closest to full without
    overshoot), falling back to the smallest-overshoot candidate only if the
    reservoir's minimum already exceeds the cap (degenerate; cannot happen for a
    realistic prompt). ctx_cap=None is the EXACT legacy behavior (byte-identical;
    every existing test keys off the uncapped path).
    """
    reservoir = _build_reservoir(rng, base_text, target_tokens, filler_variant)
    lo, hi = 0, len(reservoir)
    # When a hard ctx ceiling is supplied, aim at-or-below it (a fill target that
    # is itself above the cap is meaningless — we can never legally emit it).
    eff_target = target_tokens if ctx_cap is None else min(target_tokens, ctx_cap)
    abs_tol = max(1, int(round(eff_target * tol)))

    # Track the best LEGAL (<= ctx_cap) candidate separately so the cap is a hard
    # invariant, not merely a target. best_* is the closest-to-eff_target probe;
    # cap_* is the largest measured count that is still <= ctx_cap.
    cap_filler: str | None = None
    cap_tok = -1
    # The ceiling is the BINDING constraint iff the tolerance band around the
    # aimed-at target can reach above it — then the uncapped optimum could land
    # over ctx_cap and the clamp matters. Deterministic in (target, tol, cap);
    # independent of which probes the bisection happened to sample.
    ceiling_binds = bool(ctx_cap is not None and (eff_target + abs_tol) > ctx_cap)

    def _consider(filler: str, tok: int) -> None:
        nonlocal cap_filler, cap_tok
        if ctx_cap is not None and tok <= ctx_cap and tok > cap_tok:
            cap_filler, cap_tok = filler, tok

    def measure(mid: int) -> tuple[str, int]:
        filler = reservoir[:mid]
        prompt = assemble(filler)
        tok = int(token_counter(prompt))
        _consider(filler, tok)
        return filler, tok

    def _capped(final_tok: int) -> bool:
        # Reported capped iff the ceiling was the binding constraint and the
        # emitted prompt is legally at/under it.
        return bool(ceiling_binds and ctx_cap is not None and final_tok <= ctx_cap)

    def _result(best_filler: str, best_tok: int) -> tuple[str, int, bool]:
        # Enforce the hard ceiling: if the best-by-distance candidate overshoots
        # the cap, prefer the largest legal (<= cap) candidate we ever measured.
        if ctx_cap is not None and best_tok > ctx_cap and cap_filler is not None:
            return cap_filler, cap_tok, _capped(cap_tok)
        return best_filler, best_tok, _capped(best_tok)

    best_filler, best_tok = measure(hi)
    best_diff = abs(best_tok - eff_target)
    # If even the full reservoir is short of target, return it (best effort) —
    # the downstream realized-vs-target assert will still flag any saturation.
    if best_tok < eff_target - abs_tol:
        # Try lo too in case empty is closer (it never is for positive target),
        # then return the full reservoir as the closest-achievable.
        return _result(best_filler, best_tok)

    for _ in range(max_iter):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        filler, tok = measure(mid)
        diff = abs(tok - eff_target)
        if diff < best_diff:
            best_diff, best_filler, best_tok = diff, filler, tok
        if diff <= abs_tol and not (ctx_cap is not None and tok > ctx_cap):
            # Within tolerance AND legal (never early-return on an over-cap hit).
            return filler, tok, _capped(tok)
        if tok < eff_target:
            lo = mid
        else:
            hi = mid
    # Exhausted: evaluate the final boundary candidates and keep the closest.
    for cand_mid in (lo, hi):
        filler, tok = measure(cand_mid)
        diff = abs(tok - eff_target)
        if diff < best_diff:
            best_diff, best_filler, best_tok = diff, filler, tok
    return _result(best_filler, best_tok)


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------

# Entity inventory used by Family A1 and B2 (deterministic shuffle per seed).
_A1_NAMES = ["Yuki", "Marta", "Kenji", "Ines", "Pavel", "Amara", "Nico", "Linnea"]
_A1_LANDMARKS = [
    ("Semper Opera House", "Germany"),
    ("Brandenburg Gate", "Germany"),
    ("Cologne Cathedral", "Germany"),
    ("Reichstag Building", "Germany"),
]

_B2_OWNERS = ["Carlos", "Mei", "Theo", "Priya", "Sven", "Aisha"]
_B2_BUSINESSES = ["Cafe Estrella", "Cafe Hokkaido", "Cafe Aurora", "Cafe Mirage"]
_B2_YEARS = [1987, 1989, 1991, 1993]
_B2_YEAR_FACTS = {
    1987: "the year the Berlin Wall began to crack",
    1989: "the year the Berlin Wall fell",
    1991: "the year the Soviet Union dissolved",
    1993: "the year the European Union was formally established",
}

_D1_INSPECTORS = ["Dr. Patel", "Ms. Okafor", "Mr. Tanaka", "Capt. Ramirez"]
_D1_MONTHS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

# Entity inventory for the E-series (RULER CWE-K / IFEval / tool-call / code-edit).
_E1_KEY_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]

_E2_STRINGS = ["ready", "pending", "active", "queued", "done", "failed", "idle"]

_E3_FUNCS = [
    {
        "name": "get_weather",
        "schema": {
            "city": "string",
            "units": "string",
        },
        "city_pool": ["Berlin", "Tokyo", "Cairo", "Lima", "Oslo", "Perth"],
        "units_pool": ["celsius", "fahrenheit"],
        "template": (
            "What is the weather in {city}? Report it in {units}."
        ),
    },
    {
        "name": "schedule_meeting",
        "schema": {
            "attendees": "integer",
            "topic": "string",
        },
        "topic_pool": ["budget", "hiring", "roadmap", "security", "launch"],
        "template": (
            "Set up a meeting about {topic} for {attendees} attendees."
        ),
    },
    {
        "name": "convert_currency",
        "schema": {
            "amount": "integer",
            "currency": "string",
        },
        "currency_pool": ["USD", "EUR", "JPY", "GBP", "CHF"],
        "template": (
            "Convert {amount} units into {currency}."
        ),
    },
]

# E4 buggy/fixed function pairs. {n} substitutes a deterministic literal so each
# seed produces a distinct-but-canonical pair. The "bug" is described in prose.
#
# SCORER COUPLING (keep in sync): quality_scorer._ast_semantic_signature is tuned
# to exactly the bug-vs-fix node classes these 3 templates exercise — a Constant
# flip (range(1)->range(0)), a BinOp flip (x-{n} -> x+{n}), and a Compare flip
# (value>{n} -> value>={n}). If you ADD an E4 template whose bug-vs-fix difference
# is a DIFFERENT node class (e.g. a swapped call, a renamed attribute, a string
# literal), revisit _ast_semantic_signature so the new shape is still detected as
# a semantic (not cosmetic) mismatch.
_E4_TEMPLATES = [
    {
        "bug": "the loop starts at 1 instead of 0, skipping the first element",
        "buggy": (
            "def total(items):\n"
            "    s = 0\n"
            "    for i in range(1, len(items)):\n"
            "        s = s + items[i]\n"
            "    return s\n"
        ),
        "fixed": (
            "def total(items):\n"
            "    s = 0\n"
            "    for i in range(0, len(items)):\n"
            "        s = s + items[i]\n"
            "    return s\n"
        ),
    },
    {
        "bug": "it returns x - {n} but should return x + {n}",
        "buggy": (
            "def adjust(x):\n"
            "    return x - {n}\n"
        ),
        "fixed": (
            "def adjust(x):\n"
            "    return x + {n}\n"
        ),
    },
    {
        "bug": "the comparison uses > but should use >= to include the boundary",
        "buggy": (
            "def above(value):\n"
            "    return value > {n}\n"
        ),
        "fixed": (
            "def above(value):\n"
            "    return value >= {n}\n"
        ),
    },
]


def _seed_short(seed: int) -> str:
    """6-char hex tag for code-recall identifiers."""
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:6]


def _random_uuid8(rng: random.Random) -> str:
    """Deterministic 8-char alphanumeric uuid via the seeded RNG."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(8))


def _build_family_a1(rng: random.Random) -> dict:
    name = rng.choice(_A1_NAMES)
    landmark, country = rng.choice(_A1_LANDMARKS)
    needle = f"{name} lives in the apartment next to the {landmark}."
    question = f"Which character has been to {country}?"
    return {
        "needle": needle,
        "question": question,
        "expected_answer": name,
        "filler_key": "gutenberg",
    }


def _build_family_a2(rng: random.Random) -> dict:
    uuid8 = _random_uuid8(rng)
    secret8 = _random_uuid8(rng)
    needle = f"The activation code for sector {uuid8} is {secret8}."
    question = f"What is the activation code for sector {uuid8}?"
    return {
        "needle": needle,
        "question": question,
        "expected_answer": secret8,
        "filler_key": "gutenberg",
    }


def _build_family_a3(rng: random.Random) -> dict:
    keys = [_random_uuid8(rng) for _ in range(4)]
    values = [_random_uuid8(rng) for _ in range(4)]
    needle_lines = [f"The badge for {k} is {v}." for k, v in zip(keys, values, strict=False)]
    needle = " ".join(needle_lines)
    pick = rng.randrange(4)
    question = f"What is the badge for {keys[pick]}?"
    return {
        "needle": needle,
        "question": question,
        "expected_answer": values[pick],
        "filler_key": "cc_news",
    }


def _build_family_a4(rng: random.Random) -> dict:
    x1 = rng.randint(1, 50)
    add = rng.randint(1, 20)
    sub = rng.randint(1, 10)
    x2 = x1 + add
    x3 = x2 * 2
    x4 = x3 - sub
    assignments = [
        f"Let X1 = {x1}.",
        f"Let X2 = X1 + {add} (so X2 = {x2}).",
        f"Let X3 = X2 * 2 (so X3 = {x3}).",
        f"Let X4 = X3 - {sub} (so X4 = {x4}).",
    ]
    # Scatter assignments by separating them with filler gap markers.
    needle = " ".join(assignments)
    question = "What is X4?"
    return {
        "needle": needle,
        "question": question,
        "expected_answer": str(x4),
        "filler_key": "gutenberg",
        "_a4_chain": {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "add": add, "sub": sub},
    }


def _build_family_b1(rng: random.Random) -> dict:
    val1 = rng.randint(2, 20)
    val2 = rng.randint(2, 9)
    a = val1
    b = a * val2
    c = b + a
    needle = (
        f"A = {val1}. "
        f"B = A * {val2} (so B = {b}). "
        f"C = B + A (so C = {c})."
    )
    question = "What is C?"
    return {
        "needle": needle,
        "question": question,
        "expected_answer": str(c),
        "filler_key": "gutenberg",
    }


def _build_family_b2(rng: random.Random) -> dict:
    owner = rng.choice(_B2_OWNERS)
    business = rng.choice(_B2_BUSINESSES)
    year = rng.choice(_B2_YEARS)
    fact_year = _B2_YEAR_FACTS[year]
    needle = (
        f"{owner} owns {business}. "
        f"{business} opened in {year}. "
        f"{year} was {fact_year}."
    )
    # 2-fact recombination: who owns the business that opened in the year of <fact>?
    question = (
        f"Who owns the business that opened in the year that was {fact_year}?"
    )
    return {
        "needle": needle,
        "question": question,
        "expected_answer": owner,
        "filler_key": "cc_news",
    }


def _build_family_c1(rng: random.Random, seed: int) -> dict:
    magic = rng.randint(7, 99)
    short = _seed_short(seed)
    func_name = f"secret_func_{short}"
    needle = (
        f"def {func_name}(x):\n"
        f"    return x * {magic}\n"
    )
    question = f"What does {func_name}(7) return?"
    return {
        "needle": needle,
        "question": question,
        "expected_answer": str(7 * magic),
        "filler_key": "python",
    }


def _build_family_d1(rng: random.Random) -> dict:
    volume = rng.randint(500, 9_999)
    inspector = rng.choice(_D1_INSPECTORS)
    sign_month = rng.choice(_D1_MONTHS)
    sign_day = rng.randint(1, 28)
    sign_year = rng.randint(2024, 2027)
    audit_month = rng.choice(_D1_MONTHS)
    audit_day = rng.randint(1, 28)
    audit_year = sign_year + rng.choice([0, 1])
    sign_date = f"{sign_month} {sign_day}, {sign_year}"
    audit_date = f"{audit_month} {audit_day}, {audit_year}"
    fact1 = f"The reactor coolant volume measured {volume} liters."
    fact2 = f"{inspector} signed the report on {sign_date}."
    fact3 = f"The next audit is scheduled for {audit_date}."
    needle = f"{fact1} {fact2} {fact3}"
    question = "List the three facts as bullets (volume, inspector, next audit)."
    expected = {
        "volume": str(volume),
        "inspector": inspector,
        "next_audit": audit_date,
    }
    return {
        "needle": needle,
        "question": question,
        "expected_answer": expected,
        "filler_key": "gutenberg",
    }


def _build_family_e1(rng: random.Random) -> dict:
    """RULER CWE-K: 16 scattered (key,value) pairs; ask for 4 specific values."""
    words = _E1_KEY_WORDS[:]
    rng.shuffle(words)
    keys = words[:16]
    values = [_random_uuid8(rng) for _ in range(16)]
    pairs = list(zip(keys, values, strict=False))
    needle_lines = [f"The code for {k} is {v}." for k, v in pairs]
    needle = " ".join(needle_lines)
    ask_idx = sorted(rng.sample(range(16), 4))
    ask_keys = [keys[i] for i in ask_idx]
    expected = [values[i] for i in ask_idx]
    question = (
        "Give the codes for the following keys, in this exact order: "
        + ", ".join(ask_keys) + "."
    )
    return {
        "needle": needle,
        "question": question,
        "expected_answer": expected,
        "filler_key": "cc_news",
    }


def _build_family_e2(rng: random.Random) -> dict:
    """IFEval-style: return ONLY a JSON object with keys a (int), b (str), c (bool)."""
    a_val = rng.randint(1, 999)
    b_val = rng.choice(_E2_STRINGS)
    c_val = rng.choice([True, False])
    expected = {"a": a_val, "b": b_val, "c": c_val}
    needle = (
        f"For this task the required field values are: a = {a_val}, "
        f"b = \"{b_val}\", c = {str(c_val).lower()}."
    )
    question = (
        "Return ONLY a single JSON object (no prose, no code fence) with exactly "
        "these keys: a (integer), b (string), c (boolean), set to the required "
        "field values stated in the passage."
    )
    return {
        "needle": needle,
        "question": question,
        "expected_answer": expected,
        "filler_key": "cc_news",
    }


def _build_family_e3(rng: random.Random) -> dict:
    """Function-calling: emit a JSON tool call {name, arguments} matching a schema."""
    spec = rng.choice(_E3_FUNCS)
    name = spec["name"]
    schema = dict(spec["schema"])
    args: dict[str, object] = {}
    fields: dict[str, str] = {}
    if name == "get_weather":
        city = rng.choice(spec["city_pool"])
        units = rng.choice(spec["units_pool"])
        args = {"city": city, "units": units}
        fields = {"city": city, "units": units}
    elif name == "schedule_meeting":
        attendees = rng.randint(2, 40)
        topic = rng.choice(spec["topic_pool"])
        args = {"attendees": attendees, "topic": topic}
        fields = {"attendees": str(attendees), "topic": topic}
    else:  # convert_currency
        amount = rng.randint(1, 9999)
        currency = rng.choice(spec["currency_pool"])
        args = {"amount": amount, "currency": currency}
        fields = {"amount": str(amount), "currency": currency}
    user_query = spec["template"].format(**fields)
    schema_text = json.dumps(schema, sort_keys=True)
    needle = (
        f"Available function: {name} with arguments schema {schema_text}. "
        f"User request: {user_query}"
    )
    question = (
        "Respond with ONLY a single JSON object representing the tool call, with "
        "keys \"name\" (the function name) and \"arguments\" (an object whose fields "
        "match the schema and fill in the user request)."
    )
    expected = {"name": name, "arguments": args, "schema": schema}
    return {
        "needle": needle,
        "question": question,
        "expected_answer": expected,
        "filler_key": "python",
    }


def _build_family_e4(rng: random.Random) -> dict:
    """Code-edit: fix one described bug, return only the corrected function."""
    tmpl = rng.choice(_E4_TEMPLATES)
    n = rng.randint(1, 99)
    buggy = tmpl["buggy"].replace("{n}", str(n))
    fixed = tmpl["fixed"].replace("{n}", str(n))
    bug_desc = tmpl["bug"].replace("{n}", str(n))
    needle = (
        "Here is a Python function with a bug. "
        f"The bug: {bug_desc}.\n{buggy}"
    )
    question = (
        "Return ONLY the corrected Python function (no prose, no explanation). "
        "Fix the described bug and keep everything else identical."
    )
    return {
        "needle": needle,
        "question": question,
        "expected_answer": fixed,
        "filler_key": "python",
    }


# E5 semantic-hop code-symbol needle (perfect-retrieval control).
#
# The answer requires a TWO-PREDICATE HOP ("the function that calls parse_config
# AND returns a dict"), and — critically — the answer is NOT a unique copyable
# token:
#   * the target name is drawn from the SAME generic pool as the decoys, and
#   * the target name is ECHOED on a NON-answer line (a decoy CALLS it), and
#   * a SECOND function ALSO calls parse_config but returns a LIST — so
#     predicate-1 ("calls parse_config") is provably NON-unique and predicate-2
#     ("returns a dict") is load-bearing; the answer can no longer be found by
#     predicate-1 alone.
# so "emit the only matching identifier" cannot win the control (NoLiMa trap,
# arXiv 2502.05167). At least three functions in the needle are structurally
# similar (all have a body that `return`s), and two of them call parse_config, so
# the model cannot shortcut on either "the one that returns" or "the one that
# calls parse_config". Determinism: rng.sample over a fixed pool with
# random.Random(seed) is byte-stable.
_E5_HELPERS = ["normalize_path", "read_lines", "split_pairs", "merge_dicts",
               "load_table", "scan_dir", "build_index"]


def _build_family_e5(rng: random.Random, seed: int) -> dict:
    """Code-symbol semantic hop: name the function that calls parse_config and
    returns a DICT. The answer is a GENERIC function name (also a decoy-pool
    member and echoed on a non-answer call line), so it cannot be won by copying
    the only special-looking identifier. Critically, a SECOND function ALSO calls
    parse_config but returns a LIST — so predicate-1 ("calls parse_config") is
    provably non-unique and the model MUST use predicate-2 ("returns a dict") to
    disambiguate (the answer is NOT identifiable by predicate-1 alone). NoLiMa
    trap, arXiv 2502.05167. Deterministic per seed; exact-match scored on the
    function NAME."""
    names = rng.sample(_E5_HELPERS, 5)   # 5 generic names; none is "special"
    target = names[0]                    # the answer — also echoed below
    d1, d2, d3, dlist = names[1], names[2], names[3], names[4]
    needle = (
        # decoy 1: returns, but does NOT call parse_config
        f"def {d1}(p):\n    return p.strip()\n\n"
        # the ANSWER: calls parse_config AND returns a dict
        f"def {target}(path):\n"
        f"    raw = parse_config(path)\n"
        f"    return dict(raw)\n\n"
        # DECOY: ALSO calls parse_config but returns a LIST, not a dict.
        # Now TWO functions call parse_config, so predicate-1 alone is NOT
        # uniquely identifying — the model must use predicate-2 ("returns a
        # dict") to pick `target`.
        f"def {dlist}(p):\n"
        f"    cfg = parse_config(p)\n"
        f"    return list(cfg)\n\n"
        # decoy 2: structurally similar (also returns) and ECHOES the target name
        # as a plain call (so `target` appears on a NON-answer line — copying the
        # token is now ambiguous), but this fn does NOT call parse_config.
        f"def {d2}(xs):\n    return {target}(xs[0]) if xs else {d3}(xs)\n\n"
        # decoy 3: returns, unrelated
        f"def {d3}(xs):\n    return list(xs)\n"
    )
    question = (
        "Exactly one function in the passage both calls parse_config and returns "
        "a dict. Name that function. Answer with ONLY the function name."
    )
    return {
        "needle": needle,
        "question": question,
        "expected_answer": target,
        "filler_key": "python",
    }


_BUILDERS = {
    "A1": lambda rng, seed: _build_family_a1(rng),
    "A2": lambda rng, seed: _build_family_a2(rng),
    "A3": lambda rng, seed: _build_family_a3(rng),
    "A4": lambda rng, seed: _build_family_a4(rng),
    "B1": lambda rng, seed: _build_family_b1(rng),
    "B2": lambda rng, seed: _build_family_b2(rng),
    "C1": lambda rng, seed: _build_family_c1(rng, seed),
    "D1": lambda rng, seed: _build_family_d1(rng),
    "E1": lambda rng, seed: _build_family_e1(rng),
    "E2": lambda rng, seed: _build_family_e2(rng),
    "E3": lambda rng, seed: _build_family_e3(rng),
    "E4": lambda rng, seed: _build_family_e4(rng),
    "E5": lambda rng, seed: _build_family_e5(rng, seed),
}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def generate_prompt(
    family: str,
    seed: int,
    target_tokens: int,
    position: float,
    corpus_dir: Path,
    filler_variant: str = "coherent",
    token_counter: Callable[[str], int] | None = None,
    calibration_tol: float = CALIBRATION_TOL,
    ctx_cap: int | None = None,
    position_strategy: str = "fixed",
    swa_window: int | None = None,
) -> dict:
    """Generate a deterministic benchmark prompt for the given family and seed.

    filler_variant selects the haystack composition:
      "coherent"      -> the family's corpus key (default; unchanged behavior)
      "whitespace"    -> blank lines (pure-length pad, zero distractor signal)
      "masked_repeat" -> one neutral comment line tiled (repetition-only pad)
    An unknown variant falls through to the coherent build (no exception).

    token_counter (measured-calibration, fill-bug fix):
      None  -> EXACT legacy behavior: the filler budget is sized via
               estimate_tokens. Byte-identical to the pre-calibration code for a
               fixed (seed, target, corpus); preserves determinism + all tests.
      callable(text) -> int  -> MEASURED-calibration path: binary-search the
               filler char-length so the counter's measurement of the FULL
               assembled prompt lands within calibration_tol of target_tokens.
               In production this is a /tokenize closure; in tests it is a
               deterministic ~3.7-chars/tok stub. If the counter raises on its
               first probe, we fall back to the legacy estimate path and stamp
               calibration='estimate' (never propagate the exception).

    ctx_cap (extreme-corner ctx overshoot guard): an optional
    HARD ceiling on the emitted prompt's measured tokens. The tolerance band can
    otherwise land a calibrated prompt at target*(1+tol); with target ~= 0.98*ctx
    and a denser-than-expected real tokenizer that can creep just over the
    server's hard `-c` cap and trip HTTP 400 exceed_context. When set, the
    measured-calibration search aims at min(target_tokens, ctx_cap) and NEVER
    returns a prompt whose measured tokens exceed ctx_cap. Inert on the legacy
    (token_counter=None) path. In production pass ctx_cap=ctx_tier; varies per
    model and is cheap insurance before a multi-model sweep.

    position_strategy (F0 PLACEMENT): selects HOW the requested `position` maps to
    the EFFECTIVE needle position. DEFAULTS to "fixed" — byte-identical to the
    pre-strategy code path. See _resolve_position for the full contract:
      "fixed"    -> use `position` verbatim (TODAY's behaviour; the DEFAULT).
      "jitter"   -> seeded position jitter via random.Random(seed): perturb by a
                    deterministic offset in +/-POSITION_JITTER_HALF_WIDTH. Lets the
                    driver RETRY-on-empty with a DIFFERENT seeded position by passing
                    a different `seed` (same seed -> same offset, every time).
      "adaptive" -> needle-at-START when an SWA window is in play (swa_window a
                    positive int < target_tokens), else a NO-OP (verbatim). A DENSE
                    model passes swa_window=None and is therefore UNAFFECTED.
    An unknown strategy falls through to "fixed" (no exception).

    swa_window (F0 PLACEMENT): the model's sliding-window-attention span in tokens,
    or None for a DENSE (full-attention) model. ONLY consulted by the "adaptive"
    strategy; inert for "fixed"/"jitter". When None (the default) adaptive is a
    no-op, so a dense vehicle is byte-identical to "fixed".

    Added return keys: 'prompt_tokens_measured' (int when measured, else None),
    'calibration' ('measured' | 'estimate'), and 'calibration_capped' (True when
    a measured prompt was held at/under ctx_cap, i.e. the ceiling bound; False
    otherwise; None on the legacy path). Placement keys: 'position_strategy' (the
    requested strategy, echoed), 'effective_position' (the resolved position
    actually used for insertion; equals `position` on the fixed/no-op path), and
    'swa_window' (echoed).
    """
    if family not in _BUILDERS:
        raise ValueError(f"Unknown family {family!r}. Expected one of {FAMILIES}.")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    rng = random.Random(seed)
    spec = _BUILDERS[family](rng, seed)
    needle = spec["needle"]
    filler_key = spec["filler_key"]

    corpus, corpus_sources = _load_corpus_with_sources(Path(corpus_dir))
    base_text = corpus.get(filler_key) or _FALLBACK_GUTENBERG
    corpus_source = corpus_sources.get(filler_key, "fallback")
    corpus_all_real = all(v == "real" for v in corpus_sources.values())

    system_prompt = (
        "You are a careful reader. Read the passage and answer the question "
        "using only information present in the passage. Give a concise answer."
    )

    # F0 PLACEMENT: resolve the EFFECTIVE position ONCE, up front, so the
    # calibration probes and the final emitted prompt all splice the needle at the
    # SAME offset (the measured prompt IS the returned prompt). _assemble then uses
    # the resolved position with strategy="fixed" (verbatim) — never re-resolving.
    # On the default "fixed"/dense-adaptive path effective_position == position, so
    # the assembled prompt is byte-identical to the pre-strategy code.
    effective_position = _resolve_position(
        position, strategy=position_strategy, seed=seed,
        swa_window=swa_window, target_tokens=target_tokens,
    )

    def _assemble(filler: str) -> str:
        """Assemble the full prompt_text from a filler string. Used identically
        by the legacy path, the measured-calibration binary search, and the
        final returned prompt — so the measured prompt IS the returned prompt."""
        haystack = _insert_at_position(filler, needle, effective_position)
        return (
            f"{haystack}\n\n"
            f"Question: {spec['question']}\n"
            f"Answer:"
        )

    prompt_tokens_measured: int | None = None
    calibration = "estimate"
    calibration_capped: bool | None = None

    if token_counter is not None:
        try:
            filler, prompt_tokens_measured, calibration_capped = _calibrate_filler(
                rng=rng,
                base_text=base_text,
                needle=needle,
                assemble=_assemble,
                target_tokens=target_tokens,
                token_counter=token_counter,
                position=position,
                filler_variant=filler_variant,
                tol=calibration_tol,
                ctx_cap=ctx_cap,
            )
            prompt_text = _assemble(filler)
            calibration = "measured"
            # calibration_capped comes straight from the calibrator: True iff a
            # hard ctx ceiling was supplied, a probe actually exceeded it, and the
            # emitted prompt was held at/under it — i.e. the would-be
            # target*(1+tol) overshoot was clamped (the extreme-corner near-ctx
            # case). False on the measured path when no cap bound; None on legacy.
        except Exception:
            # Counter failed (e.g. /tokenize 404 / transport error). Fall back to
            # the deterministic estimate path; the realized-vs-target assert
            # downstream still catches any saturation. Never propagate.
            token_counter = None
            prompt_tokens_measured = None
            calibration = "estimate"

    if token_counter is None:
        # Legacy path (verbatim): size the filler budget with estimate_tokens.
        # Re-seed the RNG so the estimate path is byte-identical whether or not a
        # (later-failing) counter was passed — calibration must not perturb the
        # determinism contract on the fallback.
        rng = random.Random(seed)
        _ = _BUILDERS[family](rng, seed)  # advance RNG exactly as the real build did
        needle_tokens = estimate_tokens(needle)
        question_tokens = estimate_tokens(spec["question"])
        filler_budget = max(64, target_tokens - needle_tokens - question_tokens - 16)
        if filler_variant == "whitespace":
            filler = _build_whitespace_filler(filler_budget)
        elif filler_variant == "masked_repeat":
            filler = _build_masked_repeat_filler(filler_budget)
        else:
            filler = _expand_to_target(rng, base_text, filler_budget)
        prompt_text = _assemble(filler)

    actual_tokens_est = estimate_tokens(prompt_text)

    return {
        "family": family,
        "seed": seed,
        "target_tokens": target_tokens,
        "actual_tokens_est": actual_tokens_est,
        "prompt_tokens_measured": prompt_tokens_measured,
        "calibration": calibration,
        "calibration_capped": calibration_capped,
        "position": position,
        "position_strategy": position_strategy,
        "effective_position": effective_position,
        "swa_window": swa_window,
        "prompt_text": prompt_text,
        "expected_answer": spec["expected_answer"],
        "needle_text": needle,
        "system_prompt": system_prompt,
        "corpus_source": corpus_source,
        "corpus_all_real": corpus_all_real,
        "filler_variant": filler_variant,
    }


__all__ = [
    "generate_prompt",
    "list_families",
    "estimate_tokens",
    "seed_from_tuple",
    "FAMILIES",
    "_calibrate_filler",
    "_resolve_position",
    "REAL_CHARS_PER_TOK_GUESS",
    "CALIBRATION_TOL",
    "CALIBRATION_MAX_ITER",
    "POSITION_STRATEGIES",
    "POSITION_JITTER_HALF_WIDTH",
    "ADAPTIVE_SWA_START_POSITION",
]

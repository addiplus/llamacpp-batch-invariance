"""Pure-function scoring for LLM responses against expected answers.

Per-family rules:
  A1-A4, B1, B2, C1: exact match (with numeric extraction for numeric expected).
  D1: dict of facts, partial credit by hit count (3=1.0, 2=0.67, 1=0.33, 0=0.0).
  E1: multi-key retrieval, ordered list of 4 values, partial credit /4.
  E2: format adherence, first JSON object, key/type/value match, partial by key.
  E3: function calling, tool-call JSON, name + arguments match, partial by arg.
  E4: code edit, extract code block, AST-equivalence else token edit-distance.

Also provides Wilson 95% lower/upper confidence bounds for binomial proportions
and an aggregator over per-trial scores. No LLM-as-judge.
"""
from __future__ import annotations

import ast
import json
import math
import re

_EXACT_MATCH_FAMILIES = {"A1", "A2", "A3", "B2", "E5"}
_NUMERIC_FAMILIES = {"A4", "B1", "C1"}
_SUMMARY_FAMILIES = {"D1"}
_MULTI_KEY_FAMILIES = {"E1"}
_FORMAT_FAMILIES = {"E2"}
_FUNCTION_CALL_FAMILIES = {"E3"}
_CODE_EDIT_FAMILIES = {"E4"}
_KNOWN_FAMILIES = (
    _EXACT_MATCH_FAMILIES
    | _NUMERIC_FAMILIES
    | _SUMMARY_FAMILIES
    | _MULTI_KEY_FAMILIES
    | _FORMAT_FAMILIES
    | _FUNCTION_CALL_FAMILIES
    | _CODE_EDIT_FAMILIES
)

_STRIP_CHARS = " \t\n\r\"'`.,;:!?()[]{}<>"
_INT_RE = re.compile(r"-?\d+")
_QUOTED_RE = re.compile(r"[\"'“”‘’«»]([^\"'“”‘’«»]+)[\"'“”‘’«»]")


def _normalize(s: str) -> str:
    """Lowercase, strip whitespace and surrounding quotes/punctuation."""
    if s is None:
        return ""
    return str(s).strip().strip(_STRIP_CHARS).strip().lower()


def _extract_last_number(s: str) -> str | None:
    """Return the last integer-looking token in s as a digit string, or None."""
    if not s:
        return None
    matches = _INT_RE.findall(s)
    if not matches:
        return None
    return matches[-1]


def _extract_quoted(s: str) -> str | None:
    """Return the first quoted substring's contents, or None."""
    if not s:
        return None
    m = _QUOTED_RE.search(s)
    if not m:
        return None
    return m.group(1)


def _contains_normalized(needle: str, haystack: str) -> bool:
    """True if normalize(needle) is a substring of normalize(haystack)."""
    n = _normalize(needle)
    if not n:
        return False
    return n in _normalize(haystack)


def _score_exact(expected: str, response_text: str) -> tuple[float, str | None, str]:
    exp_norm = _normalize(expected)
    resp_norm = _normalize(response_text)
    if not exp_norm:
        return 0.0, None, "empty expected"
    if not resp_norm:
        return 0.0, None, "empty response"
    if resp_norm == exp_norm:
        return 1.0, response_text.strip(), "exact match"
    if exp_norm in resp_norm:
        return 1.0, exp_norm, "substring match"
    quoted = _extract_quoted(response_text)
    if quoted is not None and _normalize(quoted) == exp_norm:
        return 1.0, quoted, "quoted match"
    return 0.0, response_text.strip() or None, f"mismatch: expected {expected!r}"


def _score_numeric(expected: str, response_text: str) -> tuple[float, str | None, str]:
    exp_norm = _normalize(expected)
    if not exp_norm:
        return 0.0, None, "empty expected"
    if not response_text or not response_text.strip():
        return 0.0, None, "empty response"
    exp_digits = _extract_last_number(exp_norm)
    last = _extract_last_number(response_text)
    if exp_digits is None:
        if exp_norm in _normalize(response_text):
            return 1.0, exp_norm, "exact match"
        return 0.0, last, "no numeric in expected and no substring match"
    if last is None:
        return 0.0, None, f"no number in response; expected {exp_digits}"
    try:
        if int(last) == int(exp_digits):
            return 1.0, last, "numeric match"
        return 0.0, last, f"wrong value {last} vs {exp_digits}"
    except ValueError:
        return 0.0, last, "numeric parse failure"


def _score_summary(expected: dict, response_text: str) -> tuple[float, str, str]:
    if not isinstance(expected, dict) or not expected:
        return 0.0, "", "empty expected dict"
    if not response_text or not response_text.strip():
        return 0.0, "", "empty response"
    keys = list(expected.keys())
    hits = []
    misses = []
    for k in keys:
        v = expected[k]
        if v is None:
            continue
        if _contains_normalized(str(v), response_text):
            hits.append(k)
        else:
            misses.append(k)
    n_keys = len(hits) + len(misses)
    if n_keys == 0:
        return 0.0, "", "no checkable facts"
    hit_count = len(hits)
    if n_keys == 3:
        partial = {3: 1.0, 2: 0.67, 1: 0.33, 0: 0.0}
        score = partial[hit_count]
    else:
        score = round(hit_count / n_keys, 2)
    extracted = ", ".join(hits) if hits else ""
    reason = f"{hit_count}/{n_keys} facts matched"
    if misses:
        reason += f" (missed: {', '.join(misses)})"
    return score, extracted, reason


# ---------------------------------------------------------------------------
# E-family helpers: JSON extraction, type coercion, code blocks, AST normalize
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```", re.DOTALL)
# Whole-line fence markers only (``` or ```lang). Restricted to a full line so a
# literal triple-backtick INSIDE a string/code is never stripped.
_FENCE_LINE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*$")


def _find_first_json_object(text: str) -> dict | None:
    """Return the first balanced top-level JSON object parsed from text, or None.

    Scans for a '{', tracks brace depth while skipping over string literals, and
    tries json.loads on each candidate span. Never raises.
    """
    if not text:
        return None
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            obj = json.loads(candidate)
                        except (ValueError, TypeError):
                            obj = None
                        if isinstance(obj, dict):
                            return obj
                        break
            j += 1
        i += 1
    return None


def _bool_equal(expected, actual) -> bool:
    """Strict bool comparison: both must be real bools and equal."""
    return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual


def _int_equal(expected, actual) -> bool:
    """Integer comparison, rejecting bools (bool is a subclass of int)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    if not isinstance(actual, int):
        return False
    return expected == actual


def _typed_value_equal(expected, actual) -> bool:
    """Compare one expected field against an actual value matching type AND value."""
    if isinstance(expected, bool):
        return _bool_equal(expected, actual)
    if isinstance(expected, int):
        return _int_equal(expected, actual)
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return abs(float(expected) - float(actual)) < 1e-9
    if isinstance(expected, str):
        return isinstance(actual, str) and expected == actual
    # Fallback: structural equality for nested lists/dicts/None.
    return type(expected) is type(actual) and expected == actual


def _extract_code_block(text: str) -> str:
    """Return code from text, tolerant of stray/unbalanced ``` fences.

    Strategy (E4 false-negative fix):
      1. Prefer a balanced fenced block (the legacy ``` ... ``` regex).
      2. If no balanced fence matched, the text may carry an UNbalanced fence:
         a dropped opening (correct code with a dangling trailing ```), or an
         UNTERMINATED block (model hit the token cap mid-emit). In that case
         locate the whole-line fence markers (^```[lang]?$ only, never a
         backtick embedded in a string literal) and:
           - if there is exactly one fence line, treat it as an opening and
             return everything AFTER it (unterminated block);
           - otherwise strip every fence line and keep the rest.
      3. With no fence markers at all, return the raw text unchanged (prose with
         no code still flows to ast.parse and fails, preserving the
         no-code -> 0.0 contract).
    A surviving stray/opening ``` therefore no longer poisons ast.parse, so a
    genuinely-correct answer becomes AST-equivalent -> 1.0.
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip("\n")
    lines = text.split("\n")
    fence_idx = [i for i, ln in enumerate(lines) if _FENCE_LINE_RE.match(ln)]
    if not fence_idx:
        return text
    if len(fence_idx) == 1:
        # Single dangling fence: opening -> keep what follows; trailing-only ->
        # keep what precedes (whichever side carries the body).
        idx = fence_idx[0]
        after = "\n".join(lines[idx + 1:])
        before = "\n".join(lines[:idx])
        kept = after if after.strip() else before
        return kept.strip("\n")
    kept_lines = [ln for i, ln in enumerate(lines) if i not in set(fence_idx)]
    return "\n".join(kept_lines).strip("\n")


class _RangeZeroNormalizer(ast.NodeTransformer):
    """Collapse `range(0, X)` -> `range(X)` so the idiomatic E4-T0 bug-fix is
    recognised as AST-equivalent to the reference (E4 false-negative fix).

    ROOT CAUSE: template T0's reference fixed answer is `range(0, len(items))`.
    The model emits the idiomatic correct `range(len(items))`. Bare `ast.dump`
    sees different Call arg-lists, AND `_ast_semantic_signature` collects the
    redundant literal `0` into the constant multiset (alongside the `s = 0`
    zero), so a CORRECT fix is graded ~0.30 'operator/constant mismatch'.

    The transform ONLY collapses a LITERAL 0 first arg of a 2-arg, no-keyword
    `range(...)`. Applied UNIFORMLY to expected and candidate it makes the
    idiomatic fix reach the 1.0 AST-equivalent route while leaving every genuine
    survivor FAILing: `range(1, len)` keeps `const:1`, `range(2, len)` keeps
    `const:2`, the `x-7` vs `x+7` BinOp flip and `>` vs `>=` Compare flip are
    untouched. `range(3, n)` (non-zero const) and `range(0, n, 1)` (3-arg) are
    left intact. The `not isinstance(..., bool)` guard is load-bearing: Python
    `False == 0` is True, so without it `range(False, n)` would be wrongly
    collapsed.
    """

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 2
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 0
            and not isinstance(node.args[0].value, bool)
        ):
            node.args = [node.args[1]]
        return node


def _normalize_ast_dump(source: str) -> str | None:
    """Parse source and return a normalized ast.dump (no attributes), or None.

    On 3.9+ ast.dump accepts include_attributes; we always pass it False so
    line/col info is dropped. Never raises -- returns None on syntax error.

    `range(0, X)` is normalized to `range(X)` first (E4-T0 false-negative fix),
    UNIFORMLY on expected and candidate, so the idiomatic `range(len(items))`
    fix reaches the 1.0 AST-equivalent route directly.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    tree = _RangeZeroNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.dump(tree, include_attributes=False)
    except TypeError:
        return ast.dump(tree)


def _token_edit_ratio(a: str, b: str) -> float:
    """Similarity in [0,1] over Python token-ish words via Levenshtein on tokens."""
    ta = re.findall(r"\w+|[^\w\s]", a)
    tb = re.findall(r"\w+|[^\w\s]", b)
    if not ta and not tb:
        return 1.0
    la, lb = len(ta), len(tb)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = ta[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == tb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[lb]
    denom = max(la, lb)
    if denom == 0:
        return 1.0
    return max(0.0, 1.0 - dist / denom)


def _score_multi_key(expected, response_text: str) -> tuple[float, str, str]:
    """E1: expected is an ordered list of values; credit per value found in text."""
    if isinstance(expected, str):
        try:
            parsed = json.loads(expected)
            if isinstance(parsed, list):
                expected = parsed
        except (ValueError, TypeError):
            pass
    if not isinstance(expected, (list, tuple)) or not expected:
        return 0.0, "", "empty or non-list expected"
    n = len(expected)
    if not response_text or not response_text.strip():
        return 0.0, "", "empty response"
    hits = []
    for v in expected:
        if v is None:
            continue
        if _contains_normalized(str(v), response_text):
            hits.append(str(v))
    found = len(hits)
    score = round(found / n, 4)
    extracted = ", ".join(hits) if hits else ""
    reason = f"{found}/{n} keys matched"
    return score, extracted, reason


def _score_format(expected: dict, response_text: str) -> tuple[float, str, str]:
    """E2: extract first JSON object; credit per key with matching presence/type/value."""
    if not isinstance(expected, dict) or not expected:
        return 0.0, "", "empty expected dict"
    if not response_text or not response_text.strip():
        return 0.0, "", "empty response"
    obj = _find_first_json_object(response_text)
    if obj is None:
        return 0.0, "", "no valid JSON object in response"
    keys = list(expected.keys())
    n = len(keys)
    correct = 0
    bad = []
    for k in keys:
        if k in obj and _typed_value_equal(expected[k], obj[k]):
            correct += 1
        else:
            bad.append(k)
    score = round(correct / n, 4)
    extracted = json.dumps(obj, sort_keys=True)
    reason = f"{correct}/{n} keys correct"
    if bad:
        reason += f" (wrong/missing: {', '.join(bad)})"
    return score, extracted, reason


def _score_function_call(expected, response_text: str) -> tuple[float, str, str]:
    """E3: expected dict with 'name' and 'arguments'; parse tool-call JSON and match.

    Score: name must match for any credit beyond arguments fraction; final score is
    average of (name correct) and (fraction of argument fields correct), each in
    [0,1], so a perfect call -> 1.0, perfect args + wrong name -> 0.5 of arg weight.
    To keep the contract simple and per task spec (1.0 only if name AND arguments
    match, partial by fraction of correct argument fields), name acts as a gate:
    if name mismatches, score is the argument fraction scaled by 0.0 for the name
    component -> we report fraction of correct arg fields but cap pass via threshold.
    """
    if not isinstance(expected, dict):
        return 0.0, "", "expected must be a dict"
    exp_name = expected.get("name")
    exp_args = expected.get("arguments")
    if not isinstance(exp_args, dict):
        exp_args = {}
    if not response_text or not response_text.strip():
        return 0.0, "", "empty response"
    obj = _find_first_json_object(response_text)
    if obj is None:
        return 0.0, "", "no valid JSON tool call in response"
    got_name = obj.get("name")
    got_args = obj.get("arguments")
    if not isinstance(got_args, dict):
        got_args = {}
    name_ok = (exp_name is None) or (got_name == exp_name)
    arg_keys = list(exp_args.keys())
    if arg_keys:
        correct = sum(
            1 for k in arg_keys if k in got_args and _typed_value_equal(exp_args[k], got_args[k])
        )
        arg_frac = correct / len(arg_keys)
    else:
        arg_frac = 1.0
        correct = 0
    if name_ok:
        score = round(arg_frac, 4)
        reason = f"name ok; {correct}/{len(arg_keys)} args correct"
    else:
        # Name gate failed: cap below pass threshold, still reflect arg progress.
        score = round(min(arg_frac, 0.49) * 0.5, 4)
        reason = (
            f"name mismatch ({got_name!r} vs {exp_name!r}); "
            f"{correct}/{len(arg_keys)} args correct"
        )
    extracted = json.dumps({"name": got_name, "arguments": got_args}, sort_keys=True)
    return score, extracted, reason


# E4 false-positive guard: a non-AST-equal answer that flips an OPERATOR,
# COMPARATOR, or CONSTANT did NOT fix the bug, so it can never reach the 0.5
# pass threshold. Cap such answers here (well below 0.5); only AST-equivalence
# earns a pass. Cosmetic-only diffs (formatting / names / docstrings) fall to a
# capped token ratio in [_E4_COSMETIC_FLOOR, _E4_SEMANTIC_CAP].
_E4_SEMANTIC_CAP = 0.30   # hard ceiling for any operator/comparator/constant flip
_E4_COSMETIC_FLOOR = 0.30  # floor for cosmetic-only diffs (keeps them in (0,0.5))


def _ast_semantic_signature(source: str):
    """Multiset signature of the bug-bearing nodes: comparators, binary/unary/bool
    operators, and literal constants. Returns a sorted tuple, or None on parse fail.

    Deliberately EXCLUDES Name ids and arg names so a cosmetic rename (a/b -> x/y)
    is NOT counted as a semantic flip; those are caught only by the token ratio and
    stay sub-pass anyway. The operator/comparator/constant multiset is what makes
    `>` vs `>=` (Gt != GtE) and `x - 7` vs `x + 7` (Sub != Add) detectably different.

    TUNED TO THE CURRENT E4 TEMPLATE SET (prompt_generator._E4_TEMPLATES): all
    reference constants are the integer {n} or 0/1, never a string. Because every
    ast.Constant (incl. strings) goes into the multiset via repr(node.value), a
    CORRECT fix that ALSO adds a docstring / string literal would differ from the
    reference const-multiset and be capped at the FAIL ceiling. That is INERT here
    (no E4 reference answer contains a string constant, so it cannot misfire in
    this template set) but is a latent over-strictness: if E4 templates ever emit
    string constants, exclude string Constants from the multiset (or compare
    strings only when the reference has one).

    DEFENSE-IN-DEPTH: the tree is first run through
    `_RangeZeroNormalizer` so a `range(0, X)` -> `range(X)` collapse removes the
    redundant literal `0` from the constant multiset BEFORE the walk. Even when
    the answer is NOT fully AST-equal (e.g. it also renames a var), the constant
    `0` from a normalized `range` no longer pollutes the signature, so the
    multiset reflects only genuine operator/comparator/constant flips.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    tree = _RangeZeroNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    ops: list[str] = []
    consts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            ops.extend("cmp:" + type(op).__name__ for op in node.ops)
        elif isinstance(node, ast.BinOp):
            ops.append("bin:" + type(node.op).__name__)
        elif isinstance(node, ast.UnaryOp):
            ops.append("un:" + type(node.op).__name__)
        elif isinstance(node, ast.BoolOp):
            ops.append("bool:" + type(node.op).__name__)
        elif isinstance(node, ast.AugAssign):
            ops.append("aug:" + type(node.op).__name__)
        elif isinstance(node, ast.Constant):
            # repr keeps type+value distinct (7 != 7.0 != '7' != True).
            consts.append("const:" + repr(node.value))
    return tuple(sorted(ops)) + tuple(sorted(consts))


def _score_code_edit(expected: str, response_text: str) -> tuple[float, str, str]:
    """E4: AST-equivalence is the ONLY route to a pass.

    Routing (false-positive fix):
      - exp & cand both parse and AST dumps equal           -> 1.0 (fix verified)
      - both parse, ASTs differ, operator/comparator/constant
        multiset differs (the bug was NOT fixed)            -> <=0.30 FAIL
      - both parse, ASTs differ, only cosmetic differences  -> capped token ratio
                                                               in [0.30,0.49] FAIL
      - cand does not parse (even after fence-stripping)     -> 0.0
    Guarantees a surviving `>` instead of `>=`, or `x-7` for `x+7`, can never reach
    0.5; only a genuine fix (AST-equivalent to the fixed reference) passes E4.
    """
    exp_src = expected if isinstance(expected, str) else str(expected)
    if not exp_src.strip():
        return 0.0, "", "empty expected"
    if not response_text or not response_text.strip():
        return 0.0, "", "empty response"
    cand = _extract_code_block(response_text)
    if not cand.strip():
        return 0.0, "", "no code in response"
    cand_dump = _normalize_ast_dump(cand)
    if cand_dump is None:
        return 0.0, cand.strip(), "candidate did not parse"
    exp_dump = _normalize_ast_dump(exp_src)
    if exp_dump is not None and cand_dump == exp_dump:
        return 1.0, cand.strip(), "AST-equivalent"
    # Not AST-equivalent: an E4 answer that is not AST-equal did NOT verify the fix.
    ratio = _token_edit_ratio(exp_src, cand)
    exp_sig = _ast_semantic_signature(exp_src)
    cand_sig = _ast_semantic_signature(cand)
    if exp_sig is not None and cand_sig is not None and exp_sig != cand_sig:
        score = round(min(ratio, _E4_SEMANTIC_CAP), 4)
        reason = (
            "operator/constant mismatch (bug not fixed); "
            f"capped at {score} (token similarity {round(ratio, 4)})"
        )
        return score, cand.strip(), reason
    # Cosmetic-only difference (formatting / names / docstrings): still sub-pass,
    # since a non-AST-equal E4 answer is by definition not a verified fix.
    score = round(min(ratio, _E4_COSMETIC_FLOOR + 0.19), 4)  # cap at 0.49
    reason = f"AST mismatch (cosmetic); token similarity {round(ratio, 4)}, capped {score}"
    return score, cand.strip(), reason


_REASONING_TAG_RE = re.compile(r"</?(?:think|thinking|reason|reasoning)\b", re.IGNORECASE)


def _has_reasoning_markers(text: str) -> bool:
    """Detect chain-of-thought tags (<think>, </reasoning>, ...) in text.

    Detection ONLY -- never mutates. Flags a response whose reasoning leaked into
    the scored content, signalling the server reasoning-separation contract was
    violated for that trial. We deliberately do not strip: a bare </think> is
    ambiguous (a legitimate answer may contain the literal string), so we surface
    the violation instead of silently truncating the answer; exists to catch a
    future server/model/config regression.
    """
    if not text:
        return False
    return bool(_REASONING_TAG_RE.search(text))


def _max_ngram_repeat_ratio(text: str, n: int = 3) -> float:
    """Repeated-gram MASS in [0,1]: fraction of n-gram slots that are REPEATS.

    repeated_mass = sum(count - 1 for each n-gram with count > 1) / total_grams.
    0.0 if too short. Detects runaway repetition (the high-fill collapse mode)
    without an LLM. Unlike a single-most-frequent-gram ratio (which tops out near
    0.34 for a 3-distinct-gram cycle and so MISSES multi-token loops), this rises
    toward 1.0 for any tight cycle: e.g. "a b c " * 40 -> 115/118 ~= 0.975, while
    legit non-repeating code -> ~0.0.
    """
    if not text:
        return 0.0
    toks = re.findall(r"\S+", text)
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    counts: dict[tuple, int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


# Canonical failure-mode labels for the quality-collapse classifier.
# Pure-function, no LLM-judge. Mirrors the reasoning_leak detector's discipline.
FAILURE_MODES = (
    "ok", "empty", "reasoning_leak", "runaway_repetition", "no_valid_json",
    "parse_fail", "http_timeout", "oom", "loop", "premature_eos",
    "instruction_fade", "error_other",
)


def classify_failure_mode(
    family: str,
    score: float,
    response_text: str,
    reason: str = "",
    error: object = None,
    completion_tokens: object = None,
    canary: object = None,
) -> str:
    """Map a scored trial onto ONE canonical failure-mode label.

    Precedence: hard transport errors (oom/http_timeout/error_other) ->
    empty/premature_eos -> reasoning_leak -> runaway_repetition ->
    scorer-reason (no_valid_json / parse_fail) -> instruction_fade -> ok.
    Never raises; unknown inputs fall back to 'ok' (scored fine) or 'error_other'.
    NOTE: 'http_timeout' is a THROUGHPUT signal, NOT a quality collapse; a
    downstream analyzer should keep it out of the quality-collapse mode set.
    """
    err = "" if error is None else str(error)
    low_err = err.lower()
    if err:
        # 'oom' as a bare substring also matches boom/zoom/room/doom/etc., so a
        # real OOM could be misclassified AND (worse) a non-OOM "boom" string
        # could be tagged 'oom' and silently dropped from the collapse/throughput
        # buckets. Use a token/phrase match instead: the literal phrases
        # "out of memory"/"cuda error", the WORD-boundaried token \boom\b, or the
        # original-case acronym "OOM".
        if ("out of memory" in low_err or "cuda error" in low_err
                or re.search(r"\boom\b", low_err) or "OOM" in err):
            return "oom"
        if "timeout" in low_err or "timed out" in low_err:
            return "http_timeout"
        return "error_other"
    text = response_text or ""
    if not text.strip():
        try:
            if completion_tokens is not None and int(completion_tokens) == 0:
                return "premature_eos"
        except (TypeError, ValueError):
            pass
        return "empty"
    if _has_reasoning_markers(text):
        return "reasoning_leak"
    if _max_ngram_repeat_ratio(text) >= 0.5:
        return "runaway_repetition"
    low_reason = (reason or "").lower()
    if "no valid json" in low_reason:
        return "no_valid_json"
    if "did not parse" in low_reason:
        return "parse_fail"
    if canary is not None and str(canary) and str(canary) not in text:
        return "instruction_fade"
    return "ok"


# ---------------------------------------------------------------------------
# F0 RETRY/EMPTY: empty / premature-EOS completions are FIRST-CLASS.
#
# An empty or instant-EOS completion is a 200-OK-but-blank response (the starved /
# SWA-dead-zone failure the placement strategies exist to dodge). It MUST be
# distinguishable from a genuine scored 'ok' so the driver can RETRY-on-empty with
# a DIFFERENT seeded position. The two canonical empty-class labels are 'empty'
# (blank content) and 'premature_eos' (blank content with completion_tokens == 0).
# An empty must NEVER silently look like a pass, and classify_failure_mode already
# routes blank content to one of these (never to 'ok'); the helpers below expose
# that signal as a structured, additive contract for the driver.
# ---------------------------------------------------------------------------

# The empty-class failure modes (subset of FAILURE_MODES). Both mean "200 OK but no
# usable content", the RETRY-on-empty trigger.
EMPTY_FAILURE_MODES: frozenset = frozenset({"empty", "premature_eos"})

# The driver should RETRY (with a different seeded position) on exactly these modes.
# Kept identical to EMPTY_FAILURE_MODES today, but named separately so the retry
# policy can be widened later WITHOUT changing what "empty" means. (Leniency note:
# this set only ever GROWS the retry trigger; it never reclassifies an empty as ok.)
RETRYABLE_EMPTY_MODES: frozenset = EMPTY_FAILURE_MODES


def is_empty_failure_mode(mode: object) -> bool:
    """True iff `mode` is an empty-class failure mode ('empty' or 'premature_eos').

    Pure, never raises, accepts any object (non-str -> False). The driver calls this
    on a result's `failure_mode` to decide whether to RETRY-on-empty.
    """
    return mode in EMPTY_FAILURE_MODES


def classify_failure_detail(
    family: str,
    score: float,
    response_text: str,
    reason: str = "",
    error: object = None,
    completion_tokens: object = None,
    canary: object = None,
) -> dict:
    """Structured companion to classify_failure_mode for the RETRY-on-empty driver.

    ADDITIVE: classify_failure_mode is UNCHANGED (still the one-shot string mapper
    every existing caller uses). This returns a dict so the driver gets enough
    signal to retry an empty/instant-EOS completion with a different seeded position,
    without re-deriving the classification.

    Returns (keys are STABLE; only ever ADDED to):
      {
        "failure_mode":   <one canonical FAILURE_MODES label> (== classify_failure_mode),
        "is_empty":       bool, True iff failure_mode in EMPTY_FAILURE_MODES,
        "retryable_empty": bool, True iff the driver SHOULD retry-on-empty
                                   (failure_mode in RETRYABLE_EMPTY_MODES). An empty
                                   completion is ALWAYS retryable_empty; a scored
                                   'ok'/'wrong-answer'/'oom'/'timeout' is NOT.
        "is_ok":          bool, True iff failure_mode == "ok" (passed cleanly). An
                                   empty is NEVER is_ok (the never-silently-a-pass
                                   invariant, surfaced explicitly).
        "completion_tokens": int|None, echoed (0 distinguishes premature_eos from a
                                   non-zero-token blank 'empty'); None when unknown.
      }

    Pure + deterministic; never raises (mirrors classify_failure_mode).
    """
    mode = classify_failure_mode(
        family, score, response_text, reason=reason, error=error,
        completion_tokens=completion_tokens, canary=canary,
    )
    ct: object = completion_tokens
    try:
        ct = None if completion_tokens is None else int(completion_tokens)
    except (TypeError, ValueError):
        ct = None
    is_empty = mode in EMPTY_FAILURE_MODES
    return {
        "failure_mode": mode,
        "is_empty": is_empty,
        "retryable_empty": mode in RETRYABLE_EMPTY_MODES,
        # Defence-in-depth on the never-silently-a-pass invariant: an empty is_ok is
        # impossible by construction (classify_failure_mode returns ok only on
        # non-blank content), but we assert it explicitly so a future regression that
        # ever returned ok for blank content would flip is_ok False here too.
        "is_ok": (mode == "ok") and not is_empty,
        "completion_tokens": ct,
    }


def score_response(
    family: str,
    expected_answer: str | dict,
    response_text: str,
) -> dict:
    """Score a single LLM response against an expected answer.

    Returns dict with: family, score, passed, extracted_answer, expected_answer, reason.
    Raises ValueError for unknown family.
    """
    if family not in _KNOWN_FAMILIES:
        raise ValueError(
            f"Unknown family {family!r}; expected one of {sorted(_KNOWN_FAMILIES)}"
        )
    if response_text is None:
        response_text = ""
    if family in _SUMMARY_FAMILIES:
        if not isinstance(expected_answer, dict):
            raise ValueError(f"family {family} requires dict expected_answer")
        score, extracted, reason = _score_summary(expected_answer, response_text)
    elif family in _MULTI_KEY_FAMILIES:
        score, extracted, reason = _score_multi_key(expected_answer, response_text)
    elif family in _FORMAT_FAMILIES:
        if not isinstance(expected_answer, dict):
            raise ValueError(f"family {family} requires dict expected_answer")
        score, extracted, reason = _score_format(expected_answer, response_text)
    elif family in _FUNCTION_CALL_FAMILIES:
        if not isinstance(expected_answer, dict):
            raise ValueError(f"family {family} requires dict expected_answer")
        score, extracted, reason = _score_function_call(expected_answer, response_text)
    elif family in _CODE_EDIT_FAMILIES:
        if not isinstance(expected_answer, str):
            expected_answer = str(expected_answer)
        score, extracted, reason = _score_code_edit(expected_answer, response_text)
    elif family in _NUMERIC_FAMILIES:
        if not isinstance(expected_answer, str):
            expected_answer = str(expected_answer)
        score, extracted, reason = _score_numeric(expected_answer, response_text)
    else:
        if not isinstance(expected_answer, str):
            expected_answer = str(expected_answer)
        score, extracted, reason = _score_exact(expected_answer, response_text)
    passed = score >= 0.5
    leak = _has_reasoning_markers(response_text)
    if leak:
        reason = reason + (
            " | WARN_REASONING_LEAK: chain-of-thought markers in scored content "
            "(contract violation; treat score as unreliable)"
        )
    return {
        "family": family,
        "score": float(score),
        "passed": bool(passed),
        "extracted_answer": extracted,
        "expected_answer": expected_answer,
        "reason": reason,
        "reasoning_leak": leak,
    }


def wilson_lower_ci(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound at the chosen confidence (default 95%).

    Returns 0.0 if n == 0.
    """
    if n <= 0:
        return 0.0
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} must be in [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    val = (center - margin) / denom
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def wilson_upper_ci(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score upper bound (symmetric to lower)."""
    if n <= 0:
        return 0.0
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} must be in [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    val = (center + margin) / denom
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def aggregate_trials(scores: list[float]) -> dict:
    """Aggregate per-trial scores. Empty list returns zeros (no exception)."""
    if not scores:
        return {
            "n": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "pass_rate": 0.0,
            "wilson_95_lower": 0.0,
            "wilson_95_upper": 0.0,
        }
    n = len(scores)
    mean = sum(scores) / n
    passed = sum(1 for s in scores if s >= 0.5)
    pass_rate = passed / n
    return {
        "n": n,
        "mean": mean,
        "min": min(scores),
        "max": max(scores),
        "pass_rate": pass_rate,
        "wilson_95_lower": wilson_lower_ci(passed, n),
        "wilson_95_upper": wilson_upper_ci(passed, n),
    }

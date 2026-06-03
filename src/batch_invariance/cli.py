#!/usr/bin/env python3
"""cli.py -- the ``batch-invariance`` command-line entry point (SUBCOMMAND dispatcher).

``batch-invariance <subcommand> ...`` exposes four subcommands, each a thin front-end
over the package's already-tested logic (the gate/diff/cert LOGIC is NOT re-implemented
here -- this module only wires parsers and calls the existing functions):

  * ``run-mock``    -- run the full A/B/C gate against the IN-PROCESS zero-GPU mock
                       (no model, no GPU). Default writes a GREEN, ``source=mock``,
                       NON-promotable cert; ``--batch-divergence`` drives the SAME gate
                       RED. The reusable mock-driving core (:func:`run_mock_gate`) is
                       first-class in the package; ``examples/run_gate.py`` calls it too.
  * ``run-live``    -- the LIVE gate against YOUR OpenAI-compatible server. Supports both
                       the explicit ``--server-bin/--model/...`` form and the profile form
                       (``--profile <name> --model-alias <alias>``, which loads
                       ``profiles/<name>/models.toml`` for the alias->gguf+flags mapping and
                       the profile's scorer + work-set). All real work + the ``EXIT_*``
                       contract live in ``live_invariance.run_main``.
  * ``verify-cert`` -- load a cert JSON and print its status + whether
                       ``invariance_diff.is_promotable`` (and why / why-not). Exit 0 when
                       the cert parses (promotable or not), non-zero on a malformed cert.
  * ``plan-n``      -- print the per-N footprint / KV plan + memory projection (the same
                       planning math ``run-live --dry-run`` prints), launching NOTHING.

The console script ``batch-invariance`` (declared in ``pyproject.toml``) maps to
:func:`main`, which now dispatches subcommands. ``main(argv=None) -> int`` is preserved as
the entry; a bare ``batch-invariance`` with no subcommand prints help and exits non-zero.

Stdlib-only at runtime (argparse / json / os / pathlib + the local package).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import invariance_diff as idiff
from . import kv_budget
from .live_invariance import (
    ACK_FLAG,
    DEFAULT_ARM_DEADLINE_S,
    DEFAULT_BASELINE_PORT,
    DEFAULT_EXPERIMENT_PORT,
    DEFAULT_GATE_PASSES,
    DEFAULT_MODEL_ID,
    DEFAULT_READY_TIMEOUT_S,
    DEFAULT_REQ_TIMEOUT_S,
    EXIT_ERROR,
    MAX_FOOTPRINT_GB,
    run_main,
)

# Exit codes specific to the non-run subcommands (run-live re-uses live_invariance's
# EXIT_* contract via run_main). Kept small + explicit so a caller can branch on them.
EXIT_OK = 0
EXIT_USAGE = 2            # argparse-style usage error (bare cmd / bad cert path)


# ===========================================================================
# run-mock -- the reusable, first-class mock-driving gate core.
# ===========================================================================
# A fake Popen handle: the driver only ever reads ``.pid`` and calls ``.wait`` -- it
# never really spawns when ``arm_base_url`` is injected (the test seam). Lifted here
# (out of examples/conftest) so ``run-mock`` is first-class in the package and the
# example + tests can all reuse ONE implementation.
class _FakeProc:
    """Stand-in for ``subprocess.Popen`` exposing only what the driver touches."""

    pid = 4242

    def wait(self, timeout=None):  # noqa: ANN001 - mirrors Popen.wait
        return 0


def _fake_popen(*_args, **_kwargs):
    return _FakeProc()


# A built-in RED-driver work-set: each row's ``expected_answer`` EQUALS its prompt. The
# mock's score-divergence knob echoes the user prompt back on a CO-BATCHED request, so the
# echoed (concurrent) completion scores 1.0 while the serial arm's canned "x x x" scores
# 0.0 -> a GENUINE score/passed/failure_mode divergence the gate catches as RED (not a
# token-only AMBER). With the knob OFF this same work-set is honestly score-clean.
_MOCK_RED_ROWS = [
    {"item": "the hidden needle is four two four two",
     "expected_answer": "the hidden needle is four two four two",
     "family": "retrieval", "fill": 0.1},
    {"item": "secret marker alpha bravo charlie",
     "expected_answer": "secret marker alpha bravo charlie",
     "family": "retrieval", "fill": 0.2},
    {"item": "remember the phrase delta echo foxtrot",
     "expected_answer": "remember the phrase delta echo foxtrot",
     "family": "retrieval", "fill": 0.3},
]


def run_mock_gate(
    *,
    batch_divergence: bool = False,
    out_dir: str | None = None,
    ctx: int = 2048,
    parallel: int = 4,
    reps: int = 2,
    gate_passes: int = 6,
    n_predict: int = 8,
    serve_sleep: float = 0.03,
    model_alias: str = "demo-model",
) -> dict:
    """Drive the REAL A/B/C gate against the in-process mock; return the cert dict.

    No GPU, no model, no real ``llama-server``: the in-process mock serves over loopback
    and the driver's documented TEST SEAM (``arm_base_url`` aims the arms at the mock;
    ``popen``/``kill``/``mem_reader``/``wait_ready``/``is_port_free`` are no-op fakes) runs
    the genuine ``run_arm`` -> ``score_one`` -> ``invariance_diff`` path with NO real
    launch. This is the SAME path the conftest harness + the README ``run-mock`` command
    exercise, lifted here so it is first-class in the package.

    * ``batch_divergence=False`` (default): the honest mock (content independent of batch
      composition). With ``cert_source='mock'`` the run reaches the literal ``green`` status
      but is NON-promotable BY CONSTRUCTION (a real GREEN must be earned live).
    * ``batch_divergence=True``: arms the mock's score-divergence knob against the built-in
      RED work-set so a co-batched completion scores differently from serial -> the gate
      goes RED (``status='failed'``), non-promotable.

    Imports the driver + mock lazily so a bare ``import batch_invariance.cli`` does not pull
    the subprocess/threading machinery until a mock run is actually requested.
    """
    import tempfile
    import threading

    from . import mock_llama_server as mock
    from .live_invariance import LiveInvarianceDriver

    out_dir = out_dir or tempfile.mkdtemp(prefix="bi-mock-")
    os.makedirs(out_dir, exist_ok=True)

    # The RED knob is the GENUINE score-flip (score_divergence); a clean run leaves every
    # knob off. The built-in RED work-set is honestly clean when the knob is off, so the
    # SAME work-set serves both modes (its expected==prompt only matters once the mock
    # echoes the prompt on a co-batched request).
    knobs = {"score_divergence": True} if batch_divergence else {}
    cert_source = "live" if batch_divergence else "mock"

    # Build the work-set as a JSON file the loader consumes (the driver re-derives seeds
    # and -- live -- token counts; here /tokenize hits the mock).
    ws_dir = tempfile.mkdtemp(prefix="bi-mock-ws-")
    ws_path = os.path.join(ws_dir, "workset.json")
    Path(ws_path).write_text(json.dumps(_MOCK_RED_ROWS), encoding="utf-8")

    httpd = mock.serve(0, slots=max(2, int(parallel)), serve_sleep=float(serve_sleep),
                       ready_log=False, **knobs)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        drv = LiveInvarianceDriver(
            server_bin="/nonexistent/llama-server",
            model_path="model.gguf",
            model_alias=str(model_alias),
            port=18888,
            ctx=int(ctx),
            parallel=int(parallel),
            n_predict=int(n_predict),
            out_dir=str(out_dir),
            cert_source=cert_source,
            reps=int(reps),
            gate_passes=int(gate_passes),
            workset_path=ws_path,
            popen=_fake_popen,
            kill=lambda _pid, _sig: None,
            mem_reader=lambda: None,                 # off-Linux / unreadable -> permissive
            wait_ready=lambda _port, _timeout, host="127.0.0.1": 0,
            is_port_free=lambda _port, host="127.0.0.1": True,
            grace_s=0.0,                             # fake pid -> no real reap -> no grace
            arm_base_url=base_url,
            gguf_gb=1.0,                             # makes the footprint guard inert
        )
        return drv.run()
    finally:
        try:
            httpd.stop_hangs()
        except Exception:
            pass
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            t.join(timeout=5)
        except Exception:
            pass


def _cmd_run_mock(args: argparse.Namespace) -> int:
    """``run-mock``: drive the gate against the mock, persist the cert, print the verdict.

    Default -> GREEN (source=mock, non-promotable). ``--batch-divergence`` -> RED. Returns
    0 when the produced verdict MATCHES the documented contract for the requested mode
    (clean->green / divergence->failed), non-zero otherwise, so CI can gate on the code."""
    out_dir = args.out_dir or None
    cert = run_mock_gate(
        batch_divergence=bool(args.batch_divergence),
        out_dir=out_dir,
        ctx=int(args.ctx),
        parallel=int(args.parallel),
    )
    # Persist the cert as an inspectable artifact under <out-dir>/dispatch-cert/ (when an
    # explicit --out-dir was given; otherwise run_mock_gate already used a temp dir).
    path = None
    if out_dir:
        cert_dir = os.path.join(out_dir, "dispatch-cert")
        path = idiff.persist_cert(cert_dir, cert)

    status = cert.get("status")
    source = cert.get("source")
    promotable = idiff.is_promotable(cert)
    ac = (cert.get("divergence_report") or {}).get("AC") or {}
    mode = ("--batch-divergence (RED expected)" if args.batch_divergence
            else "default (GREEN expected)")
    print("=" * 72)
    print("batch-invariance run-mock (in-process mock; NO GPU, NO model)")
    print(f"  mode               = {mode}")
    print(f"  status             = {status}")
    print(f"  source             = {source}")
    print(f"  AC.n_divergent     = {ac.get('n_divergent')}")
    print(f"  overlap_ok         = {(cert.get('overlap') or {}).get('overlap_ok')}")
    print(f"  is_promotable      = {promotable}  (mock certs are NON-promotable by construction)")
    if path:
        print(f"  cert               -> {path}")

    if args.batch_divergence:
        ok = (status == idiff.STATUS_FAILED) and not promotable
        print("  VERDICT: RED -- the gate detected a co-batching score divergence."
              if ok else f"  UNEXPECTED: expected RED/failed, got status={status}.")
    else:
        ok = (status == idiff.STATUS_GREEN) and (source == "mock") and not promotable
        print("  VERDICT: GREEN (source=mock, non-promotable) -- the gate reached green on the "
              "honest mock; a real GREEN must be earned live."
              if ok else f"  UNEXPECTED: expected green/mock/non-promotable, got "
                         f"status={status} source={source} promotable={promotable}.")
    return EXIT_OK if ok else EXIT_ERROR


# ===========================================================================
# run-live -- the existing flat behavior, now under a subcommand (+ profile form).
# ===========================================================================
def _add_run_live_args(p: argparse.ArgumentParser) -> None:
    """Attach every run-live flag (kept from the original flat parser, + profile form)."""
    # ---- inputs: EITHER the explicit form OR the profile form. Both are accepted; the
    # profile resolver (in _resolve_live_inputs) fills server-bin/model/flags/scorer/workset
    # from profiles/<name>/models.toml when --profile is given, so the explicit flags are
    # NOT required when a profile + alias are supplied. ----
    p.add_argument("--server-bin", default=None,
                   help="path to the server binary (e.g. llama-server). Required UNLESS "
                        "--profile is given.")
    p.add_argument("--model", default=None,
                   help="path to the model gguf. Required UNLESS --profile + --model-alias "
                        "resolve it from the profile registry.")
    p.add_argument("--out-dir", required=True, help="cert + logs land here")
    p.add_argument("--ctx", type=int, required=True,
                   help="per-slot context size (e.g. 8192 or 32768)")
    p.add_argument("--workset", default=None,
                   help="path to the work-set JSON (a top-level list of row templates or a "
                        "bare list of prompt strings). Required UNLESS --profile supplies a "
                        "work-set.")
    p.add_argument("--model-alias", default=DEFAULT_MODEL_ID,
                   help="stable model id stamped into the cert + seeds. With --profile this "
                        "ALSO selects which [[models]] entry (alias->gguf+flags) to use.")

    # ---- profile form ----
    p.add_argument("--profile", default=None,
                   help="load profiles/<name>/models.toml (the alias->gguf+server-flags "
                        "registry) and the profile's scorer + work-set. Use with "
                        "--model-alias to pick the model entry. A built-in profile name "
                        "(e.g. 'spark') resolves to the packaged profile dir.")
    p.add_argument("--models-dir", default=None,
                   help="substitute for the {MODELS_DIR} placeholder in a profile's gguf "
                        "paths (default: the MODELS_DIR env var, else left as-is).")
    p.add_argument("--profile-corner", default="balanced",
                   help="for a profile that synthesizes its work-set, which sampling corner "
                        "to materialize (default 'balanced').")

    # ---- ports ----
    # The experiment port defaults to baseline+1 (NOT the bare DEFAULT_EXPERIMENT_PORT,
    # which equals DEFAULT_BASELINE_PORT) so the documented `run-live ...` command -- which
    # does not specify ports -- never self-collides with the baseline guard. Both remain
    # overridable; choose your own free experiment port on a real host.
    _default_experiment_port = DEFAULT_EXPERIMENT_PORT + 1
    p.add_argument("--experiment-port", type=int, default=_default_experiment_port,
                   help=f"the port the experiment server binds AND arms dispatch to "
                        f"(default {_default_experiment_port}). HARD-refused if it equals the "
                        f"baseline or any --untouchable-port.")
    p.add_argument("--baseline-port", type=int, default=DEFAULT_BASELINE_PORT,
                   help=f"a server you already have running that must NEVER be touched "
                        f"(default {DEFAULT_BASELINE_PORT}). The experiment port must differ "
                        f"from it.")
    p.add_argument("--untouchable-port", type=int, action="append", default=None,
                   help="an extra port to add to the untouchable set (repeatable).")
    p.add_argument("--untouchable-pid", type=int, action="append", default=None,
                   help="a PID the teardown must never signal (repeatable). Defaults to none.")
    p.add_argument("--server-flag", action="append", default=None,
                   help="override a single server launch flag (repeatable; replaces the "
                        "default flag set entirely when any are given).")

    # ---- gate knobs ----
    p.add_argument("--parallel", type=int, default=4, help="N slots for arms B/C")
    p.add_argument("--n-predict", type=int, default=128, help="max_tokens per request")
    p.add_argument("--reps", type=int, default=3,
                   help="R seeds per prompt (the work-set's R dimension). NOTE: T (the number "
                        "of ARM_C concurrent re-passes) is decoupled via --gate-passes; --reps "
                        "only sizes the work-set's seed dimension.")
    p.add_argument("--gate-passes", type=int, default=DEFAULT_GATE_PASSES,
                   help=f"T = number of ARM_C concurrent re-passes the AC gate unions over "
                        f"(detection power; default {DEFAULT_GATE_PASSES}). DECOUPLED from "
                        f"--reps: more passes sample more co-batch interleavings so a STOCHASTIC "
                        f"divergence is more likely to surface (any pass divergent => RED). "
                        f"Raising T can only make the verdict MORE conservative.")
    p.add_argument("--scorer", default=None,
                   help="pluggable scorer as 'pkg.mod:fn' (resolved via importlib). The "
                        "callable is (response_text, expected_answer, *, item=None) -> "
                        "(score, passed, failure_mode). Defaults to a builtin exact-match "
                        "scorer (or the profile's scorer when --profile is given).")
    p.add_argument("--invariant-fields", default=None,
                   help="comma-separated list of result fields that MUST match across arms "
                        "(default: the generic 5-tuple score,passed,expected_answer,"
                        "prompt_tokens_measured,failure_mode). A consumer that surfaces extra "
                        "score-bearing fields can extend this.")
    p.add_argument("--logit-drift-eps", type=float, default=0.0,
                   help="forensic logprob-drift tolerance forwarded to the diff (default 0.0). "
                        "Consumed by invariance_diff; the driver only plumbs it. Cannot make "
                        "the verdict more lenient.")
    p.add_argument("--ctx-sweep", action="store_true",
                   help="enable the diff's ctx-vs-divergence sweep instrumentation (default "
                        "off). Consumed by invariance_diff; the driver only plumbs the flag.")
    p.add_argument("--kv", default="q8_0", help="KV cache quant label (for the footprint math)")
    p.add_argument("--seed", type=int, default=1234, help="seed_base for the work-set")
    p.add_argument("--n-probs", type=int, default=0, help="optional logit smoke (forensic)")

    # ---- footprint / safety ----
    p.add_argument("--max-footprint-gb", type=float, default=MAX_FOOTPRINT_GB,
                   help=f"HARD experiment-footprint cap in GB (default {MAX_FOOTPRINT_GB}); "
                        f"precheck S0 REFUSES a launch whose model+KV projection exceeds it. "
                        f"Raise ONLY for an intentional larger run.")
    p.add_argument("--gguf-gb", type=float, default=None,
                   help="override the on-disk model-size probe (GB), e.g. when the gguf is "
                        "remote/absent on this machine. Used only by the footprint guard.")
    p.add_argument("--req-timeout", type=int, default=DEFAULT_REQ_TIMEOUT_S,
                   help="per-request HTTP timeout (seconds)")
    p.add_argument("--ready-timeout", type=int, default=DEFAULT_READY_TIMEOUT_S,
                   help="how long to wait for the server to become ready (seconds)")
    p.add_argument("--arm-deadline", type=float, default=DEFAULT_ARM_DEADLINE_S,
                   help="whole-arm wall-clock backstop (seconds)")
    p.add_argument("--cert-source", default="mock",
                   help="MUST be 'live' to write a promotable cert (a 'mock' cert never "
                        "promotes by construction)")

    # ---- placement (F0) ----
    p.add_argument("--position-strategy", choices=("fixed", "jitter", "adaptive"),
                   default="fixed",
                   help="needle placement: 'fixed' = the per-row fractional position "
                        "(DEFAULT); 'jitter' = seeded micro-jitter around it; 'adaptive' = "
                        "when SWA is detected (0 < n_swa < ctx) place the needle at the START "
                        "(inside the sliding window) so an SWA model does not instant-EOS to "
                        "empty. adaptive is a NO-OP when SWA is absent (dense models).")
    p.add_argument("--empty-retries", type=int, default=0,
                   help="on a completion classified failure_mode='empty'/'premature_eos', "
                        "retry up to this many times (default 0 == no retry). A retry re-fires "
                        "the SAME minted test_id so A and C still compare on the same id.")
    p.add_argument("--swa-window", type=int, default=None,
                   help="override the sliding-window size (n_swa) instead of probing /props. "
                        "None (default) => probe; 0 => force 'no SWA' (dense); >0 => force that "
                        "window. Only consulted by the adaptive placement strategy.")

    # ---- preview / ack ----
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan + footprint math, launch NOTHING")
    p.add_argument(ACK_FLAG, dest="ack", action="store_true",
                   help="REQUIRED to launch when --cert-source live and not --dry-run")


# ---- profile resolution (the --profile form) --------------------------------------
def _builtin_profiles_dir() -> Path:
    """The packaged profiles dir (``batch_invariance/profiles``)."""
    return Path(__file__).resolve().parent / "profiles"


def _resolve_profile_dir(name: str) -> Path:
    """Resolve a ``--profile`` name to its directory.

    Tries, in order: a path that exists as given (``profiles/<name>`` relative to CWD or an
    absolute/relative dir the user passed), then the packaged ``batch_invariance/profiles/
    <name>``. Raises ``FileNotFoundError`` if neither exists."""
    cand = Path(name)
    if cand.is_dir():
        return cand
    cwd_cand = Path.cwd() / "profiles" / name
    if cwd_cand.is_dir():
        return cwd_cand
    pkg_cand = _builtin_profiles_dir() / name
    if pkg_cand.is_dir():
        return pkg_cand
    raise FileNotFoundError(
        f"profile {name!r} not found (looked for {cand}, {cwd_cand}, {pkg_cand})")


def _load_toml(path: Path) -> dict:
    """Load a TOML file with the stdlib ``tomllib`` (py3.11+).

    The package targets py3.10-3.13; ``tomllib`` is stdlib on 3.11+. On 3.10 (no
    ``tomllib``) a clear error is raised telling the operator to use the explicit
    ``--server-bin/--model`` form instead -- the profile form is an optional convenience,
    not required for any core README command."""
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError as exc:  # pragma: no cover - only on py3.10
        raise RuntimeError(
            "the --profile form needs TOML support (stdlib 'tomllib', Python 3.11+). "
            "On Python 3.10 use the explicit --server-bin/--model/--server-flag form "
            "instead (the profile is only a convenience for the alias->gguf mapping)."
        ) from exc
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_live_inputs(args: argparse.Namespace) -> None:
    """Fill server-bin/model/flags/scorer/workset from a profile when --profile is given.

    MUTATES ``args`` in place so the rest of run-live (and ``run_main``) sees a fully
    populated namespace identical to the explicit form. When no --profile is given this is
    a no-op (the explicit flags are used verbatim). The profile's ``models.toml`` maps the
    ``--model-alias`` to a gguf path + server flags; a missing ``models.toml`` (only the
    ``.example`` shipped) raises a clear "copy the example first" error.

    Resolution precedence for each field: an EXPLICIT CLI flag always wins over the
    profile (so a user can override one field), and the profile fills only what the user
    left unset."""
    if not getattr(args, "profile", None):
        return
    prof_dir = _resolve_profile_dir(args.profile)

    models_toml = prof_dir / "models.toml"
    if not models_toml.is_file():
        example = prof_dir / "models.example.toml"
        hint = (f" Copy the example first: cp {example} {models_toml}"
                if example.is_file() else "")
        raise FileNotFoundError(
            f"profile {args.profile!r}: no models.toml at {models_toml}.{hint}")
    reg = _load_toml(models_toml)

    defaults = reg.get("defaults") or {}
    models = reg.get("models") or []
    alias = args.model_alias
    entry = next((m for m in models if str(m.get("alias")) == str(alias)), None)
    if entry is None:
        known = [str(m.get("alias")) for m in models]
        raise KeyError(
            f"profile {args.profile!r}: no model alias {alias!r} in {models_toml} "
            f"(known aliases: {known})")

    models_dir = args.models_dir or os.environ.get("MODELS_DIR")

    def _subst(val: str) -> str:
        if "{MODELS_DIR}" in val:
            if not models_dir:
                raise ValueError(
                    f"profile path {val!r} uses the {{MODELS_DIR}} placeholder; pass "
                    f"--models-dir or set the MODELS_DIR env var.")
            return val.replace("{MODELS_DIR}", str(models_dir))
        return val

    # gguf path (substitute {MODELS_DIR}) -- only when --model was not given explicitly.
    if not args.model:
        args.model = _subst(str(entry.get("gguf", "")))

    # server binary (optional in the registry): an entry's server_bin overrides the
    # [defaults] server_bin; used only when the user gave no explicit --server-bin. The
    # example registry leaves it unset (it is host-specific), so an operator either adds it
    # to their models.toml OR passes --server-bin -- both make the documented command work.
    if not getattr(args, "server_bin", None):
        sb = entry.get("server_bin") or defaults.get("server_bin")
        if sb:
            args.server_bin = _subst(str(sb))

    # server flags: entry overrides defaults; only used when the user gave none explicitly.
    if not getattr(args, "server_flag", None):
        flags: list[str] = []
        flags.extend(str(x) for x in (defaults.get("server_flags") or []))
        flags.extend(str(x) for x in (entry.get("server_flags") or []))
        args.server_flag = flags or None

    # KV label from the entry (only when the user left the default and the entry names one).
    if entry.get("kv") and args.kv == "q8_0":
        args.kv = str(entry.get("kv"))

    # The profile's scorer + work-set (only when the user did not supply them). The spark
    # profile ships a scorer entry point and a work-set materializer; resolve generically
    # via a per-profile convention: a sibling ``<name>_scorer:score`` and a materialized
    # work-set JSON under the out-dir. We keep this DATA-driven where possible.
    _apply_profile_scorer_and_workset(args, prof_dir)


def _apply_profile_scorer_and_workset(args: argparse.Namespace, prof_dir: Path) -> None:
    """Fill --scorer and --workset from the profile package when the user left them unset.

    Convention (matches the shipped ``spark`` profile, see its package docstring):
      * scorer  -> ``batch_invariance.profiles.<name>.<name>_scorer:score`` if that module
                   exists; else left at the generic exact-match default.
      * workset -> materialized by the profile's ``<name>_workset.write_workset_json`` into
                   ``<out-dir>/<name>_workset.json`` if that helper exists; else the user
                   MUST pass --workset.
    Only fires for a PACKAGED profile (one importable as ``batch_invariance.profiles.<name>``);
    a bare external profile dir supplies only the model registry and the user brings their
    own --scorer/--workset. Best-effort: any import failure leaves the field unset (the
    generic default / a clear "missing --workset" error downstream)."""
    import importlib

    name = prof_dir.name
    pkg = f"batch_invariance.profiles.{name}"

    if not getattr(args, "scorer", None):
        scorer_mod = f"{pkg}.{name}_scorer"
        try:
            importlib.import_module(scorer_mod)
            args.scorer = f"{scorer_mod}:score"
        except Exception:
            pass  # no packaged scorer -> generic exact-match default

    if not getattr(args, "workset", None):
        ws_mod_name = f"{pkg}.{name}_workset"
        try:
            ws_mod = importlib.import_module(ws_mod_name)
            writer = getattr(ws_mod, "write_workset_json", None)
            if callable(writer):
                os.makedirs(args.out_dir, exist_ok=True)
                ws_path = os.path.join(args.out_dir, f"{name}_workset.json")
                writer(ws_path, args.profile_corner, ctx=int(args.ctx),
                       model_id=args.model_alias)
                args.workset = ws_path
        except Exception:
            pass  # no packaged work-set materializer -> user must pass --workset


def _cmd_run_live(args: argparse.Namespace) -> int:
    """``run-live``: resolve inputs (incl. the profile form), then delegate to run_main.

    All real work + the EXIT_* contract live in ``live_invariance.run_main``. This wrapper
    only (a) resolves the profile form into the explicit fields and (b) enforces that the
    required inputs ended up present (so a missing server-bin/model/workset is a clean
    usage error, not a deep AttributeError)."""
    try:
        _resolve_live_inputs(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"run-live: {exc}", file=sys.stderr)
        return EXIT_ERROR

    missing = [name for name, val in (("--server-bin", args.server_bin),
                                      ("--model", args.model),
                                      ("--workset", args.workset)) if not val]
    if missing:
        print(f"run-live: missing required input(s) {missing}. Provide them explicitly, or "
              f"use --profile <name> --model-alias <alias> to resolve them from a profile.",
              file=sys.stderr)
        return EXIT_ERROR

    # --dry-run is a PREVIEW: it must not require the model file to exist on this box (you
    # may be writing the command on a CPU-only CI host before the gguf is present). The
    # footprint projection needs a weight size, so when none was given AND the model file is
    # absent, inject a clearly-labeled placeholder (0.0 GB) so the preview prints instead of
    # refusing. A REAL run (no --dry-run) is untouched: it still demands a sizeable gguf or
    # an explicit --gguf-gb. Use `plan-n --gguf-gb <GB>` for an accurate off-box projection.
    if getattr(args, "dry_run", False) and getattr(args, "gguf_gb", None) is None:
        model_path = args.model or ""
        if not (model_path and os.path.exists(model_path)):
            args.gguf_gb = 0.0

    return run_main(args)


# ===========================================================================
# verify-cert -- load a cert JSON and report status + is_promotable (and why).
# ===========================================================================
def _cmd_verify_cert(args: argparse.Namespace) -> int:
    """``verify-cert PATH``: load a cert JSON, print status + is_promotable + the reasons.

    Exit 0 when the cert PARSES (whether or not it is promotable -- a validly-parsed RED
    cert is a successful verification); non-zero (EXIT_ERROR) only when the file is missing
    or not a valid cert JSON object."""
    path = args.path
    try:
        with open(path, encoding="utf-8") as f:
            cert = json.load(f)
    except FileNotFoundError:
        print(f"verify-cert: no such file: {path}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        print(f"verify-cert: {path} is not valid cert JSON: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if not isinstance(cert, dict):
        print(f"verify-cert: {path} is not a cert object (got {type(cert).__name__})",
              file=sys.stderr)
        return EXIT_ERROR

    status = cert.get("status")
    source = cert.get("source")
    promotable = idiff.is_promotable(cert)
    is_green = idiff.cert_is_green(cert, require_source="live")
    overlap_ok = cert.get("overlap_ok")
    if overlap_ok is None and isinstance(cert.get("overlap"), dict):
        overlap_ok = cert["overlap"].get("overlap_ok")
    gate = cert.get("gate")
    cf = cert.get("completion_floor") or {}
    cov = cert.get("cobatch_coverage") or {}

    # A plain-English reason for the promotion verdict (mirrors is_promotable's gates).
    why: list[str] = []
    if cert.get("status") != "green":
        why.append(f"status is {status!r} (only the literal 'green' can promote)")
    if source != "live":
        why.append(f"source is {source!r} (only 'live' can promote; mock/soft-green never)")
    if not overlap_ok:
        why.append("overlap_ok is not true (no real co-batching observed)")
    if gate != "abc_union":
        why.append(f"gate is {gate!r} (the strong 'abc_union' producer is required)")
    if isinstance(cf, dict) and not cf.get("completion_floor_ok", True):
        why.append("completion_floor_ok is false (too few trials genuinely scored)")
    if (isinstance(cov, dict) and cov.get("coverage_checked")
            and not cov.get("cobatch_coverage_ok", True)):
        why.append("cobatch_coverage was checked and insufficient/uncertain")

    print(f"cert: {path}")
    print(f"  model={cert.get('model')} ctx={cert.get('ctx')} N={cert.get('dispatch_n')}")
    print(f"  status        = {status}")
    print(f"  source        = {source}")
    print(f"  gate          = {gate}")
    print(f"  overlap_ok    = {overlap_ok}")
    print(f"  cert_is_green = {is_green}  (literal green + source=='live')")
    print(f"  is_promotable = {promotable}")
    if promotable:
        print("  -> PROMOTABLE: this cert licenses its exact (model, ctx, N) cell.")
    else:
        print("  -> NOT promotable. Why:")
        for r in (why or ["(unknown)"]):
            print(f"       - {r}")
        ac = (cert.get("divergence_report") or {}).get("AC") or {}
        if cert.get("status") == idiff.STATUS_FAILED:
            print(f"     (RED: AC.n_divergent={ac.get('n_divergent')}; "
                  f"divergence_class={cert.get('divergence_class')})")
    # Exit 0 for any validly-parsed cert (promotable OR a clean RED is a successful verify).
    return EXIT_OK


# ===========================================================================
# plan-n -- the per-N footprint / KV plan + memory projection (launches nothing).
# ===========================================================================
def _cmd_plan_n(args: argparse.Namespace) -> int:
    """``plan-n``: print the per-N KV/footprint projection for --model/--ctx/--parallel.

    Uses the SAME planning math the gate's footprint guard + ``run-live --dry-run`` use
    (``kv_budget`` + the S0 cap), launching NOTHING. ``--gguf-gb`` supplies the weight size
    when the model file is absent on this box (e.g. planning on a CPU-only CI host). When a
    real --model path exists and no --gguf-gb override is given, its on-disk size is used."""
    ctx = int(args.ctx)
    kv_label = str(args.kv)

    # Weight size: explicit --gguf-gb override, else probe the file, else fail clear.
    if args.gguf_gb is not None:
        gguf_gb = float(args.gguf_gb)
        gguf_src = f"--gguf-gb override ({gguf_gb:.2f}GB)"
    else:
        try:
            gguf_gb = kv_budget.gguf_size_gb(args.model) if args.model else None
        except (FileNotFoundError, OSError):
            gguf_gb = None
        if gguf_gb is None:
            print("plan-n: cannot size the model weights. Pass --gguf-gb <GB> (the model "
                  "file is absent/remote on this box), or point --model at a real gguf.",
                  file=sys.stderr)
            return EXIT_ERROR
        gguf_src = f"on-disk probe of {args.model} ({gguf_gb:.2f}GB)"

    cap = float(args.max_footprint_gb)
    overhead = float(kv_budget.PER_SERVER_OVERHEAD_GB)
    floor = float(kv_budget.HARD_FLOOR_GB)

    # Per-N grid: 1 .. --parallel (the gate launches A at N=1 and B/C at N=--parallel; the
    # grid shows the projection at every N up to the requested one so an operator can see
    # where the S0 cap bites).
    n_target = max(1, int(args.parallel))
    ns = sorted(set([1, n_target] + list(range(1, n_target + 1))))

    print("=" * 72)
    print("batch-invariance plan-n (KV / footprint projection; launches NOTHING)")
    print(f"  model={args.model_alias}  weights={gguf_src}")
    print(f"  ctx={ctx}  kv_quant={kv_label}  kv_factor={kv_budget.kv_factor(kv_label):.3f}")
    print(f"  cap(--max-footprint-gb)={cap:.2f}GB  per_server_overhead={overhead:.2f}GB  "
          f"hard_floor={floor:.2f}GB")
    print("  (an N-slot server launches with -c = ctx*N so every slot gets the full ctx)")
    print("")
    print(f"  {'N':>4}  {'server_ctx':>11}  {'KV(GB)':>9}  {'projected(GB)':>14}  {'<=cap?':>7}")
    print(f"  {'-'*4}  {'-'*11}  {'-'*9}  {'-'*14}  {'-'*7}")
    for n in ns:
        server_ctx = ctx * max(1, n)
        kv = kv_budget.kv_gb(ctx, n, kv_label)
        projected = float(gguf_gb) + kv + overhead
        within = projected <= cap + 1e-9
        print(f"  {n:>4}  {server_ctx:>11}  {kv:>9.2f}  {projected:>14.2f}  "
              f"{'OK' if within else 'EXCEEDS':>7}")
    print("")
    proj_target = float(gguf_gb) + kv_budget.kv_gb(ctx, n_target, kv_label) + overhead
    print(f"  at N={n_target}: projected {proj_target:.2f}GB "
          f"({'within' if proj_target <= cap + 1e-9 else 'EXCEEDS'} the {cap:.2f}GB cap; "
          f"need {proj_target + floor:.2f}GB live MemAvailable incl. the {floor:.2f}GB floor)")
    print("  NOTHING launched (plan-n is a pure projection).")
    return EXIT_OK


# ===========================================================================
# Parser assembly + dispatch.
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the subcommand parser. ``batch-invariance <cmd> --help`` shows each cmd's flags.

    Four subcommands: ``run-mock`` / ``run-live`` / ``verify-cert`` / ``plan-n``. The
    top-level ``--help`` lists them; a bare invocation (no subcommand) prints help and exits
    non-zero (handled in :func:`main`)."""
    p = argparse.ArgumentParser(
        prog="batch-invariance",
        description="Per-cell batch-invariance verifier for OpenAI-compatible local LLM "
                    "servers (llama.cpp continuous batching). Detects when concurrent/batched "
                    "dispatch changes scored outputs vs serial.")
    sub = p.add_subparsers(dest="command", metavar="<command>",
                           title="commands",
                           description="run 'batch-invariance <command> --help' for a "
                                       "command's flags")

    # ---- run-mock ----
    pm = sub.add_parser(
        "run-mock",
        help="run the full A/B/C gate against the in-process zero-GPU mock (no model/GPU)",
        description="Run the full A/B/C gate against the IN-PROCESS mock server -- NO model, "
                    "NO GPU. Default writes a GREEN, source=mock, NON-promotable cert; "
                    "--batch-divergence drives the SAME gate RED. Proves the gate bites with "
                    "zero hardware (a mock cert can NEVER promote -- a real GREEN must be "
                    "earned live).")
    pm.add_argument("--out-dir", default=None,
                    help="cert + logs land here (default: a temp dir). The cert is written "
                         "under <out-dir>/dispatch-cert/.")
    pm.add_argument("--batch-divergence", action="store_true",
                    help="arm the mock so a co-batched completion scores differently from "
                         "serial -> the gate goes RED. Default OFF (honest mock -> green).")
    pm.add_argument("--ctx", type=int, default=2048,
                    help="per-slot context for the mock run (default 2048; footprint math "
                         "only -- the mock has no real KV).")
    pm.add_argument("--parallel", type=int, default=4,
                    help="N slots for arms B/C (default 4).")
    pm.set_defaults(func=_cmd_run_mock)

    # ---- run-live ----
    pl = sub.add_parser(
        "run-live",
        help="run the live gate against YOUR OpenAI-compatible server (one real cell)",
        description="Live batch-invariance gate: does batched concurrent dispatch produce the "
                    "same scored outputs as serial dispatch, on YOUR OpenAI-compatible server? "
                    "Writes one signed-by-construction cert per (model, ctx, N). Use the "
                    "explicit --server-bin/--model form OR the profile form (--profile <name> "
                    "--model-alias <alias>).")
    _add_run_live_args(pl)
    pl.set_defaults(func=_cmd_run_live)

    # ---- verify-cert ----
    pv = sub.add_parser(
        "verify-cert",
        help="load a cert JSON and print its status + whether it is_promotable (and why)",
        description="Load a cert JSON and print its status, source, and the machine-readable "
                    "is_promotable() verdict (with the reason it is / is not promotable). Exit "
                    "0 for any validly-parsed cert (promotable OR a clean RED); non-zero only "
                    "on a malformed/missing cert.")
    pv.add_argument("path", help="path to the cert JSON file")
    pv.set_defaults(func=_cmd_verify_cert)

    # ---- plan-n ----
    pp = sub.add_parser(
        "plan-n",
        help="print the per-N footprint/KV plan + memory projection (launches nothing)",
        description="Print the per-N KV-cache + footprint projection for a (model, ctx, N) "
                    "plan -- the SAME math the gate's footprint guard uses -- launching "
                    "NOTHING. Use --gguf-gb to size the weights when the model file is absent "
                    "on this box.")
    pp.add_argument("--model", default=None,
                    help="path to the model gguf (its on-disk size sizes the weights, unless "
                         "--gguf-gb overrides). Optional when --gguf-gb is given.")
    pp.add_argument("--model-alias", default=DEFAULT_MODEL_ID,
                    help="label for the plan output (default 'model').")
    pp.add_argument("--ctx", type=int, required=True,
                    help="per-slot context size (e.g. 8192 or 32768).")
    pp.add_argument("--parallel", type=int, default=4,
                    help="the max N to project up to (the grid shows N=1..parallel).")
    pp.add_argument("--gguf-gb", type=float, default=None,
                    help="weight size in GB (override the on-disk probe; required when the "
                         "model file is absent/remote on this box).")
    pp.add_argument("--kv", default="q8_0", help="KV cache quant label (footprint math).")
    pp.add_argument("--max-footprint-gb", type=float, default=MAX_FOOTPRINT_GB,
                    help=f"the HARD footprint cap to check each N against (default "
                         f"{MAX_FOOTPRINT_GB}GB; the S0 precheck refuses a launch above it).")
    pp.set_defaults(func=_cmd_plan_n)

    return p


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch the chosen subcommand; return an exit code.

    A bare ``batch-invariance`` (no subcommand) prints help and returns a non-zero usage
    code (never crashes). Each subcommand's handler returns its own exit code (run-live
    folds in ``live_invariance``'s EXIT_* contract)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    return func(args)


if __name__ == "__main__":
    sys.exit(main())

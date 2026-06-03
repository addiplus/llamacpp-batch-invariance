#!/usr/bin/env python3
"""examples/run_gate.py -- run the batch-invariance gate against the bundled mock (no GPU).

This is the offline driver ``examples/run.sh`` invokes. It needs NO real ``llama-server``,
NO model, and NO GPU: it drives the REAL A/B/C gate against the in-process mock via the
package's FIRST-CLASS mock-driving core (:func:`batch_invariance.cli.run_mock_gate` -- the
same core the ``batch-invariance run-mock`` subcommand uses). It exists to SHOW the gate
going RED on a divergent run and reaching a clean (non-promotable) GREEN on an honest run.

Modes:

  (default)    run the gate twice against the mock:
                 1. ``--batch-divergence`` ON  -> a genuine score divergence -> RED cert.
                 2. all knobs OFF (honest mock) -> the gate reaches the literal "green"
                    status; the cert is ``source=mock`` and NON-promotable by construction
                    (a real GREEN must be earned against a real server).

  --dry-run    print the plan + KV/footprint projection and launch NOTHING (no mock, no
               arms). This mirrors the real CLI's ``plan-n`` / ``run-live --dry-run`` preview.

Everything here is Python standard library only. The certs are written under ``--out-dir``
(default: a temp dir) so you can inspect the JSON artifacts.

NOTE: this script is now a THIN wrapper over the package's ``run_mock_gate`` core; the
canonical user-facing entry point is the ``batch-invariance run-mock`` subcommand
(``batch-invariance run-mock --out-dir ./certs [--batch-divergence]``). This file is kept
as the zero-install on-ramp that runs straight from a checkout.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Allow running straight from a checkout ("python examples/run_gate.py") without an install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# So `--scorer examples.scorer:...` resolves when run from the repo root.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from batch_invariance import invariance_diff as idiff  # noqa: E402
from batch_invariance import kv_budget  # noqa: E402
from batch_invariance.cli import run_mock_gate  # noqa: E402
from batch_invariance.live_invariance import DEFAULT_SERVER_FLAGS  # noqa: E402


def _print_verdict(label, cert):
    ac = (cert.get("divergence_report") or {}).get("AC") or {}
    promotable = idiff.is_promotable(cert)
    print(f"  [{label}]")
    print(f"    status              = {cert['status']}")
    print(f"    source              = {cert['source']}")
    print(f"    AC.n_divergent      = {ac.get('n_divergent')}")
    print(f"    divergence_class    = {cert.get('divergence_class')}")
    print(f"    overlap_ok          = {cert['overlap']['overlap_ok']} "
          f"(server_peak_busy_slots={cert['overlap']['server_peak_busy_slots']})")
    print(f"    completion_floor_ok = {cert['completion_floor']['completion_floor_ok']} "
          f"(n_ok_a={cert['completion_floor']['n_ok_a']}, "
          f"n_ok_c={cert['completion_floor']['n_ok_c']})")
    print(f"    is_promotable       = {promotable}")
    return promotable


def _dry_run(args) -> int:
    """Print the plan + KV/footprint projection; launch nothing (no mock, no arms)."""
    server_ctx = int(args.ctx) * max(1, int(args.parallel))
    kv = kv_budget.kv_gb(int(args.ctx), int(args.parallel), args.kv)
    projected = 1.0 + kv + kv_budget.PER_SERVER_OVERHEAD_GB   # demo weights = 1.0 GB
    print("=== batch-invariance example - DRY RUN (launches NOTHING) ===")
    print(f"workset           = {args.workset}")
    print("model_alias       = demo-model   (gguf: model.gguf, weights assumed 1.0 GB)")
    print(f"ctx               = {args.ctx}   dispatch_n(N) = {args.parallel}   "
          f"server_ctx=ctx*N = {server_ctx}")
    print(f"n_predict         = {args.n_predict}   reps = {args.reps}   "
          f"gate_passes(T) = {args.gate_passes}")
    print(f"KV(ctx,N)={kv:.2f}GB + weights(1.0GB) + overhead "
          f"({kv_budget.PER_SERVER_OVERHEAD_GB}GB) -> projected {projected:.2f}GB")
    print(f"server flags      = {' '.join(DEFAULT_SERVER_FLAGS)}")
    print("arms              = A(--parallel 1) ; B/C(--parallel "
          f"{args.parallel}, ctx*N={server_ctx})")
    print("NO mock started, NO arms run, NO cert written (dry-run).")
    print("(tip: the packaged equivalent is `batch-invariance plan-n --ctx ... --parallel "
          "... --gguf-gb ...`.)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the batch-invariance gate against the bundled mock (no GPU). Thin "
                    "wrapper over batch_invariance.cli.run_mock_gate.")
    ap.add_argument("--workset",
                    default=os.path.join(os.path.dirname(__file__), "workset.json"),
                    help="(dry-run display only) the work-set the packaged demo references. "
                         "The live gate runs use run_mock_gate's built-in RED work-set.")
    ap.add_argument("--out-dir", default=None,
                    help="where certs are written (default: a temp dir)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv", default="q8_0", help="KV cache quant label (footprint math)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gate-passes", type=int, default=6)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan + footprint projection and launch nothing")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run(args)

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="bi-example-")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("batch-invariance example: driving the REAL A/B/C gate against the mock")
    print(f"(stdlib only, no GPU, no model; certs -> {out_dir})")
    print("=" * 72)

    common = dict(out_dir=out_dir, ctx=args.ctx, parallel=args.parallel,
                  reps=args.reps, gate_passes=args.gate_passes, n_predict=args.n_predict)

    # 1) DIVERGENT run -> RED. run_mock_gate(batch_divergence=True) arms the mock's
    #    score-divergence knob against its built-in RED work-set: a co-batched completion
    #    scores differently from serial -> a genuine score divergence the gate catches as RED.
    print("\n[1] --batch-divergence ON  (a co-batched completion scores differently):")
    red = run_mock_gate(batch_divergence=True, **common)
    red_promotable = _print_verdict("DIVERGENT run", red)

    # 2) HONEST run -> the gate reaches the literal green status, but a mock-sourced cert is
    #    NON-promotable by construction (a real GREEN must be earned against a real server).
    print("\n[2] all divergence knobs OFF, source=mock (honest, vacuous-by-design):")
    green = run_mock_gate(batch_divergence=False, **common)
    green_promotable = _print_verdict("HONEST run", green)

    print("\n" + "=" * 72)
    cert_dir = os.path.join(out_dir, "dispatch-cert")
    # Persist both certs as inspectable artifacts (the driver returns the dict; persist it).
    idiff.persist_cert(cert_dir, red)
    idiff.persist_cert(cert_dir, green)
    print(f"certs written under: {cert_dir}")
    for name in sorted(os.listdir(cert_dir)) if os.path.isdir(cert_dir) else []:
        print(f"  - {name}")

    # The example ASSERTS its own headline contract so run.sh can gate on the exit code:
    #   * the divergent run is RED and NON-promotable, and
    #   * the honest mock-sourced run, while green-status, is also NON-promotable.
    ok = (red["status"] == idiff.STATUS_FAILED and not red_promotable
          and green["status"] == idiff.STATUS_GREEN and not green_promotable)
    print("\nRESULT:",
          "OK -- gate went RED on divergence and refused to promote the mock green."
          if ok else "UNEXPECTED -- see the verdicts above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

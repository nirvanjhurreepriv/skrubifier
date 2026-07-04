"""
CLI: python -m skrubifier convert <source.py> --out <converted.py>

This wires analyzer -> converter -> validator together. It expects the
source script to define, at module level:
    PIPELINE      : an (unfitted) sklearn Pipeline/ColumnTransformer + estimator
    FEATURE_COLUMNS, TARGET_COLUMN, TASK  : strings/lists
    df            : a pandas DataFrame sample (used for dtype inference only)
Scripts that don't follow this convention fall back to AST analysis
(`analyze_source`), which produces a coarser IR the LLM has to do more work
to fill in (flagged in PipelineIR.notes).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from .analyzer import analyze_estimator, analyze_source
from .converter import convert, default_anthropic_llm_call, default_openai_compatible_llm_call
from .validator import runtime_check, static_check


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("source_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(prog="skrubifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    conv = sub.add_parser("convert")
    conv.add_argument("source", help="path to source pipeline .py")
    conv.add_argument("--out", required=True)
    conv.add_argument("--backend", choices=["anthropic", "gwdg"], default="gwdg",
                       help="LLM backend: 'gwdg' uses the free open-weight SAIA API "
                            "(default), 'anthropic' uses the Anthropic API directly")
    conv.add_argument("--model", default=None,
                       help="model name; defaults to a sensible choice per backend")
    conv.add_argument("--no-repair", action="store_true",
                       help="single-shot conversion, skip static-check repair loop")
    args = ap.parse_args()

    if args.cmd == "convert":
        with open(args.source) as f:
            source_text = f.read()

        try:
            mod = _load_module(args.source)
            ir = analyze_estimator(
                mod.PIPELINE, mod.FEATURE_COLUMNS, mod.TARGET_COLUMN, mod.TASK,
                df_dtypes={c: str(t) for c, t in mod.df.dtypes.items()} if hasattr(mod, "df") else None,
            )
        except AttributeError:
            print("[skrubifier] source does not expose PIPELINE/FEATURE_COLUMNS/... "
                  "falling back to AST analysis", file=sys.stderr)
            # Prefer TARGET_COLUMN from the already-loaded module; fall back to AST scan
            target_col = getattr(mod, "TARGET_COLUMN", None)
            ir = analyze_source(args.source, target_column=target_col)

        if ir.notes:
            for n in ir.notes:
                print(f"[skrubifier] note: {n}", file=sys.stderr)

        if args.backend == "anthropic":
            llm_call = default_anthropic_llm_call(model=args.model or "claude-sonnet-4-6")
        else:
            llm_call = default_openai_compatible_llm_call(model=args.model or "qwen3-coder-next")

        if args.no_repair:
            validate_fn = None
        else:
            # Combined validator: static check first (fast), then execute the
            # candidate script in a subprocess from the source directory to
            # catch runtime DataOps API errors that static_check() misses.
            source_dir = os.path.dirname(os.path.abspath(args.source))

            def validate_fn(code: str) -> dict:
                static = static_check(code)
                if not static["ok"]:
                    return static
                return runtime_check(code, working_dir=source_dir)

        result = convert(ir, source_text, llm_call, validate_fn=validate_fn)

        with open(args.out, "w") as f:
            f.write(result.code)

        print(f"[skrubifier] wrote {args.out} (attempts={result.attempts}, "
              f"static_ok={result.validation_report and result.validation_report.get('ok')})")


if __name__ == "__main__":
    main()

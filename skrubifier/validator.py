"""
Stage 3: Validator.

Two levels of check, cheapest first:
  1. STATIC   — does the candidate script import cleanly, does it actually
     call skrub.var / .skb.apply / .skb.make_learner (i.e. it isn't just
     eager pandas/sklearn code with skrub imported and unused)?
  2. DYNAMIC  — exec the script in a subprocess with the real dataframes
     injected as the values for skrub.var(...), fit the resulting learner,
     and compare predictions/metric against the original pipeline's on a
     held-out split.

Both are exposed separately so `converter.convert(..., validate_fn=...)` can
use just the cheap static check for the LLM repair loop (fast) and reserve
the full dynamic run for the final report.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

REQUIRED_CALLS = {"var", "apply", "make_learner"}


def static_check(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "detail": f"SyntaxError: {e}"}

    calls_seen = set()
    uses_skb_namespace = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in REQUIRED_CALLS:
                calls_seen.add(node.attr)
            if node.attr == "skb":
                uses_skb_namespace = True

    missing = REQUIRED_CALLS - calls_seen
    if missing:
        return {"ok": False, "detail": f"Missing required skrub DataOps calls: {sorted(missing)}. "
                                        f"Script must build a DAG via .skb.apply/.skb.make_learner, "
                                        f"not eager pandas/sklearn code."}
    if not uses_skb_namespace:
        return {"ok": False, "detail": "No `.skb` accessor usage found — code does not appear to "
                                        "operate on DataOps objects."}
    return {"ok": True, "detail": "static check passed"}


def runtime_check(code: str, working_dir: str, timeout_s: int = 60) -> dict:
    """Execute the candidate script in a subprocess to catch runtime errors
    that appear when the DataOps DAG is constructed (at script definition time).

    All Phase-2 failure categories are detectable at this stage — before
    learner.fit() is ever called:
      - skrub.var("name", None) + .drop()/.skb.select() -> NoneType AttributeError
      - missing imports (NameError: name 'pd' is not defined)
      - hallucinated skrub attrs (AttributeError: module 'skrub' has no attr ...)
      - wrong constructor params (TypeError: __init__() unexpected keyword arg)
      - forward/undefined variable references (NameError)
      - pd.concat([DataOp, ...]) (TypeError: cannot concatenate DataOp)

    Returns the same dict schema as static_check so it can be composed with it.
    """
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=working_dir
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=working_dir,
            env=os.environ.copy(),
        )
        os.unlink(tmp_path)
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return {
            "ok": False,
            "detail": (
                f"Script execution timed out after {timeout_s}s. "
                "This may indicate loading a very large dataset inline. "
                "If the script loads data at module level, consider subsampling "
                "or loading lazily via skrub.var(...).skb.apply_func(pd.read_parquet)."
            ),
        }

    if proc.returncode != 0:
        err = proc.stderr.strip()[-3000:] if proc.stderr else "(no stderr captured)"
        return {
            "ok": False,
            "detail": (
                f"Runtime error when executing the converted script "
                f"(exit code {proc.returncode}):\n{err}"
            ),
        }

    return {"ok": True, "detail": "static and runtime checks passed"}


@dataclass
class DynamicCheckResult:
    ok: bool
    detail: str
    original_metric: Optional[float] = None
    converted_metric: Optional[float] = None
    metric_delta: Optional[float] = None


def dynamic_check(
    converted_script_path: str,
    harness_snippet: str,
    python_executable: str = sys.executable,
    timeout_s: int = 600,
    use_stratum: bool = True,
) -> DynamicCheckResult:
    """
    `harness_snippet` is a small script (provided per-example, see
    examples/*/harness.py) that:
      1. loads the real dataframe(s),
      2. runs the ORIGINAL sklearn pipeline, records `original_metric`,
      3. execs the converted script's namespace to get `learner`,
      4. fits/evaluates it the same way, records `converted_metric`,
      5. prints a JSON line: {"original_metric": ..., "converted_metric": ...}

    If `use_stratum=True`, sets STRATUM_RUST_BACKEND=1 in the subprocess
    environment. Converted scripts import stratum with a fallback to plain
    skrub (see prompts.STRATUM_NOTE), so this flag only has an effect when
    stratum is actually installed; harnesses can read it via
    `os.environ.get("STRATUM_RUST_BACKEND")` and call
    `stratum.set_config(rust_backend=True, scheduler=True)` accordingly.
    Kept as a subprocess call (rather than importing in-process) so a
    crashing candidate script can't take down the calling process, and so
    stray global state between original/converted runs can't leak.
    """
    import os
    env = {**os.environ, "STRATUM_RUST_BACKEND": "1" if use_stratum else "0"}

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness_snippet)
        harness_path = f.name

    try:
        proc = subprocess.run(
            [python_executable, harness_path, converted_script_path],
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return DynamicCheckResult(ok=False, detail=f"Timed out after {timeout_s}s")

    if proc.returncode != 0:
        return DynamicCheckResult(ok=False, detail=f"Runtime error:\n{proc.stderr[-3000:]}")

    import json
    try:
        last_line = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")][-1]
        result = json.loads(last_line)
    except Exception:
        return DynamicCheckResult(ok=False, detail=f"Could not parse harness output:\n{proc.stdout[-2000:]}")

    orig = result.get("original_metric")
    conv = result.get("converted_metric")
    if orig is None or conv is None:
        return DynamicCheckResult(ok=False, detail=f"Harness did not report both metrics: {result}")

    delta = abs(orig - conv)
    # tolerance: pipelines using randomized models (RF, GBMs) won't match bit
    # for bit even with fixed seeds once re-expressed as a DAG, so we check
    # "close enough to be the same model", not exact equality.
    tol = max(0.02, 0.03 * abs(orig))
    ok = delta <= tol
    return DynamicCheckResult(
        ok=ok,
        detail="within tolerance" if ok else f"metric delta {delta:.4f} exceeds tolerance {tol:.4f}",
        original_metric=orig, converted_metric=conv, metric_delta=delta,
    )

"""
Dynamic validation runner for Phase 2 LLM-generated scripts.
Each example is run in a fresh subprocess via its own mini-harness script
so sys.path / sys.modules state never leaks between tests.

Run from the project root:
    .venv/bin/python results/llm_conversions/run_dynamic_tests.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # project root
VENV_PY = ROOT / ".venv" / "bin" / "python"

MINI_HARNESSES = {}

# ── Each mini-harness is a self-contained script ────────────────────────────

MINI_HARNESSES["01"] = """\
import sys, os, json, runpy
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])
PIPELINE.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN])
orig = roc_auc_score(te[TARGET_COLUMN], PIPELINE.predict_proba(te[FEATURE_COLUMNS])[:, 1])

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = roc_auc_score(te[TARGET_COLUMN], learner.predict_proba({{"df": te}})[:, 1])
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["02"] = """\
import sys, os, json, runpy
import pandas as pd, numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0)
y_log_tr = np.log1p(tr[TARGET_COLUMN])
y_log_te = np.log1p(te[TARGET_COLUMN])
PIPELINE.fit(tr[FEATURE_COLUMNS], y_log_tr)
orig = r2_score(y_log_te, PIPELINE.predict(te[FEATURE_COLUMNS]))

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
pred = learner.predict({{"df": te}})
# LLM applies log1p to target before mark_as_y; predictions are in log scale
conv = r2_score(y_log_te, pred)
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["03"] = """\
import sys, os, json, runpy
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import build_features, FEATURE_COLUMNS, TARGET_COLUMN
baskets = pd.read_csv("baskets.csv")
products = pd.read_csv("products.csv")
tr_b, te_b = train_test_split(baskets, test_size=0.2, random_state=0,
                               stratify=baskets[TARGET_COLUMN])
tr_p = products[products["basket_ID"].isin(tr_b["basket_ID"])]
te_p = products[products["basket_ID"].isin(te_b["basket_ID"])]
tr_m = build_features(tr_b, tr_p)
te_m = build_features(te_b, te_p)
model = HistGradientBoostingClassifier(random_state=0)
model.fit(tr_m[FEATURE_COLUMNS], tr_m[TARGET_COLUMN])
orig = roc_auc_score(te_m[TARGET_COLUMN], model.predict_proba(te_m[FEATURE_COLUMNS])[:, 1])

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"baskets": tr_b, "products": tr_p}})
conv = roc_auc_score(te_b[TARGET_COLUMN],
                     learner.predict_proba({{"baskets": te_b, "products": te_p}})[:, 1])
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["04"] = """\
import sys, os, json, runpy
import pandas as pd, numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN, clean_nyc
full = pd.read_parquet("data/train1_subsampled.parquet")
df = full.sample(n=10_000, random_state=0).reset_index(drop=True)
df_clean = clean_nyc(df)
tr, te = train_test_split(df_clean, test_size=0.2, random_state=0)
PIPELINE.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN])
orig = r2_score(te[TARGET_COLUMN], PIPELINE.predict(te[FEATURE_COLUMNS]))

# LLM script uses skrub.var("df", ...) with inline pd.read_parquet;
# pass df directly as "df" to override
ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = r2_score(te[TARGET_COLUMN], learner.predict({{"df": te}}))
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["05"] = """\
import sys, os, json, runpy
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN, CLASSES
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])
le = LabelEncoder().fit(CLASSES)
y_tr = le.transform(tr[TARGET_COLUMN])
y_te_bin = label_binarize(le.transform(te[TARGET_COLUMN]), classes=range(9))
PIPELINE.fit(tr[FEATURE_COLUMNS], y_tr)
orig = roc_auc_score(y_te_bin, PIPELINE.predict_proba(te[FEATURE_COLUMNS]),
                     multi_class="ovr", average="macro")

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = roc_auc_score(y_te_bin, learner.predict_proba({{"df": te}}),
                     multi_class="ovr", average="macro")
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["06"] = """\
import sys, os, json, runpy
import pandas as pd, numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0)
y_log_tr = np.log1p(tr[TARGET_COLUMN])
y_log_te = np.log1p(te[TARGET_COLUMN])
PIPELINE.fit(tr[FEATURE_COLUMNS], y_log_tr)
orig = r2_score(y_log_te, PIPELINE.predict(te[FEATURE_COLUMNS]))

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = r2_score(y_log_te, learner.predict({{"df": te}}))
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["07"] = """\
import sys, os, json, runpy
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])
PIPELINE.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN])
orig = roc_auc_score(te[TARGET_COLUMN], PIPELINE.predict_proba(te[FEATURE_COLUMNS])[:, 1])

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = roc_auc_score(te[TARGET_COLUMN], learner.predict_proba({{"df": te}})[:, 1])
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["08"] = """\
import sys, os, json, runpy
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, TEXT_COL, TARGET_COLUMN
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])
PIPELINE.fit(tr[TEXT_COL], tr[TARGET_COLUMN])
classes = PIPELINE.classes_
y_bin = label_binarize(te[TARGET_COLUMN], classes=classes)
orig = roc_auc_score(y_bin, PIPELINE.predict_proba(te[TEXT_COL]),
                     multi_class="ovr", average="macro")

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = roc_auc_score(y_bin, learner.predict_proba({{"df": te}}),
                     multi_class="ovr", average="macro")
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["09"] = """\
import sys, os, json, runpy
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import build_features, FEATURE_COLUMNS, TARGET_COLUMN
app = pd.read_csv("application_train.csv")
bureau = pd.read_csv("bureau.csv")
tr_a, te_a = train_test_split(app, test_size=0.2, random_state=0, stratify=app[TARGET_COLUMN])
tr_b = bureau[bureau["SK_ID_CURR"].isin(tr_a["SK_ID_CURR"])]
te_b = bureau[bureau["SK_ID_CURR"].isin(te_a["SK_ID_CURR"])]
tr_m = build_features(tr_a, tr_b)
te_m = build_features(te_a, te_b)
model = HistGradientBoostingClassifier(random_state=0)
model.fit(tr_m[FEATURE_COLUMNS], tr_m[TARGET_COLUMN])
orig = roc_auc_score(te_m[TARGET_COLUMN], model.predict_proba(te_m[FEATURE_COLUMNS])[:, 1])

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
# LLM uses skrub.var("app", ...) and skrub.var("bureau", ...)
learner.fit({{"app": tr_a, "bureau": tr_b}})
conv = roc_auc_score(te_a[TARGET_COLUMN],
                     learner.predict_proba({{"app": te_a, "bureau": te_b}})[:, 1])
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

MINI_HARNESSES["10"] = """\
import sys, os, json, runpy
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CONVERTED = sys.argv[1]
os.chdir(r"{ex_dir}")
sys.path.insert(0, r"{ex_dir}")
from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN
df = pd.read_csv("train.csv")
tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])
PIPELINE.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN])
orig = roc_auc_score(te[TARGET_COLUMN], PIPELINE.predict_proba(te[FEATURE_COLUMNS])[:, 1])

ns = runpy.run_path(CONVERTED, run_name="__converted__")
learner = ns["learner"]
learner.fit({{"df": tr}})
conv = roc_auc_score(te[TARGET_COLUMN], learner.predict_proba({{"df": te}})[:, 1])
print(json.dumps({{"original_metric": orig, "converted_metric": conv}}))
"""

EXAMPLE_DIRS = {
    "01": "01_titanic",
    "02": "02_house_prices_xgb",
    "03": "03_credit_fraud_multitable",
    "04": "04_nyc_taxi_fare",
    "05": "05_otto_group",
    "06": "06_allstate_claims_severity",
    "07": "07_random_acts_of_pizza",
    "08": "08_spooky_author",
    "09": "09_home_credit",
    "10": "10_santander",
}


def tol(orig):
    return max(0.02, 0.03 * abs(orig))


def run_one(ex: str) -> dict:
    ex_dir = str(ROOT / "examples" / EXAMPLE_DIRS[ex])
    converted = str(ROOT / "results" / "llm_conversions" / f"{EXAMPLE_DIRS[ex]}.py")
    script_body = MINI_HARNESSES[ex].format(ex_dir=ex_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_body)
        harness_path = f.name

    try:
        result = subprocess.run(
            [str(VENV_PY), harness_path, converted],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            err_lines = result.stderr.strip().splitlines()
            short = err_lines[-1] if err_lines else "unknown error"
            return {"ex": ex, "original_metric": None, "converted_metric": None,
                    "delta": None, "tol": None, "pass": False, "error": short,
                    "stderr": result.stderr[-2000:]}
        # find the last JSON line in stdout
        for line in reversed(result.stdout.strip().splitlines()):
            try:
                data = json.loads(line)
                orig = data["original_metric"]
                conv = data["converted_metric"]
                delta = abs(conv - orig)
                t = tol(orig)
                return {"ex": ex, "original_metric": round(orig, 4),
                        "converted_metric": round(conv, 4),
                        "delta": round(delta, 4), "tol": round(t, 4),
                        "pass": delta <= t, "error": None}
            except (json.JSONDecodeError, KeyError):
                continue
        return {"ex": ex, "original_metric": None, "converted_metric": None,
                "delta": None, "tol": None, "pass": False,
                "error": "no JSON output", "stdout": result.stdout[-500:]}
    except subprocess.TimeoutExpired:
        return {"ex": ex, "original_metric": None, "converted_metric": None,
                "delta": None, "tol": None, "pass": False, "error": "timeout (300s)"}
    finally:
        os.unlink(harness_path)


if __name__ == "__main__":
    results = []
    for ex in sorted(EXAMPLE_DIRS.keys()):
        print(f"Running example {ex}...", file=sys.stderr)
        r = run_one(ex)
        status = "PASS" if r["pass"] else f"FAIL: {r.get('error', '')}"
        print(f"  {status}", file=sys.stderr)
        if r.get("stderr"):
            # print last few lines of stderr for debugging
            last = r["stderr"].strip().splitlines()[-5:]
            for ln in last:
                print(f"    {ln}", file=sys.stderr)
        results.append(r)
    print(json.dumps(results, indent=2))

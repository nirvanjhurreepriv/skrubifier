import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skrubifier.analyzer import analyze_estimator, analyze_source
from skrubifier.validator import static_check

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_titanic_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "titanic_source", EXAMPLES / "01_titanic" / "source_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_estimator_analyzer_captures_column_groups():
    mod = _load_titanic_pipeline()
    ir = analyze_estimator(mod.PIPELINE, mod.FEATURE_COLUMNS, mod.TARGET_COLUMN, mod.TASK)
    group_names = {g.name for g in ir.column_groups}
    assert group_names == {"preprocessor.num", "preprocessor.cat"}
    assert ir.estimator.class_or_func == "RandomForestClassifier"
    assert ir.estimator.params["n_estimators"] == 300


def test_estimator_analyzer_tags_every_step_in_a_branch():
    # regression test for the nested-sub-pipeline applies_to bug
    mod = _load_titanic_pipeline()
    ir = analyze_estimator(mod.PIPELINE, mod.FEATURE_COLUMNS, mod.TARGET_COLUMN, mod.TASK)
    num_steps = [s for s in ir.steps if s.applies_to == "preprocessor.num"]
    cat_steps = [s for s in ir.steps if s.applies_to == "preprocessor.cat"]
    assert {s.class_or_func for s in num_steps} == {"SimpleImputer", "StandardScaler"}
    assert {s.class_or_func for s in cat_steps} == {"SimpleImputer", "OneHotEncoder"}


def test_ir_is_json_serializable():
    mod = _load_titanic_pipeline()
    ir = analyze_estimator(mod.PIPELINE, mod.FEATURE_COLUMNS, mod.TARGET_COLUMN, mod.TASK)
    json.dumps(ir.to_dict(), default=str)  # must not raise


def test_ast_fallback_finds_groupby_and_merge_steps():
    ir = analyze_source(str(EXAMPLES / "03_credit_fraud_multitable" / "source_pipeline.py"))
    kinds = [s.kind for s in ir.steps]
    assert "groupby_agg" in kinds
    assert "join" in kinds
    assert any(s.source_snippet for s in ir.steps)  # ground-truth code preserved
    assert ir.notes  # AST-fallback warning present


def test_static_check_accepts_all_hand_written_converted_scripts():
    for name in [
        "01_titanic", "02_house_prices_xgb", "03_credit_fraud_multitable",
        "04_nyc_taxi_fare", "05_otto_group", "06_allstate_claims_severity",
        "07_random_acts_of_pizza", "08_spooky_author",
        "09_home_credit", "10_santander",
    ]:
        code = (EXAMPLES / name / "converted_dataops.py").read_text()
        report = static_check(code)
        assert report["ok"], f"{name}: {report['detail']}"


def test_static_check_rejects_eager_sklearn_masquerading_as_dataops():
    fake_code = '''
import skrub  # imported but unused, classic false-positive risk
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
'''
    report = static_check(fake_code)
    assert not report["ok"]


def test_static_check_rejects_syntax_errors():
    report = static_check("def broken(:\n    pass")
    assert not report["ok"]
    assert "SyntaxError" in report["detail"]


def test_estimator_analyzer_on_nyc_taxi_example():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "taxi_source", EXAMPLES / "04_nyc_taxi_fare" / "source_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ir = analyze_estimator(mod.PIPELINE, mod.FEATURE_COLUMNS, mod.TARGET_COLUMN, mod.TASK)
    assert ir.estimator.class_or_func == "LinearRegression"
    assert ir.task == "regression"


def test_estimator_analyzer_captures_stacking_base_estimators():
    # regression test: sklearn's nested-estimator params (StackingRegressor's
    # `estimators`/`final_estimator`) aren't JSON-primitive and were
    # previously silently dropped by _params_of; they must now show up as
    # separate StepIR entries instead of vanishing.
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    stack = StackingRegressor(
        estimators=[("rf1", RandomForestRegressor(n_estimators=10)),
                    ("rf2", RandomForestRegressor(n_estimators=20))],
        final_estimator=LinearRegression(),
    )
    pipe = Pipeline([("model", stack)])
    ir = analyze_estimator(pipe, ["a", "b"], "y", "regression")

    base_names = {s.class_or_func for s in ir.steps if "base_estimator" in (s.applies_to or "")}
    meta_names = {s.class_or_func for s in ir.steps if s.applies_to == "StackingRegressor.final_estimator"}
    assert base_names == {"RandomForestRegressor"}
    assert meta_names == {"LinearRegression"}
    assert ir.notes  # analyzer must flag that there's no single skrub equivalent

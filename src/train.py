"""Train classification and regression models, log to MLflow, save artifacts."""
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.data_utils import (add_features, add_targets, clean,
                            get_feature_columns, load_data)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def build_preprocessor(num_cols, cat_cols, scale=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def classifier_candidates():
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=2000), True),
        "RandomForest": (RandomForestClassifier(
            n_estimators=C.RF_N_ESTIMATORS, max_depth=C.RF_MAX_DEPTH,
            min_samples_leaf=C.RF_MIN_SAMPLES_LEAF,
            random_state=C.RANDOM_STATE, n_jobs=-1), False),
    }
    if HAS_XGB:
        models["XGBoost"] = (XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss",
            random_state=C.RANDOM_STATE), False)
    return models


def regressor_candidates():
    models = {
        "LinearRegression": (LinearRegression(), True),
        "RandomForest": (RandomForestRegressor(
            n_estimators=C.RF_N_ESTIMATORS, max_depth=C.RF_MAX_DEPTH,
            min_samples_leaf=C.RF_MIN_SAMPLES_LEAF,
            random_state=C.RANDOM_STATE, n_jobs=-1), False),
    }
    if HAS_XGB:
        models["XGBoost"] = (XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=C.RANDOM_STATE), False)
    return models


def maybe_sample(name, data):
    """Random Forest is slow on very large data, so it trains on a sample."""
    if name == "RandomForest" and len(data) > C.RF_TRAIN_SAMPLE:
        return data.sample(C.RF_TRAIN_SAMPLE, random_state=C.RANDOM_STATE)
    return data


def evaluate_classifier(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_regressor(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def top_features(pipe, n=15):
    names = pipe.named_steps["prep"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        return {}
    order = np.argsort(imp)[::-1][:n]
    return {str(names[i]): float(imp[i]) for i in order}


def log_mlflow(run_name, params, metrics, pipe):
    if not HAS_MLFLOW:
        return
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, "model")


def main():
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = add_targets(add_features(clean(load_data())))
    print(f"Rows after cleaning: {len(df):,}")
    print("Class balance:", df[C.TARGET_CLS].value_counts().to_dict())

    idx_train, idx_test = train_test_split(
        df.index, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE,
        stratify=df[C.TARGET_CLS])
    train, test = df.loc[idx_train], df.loc[idx_test]
    report = {"classification": {}, "regression": {}}

    # ---------- Classification ----------
    num_c, cat_c = get_feature_columns(df, include_price=False)
    cls_cols = num_c + cat_c
    best_cls = None
    for name, (model, scale) in classifier_candidates().items():
        pipe = Pipeline([("prep", build_preprocessor(num_c, cat_c, scale)),
                         ("model", model)])
        fit_df = maybe_sample(name, train)
        pipe.fit(fit_df[cls_cols], fit_df[C.TARGET_CLS])
        y_pred = pipe.predict(test[cls_cols])
        y_proba = pipe.predict_proba(test[cls_cols])[:, 1]
        metrics = evaluate_classifier(test[C.TARGET_CLS], y_pred, y_proba)
        report["classification"][name] = metrics
        log_mlflow(f"cls_{name}", {"model": name, "task": "classification"}, metrics, pipe)
        print(f"[CLS] {name}: {metrics}")
        if best_cls is None or metrics["f1"] > best_cls["metrics"]["f1"]:
            best_cls = {"name": name, "pipe": pipe, "metrics": metrics, "y_pred": y_pred}

    report["best_classifier"] = best_cls["name"]
    report["classification_confusion_matrix"] = confusion_matrix(
        test[C.TARGET_CLS], best_cls["y_pred"]).tolist()
    report["top_features_classifier"] = top_features(best_cls["pipe"])

    # ---------- Regression ----------
    num_r, cat_r = get_feature_columns(df, include_price=True)
    reg_cols = num_r + cat_r
    y_test_reg = test[C.TARGET_REG]

    # Baseline: fixed 8% growth for every property (the rule from the brief).
    baseline_pred = test["Price_in_Lakhs"] * (1 + C.BASE_ANNUAL_GROWTH) ** C.FIVE_YEAR_HORIZON
    report["regression"]["FixedRateBaseline"] = evaluate_regressor(y_test_reg, baseline_pred)

    best_reg = None
    for name, (model, scale) in regressor_candidates().items():
        pipe = Pipeline([("prep", build_preprocessor(num_r, cat_r, scale)),
                         ("model", model)])
        # Learn the growth RATIO (future / current), then multiply by current price.
        # Trees cannot extrapolate beyond the training range, so predicting a ratio
        # keeps predictions sensible for prices outside the training range.
        fit_df = maybe_sample(name, train)
        train_ratio = fit_df[C.TARGET_REG] / fit_df["Price_in_Lakhs"]
        pipe.fit(fit_df[reg_cols], train_ratio)
        pred = pipe.predict(test[reg_cols]) * test["Price_in_Lakhs"]
        metrics = evaluate_regressor(y_test_reg, pred)
        report["regression"][name] = metrics
        log_mlflow(f"reg_{name}", {"model": name, "task": "regression"}, metrics, pipe)
        print(f"[REG] {name}: {metrics}")
        if best_reg is None or metrics["rmse"] < best_reg["metrics"]["rmse"]:
            best_reg = {"name": name, "pipe": pipe, "metrics": metrics}

    report["best_regressor"] = best_reg["name"]
    report["top_features_regressor"] = top_features(best_reg["pipe"])

    # ---------- Save ----------
    joblib.dump(best_cls["pipe"], C.MODELS_DIR / "classifier.joblib")
    joblib.dump(best_reg["pipe"], C.MODELS_DIR / "regressor.joblib")
    with open(C.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved models to {C.MODELS_DIR} and metrics to {C.REPORTS_DIR}/metrics.json")


if __name__ == "__main__":
    main()

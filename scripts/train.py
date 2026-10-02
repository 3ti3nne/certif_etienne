import json
import sys
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from bank import RANDOM_STATE, ROOT, SCENARIOS, build_pipeline, load_data, precision_at_k

SCENARIO = "S3"
CALL_SHARE = 0.10
HISTORY_SHARE = 0.80
MIN_ROC_AUC = 0.65
MODEL_DIR = ROOT / "models"
MLFLOW_MODEL_NAME = "bank-marketing"


def make_model():
    return LogisticRegression(max_iter=1000)


def split_history(df):
    # Rows are in chronological order: the oldest part is the history, the rest plays production
    cut = int(len(df) * HISTORY_SHARE)
    return df.iloc[:cut], df.iloc[cut:]


def train_and_describe(train_df, features):
    X, y = train_df[features], train_df["y"]
    pipeline = build_pipeline(features, make_model())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    description = {
        "threshold": float(np.quantile(proba, 1 - CALL_SHARE)),
        "reference_roc_auc": float(roc_auc_score(y, proba)),
        "reference_pr_auc": float(average_precision_score(y, proba)),
        "reference_precision_at_10": precision_at_k(y, proba, CALL_SHARE),
        "base_rate": float(y.mean()),
        "trained_on_rows": len(train_df),
    }
    pipeline.fit(X, y)
    return pipeline, description


def save_champion(pipeline, description, features):
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    info = {"version": version, "scenario": SCENARIO, "model": type(pipeline[-1]).__name__,
            "features": features, "call_share": CALL_SHARE, **description}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("bank-marketing")
    with mlflow.start_run(run_name=f"champion-{version}"):
        mlflow.log_params({"scenario": SCENARIO, "model": info["model"], "call_share": CALL_SHARE})
        mlflow.log_metrics({k: v for k, v in description.items() if isinstance(v, float)})
        # MLflow 3 serializes with skops, which rejects numpy types unless they are explicitly trusted
        logged = mlflow.sklearn.log_model(sk_model=pipeline, name="model", registered_model_name=MLFLOW_MODEL_NAME,
                                          skops_trusted_types=["numpy.dtype"])
    MlflowClient().set_registered_model_alias(MLFLOW_MODEL_NAME, "champion", logged.registered_model_version)
    return info


def main():
    history, _ = split_history(load_data())
    features = SCENARIOS[SCENARIO]
    pipeline, description = train_and_describe(history, features)
    print(f"Reference ROC-AUC (out-of-fold): {description['reference_roc_auc']:.3f}")
    if description["reference_roc_auc"] < MIN_ROC_AUC:
        print(f"FAILED: ROC-AUC below the {MIN_ROC_AUC} floor. Model rejected.")
        sys.exit(1)
    info = save_champion(pipeline, description, features)
    print(f"Champion {info['version']} saved (call threshold = {info['threshold']:.3f}).")


if __name__ == "__main__":
    main()

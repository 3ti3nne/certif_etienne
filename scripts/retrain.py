import json
import shutil

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from bank import NUMERIC, ROOT, age_group, load_data, psi, rates_by_group
from scripts.train import MODEL_DIR, save_champion, split_history, train_and_describe

MAX_ROC_AUC_DROP = 0.05
MAX_PSI = 0.20
MAX_CALIBRATION_GAP = 0.05
MAX_FAIRNESS_DROP = 0.05


def check_triggers(champion, info, history, new) -> list[str]:
    features = info["features"]
    proba = champion.predict_proba(new[features])[:, 1]
    triggers = []

    roc_auc = roc_auc_score(new["y"], proba)
    if roc_auc < info["reference_roc_auc"] - MAX_ROC_AUC_DROP:
        triggers.append(f"performance: ROC-AUC {roc_auc:.3f} < reference {info['reference_roc_auc']:.3f} - {MAX_ROC_AUC_DROP}")

    for feature in [f for f in features if f in NUMERIC]:
        value = psi(history[feature], new[feature])
        if value > MAX_PSI:
            triggers.append(f"drift: PSI({feature}) = {value:.2f} > {MAX_PSI}")

    gap = abs(proba.mean() - new["y"].mean())
    if gap > MAX_CALIBRATION_GAP:
        triggers.append(f"calibration: predicted {proba.mean():.1%} vs actual {new['y'].mean():.1%}")
    return triggers


def evaluate(model, threshold, data, features) -> dict:
    proba = model.predict_proba(data[features])[:, 1]
    called = (proba >= threshold).astype(int)
    fairness = rates_by_group(data["y"], called, age_group(data["age"]))
    return {"pr_auc": float(average_precision_score(data["y"], proba)),
            "disparate_impact_min": float(fairness["disparate_impact"].min())}


def should_promote(champion_scores: dict, challenger_scores: dict) -> bool:
    is_better = challenger_scores["pr_auc"] > champion_scores["pr_auc"]
    is_fair_enough = (challenger_scores["disparate_impact_min"]
                      >= champion_scores["disparate_impact_min"] - MAX_FAIRNESS_DROP)
    return is_better and is_fair_enough


def main():
    champion = joblib.load(MODEL_DIR / "model.joblib")
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    history, new = split_history(load_data())

    triggers = check_triggers(champion, info, history, new)
    if not triggers:
        print("No trigger fired: the champion stays.")
        return
    print("Triggers fired:\n- " + "\n- ".join(triggers))

    half = len(new) // 2
    new_for_training, holdout = new.iloc[:half], new.iloc[half:]
    features = info["features"]
    challenger, description = train_and_describe(pd.concat([history, new_for_training]), features)

    champion_scores = evaluate(champion, info["threshold"], holdout, features)
    challenger_scores = evaluate(challenger, description["threshold"], holdout, features)
    print(f"Champion   : {champion_scores}\nChallenger : {challenger_scores}")

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("bank-marketing")
    with mlflow.start_run(run_name="champion-vs-challenger"):
        mlflow.log_param("triggers", " | ".join(triggers)[:500])
        mlflow.log_metrics({f"champion_{k}": v for k, v in champion_scores.items()})
        mlflow.log_metrics({f"challenger_{k}": v for k, v in challenger_scores.items()})

    if not should_promote(champion_scores, challenger_scores):
        print("Challenger rejected: the champion stays.")
        return
    archive = MODEL_DIR / "archive"
    archive.mkdir(exist_ok=True)
    shutil.copy(MODEL_DIR / "model.joblib", archive / f"model_{info['version']}.joblib")
    new_info = save_champion(challenger, description, features)
    print(f"Challenger promoted: new champion {new_info['version']}.")


if __name__ == "__main__":
    main()

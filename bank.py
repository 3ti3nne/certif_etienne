from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
RANDOM_STATE = 42
TARGET = "y"

NUMERIC = ["age", "duration", "campaign", "previous",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
ALL_FEATURES = NUMERIC + ["pdays"] + CATEGORICAL

SENSITIVE = ["age", "job", "marital", "education", "default"]
CAMPAIGN_HISTORY = ["campaign", "pdays", "previous", "poutcome"]


def without(features, to_remove):
    return [f for f in features if f not in to_remove]


SCENARIOS = {
    "S1": ALL_FEATURES,
    "S2": without(ALL_FEATURES, ["duration"]),
    "S3": without(ALL_FEATURES, ["duration"] + SENSITIVE),
    "S4": without(ALL_FEATURES, ["duration"] + SENSITIVE + CAMPAIGN_HISTORY),
}


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df


def build_pipeline(features, model):
    numeric = [f for f in features if f in NUMERIC]
    categorical = [f for f in features if f in CATEGORICAL]
    steps = []
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    if "pdays" in features:
        # pdays = 999 is a "never contacted" code: impute it and add a 0/1 flag
        pdays = make_pipeline(SimpleImputer(missing_values=999, strategy="median", add_indicator=True),
                              StandardScaler())
        steps.append(("pdays", pdays, ["pdays"]))
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return Pipeline([("preparation", ColumnTransformer(steps)), ("model", model)])


def precision_at_k(y_true, proba, k=0.10):
    n = max(1, int(len(proba) * k))
    best = np.argsort(np.asarray(proba))[::-1][:n]
    return float(np.asarray(y_true)[best].mean())


def age_group(age):
    return pd.cut(age, bins=[0, 30, 45, 60, 200], labels=["<30", "30-44", "45-59", "60+"], right=False)


def rates_by_group(y_true, called, groups):
    df = pd.DataFrame({"actual": np.asarray(y_true), "called": np.asarray(called), "group": np.asarray(groups)})
    table = df.groupby("group").agg(count=("actual", "size"),
                                    subscription_rate=("actual", "mean"),
                                    call_rate=("called", "mean"))
    subscribers = df[df["actual"] == 1].groupby("group")["called"].mean()
    non_subscribers = df[df["actual"] == 0].groupby("group")["called"].mean()
    table["false_negative_rate"] = 1 - subscribers
    table["false_positive_rate"] = non_subscribers
    table["disparate_impact"] = table["call_rate"] / table["call_rate"].max()
    return table


def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))

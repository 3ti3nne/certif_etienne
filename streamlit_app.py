import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from bank import load_data, psi

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
LOG_PATH = ROOT / "logs" / "api.jsonl"
FEEDBACK_PATH = ROOT / "data" / "feedback.csv"
INDICATORS = ["campaign", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def score_page():
    st.title("Should we call this client?")
    with st.form("client"):
        col1, col2, col3 = st.columns(3)
        profile = {
            "client_id": col1.text_input("Client ID (pseudonymized)", "c-0001"),
            "housing": col1.selectbox("Housing loan?", ["no", "yes", "unknown"]),
            "loan": col1.selectbox("Personal loan?", ["no", "yes", "unknown"]),
            "contact": col1.selectbox("Phone type", ["cellular", "telephone"]),
            "month": col2.selectbox("Call month", MONTHS, index=4),
            "day_of_week": col2.selectbox("Call day", ["mon", "tue", "wed", "thu", "fri"]),
            "campaign": col2.number_input("Contacts in this campaign (including this one)", 1, 100, 1),
            "pdays": col2.number_input("Days since last contact (999 = never)", 0, 999, 999),
            "previous": col2.number_input("Contacts in previous campaigns", 0, 100, 0),
            "poutcome": col3.selectbox("Previous campaign outcome", ["nonexistent", "failure", "success"]),
            "emp_var_rate": col3.number_input("Employment variation rate", value=-1.8),
            "cons_price_idx": col3.number_input("Consumer price index", value=92.9),
            "cons_conf_idx": col3.number_input("Consumer confidence index", value=-46.2),
            "euribor3m": col3.number_input("3-month Euribor", value=1.3),
            "nr_employed": col3.number_input("Number of employees (thousands)", value=5099.1),
        }
        submitted = st.form_submit_button("Score")
    if not submitted:
        return
    try:
        response = requests.post(f"{API_URL}/predict", json=profile, timeout=5)
    except requests.ConnectionError:
        st.error("The API is not responding: start `uvicorn api.main:app`.")
        return
    if response.status_code != 200:
        st.error(f"Rejected by the API ({response.status_code}): {response.json()['detail']}")
        return
    result = response.json()
    st.metric("Subscription probability", f"{result['probability']:.0%}")
    show = {"call": st.success, "do_not_call": st.info, "advisor_review": st.warning}
    show[result["decision"]](f"Decision: {result['decision'].replace('_', ' ')} "
                             f"(threshold {result['threshold']:.0%}, model {result['model_version']})")


def dashboard_page():
    st.title("Campaign dashboard")
    if not LOG_PATH.exists():
        st.info("No activity yet.")
        return
    records = [json.loads(line)["record"] for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    logs = pd.DataFrame([r["extra"] for r in records])
    calls = logs[logs["kind"] == "request"]
    predictions = logs[logs["kind"] == "prediction"]

    st.subheader("Service health")
    c1, c2, c3 = st.columns(3)
    c1.metric("Requests received", len(calls))
    c2.metric("Error rate", f"{(calls['status'] >= 500).mean():.1%}")
    c3.metric("Response time (p95)", f"{calls['latency_ms'].quantile(0.95):.0f} ms")

    if predictions.empty:
        return
    st.subheader("Commercial activity")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients scored", len(predictions))
    c2.metric("Recommended for a call", f"{(predictions['decision'] == 'call').mean():.0%}",
              help="Expected: about 10%. Much more means the threshold no longer fits.")
    if FEEDBACK_PATH.exists():
        feedback = pd.read_csv(FEEDBACK_PATH)
        last_scores = predictions.drop_duplicates("client_id", keep="last")[["client_id", "probability"]]
        answered = feedback[feedback["outcome"].isin(["subscribed", "declined"])].merge(last_scores, on="client_id")
        if not answered.empty:
            c3.metric("Actual subscription (called clients)", f"{(answered['outcome'] == 'subscribed').mean():.0%}",
                      delta=f"predicted: {answered['probability'].mean():.0%}", delta_color="off")

    st.subheader("Do scored clients still look like the training data?")
    inputs = pd.DataFrame(list(predictions["inputs"]))
    data = load_data()
    history = data.iloc[: int(len(data) * 0.8)]
    drift = pd.DataFrame({"indicator": INDICATORS, "PSI": [psi(history[c], inputs[c]) for c in INDICATORS]})
    drift["status"] = np.select([drift["PSI"] > 0.2, drift["PSI"] > 0.1], ["🔴 strong drift", "🟠 to watch"], "🟢 stable")
    st.dataframe(drift.round(2), hide_index=True)


st.set_page_config(page_title="Term deposit campaign", layout="wide")
page = st.sidebar.radio("Page", ["Score a client", "Dashboard"])
score_page() if page == "Score a client" else dashboard_page()

import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import api.main as main
from bank import SCENARIOS, build_pipeline, load_data

VALID_CLIENT = {
    "client_id": "c-001", "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "campaign": 1, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp_var_rate": -1.8, "cons_price_idx": 92.9,
    "cons_conf_idx": -46.2, "euribor3m": 1.3, "nr_employed": 5099.1,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    features = SCENARIOS["S3"]
    data = load_data().sample(3000, random_state=0)
    model = build_pipeline(features, LogisticRegression(max_iter=1000)).fit(data[features], data["y"])
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "model_info.json").write_text(json.dumps({"version": "test", "threshold": 0.5, "features": features}))
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(main, "INFO_PATH", tmp_path / "model_info.json")
    monkeypatch.setattr(main, "LOG_PATH", tmp_path / "api.jsonl")
    monkeypatch.setattr(main, "FEEDBACK_PATH", tmp_path / "feedback.csv")
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_is_ok_when_model_is_loaded(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test"}


def test_health_is_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(main, "LOG_PATH", tmp_path / "api.jsonl")
    with TestClient(main.app) as test_client:
        assert test_client.get("/health").status_code == 503


def test_predict_returns_a_probability_and_a_decision(client):
    body = client.post("/predict", json=VALID_CLIENT).json()
    assert 0 <= body["probability"] <= 1
    assert body["decision"] in {"call", "do_not_call", "advisor_review"}


@pytest.mark.parametrize("bad_field", [{"month": "january"}, {"campaign": 0}, {"duration": 120}, {"age": 40}])
def test_predict_rejects_invalid_or_forbidden_fields(client, bad_field):
    assert client.post("/predict", json={**VALID_CLIENT, **bad_field}).status_code == 422


def test_each_prediction_is_logged_with_inputs_output_and_time(client):
    client.post("/predict", json=VALID_CLIENT)
    records = [json.loads(line)["record"] for line in main.LOG_PATH.read_text(encoding="utf-8").splitlines()]
    prediction = next(r for r in records if r["extra"].get("kind") == "prediction")
    assert prediction["extra"]["client_id"] == "c-001"
    assert "inputs" in prediction["extra"] and "probability" in prediction["extra"]
    assert prediction["time"]["timestamp"] > 0


def test_opt_out_feedback_blocks_future_calls(client):
    assert client.post("/feedback", json={"client_id": "c-001", "outcome": "opted_out"}).status_code == 201
    assert client.post("/predict", json=VALID_CLIENT).status_code == 403

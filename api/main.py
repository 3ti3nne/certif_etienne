import csv
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import ClientProfile, Feedback, Prediction

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"
LOG_PATH = ROOT / "logs" / "api.jsonl"
FEEDBACK_PATH = ROOT / "data" / "feedback.csv"

GREY_ZONE = 0.02

RENAME = {"emp_var_rate": "emp.var.rate", "cons_price_idx": "cons.price.idx",
          "cons_conf_idx": "cons.conf.idx", "nr_employed": "nr.employed"}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    handler = logger.add(LOG_PATH, serialize=True, rotation="10 MB", retention="395 days")
    if MODEL_PATH.exists():
        info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
        api_features = {RENAME.get(f, f) for f in ClientProfile.model_fields if f != "client_id"}
        if api_features != set(info["features"]):
            raise RuntimeError(f"API schema does not match the model: {sorted(api_features ^ set(info['features']))}")
        state["model"], state["info"] = joblib.load(MODEL_PATH), info
    yield
    state.clear()
    logger.remove(handler)


app = FastAPI(title="Term deposit campaign scoring", lifespan=lifespan)


@app.middleware("http")
async def log_every_request(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    logger.bind(kind="request", method=request.method, path=request.url.path, status=response.status_code,
                latency_ms=round((time.perf_counter() - start) * 1000, 1)).info("request")
    return response


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logger.bind(kind="error", path=request.url.path).exception("unexpected error")
    return JSONResponse(status_code=500, content={"detail": "Internal error, the incident has been logged"})


def opted_out_clients() -> set:
    if not FEEDBACK_PATH.exists():
        return set()
    feedback = pd.read_csv(FEEDBACK_PATH)
    return set(feedback.loc[feedback["outcome"] == "opted_out", "client_id"].astype(str))


@app.get("/health")
def health() -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": state["info"]["version"]}


@app.post("/predict")
def predict(client: ClientProfile) -> Prediction:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if client.client_id in opted_out_clients():
        raise HTTPException(status_code=403, detail="Client opted out of telemarketing: do not call")

    row = pd.DataFrame([client.model_dump(exclude={"client_id"})]).rename(columns=RENAME)
    probability = float(state["model"].predict_proba(row)[0, 1])
    threshold = state["info"]["threshold"]
    if abs(probability - threshold) < GREY_ZONE:
        decision = "advisor_review"
    elif probability >= threshold:
        decision = "call"
    else:
        decision = "do_not_call"

    result = Prediction(client_id=client.client_id, probability=round(probability, 4), decision=decision,
                        threshold=round(threshold, 4), model_version=state["info"]["version"])
    logger.bind(kind="prediction", inputs=row.iloc[0].to_dict(), **result.model_dump()).info("prediction")
    return result


@app.post("/feedback", status_code=201)
def feedback(fb: Feedback) -> dict:
    is_new_file = not FEEDBACK_PATH.exists()
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "client_id", "outcome"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), fb.client_id, fb.outcome])
    logger.bind(kind="feedback", **fb.model_dump()).info("feedback")
    return {"status": "recorded"}

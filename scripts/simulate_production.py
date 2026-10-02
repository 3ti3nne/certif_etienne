import json

import requests

from bank import load_data
from scripts.train import MODEL_DIR, split_history

API_URL = "http://127.0.0.1:8000"
N_CLIENTS = 300
TO_API_NAMES = {"emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx",
                "cons.conf.idx": "cons_conf_idx", "nr.employed": "nr_employed"}


def main():
    features = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))["features"]
    _, recent = split_history(load_data())
    sample = recent.sample(N_CLIENTS, random_state=0)
    profiles = sample[features].rename(columns=TO_API_NAMES).to_dict(orient="records")
    called = 0
    for index, profile, subscribed in zip(sample.index, profiles, sample["y"]):
        client_id = f"sim-{index}"
        result = requests.post(f"{API_URL}/predict", json={"client_id": client_id, **profile}, timeout=5).json()
        if result["decision"] == "call":
            called += 1
            outcome = "subscribed" if subscribed == 1 else "declined"
            requests.post(f"{API_URL}/feedback", json={"client_id": client_id, "outcome": outcome}, timeout=5)
    print(f"{N_CLIENTS} clients scored, {called} recommended for a call, feedback recorded.")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from bank import SCENARIOS, SENSITIVE, build_pipeline, precision_at_k, psi, rates_by_group


def test_s2_removes_duration_the_leakage():
    assert "duration" in SCENARIOS["S1"]
    assert "duration" not in SCENARIOS["S2"]


@pytest.mark.parametrize("scenario", ["S3", "S4"])
def test_sensitive_variables_are_removed(scenario):
    assert not set(SENSITIVE) & set(SCENARIOS[scenario])


def test_pipeline_turns_pdays_999_into_never_contacted_flag():
    X = pd.DataFrame({"pdays": [999, 3, 999, 6], "poutcome": ["nonexistent", "success", "nonexistent", "failure"]})
    pipeline = build_pipeline(["pdays", "poutcome"], LogisticRegression()).fit(X, [0, 1, 0, 1])
    prepared = pipeline.named_steps["preparation"].transform(X)
    # pdays gives 2 columns (value + "never contacted" flag), poutcome gives 3 one-hot columns
    assert prepared.shape == (4, 5)


def test_pipeline_ignores_a_category_never_seen_in_training():
    X = pd.DataFrame({"pdays": [999, 3], "poutcome": ["nonexistent", "success"]})
    pipeline = build_pipeline(["pdays", "poutcome"], LogisticRegression()).fit(X, [0, 1])
    new_client = pd.DataFrame({"pdays": [999], "poutcome": ["never_seen"]})
    assert pipeline.predict_proba(new_client).shape == (1, 2)


def test_precision_at_k_looks_only_at_the_best_scores():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], k=0.5) == 0.5


def test_rates_by_group_counts_selection_and_errors():
    table = rates_by_group(y_true=[1, 0, 1, 0, 1, 0], called=[1, 1, 0, 0, 1, 0], groups=["a", "a", "a", "b", "b", "b"])
    assert table.loc["a", "call_rate"] == pytest.approx(2 / 3)
    assert table.loc["a", "share_of_calls"] == pytest.approx(2 / 3)
    assert table.loc["a", "precision"] == pytest.approx(0.5)
    assert table.loc["a", "false_negative_rate"] == pytest.approx(0.5)
    assert table.loc["a", "false_positive_rate"] == pytest.approx(1.0)
    assert table.loc["b", "disparate_impact"] == pytest.approx(0.5)


def test_psi_is_zero_for_same_distribution_and_high_for_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert psi(reference, reference) == pytest.approx(0, abs=1e-6)
    assert psi(reference, reference + 2) > 0.2

from scripts.retrain import should_promote

CHAMPION = {"pr_auc": 0.40, "disparate_impact_min": 0.50}


def test_better_and_as_fair_challenger_is_promoted():
    assert should_promote(CHAMPION, {"pr_auc": 0.45, "disparate_impact_min": 0.50})


def test_challenger_not_better_is_refused():
    assert not should_promote(CHAMPION, {"pr_auc": 0.39, "disparate_impact_min": 0.60})


def test_better_but_less_fair_challenger_is_refused():
    assert not should_promote(CHAMPION, {"pr_auc": 0.50, "disparate_impact_min": 0.30})

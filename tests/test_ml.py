import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml import (  # noqa: E402
    FEATURES,
    build_panel,
    feature_importance,
    ml_backtest,
    walk_forward_predict,
)

FAST = {"n_estimators": 20}  # modèles légers pour des tests rapides


@pytest.fixture
def prices():
    """12 actions, ~7 ans de prix quotidiens (marches aléatoires)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", "2021-12-31", freq="D")
    tickers = [f"T{i}" for i in range(12)]
    data = {tk: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))) for tk in tickers}
    return pd.DataFrame(data, index=idx)


def test_build_panel_columns_and_no_nan(prices):
    panel = build_panel(prices)
    assert list(panel.columns) == FEATURES + ["target"]
    assert panel.index.names == ["date", "ticker"]
    assert not panel.isna().any().any()


def test_target_is_next_month_return(prices):
    # La cible d'un (mois, ticker) doit égaler le rendement mensuel du mois suivant.
    panel = build_panel(prices)
    monthly = prices.resample("ME").last()
    mret = monthly.pct_change(fill_method=None)
    date, ticker = panel.index[0]
    nxt = mret.index[mret.index.get_loc(date) + 1]
    assert panel.loc[(date, ticker), "target"] == pytest.approx(mret.loc[nxt, ticker])


def test_walk_forward_only_predicts_after_training_window(prices):
    panel = build_panel(prices)
    preds = walk_forward_predict(panel, min_train_months=36, params=FAST)
    dates = panel.index.get_level_values("date").unique().sort_values()
    pred_dates = preds.index.get_level_values("date").unique()
    assert pred_dates.min() == dates[36]  # rien avant la fin du warm-up
    assert np.isfinite(preds.to_numpy()).all()


def test_ml_backtest_runs_and_is_clean(prices):
    panel = build_panel(prices)
    preds = walk_forward_predict(panel, min_train_months=36, params=FAST)
    bt = ml_backtest(prices, preds, frac=0.3)
    assert list(bt.columns) == ["strategy", "benchmark"]
    assert len(bt) > 0
    assert not bt.isna().any().any()


def test_feature_importance_covers_all_features(prices):
    imp = feature_importance(build_panel(prices), params=FAST)
    assert set(imp.index) == set(FEATURES)
    assert imp.sum() > 0

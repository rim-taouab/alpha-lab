import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest import (  # noqa: E402
    backtest_momentum,
    equity_curve,
    monthly_returns,
    performance,
    performance_table,
)


@pytest.fixture
def prices():
    """3 ans de prix quotidiens : A +1 %/mois, B plat, C -0,5 %/mois."""
    idx = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    months = (idx.year - 2022) * 12 + (idx.month - 1)
    return pd.DataFrame(
        {
            "A": 100 * 1.01 ** months,
            "B": 50.0,
            "C": 80 * 0.995 ** months,
        },
        index=idx,
    )


def test_monthly_returns_values(prices):
    mret = monthly_returns(prices)
    assert mret["A"].dropna().iloc[-1] == pytest.approx(0.01)
    assert mret["C"].dropna().iloc[-1] == pytest.approx(-0.005)


def test_strategy_holds_the_winner(prices):
    # frac=0.34 sur 3 actions -> on garde 1 action : A (toujours le meilleur momentum).
    bt = backtest_momentum(prices, frac=0.34, cost_bps=0)
    # une fois A détenu en continu, le turnover est nul et strat == rendement de A.
    assert bt["strategy"].iloc[-1] == pytest.approx(0.01)


def test_benchmark_is_equal_weight_mean(prices):
    bt = backtest_momentum(prices, frac=0.34, cost_bps=0)
    mret = monthly_returns(prices)
    expected = mret.mean(axis=1).loc[bt.index[-1]]
    assert bt["benchmark"].iloc[-1] == pytest.approx(expected)


def test_costs_reduce_returns(prices):
    free = backtest_momentum(prices, frac=0.34, cost_bps=0)["strategy"]
    costly = backtest_momentum(prices, frac=0.34, cost_bps=50)["strategy"]
    assert (costly <= free + 1e-12).all()
    assert costly.sum() < free.sum()  # au moins un rebalancement payant


def test_no_lookahead_warmup_trimmed(prices):
    bt = backtest_momentum(prices, frac=0.34)
    # 12 mois de warm-up momentum -> le backtest commence après.
    assert bt.index[0] > prices.index[0] + pd.DateOffset(months=11)
    assert not bt.isna().any().any()


def test_equity_curve_monotone_for_winner(prices):
    bt = backtest_momentum(prices, frac=0.34, cost_bps=0)
    eq = equity_curve(bt["strategy"])
    assert (eq.diff().dropna() >= -1e-12).all()  # A monte -> courbe croissante
    assert eq.iloc[-1] > 1


def test_performance_keys_and_signs(prices):
    bt = backtest_momentum(prices, frac=0.34, cost_bps=0)
    perf = performance(bt["strategy"])
    assert set(perf.index) == {
        "rendement_annuel",
        "volatilite_annuelle",
        "sharpe",
        "max_drawdown",
    }
    assert perf["rendement_annuel"] > 0
    assert perf["max_drawdown"] <= 0


def test_performance_table_shape(prices):
    tbl = performance_table(backtest_momentum(prices, frac=0.34))
    assert list(tbl.index) == ["strategy", "benchmark"]
    assert tbl.shape == (2, 4)

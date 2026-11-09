import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analytics import (  # noqa: E402
    annualized_return,
    annualized_vol,
    daily_returns,
    drawdown,
    log_returns,
    max_drawdown,
    sharpe_ratio,
)
from ingest import save_to_duckdb, simulate_prices  # noqa: E402


@pytest.fixture
def prices():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame({"A": [100.0, 110.0, 99.0, 108.9], "B": [50.0, 50.0, 50.0, 50.0]}, index=idx)


def test_daily_returns(prices):
    r = daily_returns(prices)
    assert len(r) == 3
    assert r["A"].tolist() == pytest.approx([0.10, -0.10, 0.10])
    assert (r["B"] == 0).all()


def test_log_returns_sum_to_total(prices):
    lr = log_returns(prices)
    assert lr["A"].sum() == pytest.approx(np.log(108.9 / 100))


def test_annualized_return_constant_growth():
    r = pd.Series([0.001] * 252)
    assert annualized_return(r) == pytest.approx(1.001**252 - 1)


def test_annualized_vol_zero_for_flat(prices):
    assert annualized_vol(daily_returns(prices))["B"] == pytest.approx(0)


def test_drawdown_and_max(prices):
    r = daily_returns(prices)["A"]
    dd = drawdown(r)
    assert (dd <= 0).all()
    assert max_drawdown(r) == pytest.approx(99 / 110 - 1)


def test_sharpe_sign():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 1000))
    assert sharpe_ratio(r) > 0


def test_simulation_and_duckdb_roundtrip(tmp_path):
    import duckdb

    df = simulate_prices(["X", "Y"], "2024-01-01")
    db = tmp_path / "test.duckdb"
    save_to_duckdb(df, db)
    with duckdb.connect(str(db)) as con:
        n_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
        n_dupes = con.execute(
            "SELECT COUNT(*) FROM (SELECT date, ticker FROM prices GROUP BY 1, 2 HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        min_price = con.execute("SELECT MIN(close) FROM prices").fetchone()[0]
    assert n_tickers == 2
    assert n_dupes == 0
    assert min_price > 0

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest import backtest_momentum  # noqa: E402
from robustness import parameter_sweep, sharpe_ci, split_performance  # noqa: E402


@pytest.fixture
def prices():
    """4 ans de prix quotidiens : A monte, B plat, C baisse légèrement."""
    idx = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    months = (idx.year - 2021) * 12 + (idx.month - 1)
    return pd.DataFrame(
        {"A": 100 * 1.01**months, "B": 50.0, "C": 80 * 0.997**months},
        index=idx,
    )


def test_sharpe_ci_brackets_point_estimate():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, 120))
    ci = sharpe_ci(r)
    assert ci["ci_bas"] < ci["sharpe"] < ci["ci_haut"]
    assert ci["n_mois"] == 120


def test_sharpe_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)
    short = sharpe_ci(pd.Series(rng.normal(0.01, 0.04, 36)))
    long = sharpe_ci(pd.Series(rng.normal(0.01, 0.04, 360)))
    width = lambda ci: ci["ci_haut"] - ci["ci_bas"]
    assert width(long) < width(short)


def test_parameter_sweep_grid_shape(prices):
    sweep = parameter_sweep(prices, lookbacks=(6, 12), fracs=(0.34, 0.67))
    assert len(sweep) == 4  # 2 x 2
    assert set(sweep.columns) == {
        "lookback",
        "frac",
        "rendement_annuel",
        "sharpe",
        "max_drawdown",
    }


def test_split_performance_structure(prices):
    bt = backtest_momentum(prices, frac=0.34)
    sp = split_performance(bt, split="2023-01-01")
    assert set(sp.index.get_level_values(0)) == {"plein", "in_sample", "hors_echantillon"}
    assert "sharpe" in sp.columns

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from momentum import momentum, to_monthly, top_fraction  # noqa: E402


@pytest.fixture
def prices():
    """15 mois de prix quotidiens : A croît de 1 % par mois, B est plat."""
    idx = pd.date_range("2023-01-01", "2024-03-31", freq="D")  # 15 fins de mois
    months = (idx.year - 2023) * 12 + (idx.month - 1)
    a = 100 * 1.01 ** months  # +1 %/mois composé
    return pd.DataFrame({"A": a, "B": 50.0}, index=idx)


def test_to_monthly_is_month_end_last(prices):
    m = to_monthly(prices)
    assert m.index.is_month_end.all()
    assert len(m) == 15  # jan 2023 -> mar 2024


def test_momentum_12_1_value(prices):
    m = momentum(prices, lookback=12, gap=1)
    monthly = to_monthly(prices)
    # Signal du dernier mois t = M[t-1] / M[t-12] - 1, soit iloc[-2] / iloc[-13].
    attendu = monthly["A"].iloc[-2] / monthly["A"].iloc[-13] - 1
    assert m["A"].iloc[-1] == pytest.approx(attendu)


def test_momentum_warmup_is_nan(prices):
    m = momentum(prices, lookback=12, gap=1)
    # Pas assez d'historique sur les 12 premiers mois -> NaN.
    assert m["A"].iloc[:12].isna().all()
    assert m["A"].iloc[-1] == pytest.approx(m["A"].dropna().iloc[-1])


def test_momentum_ranks_winner_above_loser(prices):
    m = momentum(prices)
    last = m.iloc[-1]
    assert last["A"] > last["B"]  # A monte, B est plat


def test_momentum_invalid_params(prices):
    with pytest.raises(ValueError):
        momentum(prices, lookback=12, gap=0)
    with pytest.raises(ValueError):
        momentum(prices, lookback=1, gap=1)


def test_top_fraction_picks_winner(prices):
    mask = top_fraction(momentum(prices), frac=0.5)
    last = mask.iloc[-1]
    assert last["A"] and not last["B"]  # top 50 % de 2 actions = la meilleure


def test_top_fraction_all_false_when_no_signal(prices):
    mask = top_fraction(momentum(prices), frac=0.2)
    assert not mask.iloc[0].any()  # premier mois : aucun signal

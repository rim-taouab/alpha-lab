"""Backtest d'une stratégie momentum à rebalancement mensuel.

Principe (sans biais de look-ahead) :
    - fin de mois t : on observe le signal momentum (qui n'utilise que des
      données jusqu'à t-1), on achète le top `frac` à parts égales ;
    - on détient pendant le mois t+1 et on encaisse son rendement ;
    - on rebalance à t+1, en payant des coûts de transaction sur le turnover.

On compare à un benchmark « achat-conservation » équipondéré (toutes les actions,
à parts égales).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import max_drawdown
from momentum import momentum, to_monthly, top_fraction


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements mensuels simples à partir de prix quotidiens."""
    return to_monthly(prices).pct_change(fill_method=None)


def backtest_signal(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    frac: float = 0.2,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Moteur de backtest générique : achète le top `frac` d'un signal mensuel.

    `signal` = DataFrame (fin de mois × tickers) ; plus il est élevé, plus l'action
    est privilégiée. Colonnes renvoyées : ``strategy`` (net de coûts) et
    ``benchmark`` (équipondéré). Sans look-ahead : les poids décidés en t ne
    touchent que le rendement de t+1.
    """
    mask = top_fraction(signal, frac).astype(float)
    # poids = équipondéré sur les actions retenues (0 si aucune ce mois-là)
    weights = mask.div(mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    mret = monthly_returns(prices)
    # poids décidés en t, rendement encaissé en t+1 -> weights.shift(1)
    gross = (weights.shift(1) * mret).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1)
    cost = (cost_bps / 1e4) * turnover.shift(1)  # coût du trade de t, payé sur t+1
    strategy = gross - cost

    benchmark = mret.mean(axis=1)
    out = pd.DataFrame({"strategy": strategy, "benchmark": benchmark})

    invested = weights.shift(1).sum(axis=1).gt(0)
    first = invested.idxmax()  # premier mois réellement investi
    return out.loc[first:].dropna()


def backtest_momentum(
    prices: pd.DataFrame,
    frac: float = 0.2,
    lookback: int = 12,
    gap: int = 1,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Backtest momentum top `frac` (net de coûts en points de base).

    Enveloppe de `backtest_signal` avec le signal momentum `lookback`-`gap`.
    """
    signal = momentum(prices, lookback=lookback, gap=gap)
    return backtest_signal(prices, signal, frac=frac, cost_bps=cost_bps)


def equity_curve(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Valeur d'1 $ investi au départ (produit cumulé des rendements)."""
    return (1 + returns).cumprod()


def performance(returns: pd.Series, periods_per_year: int = 12) -> pd.Series:
    """CAGR, volatilité, Sharpe et pire drawdown d'une série de rendements."""
    n = returns.count()
    cagr = (1 + returns).prod() ** (periods_per_year / n) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    return pd.Series(
        {
            "rendement_annuel": cagr,
            "volatilite_annuelle": vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(returns),
        }
    )


def performance_table(backtest: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """Applique `performance` à chaque colonne (strategy, benchmark)."""
    return backtest.apply(performance, periods_per_year=periods_per_year).T

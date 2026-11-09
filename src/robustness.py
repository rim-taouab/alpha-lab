"""Tests de robustesse du backtest momentum.

Un backtest qui brille sur un seul jeu de paramètres, en échantillon, ne prouve
rien. Ici on vérifie trois choses :
    - sensibilité aux paramètres (lookback, taille du portefeuille) ;
    - stabilité dans le temps (in-sample vs hors-échantillon) ;
    - significativité : le Sharpe est-il distinguable du bruit (intervalle de confiance) ?
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from backtest import backtest_momentum, performance, performance_table


def sharpe_ci(
    returns: pd.Series, periods_per_year: int = 12, confidence: float = 0.95
) -> pd.Series:
    """Sharpe annualisé et son intervalle de confiance (erreur-type de Lo, 2002).

    Sous l'hypothèse de rendements i.i.d., l'écart-type du Sharpe par période vaut
    ``sqrt((1 + 0.5 * SR**2) / n)``. On annualise par ``sqrt(periods_per_year)``.
    Un intervalle qui contient 0 = Sharpe non significatif.
    """
    r = returns.dropna()
    n = len(r)
    sr = r.mean() / r.std()  # Sharpe par période
    se = np.sqrt((1 + 0.5 * sr**2) / n)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    ann = np.sqrt(periods_per_year)
    return pd.Series(
        {
            "sharpe": sr * ann,
            "ci_bas": (sr - z * se) * ann,
            "ci_haut": (sr + z * se) * ann,
            "n_mois": n,
        }
    )


def parameter_sweep(
    prices: pd.DataFrame,
    lookbacks: tuple[int, ...] = (3, 6, 9, 12),
    fracs: tuple[float, ...] = (0.1, 0.2, 0.3),
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Rejoue le backtest sur une grille (lookback × taille de portefeuille).

    Si la performance s'effondre dès qu'on bouge un paramètre, c'est du
    surajustement. On veut une zone de bons résultats, pas un pic isolé.
    """
    rows = []
    for lb in lookbacks:
        for fr in fracs:
            bt = backtest_momentum(prices, frac=fr, lookback=lb, cost_bps=cost_bps)
            perf = performance(bt["strategy"])
            rows.append(
                {
                    "lookback": lb,
                    "frac": fr,
                    "rendement_annuel": perf["rendement_annuel"],
                    "sharpe": perf["sharpe"],
                    "max_drawdown": perf["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def split_performance(backtest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Performance sur toute la période, puis in-sample et hors-échantillon.

    `split` = date de coupure (ex. '2021-01-01'). Si la stratégie marche avant la
    coupure mais pas après, le signal ne tient pas hors échantillon.
    """
    windows = {
        "plein": backtest,
        "in_sample": backtest.loc[:split],
        "hors_echantillon": backtest.loc[split:],
    }
    return pd.concat({k: performance_table(v) for k, v in windows.items()})

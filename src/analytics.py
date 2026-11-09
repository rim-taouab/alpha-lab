"""Fonctions d'analyse de base : rendements, volatilité, drawdown, corrélations."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_prices(db_path: str | Path) -> pd.DataFrame:
    """Charge les prix de clôture au format large : index = date, colonnes = tickers."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("SELECT date, ticker, close FROM prices").df()
    prices = df.pivot(index="date", columns="ticker", values="close").sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements simples : r_t = P_t / P_{t-1} - 1."""
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-rendements : ln(P_t / P_{t-1}). Additifs dans le temps."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def annualized_return(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Rendement annualisé composé (CAGR) à partir de rendements quotidiens."""
    n = returns.count()
    growth = (1 + returns).prod()
    return growth ** (TRADING_DAYS / n) - 1


def annualized_vol(returns: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Volatilité annualisée : écart-type quotidien x racine(252)."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.DataFrame | pd.Series, risk_free: float = 0.0) -> pd.Series | float:
    """Ratio de Sharpe annualisé (taux sans risque annuel en paramètre)."""
    excess = returns - risk_free / TRADING_DAYS
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def drawdown(returns: pd.Series) -> pd.Series:
    """Perte depuis le plus haut historique, à chaque date (valeurs <= 0)."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Pire perte depuis un plus haut, sur toute la période."""
    return drawdown(returns).min()


def summary_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Tableau récapitulatif par ticker."""
    return pd.DataFrame(
        {
            "rendement_annuel": annualized_return(returns),
            "volatilite_annuelle": annualized_vol(returns),
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": returns.apply(max_drawdown),
        }
    ).sort_values("sharpe", ascending=False)

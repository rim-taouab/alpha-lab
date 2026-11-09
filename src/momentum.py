"""Signal de momentum transversal (cross-sectional) 12-1.

Le momentum 12-1 : à la fin de chaque mois, on mesure la performance d'une
action sur les 12 derniers mois en EXCLUANT le mois le plus récent (le « -1 »).
Sauter le dernier mois évite l'effet de retournement à court terme — un titre qui
vient de bondir a tendance à refluer le mois suivant, ce qui brouillerait le signal.

Pipeline type :
    prices (quotidiens) -> to_monthly -> momentum -> top_fraction (portefeuille)
"""

from __future__ import annotations

import pandas as pd


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Rééchantillonne des prix quotidiens en prix de fin de mois (dernier cours du mois)."""
    return prices.resample("ME").last()


def momentum(prices: pd.DataFrame, lookback: int = 12, gap: int = 1) -> pd.DataFrame:
    """Signal momentum `lookback`-`gap`, calculé mois par mois.

    Par défaut 12-1 : rendement entre la fin du mois t-`lookback` et la fin du
    mois t-`gap`, soit ``M.shift(gap) / M.shift(lookback) - 1`` sur les prix
    mensuels ``M``. Index = fin de mois, colonnes = tickers. Les premières lignes
    sont NaN (historique insuffisant).
    """
    if gap < 1:
        raise ValueError("gap doit être >= 1 pour exclure le mois le plus récent")
    if lookback <= gap:
        raise ValueError("lookback doit être > gap")
    monthly = to_monthly(prices)
    return monthly.shift(gap) / monthly.shift(lookback) - 1


def top_fraction(signal: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    """Masque booléen : True pour les actions du meilleur `frac` du signal,
    classées transversalement (mois par mois).

    Ex. `frac=0.2` -> top 20 % de momentum chaque mois. Le seuil est au moins 1
    action. Un mois sans aucun signal valide renvoie une ligne entièrement False.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac doit être dans ]0, 1]")
    valid = signal.notna()
    ranks = signal.rank(axis=1, ascending=False, method="first")
    counts = valid.sum(axis=1)
    cutoff = (counts * frac).clip(lower=1).astype(int)
    return ranks.le(cutoff, axis=0) & valid

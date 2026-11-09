"""Modèle ML (LightGBM) en walk-forward pour classer les actions.

Au lieu d'un signal fixe (le momentum 12-1), on laisse un modèle de gradient
boosting apprendre à prédire le **rendement du mois suivant** à partir de plusieurs
caractéristiques (momentum à divers horizons, retournement court terme, volatilité).

Walk-forward (sans look-ahead) : à chaque fin de mois t, on entraîne le modèle sur
tout le passé (mois < t), on prédit le classement des actions à t, et on détient le
top `frac` pendant t+1. On ré-entraîne chaque mois. On ne teste donc jamais le
modèle sur des données qu'il a vues à l'entraînement.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from backtest import backtest_signal
from momentum import momentum, to_monthly

FEATURES = ["mom_3", "mom_6", "mom_12", "rev_1", "vol_6"]

DEFAULT_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=7,
    max_depth=3,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Construit le panel (date, ticker) des features + cible (rendement t+1).

    Toutes les features sont connues à la fin du mois t ; la cible est le rendement
    du mois t+1, donc jamais utilisée comme entrée. Les lignes incomplètes (warm-up,
    dernier mois sans cible) sont retirées.
    """
    monthly = to_monthly(prices)
    mret = monthly.pct_change(fill_method=None)
    feats = {
        "mom_3": momentum(prices, lookback=3, gap=1),
        "mom_6": momentum(prices, lookback=6, gap=1),
        "mom_12": momentum(prices, lookback=12, gap=1),
        "rev_1": mret,  # rendement du mois courant (effet de retournement)
        "vol_6": mret.rolling(6).std(),  # volatilité récente
    }
    panel = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    panel["target"] = mret.shift(-1).stack()
    panel.index.names = ["date", "ticker"]
    return panel.dropna()


def walk_forward_predict(
    panel: pd.DataFrame, min_train_months: int = 36, params: dict | None = None
) -> pd.Series:
    """Prédit le rendement t+1 mois par mois, modèle ré-entraîné à chaque pas.

    Renvoie une série de prédictions indexée (date, ticker), uniquement pour les
    mois postérieurs à la fenêtre d'entraînement initiale `min_train_months`.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    dates = panel.index.get_level_values("date").unique().sort_values()
    level_date = panel.index.get_level_values("date")
    out = []
    for t in dates[min_train_months:]:
        train = panel[level_date < t]
        test = panel[level_date == t]
        model = lgb.LGBMRegressor(**params)
        model.fit(train[FEATURES], train["target"])
        out.append(pd.Series(model.predict(test[FEATURES]), index=test.index))
    return pd.concat(out)


def ml_backtest(
    prices: pd.DataFrame, predictions: pd.Series, frac: float = 0.2, cost_bps: float = 10.0
) -> pd.DataFrame:
    """Transforme les prédictions en portefeuille (top `frac`) et le backteste."""
    signal = predictions.unstack("ticker")
    return backtest_signal(prices, signal, frac=frac, cost_bps=cost_bps)


def feature_importance(panel: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Importance des features (gain) d'un modèle entraîné sur tout le panel.

    Donne une idée de ce que le modèle exploite ; purement descriptif (entraîné sur
    l'ensemble, pas en walk-forward).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(**params).fit(panel[FEATURES], panel["target"])
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

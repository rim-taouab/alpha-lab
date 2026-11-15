"""Tests de l'attribution factorielle.

On vérifie la mécanique de la régression OLS maison et de l'attribution, sur des
données synthétiques (aucun accès réseau) :
    - la régression retrouve les bêtas et l'alpha d'un processus connu ;
    - un R² parfait et un alpha nul quand la stratégie EST une combinaison exacte
      de facteurs (pas de compétence) ;
    - la structure du tableau d'attribution multi-modèles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factors import MODELS, attribution, factor_regression, simulate_factors  # noqa: E402


def _factors(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.006, 0.045, n),
            "smb": rng.normal(0.0, 0.025, n),
            "hml": rng.normal(0.0, 0.030, n),
            "mom": rng.normal(0.004, 0.040, n),
            "rf": np.full(n, 0.002),
        },
        index=idx,
    )


def test_regression_recovers_known_coefficients():
    fac = _factors()
    rng = np.random.default_rng(1)
    # stratégie = 0.3%/mois d'alpha + expositions connues + bruit léger
    excess = (
        0.003
        + 1.1 * fac["mkt_rf"]
        - 0.4 * fac["hml"]
        + 0.6 * fac["mom"]
        + rng.normal(0, 0.002, len(fac))
    )
    reg = factor_regression(excess, fac[["mkt_rf", "smb", "hml", "mom"]])
    assert np.isclose(reg.loc["alpha", "coef"], 0.003, atol=5e-4)
    assert np.isclose(reg.loc["mkt_rf", "coef"], 1.1, atol=0.05)
    assert np.isclose(reg.loc["hml", "coef"], -0.4, atol=0.05)
    assert np.isclose(reg.loc["mom", "coef"], 0.6, atol=0.05)
    assert reg.attrs["r2"] > 0.98
    assert reg.attrs["n"] == len(fac)


def test_pure_factor_combo_has_zero_alpha_and_perfect_fit():
    fac = _factors()
    # stratégie = combinaison EXACTE de facteurs, sans alpha ni bruit
    excess = 0.9 * fac["mkt_rf"] + 0.5 * fac["mom"]
    reg = factor_regression(excess, fac[["mkt_rf", "smb", "hml", "mom"]])
    assert np.isclose(reg.loc["alpha", "coef"], 0.0, atol=1e-6)
    assert reg.attrs["r2"] > 0.999


def test_adding_momentum_factor_absorbs_alpha():
    """Une stratégie momentum voit son alpha chuter quand on ajoute le facteur Mom."""
    fac = _factors()
    strat = fac["rf"] + 0.7 * fac["mkt_rf"] + 0.6 * fac["mom"]  # edge = exposition Mom
    attr = attribution(strat, fac)
    assert abs(attr.loc["FF3+Mom", "alpha_annuel"]) < abs(attr.loc["FF3", "alpha_annuel"])
    assert attr.loc["FF3+Mom", "r2"] > attr.loc["FF3", "r2"]


def test_attribution_structure():
    fac = _factors()
    strat = fac["rf"] + fac["mkt_rf"]
    attr = attribution(strat, fac)
    assert list(attr.index) == list(MODELS)
    assert set(attr.columns) == {"alpha_annuel", "alpha_tstat", "r2", "n_mois"}


def test_simulate_factors_shape():
    fac = simulate_factors("2015-01-01")
    assert list(fac.columns) == ["mkt_rf", "smb", "hml", "mom", "rf"]
    assert len(fac) > 100
    assert fac.index.is_monotonic_increasing

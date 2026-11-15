"""Attribution factorielle (Fama-French) de la stratégie.

Une stratégie peut sembler créer de la valeur simplement parce qu'elle est
**exposée à des facteurs de risque connus** : le marché, la taille (SMB), la
value (HML), le momentum (Mom). L'attribution régresse les rendements de la
stratégie, en excès du taux sans risque, sur ces facteurs :

    r_strat - rf = alpha + b_mkt*(Mkt-RF) + b_smb*SMB + b_hml*HML + b_mom*Mom + e

- les **bêtas** disent à quels facteurs la stratégie est exposée ;
- l'**alpha** est le rendement qui reste *inexpliqué* par ces expositions ; seul
  un alpha statistiquement significatif (|t| > ~2) traduit une vraie compétence.

Si l'alpha d'une stratégie momentum s'évapore dès qu'on ajoute le facteur Mom,
c'est que son « edge » n'était que l'exposition à une prime de risque connue.

Les facteurs proviennent de la bibliothèque de Kenneth French (mensuels,
gratuits). Comme `ingest.py`, un mode ``--demo`` simule des facteurs pour
tourner sans internet.

Usage :
    python src/factors.py          # télécharge les facteurs -> table `factors`
    python src/factors.py --demo   # facteurs simulés (sans internet)
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("factors")

ROOT = Path(__file__).resolve().parents[1]

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF3_URL = f"{BASE}/F-F_Research_Data_Factors_CSV.zip"
MOM_URL = f"{BASE}/F-F_Momentum_Factor_CSV.zip"

# colonnes de facteurs stockées (taux sans risque `rf` à part)
FACTOR_COLS = ["mkt_rf", "smb", "hml", "mom"]

# modèles emboîtés : on ajoute les facteurs un bloc à la fois
MODELS = {
    "CAPM": ["mkt_rf"],
    "FF3": ["mkt_rf", "smb", "hml"],
    "FF3+Mom": ["mkt_rf", "smb", "hml", "mom"],
}


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch_zip_csv(url: str) -> str:
    """Télécharge un zip de Ken French et renvoie le texte de son unique CSV."""
    req = urllib.request.Request(url, headers={"User-Agent": "momentum-research"})
    data = urllib.request.urlopen(req, timeout=30).read()  # noqa: S310 (URL de confiance)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def _parse_french_csv(text: str, columns: list[str]) -> pd.DataFrame:
    """Parse le bloc mensuel d'un CSV de Ken French.

    Ces fichiers ont un en-tête de commentaires, puis des lignes ``YYYYMM,...``
    (rendements en %), puis un bloc annuel. On ne garde que le bloc mensuel, on
    le passe en décimal et on indexe par la fin de mois.
    """
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{6})\s*,(.*)$", line)
        if not m:
            if rows:  # on a fini le bloc mensuel, on s'arrête avant l'annuel
                break
            continue
        rows.append([m.group(1), *[float(x) for x in m.group(2).split(",")]])
    df = pd.DataFrame(rows, columns=["ym", *columns])
    df["date"] = pd.to_datetime(df["ym"], format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
    return df.set_index("date")[columns].astype(float) / 100.0


def download_factors(start: str | None = None) -> pd.DataFrame:
    """Télécharge FF3 + momentum et les fusionne (colonnes `FACTOR_COLS` + `rf`).

    Index = fin de mois, valeurs en décimal. `start` filtre la date de début.
    """
    ff3 = _parse_french_csv(_fetch_zip_csv(FF3_URL), ["Mkt-RF", "SMB", "HML", "RF"])
    mom = _parse_french_csv(_fetch_zip_csv(MOM_URL), ["Mom"])
    fac = ff3.join(mom, how="inner")
    fac.columns = ["mkt_rf", "smb", "hml", "rf", "mom"]
    fac = fac[["mkt_rf", "smb", "hml", "mom", "rf"]]
    if start is not None:
        fac = fac.loc[start:]
    return fac


def simulate_factors(start: str, seed: int = 7) -> pd.DataFrame:
    """Facteurs mensuels fictifs pour tourner sans internet (mode démo).

    Primes et volatilités plausibles ; aucune valeur économique, juste de quoi
    faire tourner le pipeline d'attribution.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="ME")
    n = len(dates)
    data = {
        "mkt_rf": rng.normal(0.006, 0.045, n),
        "smb": rng.normal(0.001, 0.025, n),
        "hml": rng.normal(0.001, 0.030, n),
        "mom": rng.normal(0.004, 0.040, n),
        "rf": np.full(n, 0.002),
    }
    return pd.DataFrame(data, index=dates)


def save_to_duckdb(fac: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = fac.reset_index(names="date")
    with duckdb.connect(str(db_path)) as con:
        con.register("df_factors", df)
        con.execute(
            """
            CREATE OR REPLACE TABLE factors AS
            SELECT CAST(date AS DATE) AS date, mkt_rf, smb, hml, mom, rf
            FROM df_factors
            ORDER BY date
            """
        )
        n = con.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    log.info("Table 'factors' écrite : %s mois -> %s", n, db_path)


def load_factors(db_path: str | Path) -> pd.DataFrame:
    """Charge les facteurs stockés : index = fin de mois, colonnes en décimal."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        fac = con.execute("SELECT * FROM factors ORDER BY date").df()
    fac["date"] = pd.to_datetime(fac["date"]).dt.to_period("M").dt.to_timestamp("M")
    return fac.set_index("date")


def factor_regression(excess_returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Régresse les rendements en excès sur les facteurs (OLS, à la main).

    `factors` = DataFrame des colonnes de facteurs à inclure. Renvoie un tableau
    indexé ``['alpha', <facteurs>]`` avec le coefficient, son erreur-type et son
    t-stat. Le R² et le nombre de mois sont dans ``.attrs['r2']`` et
    ``.attrs['n']``. L'alpha est **mensuel** (l'annualiser = × 12).
    """
    data = pd.concat([excess_returns.rename("y"), factors], axis=1).dropna()
    y = data["y"].to_numpy()
    names = ["alpha", *factors.columns]
    x = np.column_stack([np.ones(len(data)), data[list(factors.columns)].to_numpy()])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    n, k = x.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / tss if tss > 0 else np.nan

    out = pd.DataFrame({"coef": beta, "se": se, "t_stat": beta / se}, index=names)
    out.attrs["r2"] = float(r2)
    out.attrs["n"] = int(n)
    return out


def attribution(
    strategy_returns: pd.Series, factors: pd.DataFrame, models: dict | None = None
) -> pd.DataFrame:
    """Compare plusieurs modèles de facteurs sur une même stratégie.

    Pour chaque modèle (CAPM, FF3, FF3+Mom par défaut) : alpha annualisé, son
    t-stat et le R². Un alpha qui tombe sous le seuil de significativité quand on
    ajoute un facteur = l'edge venait de ce facteur, pas d'une compétence.
    """
    models = models or MODELS
    excess = strategy_returns - factors["rf"]
    rows = []
    for name, cols in models.items():
        reg = factor_regression(excess, factors[cols])
        rows.append(
            {
                "modele": name,
                "alpha_annuel": reg.loc["alpha", "coef"] * 12,
                "alpha_tstat": reg.loc["alpha", "t_stat"],
                "r2": reg.attrs["r2"],
                "n_mois": reg.attrs["n"],
            }
        )
    return pd.DataFrame(rows).set_index("modele")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="utiliser des facteurs simulés")
    args = parser.parse_args()

    cfg = load_config()
    if args.demo:
        log.info("Mode démo : génération de facteurs simulés")
        fac = simulate_factors(cfg["start_date"])
    else:
        log.info("Téléchargement des facteurs Fama-French depuis %s", cfg["start_date"])
        fac = download_factors(cfg["start_date"])
    save_to_duckdb(fac, ROOT / cfg["db_path"])


if __name__ == "__main__":
    main()

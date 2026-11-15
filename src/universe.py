"""Appartenance historique au S&P 500 (point-in-time), contre le biais du survivant.

Le backtest initial se limite aux 50 grandes capitalisations d'*aujourd'hui* : il
ignore les entreprises jadis dans l'indice puis sorties (rachats, faillites,
relégations) — exactement les perdants, ce qui embellit les résultats. Ce module
reconstruit *qui était membre de l'indice à chaque date*, pour que le backtest ne
détienne à l'instant t que les actions réellement dans l'indice à t.

Source gratuite : jeu de données « S&P 500 historical constituents » de hanshof
(format ``date,tickers`` où ``tickers`` est la liste des membres à cette date),
de 1996 à 2025. Comme ``factors.py``, un mode ``--demo`` évite tout accès réseau.

Pipeline type :
    download_membership -> to_monthly_membership -> save_to_duckdb (table `membership`)
    universe_tickers(...) donne l'union des tickers à télécharger (ingestion).
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from io import StringIO
from pathlib import Path

import duckdb
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("universe")

ROOT = Path(__file__).resolve().parents[1]

HIST_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/"
    "sp_500_historical_components.csv"
)


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(ticker: str) -> str:
    """Harmonise la notation des classes d'actions sur celle de yfinance.

    Le jeu de données mélange points et tirets (``BRK.B`` mais ``BF-B``) ; yfinance
    attend des tirets (``BRK-B``). On convertit donc les points en tirets.
    """
    return ticker.strip().replace(".", "-")


def _parse_raw(text: str) -> pd.DataFrame:
    """Transforme le CSV brut ``date,tickers`` en format long ``(date, ticker)``."""
    raw = pd.read_csv(StringIO(text))
    raw["date"] = pd.to_datetime(raw["date"])
    rows = []
    for d, tickers in zip(raw["date"], raw["tickers"]):
        for t in str(tickers).split(","):
            t = _normalize(t)
            if t:
                rows.append((d, t))
    return pd.DataFrame(rows, columns=["date", "ticker"])


def download_membership(start: str | None = None) -> pd.DataFrame:
    """Télécharge l'appartenance historique -> format long ``(date, ticker)``.

    ``date`` = date de la photo de composition ; filtrée à partir de ``start`` (en
    gardant la dernière photo antérieure, pour connaître la composition en vigueur
    au début de la fenêtre).
    """
    req = urllib.request.Request(HIST_URL, headers={"User-Agent": "momentum-research"})
    text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    long = _parse_raw(text)
    if start is not None:
        start_ts = pd.Timestamp(start)
        # garder la dernière photo < start pour couvrir le premier mois de la fenêtre
        before = long[long["date"] < start_ts]["date"]
        floor = before.max() if not before.empty else long["date"].min()
        long = long[long["date"] >= floor]
    return long.reset_index(drop=True)


def simulate_membership(tickers: list[str], start: str, seed: int = 11) -> pd.DataFrame:
    """Appartenance fictive (mode démo, sans réseau) : tous membres tout du long.

    Suffit à faire tourner le pipeline point-in-time hors ligne ; aucune valeur
    économique.
    """
    dates = pd.date_range(start=start, end=pd.Timestamp.today().normalize(), freq="MS")
    rows = [(d, _normalize(t)) for d in dates for t in tickers]
    return pd.DataFrame(rows, columns=["date", "ticker"])


def to_monthly_membership(long: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    """Rééchantillonne en appartenance de **fin de mois** (point-in-time).

    Pour chaque fin de mois, on prend la composition de la **dernière photo connue
    à cette date** (report en avant). Renvoie un format long ``(date, ticker)`` où
    ``date`` est une fin de mois.
    """
    long = long.sort_values("date")
    snapshots = {d: set(g["ticker"]) for d, g in long.groupby("date")}
    snap_dates = sorted(snapshots)

    begin = pd.Timestamp(start) if start else long["date"].min()
    # fin de mois de la dernière photo incluse (sinon on perd le mois en cours)
    end = long["date"].max() + pd.offsets.MonthEnd(0)
    month_ends = pd.date_range(start=begin, end=end, freq="ME")

    rows = []
    for me in month_ends:
        # dernière photo <= fin de mois
        prior = [d for d in snap_dates if d <= me]
        if not prior:
            continue
        for t in snapshots[prior[-1]]:
            rows.append((me, t))
    return pd.DataFrame(rows, columns=["date", "ticker"])


def universe_tickers(membership: pd.DataFrame) -> list[str]:
    """Union triée de tous les tickers apparaissant dans l'appartenance."""
    return sorted(membership["ticker"].unique())


def save_to_duckdb(membership: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.register("df_mem", membership)
        con.execute(
            """
            CREATE OR REPLACE TABLE membership AS
            SELECT CAST(date AS DATE) AS date, ticker
            FROM df_mem
            ORDER BY date, ticker
            """
        )
        n, k = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ticker) FROM membership"
        ).fetchone()
    log.info("Table 'membership' écrite : %s lignes, %s tickers distincts -> %s", n, k, db_path)


def load_membership(db_path: str | Path) -> pd.DataFrame:
    """Charge l'appartenance au format large : index = fin de mois, colonnes =
    tickers, valeurs booléennes (True = membre ce mois-là)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("SELECT date, ticker FROM membership").df()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    # pivot -> True là où (date, ticker) existe, NaN sinon -> notna() = booléen propre
    wide = (
        df.drop_duplicates(["date", "ticker"])
        .assign(member=True)
        .pivot(index="date", columns="ticker", values="member")
        .notna()
        .sort_index()
    )
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description="Appartenance historique au S&P 500.")
    parser.add_argument("--demo", action="store_true", help="appartenance simulée (sans internet)")
    args = parser.parse_args()

    cfg = load_config()
    start = cfg["start_date"]
    if args.demo:
        log.info("Mode démo : appartenance simulée depuis %s", start)
        long = simulate_membership(cfg["tickers"], start)
        monthly = to_monthly_membership(long, start=start)
    else:
        log.info("Téléchargement de l'appartenance historique S&P 500 depuis %s", start)
        long = download_membership(start)
        monthly = to_monthly_membership(long, start=start)
    save_to_duckdb(monthly, ROOT / cfg["db_path"])
    log.info("Univers (union) : %s tickers distincts", len(universe_tickers(monthly)))


if __name__ == "__main__":
    main()

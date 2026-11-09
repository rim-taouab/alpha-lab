"""Télécharge les prix quotidiens et les stocke dans DuckDB.

Usage :
    python src/ingest.py          # vraies données via yfinance
    python src/ingest.py --demo   # données simulées (sans internet), pour tester
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Télécharge les prix ajustés (dividendes et splits) au format long :
    une ligne par (date, ticker)."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Aucune donnée reçue de yfinance.")

    close = raw["Close"].reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
    volume = raw["Volume"].reset_index().melt(id_vars="Date", var_name="ticker", value_name="volume")
    df = close.merge(volume, on=["Date", "ticker"]).rename(columns={"Date": "date"})
    df = df.dropna(subset=["close"])

    missing = sorted(set(tickers) - set(df["ticker"].unique()))
    if missing:
        log.warning("Tickers sans données : %s", missing)
    return df


def simulate_prices(tickers: list[str], start: str, seed: int = 42) -> pd.DataFrame:
    """Génère des prix fictifs (mouvement brownien géométrique) pour tester
    le pipeline sans connexion internet."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    frames = []
    for ticker in tickers:
        mu, sigma = rng.uniform(0.02, 0.15), rng.uniform(0.15, 0.45)
        dt = 1 / 252
        shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), len(dates))
        prices = 100 * np.exp(np.cumsum(shocks))
        volume = rng.integers(1_000_000, 50_000_000, len(dates))
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker, "close": prices, "volume": volume}))
    return pd.concat(frames, ignore_index=True)


def save_to_duckdb(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.register("df_prices", df)
        con.execute(
            """
            CREATE OR REPLACE TABLE prices AS
            SELECT CAST(date AS DATE) AS date, ticker, close, CAST(volume AS BIGINT) AS volume
            FROM df_prices
            ORDER BY ticker, date
            """
        )
        n, k = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM prices").fetchone()
    log.info("Table 'prices' écrite : %s lignes, %s tickers -> %s", n, k, db_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="utiliser des données simulées")
    args = parser.parse_args()

    cfg = load_config()
    if args.demo:
        log.info("Mode démo : génération de prix simulés")
        df = simulate_prices(cfg["tickers"], cfg["start_date"])
    else:
        log.info("Téléchargement de %s tickers depuis %s", len(cfg["tickers"]), cfg["start_date"])
        df = download_prices(cfg["tickers"], cfg["start_date"])
    save_to_duckdb(df, ROOT / cfg["db_path"])


if __name__ == "__main__":
    main()

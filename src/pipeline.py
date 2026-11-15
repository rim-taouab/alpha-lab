"""Pipeline de bout en bout : ingestion -> facteurs -> calcul des résultats.

Ce module regroupe les étapes du projet en **fonctions appelables** et sans
look-ahead, de sorte qu'elles puissent être :
    - testées directement (``tests/test_pipeline.py``),
    - lancées en ligne de commande (``python src/pipeline.py --demo``),
    - orchestrées par Airflow (``dags/momentum_pipeline.py``), chaque tâche du
      DAG n'étant qu'un mince appel à l'une de ces fonctions.

Étapes :
    1. ``run_ingest``   -> table ``prices``   (prix quotidiens)
    2. ``run_factors``  -> table ``factors``  (facteurs Fama-French mensuels)
    3. ``run_results``  -> tables ``perf``, ``equity``, ``attribution``,
                           ``pipeline_runs`` (résultats prêts pour un dashboard)

Le mode ``demo=True`` simule prix et facteurs : le pipeline tourne alors sans
accès internet, ce qui le rend reproductible en test comme en CI.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import duckdb
import pandas as pd

from analytics import load_prices
from backtest import backtest_momentum, equity_curve, performance
from factors import (
    attribution,
    download_factors,
    load_factors,
    save_to_duckdb as save_factors,
    simulate_factors,
)
from factors import load_config as _load_config
from ingest import (
    download_prices,
    save_to_duckdb as save_prices,
    simulate_prices,
)
from ml import build_panel, ml_backtest, walk_forward_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")

ROOT = Path(__file__).resolve().parents[1]


def _db_path(db_path: str | Path | None) -> Path:
    """Chemin de la base : argument explicite, sinon celui de ``config.yaml``."""
    if db_path is not None:
        return Path(db_path)
    return ROOT / _load_config()["db_path"]


def _save_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register("_tmp", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
    con.unregister("_tmp")


# --------------------------------------------------------------------------- #
# Étape 1 — ingestion des prix
# --------------------------------------------------------------------------- #
def run_ingest(
    demo: bool = False,
    db_path: str | Path | None = None,
    tickers: list[str] | None = None,
    start: str | None = None,
) -> Path:
    """Télécharge (ou simule) les prix quotidiens -> table ``prices``."""
    cfg = _load_config()
    tickers = tickers or cfg["tickers"]
    start = start or cfg["start_date"]
    path = _db_path(db_path)

    if demo:
        log.info("Ingestion (démo) : %s tickers simulés depuis %s", len(tickers), start)
        df = simulate_prices(tickers, start)
    else:
        log.info("Ingestion : téléchargement de %s tickers depuis %s", len(tickers), start)
        df = download_prices(tickers, start)
    save_prices(df, path)
    return path


# --------------------------------------------------------------------------- #
# Étape 2 — facteurs Fama-French
# --------------------------------------------------------------------------- #
def run_factors(
    demo: bool = False,
    db_path: str | Path | None = None,
    start: str | None = None,
) -> Path:
    """Télécharge (ou simule) les facteurs mensuels -> table ``factors``."""
    cfg = _load_config()
    start = start or cfg["start_date"]
    path = _db_path(db_path)

    if demo:
        log.info("Facteurs (démo) : simulation depuis %s", start)
        fac = simulate_factors(start)
    else:
        log.info("Facteurs : téléchargement Fama-French depuis %s", start)
        fac = download_factors(start)
    save_factors(fac, path)
    return path


# --------------------------------------------------------------------------- #
# Étape 3 — calcul et persistance des résultats
# --------------------------------------------------------------------------- #
def run_results(
    db_path: str | Path | None = None,
    frac: float = 0.2,
    cost_bps: float = 10.0,
    include_ml: bool = True,
) -> dict[str, pd.DataFrame]:
    """Backteste momentum (+ ML) vs marché, attribue les facteurs, et persiste tout.

    Lit les tables ``prices`` et ``factors``, puis écrit dans la même base :
        - ``perf``         : métriques (rendement, vol, Sharpe, drawdown) par stratégie ;
        - ``equity``       : courbes de capital (1 $ investi) au format long ;
        - ``attribution``  : alpha/t-stat/R² par modèle de facteurs sur le momentum ;
        - ``pipeline_runs``: une ligne de métadonnées par exécution.

    Renvoie aussi ces tables (dict) pour inspection directe en test ou en notebook.
    """
    path = _db_path(db_path)
    prices = load_prices(path)

    # --- momentum vs marché (équipondéré) ---
    bt = backtest_momentum(prices, frac=frac, cost_bps=cost_bps)
    perf = {
        "momentum": performance(bt["strategy"]),
        "marche": performance(bt["benchmark"]),
    }
    equity_series = {
        "momentum": equity_curve(bt["strategy"]),
        "marche": equity_curve(bt["benchmark"]),
    }

    # --- modèle ML (walk-forward) ---
    if include_ml:
        panel = build_panel(prices)
        preds = walk_forward_predict(panel)
        bt_ml = ml_backtest(prices, preds, frac=frac, cost_bps=cost_bps)
        perf["ml"] = performance(bt_ml["strategy"])
        equity_series["ml"] = equity_curve(bt_ml["strategy"])

    perf_df = pd.DataFrame(perf).T
    perf_df.index.name = "strategie"

    equity_df = pd.concat(
        [s.rename("valeur").rename_axis("date").reset_index().assign(serie=name) for name, s in equity_series.items()],
        ignore_index=True,
    )[["date", "serie", "valeur"]]

    # --- attribution factorielle sur le momentum ---
    fac = load_factors(path)
    attr = attribution(bt["strategy"], fac).reset_index()

    meta = pd.DataFrame(
        [
            {
                "run_ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "n_mois": int(bt.shape[0]),
                "sharpe_momentum": float(perf_df.loc["momentum", "sharpe"]),
                "sharpe_marche": float(perf_df.loc["marche", "sharpe"]),
                "sharpe_ml": float(perf_df.loc["ml", "sharpe"]) if include_ml else None,
            }
        ]
    )

    with duckdb.connect(str(path)) as con:
        _save_table(con, "perf", perf_df.reset_index())
        _save_table(con, "equity", equity_df)
        _save_table(con, "attribution", attr)
        # on APPEND les métadonnées : garder l'historique des exécutions
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "pipeline_runs" in existing:
            con.register("_meta", meta)
            con.execute("INSERT INTO pipeline_runs SELECT * FROM _meta")
            con.unregister("_meta")
        else:
            _save_table(con, "pipeline_runs", meta)

    log.info(
        "Résultats écrits -> %s | Sharpe momentum=%.2f, marché=%.2f%s",
        path,
        perf_df.loc["momentum", "sharpe"],
        perf_df.loc["marche", "sharpe"],
        f", ml={perf_df.loc['ml', 'sharpe']:.2f}" if include_ml else "",
    )
    return {"perf": perf_df, "equity": equity_df, "attribution": attr, "pipeline_runs": meta}


def run_pipeline(
    demo: bool = False,
    db_path: str | Path | None = None,
    include_ml: bool = True,
) -> dict[str, pd.DataFrame]:
    """Enchaîne les trois étapes (pratique hors Airflow : CLI, notebook, test)."""
    run_ingest(demo=demo, db_path=db_path)
    run_factors(demo=demo, db_path=db_path)
    return run_results(db_path=db_path, include_ml=include_ml)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline momentum de bout en bout.")
    parser.add_argument("--demo", action="store_true", help="prix et facteurs simulés (sans internet)")
    parser.add_argument("--no-ml", action="store_true", help="sauter le modèle ML (plus rapide)")
    args = parser.parse_args()
    run_pipeline(demo=args.demo, include_ml=not args.no_ml)


if __name__ == "__main__":
    main()

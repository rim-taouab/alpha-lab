"""Tests du pipeline de bout en bout (``src/pipeline.py``).

Tout tourne en mode démo (prix et facteurs simulés) dans une base DuckDB
temporaire : aucun accès réseau, reproductible en CI. On vérifie que chaque
étape écrit bien sa table et que les résultats ont la forme attendue — c'est la
même logique que celle qu'Airflow orchestre, donc tester ici couvre le DAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import run_factors, run_ingest, run_pipeline, run_results  # noqa: E402

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
START = "2016-01-01"  # ~10 ans d'historique simulé -> assez pour le walk-forward ML


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "test.duckdb"


def _tables(path: Path) -> set[str]:
    with duckdb.connect(str(path), read_only=True) as con:
        return {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def test_run_ingest_writes_prices(db):
    run_ingest(demo=True, db_path=db, tickers=TICKERS, start=START)
    assert "prices" in _tables(db)
    with duckdb.connect(str(db), read_only=True) as con:
        n_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    assert n_tickers == len(TICKERS)


def test_run_factors_writes_factors(db):
    run_factors(demo=True, db_path=db, start=START)
    assert "factors" in _tables(db)
    with duckdb.connect(str(db), read_only=True) as con:
        cols = [c[0] for c in con.execute("DESCRIBE factors").fetchall()]
    assert {"mkt_rf", "smb", "hml", "mom", "rf"}.issubset(cols)


def test_run_results_shapes(db):
    run_ingest(demo=True, db_path=db, tickers=TICKERS, start=START)
    run_factors(demo=True, db_path=db, start=START)
    out = run_results(db_path=db, include_ml=False)

    # trois tables de résultats présentes
    assert {"perf", "equity", "attribution", "pipeline_runs"}.issubset(_tables(db))

    perf = out["perf"]
    assert {"momentum", "marche"}.issubset(perf.index)
    assert {"rendement_annuel", "volatilite_annuelle", "sharpe", "max_drawdown"}.issubset(perf.columns)

    # courbe de capital strictement positive (produit de (1+r))
    assert (out["equity"]["valeur"] > 0).all()

    # attribution : un modèle par ligne, colonnes attendues
    attr = out["attribution"]
    assert set(attr["modele"]) == {"CAPM", "FF3", "FF3+Mom"}
    assert {"alpha_annuel", "alpha_tstat", "r2"}.issubset(attr.columns)


def test_full_pipeline_with_ml_appends_run(db):
    run_pipeline(demo=True, db_path=db, include_ml=True)
    out1_tables = _tables(db)
    assert "ml" in run_results(db_path=db, include_ml=True)["perf"].index

    # pipeline_runs est en APPEND : une 2e exécution ajoute une ligne
    with duckdb.connect(str(db), read_only=True) as con:
        before = con.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    run_results(db_path=db, include_ml=False)
    with duckdb.connect(str(db), read_only=True) as con:
        after = con.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    assert after == before + 1
    assert "pipeline_runs" in out1_tables

"""DAG Airflow : orchestre le pipeline momentum (ingestion -> facteurs -> résultats).

Ce DAG est une **fine couche d'orchestration** : chaque tâche ne fait qu'appeler
une fonction de ``src/pipeline.py`` (déjà testée dans ``tests/test_pipeline.py``).
Toute la logique quantitative vit dans ``src/`` ; Airflow ne gère ici que
l'enchaînement, la planification et les reprises.

    ingest ─┐
            ├─> results   (results lit les deux tables écrites en amont)
    factors ┘

Configuration par variables d'environnement (lues au parsing du DAG) :
    MOMENTUM_DEMO=true|false   prix/facteurs simulés (défaut true, sans internet)
    MOMENTUM_ML=true|false     inclure le modèle ML walk-forward (défaut true)

Installation et lancement : voir la section « Orchestration Airflow » du README.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Rendre les modules du projet (src/) importables depuis le DAG, quel que soit
# l'emplacement d'AIRFLOW_HOME.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# TaskFlow API : chemin Airflow 3.x d'abord, repli sur 2.x.
try:
    from airflow.sdk import dag, task  # Airflow >= 3.0
except ImportError:  # pragma: no cover - dépend de la version installée
    from airflow.decorators import dag, task  # Airflow 2.x


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


DEMO = _flag("MOMENTUM_DEMO", True)
INCLUDE_ML = _flag("MOMENTUM_ML", True)


@dag(
    dag_id="momentum_pipeline",
    description="Ingestion des prix et facteurs, puis backtest momentum/ML vs marché.",
    schedule="@monthly",  # la stratégie rebalance en mensuel
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["momentum", "quant", "research"],
    default_args={"retries": 1},
)
def momentum_pipeline():
    @task
    def ingest() -> str:
        from pipeline import run_ingest

        return str(run_ingest(demo=DEMO))

    @task
    def factors() -> str:
        from pipeline import run_factors

        return str(run_factors(demo=DEMO))

    @task
    def results(prices_db: str, factors_db: str) -> dict:
        """Backtest + attribution + persistance. Renvoie un petit résumé (XCom)."""
        from pipeline import run_results

        out = run_results(include_ml=INCLUDE_ML)
        perf = out["perf"]
        summary = {
            "n_mois": int(out["pipeline_runs"].loc[0, "n_mois"]),
            "sharpe_momentum": round(float(perf.loc["momentum", "sharpe"]), 3),
            "sharpe_marche": round(float(perf.loc["marche", "sharpe"]), 3),
        }
        if "ml" in perf.index:
            summary["sharpe_ml"] = round(float(perf.loc["ml", "sharpe"]), 3)
        return summary

    # ingest et factors sont indépendants ; results dépend des deux.
    results(ingest(), factors())


dag = momentum_pipeline()

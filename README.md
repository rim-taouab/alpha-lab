# Momentum Research Platform

Plateforme de recherche quantitative en Python : pipeline de données de marché, puis backtest rigoureux d'une stratégie momentum, puis d'un modèle de machine learning, comparés au marché.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Démarrage

```bash
python src/ingest.py             # télécharge ~10 ans de prix pour 50 actions -> data/market.duckdb
python src/ingest.py --demo      # variante sans internet, avec des prix simulés
python src/factors.py            # facteurs Fama-French (Kenneth French) -> table `factors`
python src/pipeline.py --demo    # pipeline complet (ingestion -> facteurs -> résultats)
pytest                           # lance les tests
jupyter notebook notebooks/01_exploration.ipynb   # puis 02 à 05
```

Les tickers et la date de début se changent dans `config.yaml`.

## Structure

```
config.yaml                  paramètres (tickers, dates, base)
src/ingest.py                téléchargement (yfinance) -> DuckDB, table `prices`
src/analytics.py             rendements, volatilité, Sharpe, drawdown
src/momentum.py              signal momentum 12-1 (passage mensuel, sélection du top)
src/backtest.py              moteur de backtest mensuel, net de coûts, vs marché
src/robustness.py            IC du Sharpe, balayage de paramètres, hors-échantillon
src/ml.py                    modèle LightGBM réentraîné en walk-forward
src/factors.py               facteurs Fama-French + régression d'attribution (alpha, bêtas)
src/pipeline.py              pipeline de bout en bout (ingestion -> facteurs -> résultats persistés)
dags/momentum_pipeline.py    DAG Airflow qui orchestre le pipeline
notebooks/01_exploration     première analyse des données
notebooks/02_backtest        backtest momentum vs marché équipondéré
notebooks/03_robustesse      IC du Sharpe, sensibilité, hors-échantillon
notebooks/04_ml              modèle ML vs momentum vs marché
notebooks/05_factors         attribution factorielle (l'alpha survit-il ?)
tests/                       tests pytest (un par module de src/)
data/                        base DuckDB (non versionnée)
```

## Feuille de route

- [x] **Étape 1** : ingestion des prix et exploration (notebook `01`)
- [x] **Étape 2** : signal momentum 12-1 en Python (passage en mensuel, sélection du top) et tests
- [x] **Étape 3** : backtest momentum mensuel (top 20 %, coûts de 10 pb), comparé à l'achat-conservation (notebook `02`)
- [x] **Étape 4** : rigueur : hors échantillon, sensibilité aux paramètres, intervalles de confiance du Sharpe (`robustness.py` + tests)
- [x] **Étape 5** : modèle ML (LightGBM) réentraîné en walk-forward (`ml.py` + tests, notebook `04`)
- [x] **Étape 6a** : attribution factorielle Fama-French (`factors.py` + tests, notebook `05`)
- [x] **Étape 6b** : pipeline de bout en bout (`pipeline.py` + tests) orchestré par Airflow (`dags/`)
- [ ] **Étape 6c** : dashboard Streamlit sur les tables de résultats

> Résultat de l'étape 5 : sur la période commune, le modèle ML **ne bat ni** le momentum simple **ni** le marché (Sharpe 0,86 contre 1,05 et 1,15, drawdown -36 %). La complexité n'apporte rien ici — c'est un résultat de contrôle, pas un échec.
>
> Résultat de l'étape 6a : la stratégie n'a **pas d'alpha significatif**. Son bêta au facteur momentum est de 0,56 (t=9) et le R² passe à 0,79 une fois `Mom` inclus — l'edge est l'exposition à une **prime de risque connue**, pas une compétence. Conclusion conforme à la littérature.

## Orchestration Airflow

Le pipeline est d'abord un module testable, `src/pipeline.py`, qui enchaîne trois étapes :

1. `run_ingest` → table `prices` (prix quotidiens) ;
2. `run_factors` → table `factors` (facteurs mensuels) ;
3. `run_results` → tables `perf`, `equity`, `attribution`, `pipeline_runs` (résultats prêts pour un dashboard).

On peut le lancer seul, sans Airflow :

```bash
python src/pipeline.py --demo      # tout simulé, sans internet
python src/pipeline.py             # vraies données (yfinance + Kenneth French)
python src/pipeline.py --demo --no-ml   # plus rapide, sans le modèle ML
```

Le DAG `dags/momentum_pipeline.py` n'est qu'une fine couche d'orchestration par-dessus : `ingest` et `factors` tournent en parallèle, puis `results`. **Airflow est gratuit et open-source** (licence Apache 2.0) ; seules les versions cloud managées sont payantes. Il faut **Airflow ≥ 3.1** pour Python 3.13.

```bash
# 1. Installer Airflow avec son fichier de contraintes (voir requirements-airflow.txt)
AIRFLOW_VERSION=3.1.0; PYTHON_VERSION=3.13
CONSTRAINT="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow[standard]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT}"

# 2. Pointer Airflow sur le dossier dags/ du projet
export AIRFLOW_HOME="$PWD/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MOMENTUM_DEMO=true          # prix/facteurs simulés ; false = vraies données
export MOMENTUM_ML=true            # inclure le modèle ML (false = plus rapide)

# 3. Vérifier que le DAG se charge, puis le lancer
airflow dags list
airflow dags test momentum_pipeline      # exécute le DAG une fois, en local
# ou, pour l'interface web + le scheduler :
airflow standalone                       # http://localhost:8080
```

Les résultats atterrissent dans les tables DuckDB (`data/market.duckdb`), que le futur dashboard Streamlit (étape 6c) lira directement.

## Limites connues

- **Biais du survivant** : les 50 actions sont les grandes capitalisations d'aujourd'hui. Les entreprises disparues depuis 2015 manquent, ce qui embellit les résultats.
- **Données** : yfinance est gratuit mais non officiel ; il peut échouer ou limiter les requêtes. Relancer le script suffit en général.

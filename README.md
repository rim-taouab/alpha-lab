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
pytest                           # lance les tests
jupyter notebook notebooks/01_exploration.ipynb
```

Les tickers et la date de début se changent dans `config.yaml`.

## Structure

```
config.yaml                  paramètres (tickers, dates, base)
src/ingest.py                téléchargement (yfinance) -> DuckDB, table `prices`
src/analytics.py             rendements, volatilité, Sharpe, drawdown
notebooks/01_exploration     première analyse des données
tests/                       tests pytest
data/                        base DuckDB (non versionnée)
```

## Feuille de route

- [x] **Étape 1** : ingestion des prix et exploration
- [ ] **Étape 2** : modèles dbt (rendements, volatilité glissante, momentum 12-1) et tests de qualité
- [ ] **Étape 3** : backtest momentum mensuel (top 20 %, coûts de 10 pb), comparé à l'achat-conservation
- [ ] **Étape 4** : rigueur : hors échantillon, sensibilité aux paramètres, intervalles de confiance du Sharpe
- [ ] **Étape 5** : modèle ML (LightGBM) réentraîné en walk-forward, simulation avec capital de départ
- [ ] **Étape 6** : attribution factorielle (Fama-French), orchestration Airflow, dashboard Streamlit

## Limites connues

- **Biais du survivant** : les 50 actions sont les grandes capitalisations d'aujourd'hui. Les entreprises disparues depuis 2015 manquent, ce qui embellit les résultats.
- **Données** : yfinance est gratuit mais non officiel ; il peut échouer ou limiter les requêtes. Relancer le script suffit en général.

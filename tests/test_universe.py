"""Tests de la couche d'appartenance historique (``src/universe.py``).

Sur un petit CSV synthétique (aucun accès réseau) : on vérifie le parsing, la
normalisation des tickers, le rééchantillonnage point-in-time en fin de mois (avec
report en avant de la dernière photo), et l'écriture/lecture DuckDB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from universe import (  # noqa: E402
    _parse_raw,
    load_membership,
    save_to_duckdb,
    simulate_membership,
    to_monthly_membership,
    universe_tickers,
)

# AAA sort après mars, DDD entre en avril ; BRK.B doit devenir BRK-B.
CSV = (
    "date,tickers\n"
    "2020-01-06,\"AAA,BBB,CCC,BRK.B\"\n"
    "2020-04-15,\"BBB,CCC,DDD,BRK.B\"\n"
)


def test_parse_normalizes_dot_to_dash():
    long = _parse_raw(CSV)
    assert "BRK-B" in set(long["ticker"])
    assert "BRK.B" not in set(long["ticker"])


def test_monthly_membership_point_in_time():
    long = _parse_raw(CSV)
    monthly = to_monthly_membership(long, start="2020-01-01")
    wide = (
        monthly.assign(m=True)
        .pivot(index="date", columns="ticker", values="m")
        .notna()
    )
    jan = wide.loc[wide.index[0]]       # janvier : 1re photo
    assert jan["AAA"] and not jan["DDD"]
    apr = wide.loc[pd.Timestamp("2020-04-30")]  # après le changement du 15 avril
    assert apr["DDD"] and not apr["AAA"]


def test_universe_is_union():
    long = _parse_raw(CSV)
    monthly = to_monthly_membership(long, start="2020-01-01")
    assert set(universe_tickers(monthly)) == {"AAA", "BBB", "CCC", "DDD", "BRK-B"}


def test_save_and_load_roundtrip(tmp_path):
    long = _parse_raw(CSV)
    monthly = to_monthly_membership(long, start="2020-01-01")
    db = tmp_path / "u.duckdb"
    save_to_duckdb(monthly, db)
    wide = load_membership(db)
    assert wide.index.is_monotonic_increasing
    assert bool(wide.loc[wide.index[0], "AAA"])          # membre en janvier
    assert not bool(wide.loc[wide.index[-1], "AAA"])     # sorti en fin de période


def test_simulate_membership_offline():
    mem = simulate_membership(["AAA", "BBB"], "2020-01-01")
    assert set(mem["ticker"]) == {"AAA", "BBB"}
    assert len(mem) > 0

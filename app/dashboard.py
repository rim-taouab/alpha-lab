"""Dashboard Streamlit : résultats du pipeline momentum.

Lit les tables produites par ``src/pipeline.py`` / le DAG Airflow dans la base
DuckDB (``perf``, ``equity``, ``attribution``, ``pipeline_runs``) et les présente :
courbes de capital, métriques par stratégie, attribution factorielle.

Lancement :
    streamlit run app/dashboard.py

Prérequis : avoir au moins une fois exécuté le pipeline (``python src/pipeline.py
--demo`` ou le DAG Airflow) pour remplir les tables de résultats.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Palette catégorielle (ordre fixe, CVD-safe) — une couleur par stratégie.
COLORS = {"momentum": "#2a78d6", "marche": "#eb6834", "ml": "#1baf7a"}
LABELS = {"momentum": "Momentum", "marche": "Marché (équipondéré)", "ml": "ML (LightGBM)"}
GRID = "rgba(130,130,125,0.18)"


def _db_path() -> Path:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return ROOT / cfg["db_path"]


@st.cache_data(show_spinner=False)
def load_tables(mtime: float) -> dict[str, pd.DataFrame] | None:
    """Charge les tables de résultats. `mtime` invalide le cache quand la base change."""
    path = _db_path()
    if not path.exists():
        return None
    with duckdb.connect(str(path), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        needed = {"perf", "equity", "attribution", "pipeline_runs"}
        if not needed.issubset(tables):
            return None
        out = {t: con.execute(f"SELECT * FROM {t}").df() for t in needed}
    out["equity"]["date"] = pd.to_datetime(out["equity"]["date"])
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def equity_chart(equity: pd.DataFrame) -> go.Figure:
    """Courbe de capital (1 $ investi) par stratégie, hover unifié + label de fin."""
    fig = go.Figure()
    for serie in ["momentum", "marche", "ml"]:
        d = equity[equity["serie"] == serie]
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["valeur"],
                name=LABELS[serie],
                mode="lines",
                line=dict(color=COLORS[serie], width=2),
                hovertemplate="%{y:.2f} $<extra>" + LABELS[serie] + "</extra>",
            )
        )
        # label direct en bout de courbe (identité sans dépendre de la seule couleur)
        last = d.iloc[-1]
        fig.add_annotation(
            x=last["date"], y=last["valeur"], text=f"  {LABELS[serie]}",
            showarrow=False, xanchor="left", font=dict(color=COLORS[serie], size=12),
        )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=130, t=10, b=10),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title="Capital (1 $ investi)", gridcolor=GRID, zeroline=False),
        xaxis=dict(gridcolor=GRID, zeroline=False),
    )
    return fig


def attribution_chart(attr: pd.DataFrame) -> go.Figure:
    """t-stat de l'alpha par modèle, avec le seuil de significativité |t|=2."""
    fig = go.Figure()
    sig = attr["alpha_tstat"].abs() >= 2
    fig.add_trace(
        go.Bar(
            x=attr["modele"],
            y=attr["alpha_tstat"],
            marker_color=["#2a78d6" if s else "#9aa0a6" for s in sig],
            width=0.5,
            hovertemplate="t = %{y:.2f}<extra>%{x}</extra>",
        )
    )
    for y in (2, -2):
        fig.add_hline(y=y, line=dict(color="#e34948", width=1, dash="dash"))
    fig.add_annotation(
        xref="paper", x=1, y=2, text="seuil de significativité |t| = 2",
        showarrow=False, yanchor="bottom", xanchor="right",
        font=dict(color="#e34948", size=11),
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="t-stat de l'alpha", gridcolor=GRID, zeroline=True, zerolinecolor=GRID),
        xaxis=dict(gridcolor=GRID),
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Momentum Research", page_icon="📈", layout="wide")
st.title("📈 Momentum Research — résultats")

path = _db_path()
data = load_tables(path.stat().st_mtime if path.exists() else 0.0)

if data is None:
    st.warning(
        "Aucune table de résultats trouvée. Lance d'abord le pipeline :\n\n"
        "```bash\npython src/pipeline.py --demo\n```\n\n"
        "ou le DAG Airflow (`airflow dags test momentum_pipeline`)."
    )
    st.stop()

perf, equity, attr, runs = data["perf"], data["equity"], data["attribution"], data["pipeline_runs"]
perf = perf.set_index("strategie")
last_run = runs.sort_values("run_ts").iloc[-1]

# --- bandeau d'en-tête : contexte de l'exécution ---
c1, c2 = st.columns([3, 2])
c1.caption(
    f"Dernière exécution du pipeline : **{last_run['run_ts']}** · "
    f"{int(last_run['n_mois'])} mois de backtest"
)
c2.caption("Source : table DuckDB `data/market.duckdb` écrite par le pipeline / Airflow")

# --- KPI : une rangée de tuiles par stratégie ---
st.subheader("Performance par stratégie")
cols = st.columns(len(perf))
for col, (name, row) in zip(cols, perf.iterrows()):
    with col:
        st.markdown(f"**{LABELS.get(name, name)}**")
        st.metric("Sharpe", f"{row['sharpe']:.2f}")
        st.metric("Rendement annualisé", fmt_pct(row["rendement_annuel"]))
        st.metric("Volatilité annualisée", fmt_pct(row["volatilite_annuelle"]))
        st.metric("Pire drawdown", fmt_pct(row["max_drawdown"]))

# --- courbe de capital ---
st.subheader("Courbes de capital")
st.plotly_chart(equity_chart(equity), use_container_width=True)

# --- attribution factorielle ---
st.subheader("Attribution factorielle (momentum)")
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(attribution_chart(attr), use_container_width=True)
with right:
    st.caption(
        "Alpha annualisé restant une fois les facteurs de risque retirés. "
        "Un alpha qui perd sa significativité (|t| < 2) quand on ajoute un facteur "
        "signifie que l'« edge » venait de ce facteur, pas d'une compétence."
    )
    show = attr.assign(
        alpha_annuel=lambda d: (d["alpha_annuel"] * 100).round(1),
        alpha_tstat=lambda d: d["alpha_tstat"].round(2),
        r2=lambda d: d["r2"].round(3),
    ).rename(
        columns={"modele": "Modèle", "alpha_annuel": "Alpha annuel (%)",
                 "alpha_tstat": "t-stat", "r2": "R²", "n_mois": "N mois"}
    )
    st.dataframe(show, hide_index=True, use_container_width=True)

st.caption(
    "⚠️ En mode démo, les prix sont simulés : les chiffres n'ont aucune valeur "
    "économique, ils valident la plomberie. Relance avec `MOMENTUM_DEMO=false` "
    "pour les vraies données."
)

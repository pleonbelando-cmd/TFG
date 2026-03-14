# -*- coding: utf-8 -*-
"""
dashboard.py
Dashboard web del bot semanal de oro (XGBoost + LightGBM).
Ejecutar: streamlit run bot_trading/dashboard.py

Secciones:
  1. Señal de la semana en curso (BUY / FLAT / CASH)
  2. Métricas del backtest walk-forward 2018–2026
  3. Figuras: equity curve, retornos anuales, drawdown, DA
"""

import sys
import pathlib

# Asegurar que bot_trading/ está en el path
BOT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Bot Semanal de Oro",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Imports del bot ──────────────────────────────────────────────────────────
from data          import load_prices
from features      import build_features
from model         import (
    walk_forward, compute_da_per_model,
    retrain_full, predict_latest, load_models,
)
from backtest      import simulate_portfolio, compute_metrics
from paper_trading import get_paper_stats, INITIAL_CAPITAL as PAPER_INITIAL_CAPITAL
import config


# ─── Helpers de caché ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Descargando datos de Yahoo Finance...")
def _load_data():
    df_raw = load_prices()
    data, feat, tgt = build_features(df_raw)
    return data, feat, tgt


@st.cache_data(show_spinner="Ejecutando walk-forward (puede tardar ~30 s)...")
def _run_backtest(_data, feat, tgt):
    results = walk_forward(_data, feat, tgt)
    das     = compute_da_per_model(results)
    port_df = simulate_portfolio(results)
    metrics = compute_metrics(port_df)
    return results, das, port_df, metrics


@st.cache_resource(show_spinner="Cargando modelos...")
def _load_models_cached():
    try:
        return load_models()
    except FileNotFoundError:
        return None, None


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🥇 Bot Semanal de Oro — XGBoost + LightGBM")
st.caption(
    "Walk-forward 2018–2026 · XAUUSD · Umbral BUY > 0.55 | CASH < 0.45 | FLAT zona neutra"
)

# ─── Botón de refresco ────────────────────────────────────────────────────────

col_ref, col_info = st.columns([1, 5])
with col_ref:
    if st.button("🔄 Refrescar señal", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
with col_info:
    st.info(
        "Los datos se actualizan desde Yahoo Finance al pulsar Refrescar "
        "o al abrir el dashboard por primera vez. El backtest completo tarda ~30 segundos."
    )

st.divider()

# ─── Cargar datos y backtest ──────────────────────────────────────────────────

data, feat, tgt = _load_data()
results, das, port_df, metrics = _run_backtest(data, feat, tgt)

# ─── SECCIÓN 1: Señal de la semana ───────────────────────────────────────────

xgb_model, lgbm_model = _load_models_cached()

if xgb_model is None:
    st.warning("Modelos no encontrados. Ejecuta primero el backtest con: `python bot_trading/run.py --mode backtest`")
else:
    pred = predict_latest(data, feat, xgb_model, lgbm_model)

    st.subheader("📡 Señal para la próxima semana")

    signal_color = {
        "BUY":  "green",
        "FLAT": "orange",
        "CASH": "red",
    }.get(pred["signal_str"], "gray")

    signal_emoji = {
        "BUY":  "🟢",
        "FLAT": "🟡",
        "CASH": "🔴",
    }.get(pred["signal_str"], "⚪")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="SEÑAL",
            value=f"{signal_emoji} {pred['signal_str']}",
            delta=f"Fecha: {pred['date'].date()}",
        )

    with col2:
        st.metric(
            label="Prob. Ensemble",
            value=f"{pred['prob_ensemble']*100:.1f}%",
            delta="umbral: BUY>55% / CASH<45%",
        )

    with col3:
        st.metric(
            label="Prob. XGBoost",
            value=f"{pred['prob_xgb']*100:.1f}%",
        )

    with col4:
        prob_lgbm = pred.get("prob_lgbm")
        if prob_lgbm is not None:
            st.metric(
                label="Prob. LightGBM",
                value=f"{prob_lgbm*100:.1f}%",
            )

    # Interpretación
    interpretacion = {
        "BUY":  "Los modelos estiman probabilidad > 55% de que el oro suba esta semana. Señal de entrada larga en XAUUSD.",
        "FLAT": "Probabilidad en zona neutra (45%–55%). Sin señal clara — mantener posición actual o no operar.",
        "CASH": "Los modelos estiman probabilidad < 45% de subida. Señal de salida o no entrada.",
    }.get(pred["signal_str"], "")

    st.caption(f"**Interpretación:** {interpretacion}")

st.divider()

# ─── SECCIÓN 2: Métricas del backtest ────────────────────────────────────────

st.subheader("📊 Métricas del Backtest Walk-Forward (2018–2026)")

col_da, col_table = st.columns([1, 2])

with col_da:
    st.markdown("**Directional Accuracy (DA)**")
    da_rows = []
    labels = {"prob_xgb": "XGBoost", "prob_lgbm": "LightGBM", "prob_ensemble": "Ensemble"}
    for k, v in das.items():
        da_rows.append({
            "Modelo": labels.get(k, k),
            "DA": f"{v*100:.2f}%",
            "vs objetivo": "✓ ≥ 54%" if v >= 0.54 else "✗ < 54%",
        })
    st.dataframe(pd.DataFrame(da_rows), hide_index=True, width="stretch")

with col_table:
    st.markdown("**Métricas de rendimiento**")
    display_metrics = metrics[["Estrategia", "CAGR", "Sharpe", "Max Drawdown", "Win Rate"]].copy()
    st.dataframe(display_metrics, hide_index=True, width="stretch")

st.divider()

# ─── SECCIÓN 3: Figuras del backtest ─────────────────────────────────────────

st.subheader("📈 Figuras del Backtest")

OUTPUT_DIR = config.OUTPUT_DIR

fig_paths = {
    "equity_curve":   OUTPUT_DIR / "equity_curve.png",
    "annual_returns": OUTPUT_DIR / "annual_returns.png",
    "drawdown":       OUTPUT_DIR / "drawdown.png",
    "da_comparison":  OUTPUT_DIR / "da_comparison.png",
}

# Fila 1: equity curve (ancha) + DA comparison
row1_col1, row1_col2 = st.columns([3, 2])

with row1_col1:
    if fig_paths["equity_curve"].exists():
        st.image(str(fig_paths["equity_curve"]), caption="Equity Curve", width="stretch")
    else:
        st.warning("equity_curve.png no encontrado. Ejecuta el backtest.")

with row1_col2:
    if fig_paths["da_comparison"].exists():
        st.image(str(fig_paths["da_comparison"]), caption="Directional Accuracy", width="stretch")
    else:
        st.warning("da_comparison.png no encontrado.")

# Fila 2: retornos anuales + drawdown
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    if fig_paths["annual_returns"].exists():
        st.image(str(fig_paths["annual_returns"]), caption="Retorno Anual por Estrategia", width="stretch")
    else:
        st.warning("annual_returns.png no encontrado.")

with row2_col2:
    if fig_paths["drawdown"].exists():
        st.image(str(fig_paths["drawdown"]), caption="Drawdown (Underwater Chart)", width="stretch")
    else:
        st.warning("drawdown.png no encontrado.")

st.divider()

# ─── SECCIÓN 4: Serie histórica de señales ───────────────────────────────────

st.subheader("🗓️ Historial de señales (últimas 52 semanas)")

last_52 = results.tail(52).copy()
last_52["Señal"] = last_52["signal"].map({1: "🟢 BUY", -1: "🔴 CASH", 0: "🟡 FLAT"})
last_52["Prob Ensemble (%)"] = (last_52["prob_ensemble"] * 100).round(1)
last_52["Prob XGB (%)"]      = (last_52["prob_xgb"]      * 100).round(1)
last_52["Retorno oro (%)"]   = (last_52["gold_ret"]       * 100).round(2)
last_52["Dirección real"]    = last_52["y_true"].map({1: "⬆ Subió", 0: "⬇ Bajó"})
last_52["Acierto"]           = (
    last_52["signal"].map({1: 1, -1: 0, 0: None}) == last_52["y_true"]
).map({True: "✓", False: "✗", None: "—"})

display_cols = ["Señal", "Prob Ensemble (%)", "Prob XGB (%)", "Retorno oro (%)", "Dirección real", "Acierto"]
st.dataframe(
    last_52[display_cols].sort_index(ascending=False),
    width="stretch",
    height=400,
)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()

# ─── SECCIÓN 5: Paper Trading ─────────────────────────────────────────────────

st.subheader("📋 Paper Trading — Simulación sin dinero real")

paper = get_paper_stats()

if not paper["started"]:
    st.info(
        "El paper trading no ha comenzado todavía. "
        "Ejecuta el primer ciclo con:\n\n"
        "```bash\npython bot_trading/run.py --mode paper\n```\n\n"
        "Cada lunes vuelve a ejecutarlo para registrar la operación de la semana."
    )
else:
    state = paper["state"]

    # KPIs principales
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric(
            "Capital virtual",
            f"${paper['virtual_capital']:,.2f}",
            delta=f"{paper['total_return']:+.2f}% desde inicio",
        )
    with col_b:
        st.metric(
            "Posición actual",
            ("🟢 BUY" if state["signal"] == 1 else
             "🔴 CASH" if state["signal"] == -1 else "🟡 FLAT")
            if state["open"] else "— Sin posición",
        )
    with col_c:
        st.metric("Operaciones cerradas", paper["n_buy_trades"])
    with col_d:
        wr = paper["win_rate"]
        st.metric(
            "Win Rate (BUY)",
            f"{wr:.1f}%" if paper["n_buy_trades"] > 0 else "—",
            delta=f"Media P&L: {paper['avg_pnl_pct']:+.2f}%" if paper["n_buy_trades"] > 0 else None,
        )

    # Posición abierta actualmente
    if state["open"]:
        st.caption(
            f"Posición {state['signal_str']} abierta el {state['entry_date']} "
            f"a {state['entry_price']:.2f} USD · "
            f"Prob. ensemble: {state['prob_ensemble']*100:.1f}%"
        )

    # Tabla de operaciones
    log = paper["log"]
    if len(log) > 0:
        st.markdown("**Historial de operaciones**")
        display_log = log.copy()
        display_log["P&L (%)"] = display_log["pnl_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        display_log["P&L (USD)"] = display_log["pnl_usd"].apply(
            lambda x: f"${x:+.2f}" if pd.notna(x) else "—"
        )
        display_log["Capital tras trade"] = display_log["capital_after"].apply(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )
        show_cols = [
            "entry_date", "exit_date", "signal_str",
            "entry_price", "exit_price",
            "P&L (%)", "P&L (USD)", "Capital tras trade",
        ]
        st.dataframe(
            display_log[show_cols].sort_values("entry_date", ascending=False),
            hide_index=True,
            width="stretch",
        )

        # Curva de capital paper trading
        if len(log) > 1:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            capital_series = [PAPER_INITIAL_CAPITAL] + list(log["capital_after"].dropna())
            fig_paper, ax_paper = plt.subplots(figsize=(10, 3))
            ax_paper.plot(capital_series, marker="o", markersize=4,
                          color="#2E86AB", linewidth=2)
            ax_paper.axhline(PAPER_INITIAL_CAPITAL, color="gray",
                             linestyle="--", linewidth=0.8, label="Capital inicial")
            ax_paper.set_title("Evolución del capital virtual (Paper Trading)", fontsize=10)
            ax_paper.set_ylabel("USD")
            ax_paper.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
            )
            ax_paper.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_paper)
            plt.close(fig_paper)

st.divider()

# ─── SECCIÓN 6: Log del scheduler ────────────────────────────────────────────

st.subheader("🕐 Log del Scheduler")

LOG_FILE = config.OUTPUT_DIR / "scheduler.log"

if not LOG_FILE.exists():
    st.info(
        "El scheduler no ha arrancado todavia. Ejecuta:\n\n"
        "```bash\npython bot_trading/start_all.py\n```\n\n"
        "o haz doble clic en `bot_trading/start_all.bat`"
    )
else:
    # Leer ultimas 60 lineas del log
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    last_lines = "".join(lines[-60:])

    col_log1, col_log2 = st.columns([3, 1])
    with col_log1:
        st.text_area("Ultimas entradas del log", value=last_lines, height=250)
    with col_log2:
        st.markdown("**Estado del scheduler**")
        if lines:
            last_ts = lines[-1][:19] if len(lines[-1]) >= 19 else "—"
            st.metric("Ultima actividad", last_ts)
        st.metric("Entradas totales", len(lines))
        st.caption(f"Archivo: {LOG_FILE.name}")

st.divider()
st.caption(
    "TFG — Dinamica del precio del oro 2000-2025 · "
    "Modelos: XGBoost + LightGBM · "
    "Datos: Yahoo Finance (GC=F, DX-Y.NYB, ^GSPC, ^VIX, CL=F, SI=F)"
)

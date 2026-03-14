# -*- coding: utf-8 -*-
"""
backtest.py
Simula tres estrategias de trading con las señales del ensemble.
Genera 4 figuras PNG en output/.

Estrategias:
  1. Conservadora (L/C) — prob > 0.55 → LONG, prob < 0.45 → CASH
  2. Binaria (L/C)      — prob >= 0.50 → LONG, else → CASH
  3. Buy-and-Hold Oro   — benchmark
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import THRESH_UP, THRESH_DOWN, INITIAL_CAPITAL, OUTPUT_DIR


# ─── Simulación del portafolio ────────────────────────────────────────────────

def simulate_portfolio(results: pd.DataFrame) -> pd.DataFrame:
    """
    Simula tres estrategias a partir de las señales del ensemble.

    Frecuencia: semanal. Las posiciones del periodo t determinan el retorno t.

    Estrategia CONSERVADORA (long / cash):
        prob > THRESH_UP   → LONG  (retorno = gold_ret)
        prob < THRESH_DOWN → CASH  (retorno = 0)
        zona neutra        → mantener posición anterior

    Estrategia BINARIA (long / cash):
        prob >= 0.50 → LONG
        prob  < 0.50 → CASH

    Buy-and-Hold: benchmark pasivo.
    """
    df = results.copy()

    # ── Conservadora ─────────────────────────────────────────────────────────
    pos_cons = []
    prev = 0
    for s in df["signal"]:
        if s == 1:
            prev = 1
        elif s == -1:
            prev = 0
        pos_cons.append(prev)
    df["pos_cons"] = pos_cons

    # ── Binaria ───────────────────────────────────────────────────────────────
    df["pos_bin"] = (df["prob_ensemble"] >= 0.50).astype(int)

    # ── Retornos (posición del periodo anterior determina el retorno) ─────────
    df["ret_cons"] = df["pos_cons"].shift(1).fillna(0) * df["gold_ret"]
    df["ret_bin"]  = df["pos_bin"].shift(1).fillna(0)  * df["gold_ret"]
    df["ret_bnh"]  = df["gold_ret"]

    # ── Portafolios acumulados ────────────────────────────────────────────────
    df["port_cons"] = INITIAL_CAPITAL * (1 + df["ret_cons"]).cumprod()
    df["port_bin"]  = INITIAL_CAPITAL * (1 + df["ret_bin"]).cumprod()
    df["port_bnh"]  = INITIAL_CAPITAL * (1 + df["ret_bnh"]).cumprod()

    return df


# ─── Métricas ─────────────────────────────────────────────────────────────────

def _cagr(series: pd.Series, freq: int = 52) -> float:
    """CAGR anualizado. freq=52 para datos semanales."""
    n = len(series)
    if n < 2:
        return 0.0
    total_return = series.iloc[-1] / series.iloc[0]
    years = n / freq
    return float(total_return ** (1 / years) - 1)


def _sharpe(rets: pd.Series, freq: int = 52) -> float:
    """Sharpe anualizado (asume tasa libre de riesgo = 0)."""
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(freq))


def _max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd   = (series - peak) / peak
    return float(dd.min())


def _win_rate(rets: pd.Series) -> float:
    active = rets[rets != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def compute_metrics(port_df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de métricas: CAGR, Sharpe, Max Drawdown, Win Rate."""
    rows = []
    for name, ret_col, port_col in [
        ("Conservadora L/C (thr=0.55)", "ret_cons", "port_cons"),
        ("Binaria L/C (thr=0.50)",      "ret_bin",  "port_bin"),
        ("Buy-and-Hold Oro",            "ret_bnh",  "port_bnh"),
    ]:
        rets = port_df[ret_col].dropna()
        port = port_df[port_col].dropna()
        rows.append({
            "Estrategia":   name,
            "CAGR":         f"{_cagr(port)*100:.1f}%",
            "Sharpe":       f"{_sharpe(rets):.2f}",
            "Max Drawdown": f"{_max_drawdown(port)*100:.1f}%",
            "Win Rate":     f"{_win_rate(rets)*100:.1f}%",
            "CAGR_num":     _cagr(port) * 100,
        })
    return pd.DataFrame(rows)


# ─── Figuras ──────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "figure.dpi":        150,
    })


def plot_equity_curve(port_df: pd.DataFrame, das: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(port_df.index, port_df["port_bnh"],
            label="Buy-and-Hold Oro", color="#C9A84C", linewidth=1.5, alpha=0.8)
    ax.plot(port_df.index, port_df["port_bin"],
            label="Binaria L/C (thr=0.50)", color="#3BB273", linewidth=2)
    ax.plot(port_df.index, port_df["port_cons"],
            label="Conservadora L/C (thr=0.55)", color="#2E86AB", linewidth=2)

    da_ens = das.get("prob_ensemble", 0)
    ax.set_title(
        f"Equity Curve — Bot semanal de oro (XGBoost + LightGBM)\n"
        f"Ensemble DA: {da_ens:.1%}  |  Walk-forward 2018–2026",
        fontsize=11,
    )
    ax.set_ylabel("Valor del portafolio (USD)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = OUTPUT_DIR / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_annual_returns(port_df: pd.DataFrame) -> None:
    _style()
    df = port_df.copy()
    df["year"] = df.index.year

    annual = df.groupby("year").agg(
        ret_bnh  =("ret_bnh",  lambda x: (1 + x).prod() - 1),
        ret_bin  =("ret_bin",  lambda x: (1 + x).prod() - 1),
        ret_cons =("ret_cons", lambda x: (1 + x).prod() - 1),
    )

    x     = np.arange(len(annual))
    width = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, annual["ret_bnh"]  * 100, width,
           label="Buy-and-Hold", color="#C9A84C", alpha=0.85)
    ax.bar(x,         annual["ret_bin"]  * 100, width,
           label="Binaria L/C",  color="#3BB273", alpha=0.85)
    ax.bar(x + width, annual["ret_cons"] * 100, width,
           label="Conservadora", color="#2E86AB", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index, rotation=45, ha="right", fontsize=8)
    ax.set_title("Retorno Anual por Estrategia (%)", fontsize=11)
    ax.set_ylabel("Retorno (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = OUTPUT_DIR / "annual_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_drawdown(port_df: pd.DataFrame) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11, 4))

    for col, label, color in [
        ("port_bnh",  "Buy-and-Hold Oro",  "#C9A84C"),
        ("port_bin",  "Binaria L/C",        "#3BB273"),
        ("port_cons", "Conservadora L/C",   "#2E86AB"),
    ]:
        s  = port_df[col].dropna()
        dd = (s - s.cummax()) / s.cummax() * 100
        ax.fill_between(dd.index, dd, 0, alpha=0.20, color=color)
        ax.plot(dd.index, dd, label=label, color=color, linewidth=1.2)

    ax.set_title("Drawdown (%) — Underwater Chart  |  Semanal 2018–2026", fontsize=11)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = OUTPUT_DIR / "drawdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_da_comparison(das: dict) -> None:
    _style()
    labels = {
        "prob_xgb":      "XGBoost",
        "prob_lgbm":     "LightGBM",
        "prob_ensemble": "Ensemble",
    }
    models = [k for k in labels if k in das]
    values = [das[k] * 100 for k in models]
    names  = [labels[k] for k in models]
    colors = ["#E84855", "#FF8C00", "#2E86AB"][:len(models)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors, alpha=0.85, width=0.5)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% (azar)")
    ax.axhline(54, color="green", linestyle=":",  linewidth=1, label="54% (señal útil)")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    max_v = max(values) if values else 60
    ax.set_ylim(40, max_v + 8)
    ax.set_title("Directional Accuracy (DA) — Walk-Forward 2018–2026", fontsize=11)
    ax.set_ylabel("DA (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = OUTPUT_DIR / "da_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")


# ─── Función principal ────────────────────────────────────────────────────────

def run_backtest(results: pd.DataFrame, das: dict) -> pd.DataFrame:
    """Simula portafolios, imprime métricas y genera figuras."""
    print("\n=== Backtest de Trading (semanal) ===")

    port_df = simulate_portfolio(results)
    metrics = compute_metrics(port_df)

    print("\nMétricas de rendimiento:")
    print(metrics[["Estrategia", "CAGR", "Sharpe", "Max Drawdown", "Win Rate"]]
          .to_string(index=False))

    print("\nGenerando figuras en output/...")
    plot_equity_curve(port_df, das)
    plot_annual_returns(port_df)
    plot_drawdown(port_df)
    plot_da_comparison(das)

    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data import load_prices
    from features import build_features
    from model import walk_forward, compute_da_per_model

    df_raw = load_prices()
    data, feat, tgt = build_features(df_raw)
    results = walk_forward(data, feat, tgt)
    das     = compute_da_per_model(results)
    metrics = run_backtest(results, das)

# -*- coding: utf-8 -*-
"""
paper_trading.py
Paper trading semanal — simula operaciones sin dinero real.

Flujo semanal:
  1. Lunes: ejecutas `python bot_trading/run.py --mode paper`
  2. El bot genera la señal y registra: fecha, señal, precio_entrada
  3. El lunes siguiente vuelves a ejecutar
  4. El bot cierra la operación anterior (anota precio_salida, calcula P&L)
  5. Abre la nueva operación con la señal actual

Archivo de log: bot_trading/output/paper_trades.csv
Capital virtual: configurable (por defecto 10.000 USD)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, date

from config import OUTPUT_DIR, THRESH_UP, THRESH_DOWN

PAPER_CSV  = OUTPUT_DIR / "paper_trades.csv"
PAPER_JSON = OUTPUT_DIR / "paper_state.json"   # estado de la posición abierta
INITIAL_CAPITAL = 10_000.0                      # USD virtuales


# ─── Estado persistente ───────────────────────────────────────────────────────

def _load_state() -> dict:
    """Carga el estado de la posición abierta (o estado inicial si no existe)."""
    if PAPER_JSON.exists():
        with open(PAPER_JSON, "r") as f:
            return json.load(f)
    return {
        "open":            False,
        "signal":          0,
        "signal_str":      "FLAT",
        "entry_date":      None,
        "entry_price":     None,
        "entry_capital":   INITIAL_CAPITAL,
        "prob_ensemble":   None,
        "virtual_capital": INITIAL_CAPITAL,
    }


def _save_state(state: dict) -> None:
    with open(PAPER_JSON, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ─── Log de operaciones ───────────────────────────────────────────────────────

def _load_log() -> pd.DataFrame:
    if PAPER_CSV.exists():
        return pd.read_csv(PAPER_CSV, parse_dates=["entry_date", "exit_date"])
    return pd.DataFrame(columns=[
        "entry_date", "exit_date", "signal_str",
        "entry_price", "exit_price",
        "pnl_pct", "pnl_usd", "capital_after",
        "prob_ensemble",
    ])


def _append_trade(log: pd.DataFrame, trade: dict) -> pd.DataFrame:
    new_row = pd.DataFrame([trade])
    return pd.concat([log, new_row], ignore_index=True)


def _save_log(log: pd.DataFrame) -> None:
    log.to_csv(PAPER_CSV, index=False)


# ─── Lógica principal ─────────────────────────────────────────────────────────

def run_paper_trading(pred: dict) -> dict:
    """
    Ejecuta un ciclo de paper trading con la predicción actual.

    pred: dict con keys {date, signal, signal_str, prob_ensemble, prob_xgb,
                         prob_lgbm} de predict_latest() + gold_price actual

    Devuelve un resumen del ciclo: operación cerrada (si aplica) + nueva abierta.
    """
    state = _load_state()
    log   = _load_log()

    current_date  = pred["date"]
    current_price = pred["gold_price"]
    new_signal    = pred["signal"]
    new_signal_str = pred["signal_str"]
    current_capital = state["virtual_capital"]

    result = {
        "current_date":    current_date,
        "current_price":   current_price,
        "new_signal":      new_signal_str,
        "prob_ensemble":   pred["prob_ensemble"],
        "closed_trade":    None,
        "opened_trade":    None,
        "virtual_capital": current_capital,
        "total_return_pct": (current_capital / INITIAL_CAPITAL - 1) * 100,
        "n_trades":        len(log),
    }

    # ── PASO 1: Cerrar posición abierta si hay cambio de señal ────────────────
    if state["open"] and state["signal"] != new_signal:
        entry_price   = state["entry_price"]
        entry_capital = state["entry_capital"]

        # Solo las posiciones BUY generan P&L (estrategia L/C)
        if state["signal"] == 1:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_usd = entry_capital * (pnl_pct / 100)
        else:
            pnl_pct = 0.0
            pnl_usd = 0.0

        capital_after = entry_capital + pnl_usd
        state["virtual_capital"] = capital_after

        trade = {
            "entry_date":    state["entry_date"],
            "exit_date":     str(current_date.date()),
            "signal_str":    state["signal_str"],
            "entry_price":   round(entry_price, 2),
            "exit_price":    round(current_price, 2),
            "pnl_pct":       round(pnl_pct, 3),
            "pnl_usd":       round(pnl_usd, 2),
            "capital_after": round(capital_after, 2),
            "prob_ensemble": state["prob_ensemble"],
        }
        log    = _append_trade(log, trade)
        result["closed_trade"] = trade
        result["virtual_capital"] = capital_after
        state["open"] = False

        print(f"  [PAPER] CERRADA: {state['signal_str']} "
              f"entrada={entry_price:.2f} salida={current_price:.2f} "
              f"P&L={pnl_pct:+.2f}% ({pnl_usd:+.2f} USD)")

    # ── PASO 2: Abrir nueva posición si la señal lo indica ───────────────────
    if new_signal != 0 and not state["open"]:
        state.update({
            "open":          True,
            "signal":        new_signal,
            "signal_str":    new_signal_str,
            "entry_date":    str(current_date.date()),
            "entry_price":   current_price,
            "entry_capital": state["virtual_capital"],
            "prob_ensemble": pred["prob_ensemble"],
        })
        result["opened_trade"] = {
            "signal_str":  new_signal_str,
            "entry_price": current_price,
            "capital":     state["virtual_capital"],
        }
        print(f"  [PAPER] ABIERTA: {new_signal_str} "
              f"precio={current_price:.2f}  "
              f"capital virtual={state['virtual_capital']:.2f} USD")
    elif new_signal == 0:
        if state["open"] and state["signal"] == 0:
            pass  # FLAT: mantener
        elif not state["open"]:
            print(f"  [PAPER] FLAT — sin posicion abierta.")

    # ── Guardar estado y log ──────────────────────────────────────────────────
    result["total_return_pct"] = (state["virtual_capital"] / INITIAL_CAPITAL - 1) * 100
    result["n_trades"] = len(log)

    _save_state(state)
    _save_log(log)

    return result


def print_paper_summary(result: dict, state: dict) -> None:
    print("\n" + "=" * 60)
    print("  PAPER TRADING — RESUMEN")
    print("=" * 60)
    print(f"  Capital inicial    : {INITIAL_CAPITAL:>10,.2f} USD")
    print(f"  Capital actual     : {result['virtual_capital']:>10,.2f} USD")
    print(f"  Retorno acumulado  : {result['total_return_pct']:>+10.2f}%")
    print(f"  Operaciones totales: {result['n_trades']:>10}")
    print(f"  Posicion abierta   : {'SI - ' + state['signal_str'] if state['open'] else 'NO'}")
    if state["open"]:
        print(f"  Entrada en         : {state['entry_price']:.2f} USD")
        print(f"  Precio actual      : {result['current_price']:.2f} USD")
        unrealized = (result["current_price"] - state["entry_price"]) / state["entry_price"] * 100
        print(f"  P&L no realizado   : {unrealized:+.2f}%")
    print("=" * 60)


def get_paper_stats() -> dict:
    """Devuelve estadísticas del paper trading para el dashboard."""
    state = _load_state()
    log   = _load_log()

    if len(log) == 0:
        return {"started": False, "state": state, "log": log}

    closed = log[log["signal_str"] == "BUY"].copy()

    stats = {
        "started":         True,
        "state":           state,
        "log":             log,
        "n_trades":        len(log),
        "n_buy_trades":    len(closed),
        "win_rate":        (closed["pnl_pct"] > 0).mean() * 100 if len(closed) > 0 else 0,
        "avg_pnl_pct":     closed["pnl_pct"].mean() if len(closed) > 0 else 0,
        "total_pnl_usd":   log["pnl_usd"].sum(),
        "virtual_capital": state["virtual_capital"],
        "total_return":    (state["virtual_capital"] / INITIAL_CAPITAL - 1) * 100,
        "initial_capital": INITIAL_CAPITAL,
    }
    return stats


if __name__ == "__main__":
    stats = get_paper_stats()
    if not stats["started"]:
        print("Paper trading no iniciado. Ejecuta: python bot_trading/run.py --mode paper")
    else:
        print(f"Operaciones: {stats['n_trades']}")
        print(f"Capital: {stats['virtual_capital']:.2f} USD")
        print(f"Retorno: {stats['total_return']:+.2f}%")

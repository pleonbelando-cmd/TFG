# -*- coding: utf-8 -*-
"""
run.py
Punto de entrada del bot semanal de oro.

Modos de uso:
    # Backtest completo (sin MT5) — entrena, evalúa, genera figuras
    python bot_trading/run.py --mode backtest

    # Señal de la próxima semana (sin ejecutar)
    python bot_trading/run.py --mode signal

    # Ejecutar señal en MT5 (modo demo por defecto)
    python bot_trading/run.py --mode live
    python bot_trading/run.py --mode live --live   # REAL — usar con cuidado

Requisitos previos:
    pip install xgboost lightgbm joblib yfinance MetaTrader5
"""

import argparse
import sys
import pathlib

# Asegurar que el directorio del bot está en el path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from data          import load_prices
from features      import build_features
from model         import (
    walk_forward, compute_da_per_model,
    retrain_full, predict_latest, load_models,
)
from backtest      import run_backtest
from paper_trading import run_paper_trading, print_paper_summary, get_paper_stats, _load_state
import config


# ─── Modos ────────────────────────────────────────────────────────────────────

def mode_backtest() -> None:
    """
    Backtest completo:
    1. Descarga datos semanales
    2. Construye 15 features
    3. Walk-forward 2018–2026
    4. Métricas + 4 figuras PNG
    5. Re-entrena con todos los datos y guarda modelos .pkl
    """
    print("=" * 60)
    print("  BOT TRADING DE ORO — Modo: BACKTEST")
    print("=" * 60)

    # 1. Datos
    df_raw = load_prices()

    # 2. Features
    print("\n--- Feature Engineering ---")
    data, feat, tgt = build_features(df_raw)

    # 3. Walk-forward
    results = walk_forward(data, feat, tgt)

    # 4. DA y backtest
    das     = compute_da_per_model(results)
    metrics = run_backtest(results, das)

    # 5. Re-entrenar completo y guardar modelos
    print("\n--- Re-entrenamiento final con todos los datos ---")
    retrain_full(data, feat, tgt)

    # Resumen final
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    da_ens = das.get("prob_ensemble", 0)
    print(f"  DA Ensemble: {da_ens*100:.2f}%  {'✓ ≥ 54%' if da_ens >= 0.54 else '✗ < 54%'}")

    best_cagr = max(
        (row["CAGR_num"] for _, row in metrics.iterrows()),
        default=0,
    )
    print(f"  Mejor CAGR:  {best_cagr:.1f}%")
    print(f"  Figuras:     {config.OUTPUT_DIR}/")
    print(f"  Modelos:     {config.OUTPUT_DIR}/*.pkl")
    print("=" * 60)


def mode_signal() -> None:
    """
    Señal para la próxima semana:
    1. Descarga datos actualizados
    2. Construye features
    3. Carga modelos guardados (o re-entrena si no existen)
    4. Imprime señal + probabilidad
    """
    print("=" * 60)
    print("  BOT TRADING DE ORO — Modo: SIGNAL")
    print("=" * 60)

    # 1. Datos actualizados
    df_raw = load_prices()

    # 2. Features
    print("\n--- Feature Engineering ---")
    data, feat, tgt = build_features(df_raw)

    # 3. Cargar o re-entrenar modelos
    try:
        xgb_model, lgbm_model = load_models()
        print("  Modelos cargados desde disco.")
    except FileNotFoundError:
        print("  Modelos no encontrados. Re-entrenando...")
        xgb_model, lgbm_model = retrain_full(data, feat, tgt)

    # 4. Predicción
    pred = predict_latest(data, feat, xgb_model, lgbm_model)

    print("\n" + "=" * 60)
    print("  SEÑAL PRÓXIMA SEMANA")
    print("=" * 60)
    print(f"  Fecha referencia : {pred['date'].date()}")
    print(f"  Prob XGBoost     : {pred['prob_xgb']*100:.1f}%")
    if pred["prob_lgbm"] is not None:
        print(f"  Prob LightGBM    : {pred['prob_lgbm']*100:.1f}%")
    print(f"  Prob Ensemble    : {pred['prob_ensemble']*100:.1f}%")
    print(f"  SEÑAL            : {pred['signal_str']}  (umbral ±0.55/0.45)")
    print("=" * 60)

    return pred


def mode_paper() -> None:
    """
    Paper trading semanal — señal real, dinero virtual.
    Cada semana:
      1. Cierra la operación anterior (si la había) y calcula P&L
      2. Abre nueva operación con la señal actual
      3. Guarda registro en output/paper_trades.csv
    """
    print("=" * 60)
    print("  BOT TRADING DE ORO — Modo: PAPER TRADING")
    print("=" * 60)
    print("  Capital virtual: 10.000 USD (sin dinero real)")
    print("=" * 60)

    # 1. Datos y señal
    df_raw = load_prices()
    print("\n--- Feature Engineering ---")
    data, feat, tgt = build_features(df_raw)

    try:
        xgb_model, lgbm_model = load_models()
        print("  Modelos cargados desde disco.")
    except FileNotFoundError:
        print("  Modelos no encontrados. Re-entrenando...")
        xgb_model, lgbm_model = retrain_full(data, feat, tgt)

    pred = predict_latest(data, feat, xgb_model, lgbm_model)

    # Añadir precio actual del oro al dict de predicción
    pred["gold_price"] = float(data["gold"].iloc[-1])

    print(f"\n  Precio actual oro : {pred['gold_price']:.2f} USD")
    print(f"  Señal             : {pred['signal_str']}  "
          f"(Ensemble: {pred['prob_ensemble']*100:.1f}%)")

    # 2. Ejecutar ciclo paper trading
    print("\n--- Paper Trading ---")
    result = run_paper_trading(pred)
    state  = _load_state()
    print_paper_summary(result, state)


def mode_live(demo: bool = True) -> None:
    """
    Ejecuta la señal en MetaTrader 5.

    demo=True  → imprime la orden sin enviarla (LIVE_TRADING=False)
    demo=False → envía órdenes reales (LIVE_TRADING=True) — usar con precaución
    """
    print("=" * 60)
    mode_str = "DEMO" if demo else "*** LIVE REAL ***"
    print(f"  BOT TRADING DE ORO — Modo: LIVE [{mode_str}]")
    print("=" * 60)

    # Sobreescribir flag en config
    config.LIVE_TRADING = not demo

    # Obtener señal
    pred = mode_signal()

    # Ejecutar en MT5
    from mt5_connector import execute_signal
    print(f"\n  Ejecutando señal {pred['signal_str']} en MT5...")
    execute_signal(signal=pred["signal"], symbol=config.SYMBOL)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot semanal de oro — XGBoost + LightGBM + MT5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python bot_trading/run.py --mode backtest   # backtest completo + figuras
  python bot_trading/run.py --mode signal     # señal próxima semana
  python bot_trading/run.py --mode live       # ejecutar en MT5 (demo)
  python bot_trading/run.py --mode live --live  # ejecutar en MT5 (real)
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "signal", "live", "paper"],
        default="backtest",
        help="Modo de ejecución (default: backtest)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Activar trading real en MT5 (solo en --mode live). "
             "Sin este flag, se ejecuta en modo demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "backtest":
        mode_backtest()

    elif args.mode == "signal":
        mode_signal()

    elif args.mode == "paper":
        mode_paper()

    elif args.mode == "live":
        if args.live:
            print("\nATENCION: Modo LIVE REAL activado. Se enviarán órdenes reales.")
            confirm = input("¿Confirmar? (escribe 'SI' para continuar): ")
            if confirm.strip().upper() != "SI":
                print("Cancelado.")
                return
        mode_live(demo=not args.live)


if __name__ == "__main__":
    main()

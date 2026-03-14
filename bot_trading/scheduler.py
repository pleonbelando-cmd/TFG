# -*- coding: utf-8 -*-
"""
scheduler.py
Proceso persistente que ejecuta el bot semanal de oro de forma automatica.

Tareas programadas:
  - Cada lunes a las 09:00 -> paper trading (señal + registro P&L virtual)
  - Cada 4 semanas (lunes) -> re-entrenamiento completo del modelo
  - Cada hora -> health check (log "sigo vivo")

Arranque:
    python bot_trading/scheduler.py

Para arrancar junto con el dashboard, usa:
    python bot_trading/start_all.py
"""

import sys
import pathlib
import logging
import traceback
import time
from datetime import datetime

# Path
BOT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

import schedule
from config import OUTPUT_DIR

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = OUTPUT_DIR / "scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bot_scheduler")

# ─── Contador de semanas para re-entrenamiento ────────────────────────────────
_week_counter = {"n": 0}
REFIT_EVERY_WEEKS = 4


# ─── Tareas ───────────────────────────────────────────────────────────────────

def job_paper_trading():
    """Ejecuta el ciclo de paper trading semanal."""
    log.info("=" * 50)
    log.info("INICIO: paper trading semanal")
    log.info("=" * 50)

    try:
        from data          import load_prices
        from features      import build_features
        from model         import retrain_full, predict_latest, load_models
        from paper_trading import run_paper_trading, print_paper_summary, _load_state

        # Datos y features
        df_raw = load_prices()
        data, feat, tgt = build_features(df_raw)

        # Re-entrenar modelo cada REFIT_EVERY_WEEKS semanas
        _week_counter["n"] += 1
        if _week_counter["n"] % REFIT_EVERY_WEEKS == 0:
            log.info(f"Re-entrenamiento programado (semana {_week_counter['n']})")
            xgb_model, lgbm_model = retrain_full(data, feat, tgt)
        else:
            try:
                xgb_model, lgbm_model = load_models()
                log.info("Modelos cargados desde disco.")
            except FileNotFoundError:
                log.warning("Modelos no encontrados. Re-entrenando...")
                xgb_model, lgbm_model = retrain_full(data, feat, tgt)

        # Prediccion
        pred = predict_latest(data, feat, xgb_model, lgbm_model)
        pred["gold_price"] = float(data["gold"].iloc[-1])

        log.info(f"Señal: {pred['signal_str']}  "
                 f"Ensemble: {pred['prob_ensemble']*100:.1f}%  "
                 f"Oro: {pred['gold_price']:.2f} USD")

        # Paper trading
        result = run_paper_trading(pred)
        state  = _load_state()

        log.info(f"Capital virtual: {result['virtual_capital']:,.2f} USD  "
                 f"Retorno acumulado: {result['total_return_pct']:+.2f}%  "
                 f"Operaciones: {result['n_trades']}")

        if result["closed_trade"]:
            ct = result["closed_trade"]
            log.info(f"Cerrada: {ct['signal_str']}  "
                     f"P&L={ct['pnl_pct']:+.2f}%  ({ct['pnl_usd']:+.2f} USD)")

        if result["opened_trade"]:
            ot = result["opened_trade"]
            log.info(f"Abierta: {ot['signal_str']}  precio={ot['entry_price']:.2f}")

        log.info("FIN: paper trading OK")

    except Exception:
        log.error("ERROR en paper trading:")
        log.error(traceback.format_exc())


def job_health_check():
    """Log de que el scheduler sigue activo."""
    log.info(f"[HEALTH] Scheduler activo. Proxima ejecucion: lunes 09:00")


# ─── Programar tareas ─────────────────────────────────────────────────────────

def setup_schedule():
    # Paper trading: cada lunes a las 09:00
    schedule.every().monday.at("09:00").do(job_paper_trading)

    # Health check: cada hora
    schedule.every().hour.do(job_health_check)

    log.info("Scheduler configurado:")
    log.info("  - Paper trading: lunes 09:00")
    log.info(f"  - Re-entrenamiento: cada {REFIT_EVERY_WEEKS} semanas")
    log.info("  - Health check: cada hora")
    log.info(f"  - Log: {LOG_FILE}")


# ─── Bucle principal ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("BOT DE ORO — SCHEDULER INICIADO")
    log.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 50)

    setup_schedule()

    # Ejecutar health check inmediato al arrancar
    job_health_check()

    log.info("Esperando proxima tarea programada... (Ctrl+C para detener)")

    while True:
        schedule.run_pending()
        time.sleep(30)   # comprobar cada 30 segundos


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Scheduler detenido por el usuario.")

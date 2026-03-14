# -*- coding: utf-8 -*-
"""
config.py
Parámetros centrales para bot_trading/ — bot semanal de oro con MT5.
Importado por todos los demás módulos.
"""

# ─── Periodo de análisis ───────────────────────────────────────────────────────
START       = "2000-01-01"
END         = "2026-03-31"
TRAIN_START = "2000-01-01"
TEST_START  = "2018-01-01"   # walk-forward test desde 2018

# ─── Tickers Yahoo Finance (frecuencia semanal) ────────────────────────────────
TICKERS = {
    "gold":   "GC=F",
    "dxy":    "DX-Y.NYB",
    "sp500":  "^GSPC",
    "vix":    "^VIX",
    "wti":    "CL=F",
    "silver": "SI=F",
}

# ─── Hiperparámetros XGBoost ──────────────────────────────────────────────────
# Sin early stopping — bug confirmado con datasets pequeños
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "n_estimators":     300,
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "random_state":     42,
    "eval_metric":      "logloss",
    "refit_every":      52,   # re-entrenar cada 52 semanas (1 año)
}

# ─── Hiperparámetros LightGBM ─────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":       "binary",
    "n_estimators":    300,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 20,
    "random_state":    42,
    "verbose":         -1,
    "refit_every":     52,
}

# ─── Pesos del ensemble [XGB, LGBM] ──────────────────────────────────────────
ENSEMBLE_WEIGHTS = [0.50, 0.50]

# ─── Umbrales de señal ────────────────────────────────────────────────────────
THRESH_UP   = 0.55   # prob > 0.55 → BUY
THRESH_DOWN = 0.45   # prob < 0.45 → CASH

# ─── Mínimo de observaciones para entrenar ────────────────────────────────────
MIN_TRAIN_OBS = 200   # ~4 años de semanas

# ─── Capital inicial del backtest ─────────────────────────────────────────────
INITIAL_CAPITAL = 1000.0   # USD

# ─── MetaTrader 5 ─────────────────────────────────────────────────────────────
SYMBOL         = "XAUUSD"
MAGIC_NUMBER   = 20260314
LIVE_TRADING   = False        # True = ejecutar órdenes reales

# ─── Gestión de riesgo ────────────────────────────────────────────────────────
RISK_PCT_PER_TRADE = 0.01    # 1% del capital por trade
ATR_PERIOD         = 14
SL_ATR_MULT        = 2.0     # stop-loss = 2× ATR
TP_ATR_MULT        = 3.0     # take-profit = 3× ATR (ratio riesgo:beneficio 1:3)

# ─── Paths ────────────────────────────────────────────────────────────────────
import pathlib
BASE_DIR   = pathlib.Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Semilla global ───────────────────────────────────────────────────────────
SEED = 42

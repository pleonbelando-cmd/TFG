"""
config.py — Constantes globales del TFG "Gold Price Dynamics"

Periodo de análisis, códigos de series, episodios históricos y rutas.
"""

from pathlib import Path
from datetime import date

# ── Rutas del proyecto ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FINAL = PROJECT_ROOT / "data" / "final"
DATA_MANUAL = PROJECT_ROOT / "data" / "manual"
OUTPUT_FIGURES = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "output" / "tables"

# ── Periodo de análisis ─────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"

# ── Series FRED ──────────────────────────────────────────────────────────────
# Clave: nombre interno → Valor: código FRED
FRED_SERIES = {
    # Nota: la serie GOLDAMGBD228NLBM fue retirada de FRED.
    # El oro se descarga via Yahoo Finance (GC=F).
    "dxy_new":    "DTWEXBGS",            # Trade Weighted USD Index, Broad (2006–presente)
    "dxy_old":    "DTWEXB",              # Trade Weighted USD Index, Broad (2000–2019, legacy)
    "tips_10y":   "DFII10",              # 10-Year TIPS (real yield)
    "dgs10":      "DGS10",               # 10-Year Treasury Constant Maturity Rate
    "cpi":        "CPIAUCSL",            # CPI for All Urban Consumers (SA)
    "breakeven":  "T10YIE",              # 10-Year Breakeven Inflation Rate
    "vix":        "VIXCLS",              # CBOE Volatility Index
    "wti":        "DCOILWTICO",          # WTI Crude Oil (USD/barrel)
    "fedfunds":   "FEDFUNDS",            # Federal Funds Effective Rate
    "yield_curve": "T10Y2Y",             # Pendiente curva 10Y-2Y (ciclo económico)
    "hy_spread":  "BAMLH0A0HYM2",       # High Yield OAS spread (apetito al riesgo)
}

# ── Yahoo Finance ────────────────────────────────────────────────────────────
YAHOO_TICKERS = {
    "gold":  "GC=F",                     # Gold Futures (USD/oz) — LBMA retirado de FRED
    "sp500": "^GSPC",                    # S&P 500 Index
}

# ── Google Trends ────────────────────────────────────────────────────────────
GOOGLE_TRENDS_KEYWORD = "gold price"
GOOGLE_TRENDS_GEO = ""  # Worldwide

# ── Episodios históricos (del Capítulo 2) ────────────────────────────────────
# Cada episodio: (nombre, fecha_inicio, fecha_fin, color, alpha)
EPISODES = {
    "gfc": {
        "name": "Crisis Financiera Global",
        "start": date(2007, 8, 1),
        "end": date(2009, 6, 30),
        "color": "#D32F2F",  # rojo
        "alpha": 0.12,
    },
    "post_qe_peak": {
        "name": "Pico post-QE y corrección",
        "start": date(2011, 7, 1),
        "end": date(2013, 6, 30),
        "color": "#F57C00",  # naranja
        "alpha": 0.12,
    },
    "covid": {
        "name": "Pandemia COVID-19",
        "start": date(2020, 2, 1),
        "end": date(2020, 8, 31),
        "color": "#7B1FA2",  # púrpura
        "alpha": 0.12,
    },
    "rate_hike": {
        "name": "Ciclo de subidas de tipos",
        "start": date(2022, 3, 1),
        "end": date(2024, 7, 31),
        "color": "#1565C0",  # azul
        "alpha": 0.12,
    },
    "triple_confluence": {
        "name": "Triple confluencia 2025",
        "start": date(2025, 1, 1),
        "end": date(2025, 12, 31),
        "color": "#2E7D32",  # verde
        "alpha": 0.12,
    },
}

# ── Nombres legibles para gráficos ──────────────────────────────────────────
VARIABLE_LABELS = {
    "gold":         "Oro (USD/oz)",
    "dxy":          "DXY (Índice dólar)",
    "tips_10y":     "TIPS 10Y (%)",
    "cpi_yoy":      "Inflación CPI (% YoY)",
    "breakeven":    "Breakeven 10Y (%)",
    "vix":          "VIX",
    "sp500":        "S&P 500",
    "sp500_ret":    "Retorno S&P 500 (%)",
    "wti":          "WTI (USD/barril)",
    "cb_reserves":  "Reservas BC (Δ toneladas)",
    "google_trends":"Google Trends «gold price»",
    "etf_flows":    "ETF flows (toneladas)",
    "fedfunds":     "Fed Funds Rate (%)",
    "yield_curve":  "Pendiente curva 10Y-2Y (pp)",
    "hy_spread":    "High Yield spread (pp)",
}

# ── Signos esperados (del Capítulo 3, ecuación 3.6) ─────────────────────────
EXPECTED_SIGNS = {
    "dxy":          "negative",
    "tips_10y":     "negative",
    "cpi_yoy":      "positive",
    "breakeven":    "positive",
    "vix":          "positive",
    "sp500_ret":    "negative",
    "wti":          "positive",
    "cb_reserves":  "positive",
    "google_trends":"positive",
    "etf_flows":    "positive",
}

# ── Dataset maestro ─────────────────────────────────────────────────────────
MASTER_DATASET_NAME = "gold_macro_monthly.csv"

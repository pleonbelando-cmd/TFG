# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Academic thesis (TFG) analyzing gold price dynamics 2000–2025 using econometrics and machine learning. Written in Spanish. Python codebase for data pipeline, EDA, econometric models, and ML forecasting.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (download → clean → merge → validate)
python -m src.data.pipeline

# Skip API downloads, reuse existing raw data
python -m src.data.pipeline --skip-download

# Run individual modules
python -c "from src.data.pipeline import load_master_dataset; df = load_master_dataset()"
python -c "from src.eda.visualizations import generate_all_figures; ..."
python -c "from src.econometrics.unit_root_tests import generate_unit_root_table; ..."
```

Requires `FRED_API_KEY` in `.env` file (see `.env.example`). On Windows, use `python -X utf8` for console output with Unicode characters (Δ, ≤, etc.).

## Architecture

**Central config**: `src/config.py` — all FRED codes, Yahoo tickers, episode dates, variable labels, expected signs, and paths. Every module imports from here.

**Data pipeline** (`src/data/`): `pipeline.py` orchestrates `download.py` → `clean.py` → `merge.py`. Output: `data/final/gold_macro_monthly.csv` (~312 rows × 19 columns, monthly frequency).

**EDA** (`src/eda/`): 17 figures in `visualizations.py` + `correlation_analysis.py`, 7 tables in `descriptive_stats.py` + `correlation_analysis.py`. All output to `output/figures/` (PNG+PDF) and `output/tables/` (CSV+LaTeX).

**Econometrics** (`src/econometrics/`): Unit root (ADF+KPSS), Johansen/Engle-Granger cointegration, VIF, Granger causality. Each module has a `generate_*` function that produces tables.

**Utils** (`src/utils/`): `plotting.py` (academic matplotlib style), `episode_markers.py` (crisis shading bands on time series), `latex_tables.py` (CSV+LaTeX export).

## Key design decisions

- **Gold price from Yahoo Finance** (`GC=F`), not FRED — the LBMA series was retired from FRED.
- **DXY combines two FRED series**: `DTWEXB` (2000–2019) and `DTWEXBGS` (2006–present), rescaled on overlap.
- **TIPS proxy 2000–2002**: `DGS10 − CPI_YoY` (ex-post real rate), since DFII10 starts Jan 2003.
- **VIX uses monthly mean** (not end-of-month) because it's mean-reverting.
- **CB reserves and ETF flows** are optional manual data from World Gold Council (`data/manual/`). Pipeline assigns NaN gracefully if missing.
- **Google Trends** downloaded in 5-year overlapping chunks with rescaling factor.
- **Episode column** in master dataset classifies each month as one of 5 crisis episodes or "calma".
- Variables in the model use **log transforms** for prices (gold, DXY, SP500, WTI) and levels for rates (TIPS, VIX, CPI).

## Five historical episodes (from Chapter 2)

Referenced throughout all visualizations and conditional analyses:
1. GFC (2007-08 to 2009-06)
2. Post-QE peak (2011-07 to 2013-06)
3. COVID-19 (2020-02 to 2020-08)
4. Rate hike cycle (2022-03 to 2024-07)
5. Triple confluence 2025 (2025-01 to 2025-12)

## Thesis chapters

Chapters are markdown files at project root (`capitulo_01_introduccion.md` through `capitulo_04_datos_eda.md`). Written in academic Spanish. Cross-reference figures as "Figura 4.X" and tables as "Tabla 4.X". Chapter 3 contains the formal equation (§3.6) that defines the model specification.

## Roadmap

See `ROADMAP.md`. Currently in Phase 2 (data + EDA). Phases 3–5 pending: VAR/VECM/GARCH econometrics, ML models (XGBoost, LSTM, SHAP), and final integration.

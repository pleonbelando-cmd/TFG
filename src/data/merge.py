"""
merge.py — Fusión de series limpias en el dataset maestro.

Genera gold_macro_monthly.csv con variables en niveles, logs, retornos,
y una columna de episodio para cada observación.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    DATA_FINAL,
    DATA_PROCESSED,
    END_DATE,
    EPISODES,
    MASTER_DATASET_NAME,
    START_DATE,
)

logger = logging.getLogger(__name__)

# Columnas que se incluyen en el dataset maestro (del paso clean)
COLUMNS_TO_MERGE = [
    "gold", "dxy", "tips_10y", "cpi_yoy", "breakeven", "vix",
    "sp500", "sp500_ret", "wti", "fedfunds",
    "cb_reserves", "google_trends", "etf_flows",
    "yield_curve", "hy_spread",
]


def _assign_episode(dt: date) -> str:
    """Asigna un episodio a una fecha, o 'calma' si no pertenece a ninguno."""
    d = dt.date() if hasattr(dt, "date") else dt
    for key, ep in EPISODES.items():
        if ep["start"] <= d <= ep["end"]:
            return key
    return "calma"


def merge_series(series_dict: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """
    Fusiona todas las series limpias en un único DataFrame mensual.

    Si series_dict es None, lee desde data/processed/.
    """
    if series_dict is None:
        series_dict = {}
        for col_name in COLUMNS_TO_MERGE:
            path = DATA_PROCESSED / f"{col_name}.csv"
            if path.exists():
                s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                s.name = col_name
                series_dict[col_name] = s
            else:
                logger.warning(f"No encontrado: {path}")

    # Crear índice mensual completo
    idx = pd.date_range(START_DATE, END_DATE, freq="ME")
    df = pd.DataFrame(index=idx)
    df.index.name = "date"

    # Merge de cada serie
    for name in COLUMNS_TO_MERGE:
        if name in series_dict:
            s = series_dict[name]
            s.index = pd.DatetimeIndex(s.index)
            # Reindex al índice maestro (fin de mes)
            df[name] = s.reindex(df.index, method="nearest", tolerance=pd.Timedelta("5D"))
        else:
            df[name] = np.nan

    # ── Variables derivadas ──────────────────────────────────────────────
    # Logaritmos (para la ecuación del Cap. 3)
    for var in ["gold", "dxy", "sp500", "wti"]:
        if var in df.columns and df[var].notna().any():
            df[f"ln_{var}"] = np.log(df[var].clip(lower=0.01))

    # Retornos logarítmicos del oro
    if "gold" in df.columns:
        df["gold_ret"] = np.log(df["gold"] / df["gold"].shift(1)) * 100

    # ── Columna de episodio ──────────────────────────────────────────────
    df["episode"] = df.index.map(_assign_episode)

    # ── Recortar al periodo de análisis ──────────────────────────────────
    df = df.loc[START_DATE:END_DATE]

    return df


def save_master_dataset(df: pd.DataFrame) -> None:
    """Guarda el dataset maestro en data/final/."""
    DATA_FINAL.mkdir(parents=True, exist_ok=True)
    filepath = DATA_FINAL / MASTER_DATASET_NAME
    df.to_csv(filepath)
    logger.info(f"Dataset maestro guardado: {filepath}")
    logger.info(f"  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    logger.info(f"  Periodo: {df.index.min()} → {df.index.max()}")
    logger.info(f"  Columnas: {list(df.columns)}")

    # Resumen de NaN por columna
    nan_pct = (df.isna().sum() / len(df) * 100).round(1)
    logger.info("  NaN por columna:")
    for col, pct in nan_pct.items():
        if pct > 0:
            logger.info(f"    {col}: {pct}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = merge_series()
    save_master_dataset(df)

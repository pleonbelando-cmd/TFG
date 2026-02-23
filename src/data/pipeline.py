"""
pipeline.py — Orquestador del pipeline de datos completo.

Ejecución: python -m src.data.pipeline

Flujo: descarga → limpieza → fusión → validación.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_FINAL, END_DATE, MASTER_DATASET_NAME, START_DATE

logger = logging.getLogger(__name__)


def run_pipeline(skip_download: bool = False) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo.

    Args:
        skip_download: Si True, salta la descarga y usa datos existentes en raw/.
    """
    from src.data.clean import clean_all
    from src.data.download import download_all
    from src.data.merge import merge_series, save_master_dataset

    # ── Paso 1: Descarga ─────────────────────────────────────────────────
    if not skip_download:
        logger.info("=" * 60)
        logger.info("PASO 1: Descarga de datos")
        logger.info("=" * 60)
        download_all()
    else:
        logger.info("Descarga omitida (skip_download=True)")

    # ── Paso 2: Limpieza ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 2: Limpieza y transformación")
    logger.info("=" * 60)
    series_dict = clean_all()

    # ── Paso 3: Fusión ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 3: Fusión en dataset maestro")
    logger.info("=" * 60)
    df = merge_series(series_dict)
    save_master_dataset(df)

    # ── Paso 4: Validación ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 4: Validación")
    logger.info("=" * 60)
    validate_dataset(df)

    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validación del dataset maestro.

    Comprueba:
    1. Número de filas razonable (~312)
    2. Rango de fechas correcto
    3. No hay columnas todo-NaN (excepto datos manuales opcionales)
    4. Oro tiene datos válidos
    """
    ok = True

    # 1. Filas
    expected_min = 280
    expected_max = 320
    n_rows = len(df)
    if expected_min <= n_rows <= expected_max:
        logger.info(f"  ✓ Filas: {n_rows} (esperado ~312)")
    else:
        logger.warning(f"  ✗ Filas: {n_rows} (esperado {expected_min}-{expected_max})")
        ok = False

    # 2. Rango de fechas
    date_min = df.index.min()
    date_max = df.index.max()
    if date_min.year == 2000 and date_max.year == 2025:
        logger.info(f"  ✓ Periodo: {date_min.date()} → {date_max.date()}")
    else:
        logger.warning(f"  ✗ Periodo inesperado: {date_min.date()} → {date_max.date()}")
        ok = False

    # 3. Columnas todo-NaN
    optional_cols = {"cb_reserves", "google_trends", "etf_flows"}
    all_nan = df.columns[df.isna().all()].tolist()
    mandatory_nan = [c for c in all_nan if c not in optional_cols]
    if mandatory_nan:
        logger.warning(f"  ✗ Columnas obligatorias todo-NaN: {mandatory_nan}")
        ok = False
    if all_nan:
        opt = [c for c in all_nan if c in optional_cols]
        if opt:
            logger.info(f"  ⚠ Columnas opcionales sin datos: {opt}")

    # 4. Oro
    gold_valid = df["gold"].notna().sum()
    if gold_valid > 200:
        logger.info(f"  ✓ Oro: {gold_valid} observaciones válidas")
    else:
        logger.warning(f"  ✗ Oro: solo {gold_valid} observaciones válidas")
        ok = False

    if ok:
        logger.info("  → Validación PASADA")
    else:
        logger.warning("  → Validación con advertencias")

    return ok


def load_master_dataset() -> pd.DataFrame:
    """Carga el dataset maestro desde data/final/."""
    path = DATA_FINAL / MASTER_DATASET_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset maestro no encontrado: {path}. "
            "Ejecuta primero: python -m src.data.pipeline"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    skip = "--skip-download" in sys.argv
    run_pipeline(skip_download=skip)

"""
run_chapter6.py — Ejecutar todos los modelos de ML del Capítulo 6.

Uso:
    python run_chapter6.py

Genera:
    - output/tables/tab_6_*.csv  (métricas y predicciones de cada modelo)
    - output/figures/fig_6_*.png (predicciones, SHAP, comparativa)
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.data.pipeline import load_master_dataset
from src.ml.comparison import run_all_models

if __name__ == "__main__":
    print("=" * 60)
    print("  Capítulo 6 — Modelos de Machine Learning")
    print("  Gold Price Dynamics TFG")
    print("=" * 60)

    df = load_master_dataset()
    print(f"\nDataset cargado: {len(df)} obs × {len(df.columns)} cols")
    print(f"Periodo: {df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}\n")

    comparison_df = run_all_models(df, train_frac=0.60)

    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES — Tabla 6.5")
    print("=" * 60)
    print(comparison_df.to_string())
    print("=" * 60)
    print("\nTodas las figuras y tablas guardadas en output/")

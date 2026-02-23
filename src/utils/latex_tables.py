"""
latex_tables.py — Exportación de DataFrames a formato LaTeX y CSV.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import OUTPUT_TABLES

logger = logging.getLogger(__name__)


def save_table(
    df: pd.DataFrame,
    name: str,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
    formats: tuple = ("csv", "tex"),
) -> None:
    """
    Guarda una tabla en output/tables/ como CSV y/o LaTeX.

    Args:
        df: DataFrame a exportar.
        name: Nombre base (sin extensión), e.g. "tab_4_01_descriptive".
        caption: Título de la tabla para LaTeX.
        label: Etiqueta de referencia LaTeX, e.g. "tab:descriptive".
        float_format: Formato numérico.
        formats: Tupla de formatos de salida.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    if "csv" in formats:
        csv_path = OUTPUT_TABLES / f"{name}.csv"
        df.to_csv(csv_path)
        logger.info(f"Tabla guardada: {csv_path}")

    if "tex" in formats:
        tex_path = OUTPUT_TABLES / f"{name}.tex"
        latex_str = df.to_latex(
            float_format=float_format,
            caption=caption or None,
            label=label or None,
            escape=True,
            multirow=True,
        )
        tex_path.write_text(latex_str, encoding="utf-8")
        logger.info(f"Tabla LaTeX guardada: {tex_path}")


def format_significance(pval: float) -> str:
    """Formatea significancia estadística con asteriscos."""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    return ""


def format_number(x: float, decimals: int = 4) -> str:
    """Formatea un número con el número de decimales indicado."""
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"

"""
descriptive_stats.py — Estadísticas descriptivas y tablas para el Capítulo 4.

Genera:
- Tabla 4.1: Estadísticas descriptivas completas
- Tabla 4.3: Estadísticas condicionales (crisis vs. calma)
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import VARIABLE_LABELS
from src.utils.latex_tables import save_table

logger = logging.getLogger(__name__)

# Variables principales para las tablas descriptivas
MAIN_VARS = [
    "gold", "dxy", "tips_10y", "cpi_yoy", "breakeven", "vix",
    "sp500", "wti", "fedfunds", "cb_reserves", "google_trends", "etf_flows",
]


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla 4.1: Estadísticas descriptivas completas.

    Incluye: N, media, mediana, std, min, max, skewness, kurtosis, Jarque-Bera.
    """
    cols = [c for c in MAIN_VARS if c in df.columns and df[c].notna().any()]
    records = []

    for col in cols:
        s = df[col].dropna()
        n = len(s)
        jb_stat, jb_pval = stats.jarque_bera(s) if n > 10 else (np.nan, np.nan)

        records.append({
            "Variable": VARIABLE_LABELS.get(col, col),
            "N": n,
            "Media": s.mean(),
            "Mediana": s.median(),
            "Desv. Est.": s.std(),
            "Mín.": s.min(),
            "Máx.": s.max(),
            "Asimetría": s.skew(),
            "Curtosis": s.kurtosis(),  # exceso (Fisher)
            "JB": jb_stat,
            "JB p-val": jb_pval,
        })

    result = pd.DataFrame(records).set_index("Variable")
    return result


def compute_conditional_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla 4.3: Estadísticas condicionales — crisis vs. calma.

    Crisis = cualquier episodio distinto de 'calma'.
    """
    if "episode" not in df.columns:
        raise ValueError("El DataFrame necesita columna 'episode'")

    cols = [c for c in MAIN_VARS if c in df.columns and df[c].notna().any()]
    crisis = df[df["episode"] != "calma"]
    calma = df[df["episode"] == "calma"]

    records = []
    for col in cols:
        s_crisis = crisis[col].dropna()
        s_calma = calma[col].dropna()

        record = {
            "Variable": VARIABLE_LABELS.get(col, col),
            "N (crisis)": len(s_crisis),
            "Media (crisis)": s_crisis.mean(),
            "Std (crisis)": s_crisis.std(),
            "N (calma)": len(s_calma),
            "Media (calma)": s_calma.mean(),
            "Std (calma)": s_calma.std(),
        }

        # Test de diferencia de medias (Welch t-test)
        if len(s_crisis) > 5 and len(s_calma) > 5:
            t_stat, t_pval = stats.ttest_ind(s_crisis, s_calma, equal_var=False)
            record["t-stat"] = t_stat
            record["p-val"] = t_pval
        else:
            record["t-stat"] = np.nan
            record["p-val"] = np.nan

        records.append(record)

    result = pd.DataFrame(records).set_index("Variable")
    return result


def generate_all_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Genera y guarda todas las tablas descriptivas."""
    tables = {}

    # Tabla 4.1
    logger.info("Generando Tabla 4.1: Estadísticas descriptivas")
    tab1 = compute_descriptive_stats(df)
    save_table(
        tab1, "tab_4_01_descriptive",
        caption="Estadísticas descriptivas de las variables principales (2000--2025)",
        label="tab:descriptive",
        float_format="%.3f",
    )
    tables["descriptive"] = tab1

    # Tabla 4.3
    logger.info("Generando Tabla 4.3: Estadísticas condicionales crisis vs. calma")
    tab3 = compute_conditional_stats(df)
    save_table(
        tab3, "tab_4_03_conditional",
        caption="Estadísticas condicionales: periodos de crisis vs.\\ calma",
        label="tab:conditional",
        float_format="%.3f",
    )
    tables["conditional"] = tab3

    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_all_tables(df)

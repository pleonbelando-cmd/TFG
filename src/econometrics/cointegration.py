"""
cointegration.py — Tests de cointegración: Johansen y Engle-Granger.

Genera Tabla 4.5: Nº de vectores de cointegración y estadísticos.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint

from src.config import VARIABLE_LABELS
from src.utils.latex_tables import format_significance, save_table

logger = logging.getLogger(__name__)


def run_johansen(
    df: pd.DataFrame,
    variables: list[str],
    det_order: int = 0,
    k_ar_diff: int = 2,
) -> dict:
    """
    Ejecuta test de Johansen para cointegración multivariante.

    Args:
        df: DataFrame con las variables.
        variables: Lista de nombres de columna a incluir.
        det_order: -1 (sin determinístico), 0 (constante), 1 (tendencia).
        k_ar_diff: Número de lags en diferencias.

    Returns:
        Dict con trace stats, eigenvalue stats, valores críticos y nº de vectores.
    """
    cols = [c for c in variables if c in df.columns]
    data = df[cols].dropna()

    if len(data) < 50:
        logger.warning(f"Muestra insuficiente para Johansen: {len(data)} obs")
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    n_vars = len(cols)
    trace_records = []
    eigen_records = []

    for r in range(n_vars):
        # Trace test
        trace_records.append({
            "H0: r ≤": r,
            "Trace stat": result.lr1[r],
            "CV 10%": result.cvt[r, 0],
            "CV 5%": result.cvt[r, 1],
            "CV 1%": result.cvt[r, 2],
            "Rechaza 5%": "Sí" if result.lr1[r] > result.cvt[r, 1] else "No",
        })

        # Max eigenvalue test
        eigen_records.append({
            "H0: r =": r,
            "Max-eigen stat": result.lr2[r],
            "CV 10%": result.cvm[r, 0],
            "CV 5%": result.cvm[r, 1],
            "CV 1%": result.cvm[r, 2],
            "Rechaza 5%": "Sí" if result.lr2[r] > result.cvm[r, 1] else "No",
        })

    # Determinar número de vectores de cointegración (trace test al 5%)
    n_coint_trace = sum(1 for r in trace_records if r["Rechaza 5%"] == "Sí")
    n_coint_eigen = sum(1 for r in eigen_records if r["Rechaza 5%"] == "Sí")

    return {
        "trace": pd.DataFrame(trace_records),
        "eigenvalue": pd.DataFrame(eigen_records),
        "n_coint_trace": n_coint_trace,
        "n_coint_eigen": n_coint_eigen,
        "variables": cols,
    }


def run_engle_granger(df: pd.DataFrame, y: str, x: str) -> dict:
    """
    Ejecuta test de Engle-Granger para cointegración bivariante.

    H0: No hay cointegración entre y y x.

    Args:
        y: Variable dependiente (nombre de columna).
        x: Variable independiente (nombre de columna).
    """
    data = df[[y, x]].dropna()
    if len(data) < 30:
        return {"pval": np.nan, "stat": np.nan}

    stat, pval, crit_values = coint(data[y], data[x])

    return {
        "y": y,
        "x": x,
        "stat": stat,
        "pval": pval,
        "cv_1pct": crit_values[0],
        "cv_5pct": crit_values[1],
        "cv_10pct": crit_values[2],
    }


def generate_cointegration_tables(df: pd.DataFrame) -> dict:
    """
    Genera Tabla 4.5 con resultados de Johansen y Engle-Granger.
    """
    logger.info("Ejecutando tests de cointegración...")
    results = {}

    # ── Johansen multivariante ───────────────────────────────────────────
    # Sistema principal: variables I(1) en niveles
    i1_candidates = ["gold", "dxy", "tips_10y", "sp500", "wti"]
    i1_vars = [v for v in i1_candidates if v in df.columns and df[v].notna().sum() > 50]

    if len(i1_vars) >= 2:
        logger.info(f"  Johansen sobre: {i1_vars}")
        johansen = run_johansen(df, i1_vars)
        results["johansen"] = johansen

        if "trace" in johansen:
            # Guardar trace y eigenvalue por separado
            trace_df = johansen["trace"].copy()
            trace_df.insert(0, "Test", "Trace")
            eigen_df = johansen["eigenvalue"].copy()
            eigen_df.insert(0, "Test", "Max-Eigenvalue")

            # Renombrar columnas del eigenvalue para alinear
            eigen_df = eigen_df.rename(columns={
                "H0: r =": "H0",
                "Max-eigen stat": "Estadístico",
            })
            trace_df = trace_df.rename(columns={
                "H0: r ≤": "H0",
                "Trace stat": "Estadístico",
            })

            combined = pd.concat([trace_df, eigen_df], ignore_index=True)

            save_table(
                combined.set_index(["Test", "H0"]),
                "tab_4_05_cointegration_johansen",
                caption=f"Test de Johansen: {', '.join(i1_vars)} (det\\_order=0, lags=2)",
                label="tab:johansen",
                float_format="%.3f",
            )
            logger.info(
                f"  Johansen trace: {johansen['n_coint_trace']} vectores de cointegración; "
                f"Max-eigen: {johansen['n_coint_eigen']} vectores"
            )

    # ── Engle-Granger bivariante (robustez) ──────────────────────────────
    eg_pairs = [("gold", "tips_10y"), ("gold", "dxy"), ("gold", "wti")]
    eg_records = []

    for y, x in eg_pairs:
        if y in df.columns and x in df.columns:
            logger.info(f"  Engle-Granger: {y} ~ {x}")
            eg = run_engle_granger(df, y, x)
            y_label = VARIABLE_LABELS.get(y, y)
            x_label = VARIABLE_LABELS.get(x, x)
            eg_records.append({
                "Par": f"{y_label} ~ {x_label}",
                "EG stat": eg["stat"],
                "p-val": eg["pval"],
                "CV 1%": eg["cv_1pct"],
                "CV 5%": eg["cv_5pct"],
                "CV 10%": eg["cv_10pct"],
                "Cointegra (5%)": "Sí" if eg["pval"] < 0.05 else "No",
            })

    if eg_records:
        eg_df = pd.DataFrame(eg_records).set_index("Par")
        save_table(
            eg_df,
            "tab_4_05b_cointegration_eg",
            caption="Test de Engle-Granger: cointegración bivariante (robustez)",
            label="tab:engle_granger",
            float_format="%.3f",
        )
        results["engle_granger"] = eg_df

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_cointegration_tables(df)

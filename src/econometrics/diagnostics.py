"""
diagnostics.py — VIF (multicolinealidad) y causalidad de Granger.

Genera:
- Tabla 4.6: Causalidad de Granger bilateral
- Tabla 4.7: Variance Inflation Factors (VIF)
"""

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import VARIABLE_LABELS
from src.utils.latex_tables import format_significance, save_table

logger = logging.getLogger(__name__)

# Variables regresoras (del Capítulo 3, ecuación 3.6)
REGRESSORS = [
    "dxy", "tips_10y", "cpi_yoy", "breakeven",
    "vix", "sp500", "wti", "fedfunds",
]


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla 4.7: Variance Inflation Factors para los regresores.

    VIF > 10 indica multicolinealidad severa.
    VIF > 5 indica multicolinealidad moderada.
    """
    cols = [c for c in REGRESSORS if c in df.columns and df[c].notna().any()]
    data = df[cols].dropna()

    if len(data) < len(cols) + 10:
        logger.warning("Muestra insuficiente para VIF")
        return pd.DataFrame()

    # Añadir constante para el cálculo
    from statsmodels.tools import add_constant
    X = add_constant(data)

    records = []
    for i, col in enumerate(cols):
        # Index +1 para saltar la constante
        vif_val = variance_inflation_factor(X.values, i + 1)
        flag = ""
        if vif_val > 10:
            flag = "SEVERA"
        elif vif_val > 5:
            flag = "Moderada"

        records.append({
            "Variable": VARIABLE_LABELS.get(col, col),
            "VIF": vif_val,
            "Diagnóstico": flag,
        })

    result = pd.DataFrame(records).set_index("Variable")
    return result


def run_granger_causality(
    df: pd.DataFrame,
    target: str = "gold",
    max_lag: int = 12,
    lags_to_report: tuple = (1, 3, 6, 12),
) -> pd.DataFrame:
    """
    Tabla 4.6: Causalidad de Granger bilateral — oro ↔ cada catalizador.

    Ejecuta test F de Granger para cada par en ambas direcciones.
    Reporta p-valores para los lags especificados.

    Nota: "causalidad de Granger" = precedencia predictiva, no causalidad económica.
    """
    catalysts = [c for c in REGRESSORS if c in df.columns and df[c].notna().any()]
    records = []

    for cat in catalysts:
        # Dirección: catalizador → oro
        data_fwd = df[[target, cat]].dropna()
        if len(data_fwd) < max_lag + 20:
            continue

        label_cat = VARIABLE_LABELS.get(cat, cat)
        label_tgt = VARIABLE_LABELS.get(target, target)

        record_fwd = {"Dirección": f"{label_cat} → {label_tgt}"}
        record_bwd = {"Dirección": f"{label_tgt} → {label_cat}"}

        for lag in lags_to_report:
            if lag > max_lag:
                continue

            # Forward: cat → target
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc_fwd = grangercausalitytests(
                        data_fwd[[target, cat]], maxlag=lag, verbose=False
                    )
                pval_fwd = gc_fwd[lag][0]["ssr_ftest"][1]
                record_fwd[f"Lag {lag} p-val"] = pval_fwd
                record_fwd[f"Lag {lag} sig"] = format_significance(pval_fwd)
            except Exception:
                record_fwd[f"Lag {lag} p-val"] = np.nan
                record_fwd[f"Lag {lag} sig"] = ""

            # Backward: target → cat
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc_bwd = grangercausalitytests(
                        data_fwd[[cat, target]], maxlag=lag, verbose=False
                    )
                pval_bwd = gc_bwd[lag][0]["ssr_ftest"][1]
                record_bwd[f"Lag {lag} p-val"] = pval_bwd
                record_bwd[f"Lag {lag} sig"] = format_significance(pval_bwd)
            except Exception:
                record_bwd[f"Lag {lag} p-val"] = np.nan
                record_bwd[f"Lag {lag} sig"] = ""

        records.append(record_fwd)
        records.append(record_bwd)

    result = pd.DataFrame(records).set_index("Dirección")
    return result


def generate_diagnostic_tables(df: pd.DataFrame) -> dict:
    """Genera y guarda Tablas 4.6 (Granger) y 4.7 (VIF)."""
    results = {}

    # ── Tabla 4.7: VIF ───────────────────────────────────────────────────
    logger.info("Calculando VIF...")
    vif_df = compute_vif(df)
    if not vif_df.empty:
        save_table(
            vif_df,
            "tab_4_07_vif",
            caption="Variance Inflation Factors (VIF) de los regresores",
            label="tab:vif",
            float_format="%.2f",
        )
        n_severe = (vif_df["VIF"] > 10).sum()
        n_moderate = ((vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)).sum()
        logger.info(f"  VIF > 10 (severa): {n_severe}")
        logger.info(f"  VIF 5-10 (moderada): {n_moderate}")
        results["vif"] = vif_df

    # ── Tabla 4.6: Granger ───────────────────────────────────────────────
    logger.info("Ejecutando causalidad de Granger bilateral...")
    granger_df = run_granger_causality(df)
    if not granger_df.empty:
        save_table(
            granger_df,
            "tab_4_06_granger",
            caption="Causalidad de Granger bilateral: oro $\\leftrightarrow$ catalizadores",
            label="tab:granger",
            float_format="%.3f",
        )
        results["granger"] = granger_df

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_diagnostic_tables(df)

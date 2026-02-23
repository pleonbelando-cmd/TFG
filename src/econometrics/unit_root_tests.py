"""
unit_root_tests.py — Tests de raíz unitaria: ADF y KPSS.

Genera Tabla 4.4: Clasificación de variables como I(0) o I(1).
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import VARIABLE_LABELS
from src.utils.latex_tables import format_significance, save_table

logger = logging.getLogger(__name__)

# Variables a testear
VARS_TO_TEST = [
    "gold", "dxy", "tips_10y", "cpi_yoy", "breakeven",
    "vix", "sp500", "wti", "fedfunds",
]


def run_adf(series: pd.Series, regression: str = "c") -> dict:
    """
    Ejecuta test ADF (Augmented Dickey-Fuller).

    H0: La serie tiene raíz unitaria (no estacionaria).

    Args:
        series: Serie temporal (sin NaN).
        regression: 'c' (constante), 'ct' (constante + tendencia), 'n' (ninguno).

    Returns:
        Dict con estadístico, p-valor, lags, y valores críticos.
    """
    result = adfuller(series, autolag="AIC", regression=regression)
    return {
        "adf_stat": result[0],
        "adf_pval": result[1],
        "adf_lags": result[2],
        "adf_nobs": result[3],
        "adf_cv_1pct": result[4]["1%"],
        "adf_cv_5pct": result[4]["5%"],
        "adf_cv_10pct": result[4]["10%"],
    }


def run_kpss(series: pd.Series, regression: str = "c") -> dict:
    """
    Ejecuta test KPSS (Kwiatkowski-Phillips-Schmidt-Shin).

    H0: La serie es estacionaria.

    Args:
        series: Serie temporal (sin NaN).
        regression: 'c' (level stationarity) o 'ct' (trend stationarity).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, lags, cv = kpss(series, regression=regression, nlags="auto")

    return {
        "kpss_stat": stat,
        "kpss_pval": pval,
        "kpss_lags": lags,
        "kpss_cv_1pct": cv["1%"],
        "kpss_cv_5pct": cv["5%"],
        "kpss_cv_10pct": cv["10%"],
    }


def classify_integration_order(adf_pval: float, kpss_pval: float) -> str:
    """
    Clasifica el orden de integración basándose en ADF + KPSS.

    Estrategia de confirmación:
    - ADF rechaza (p < 0.05) + KPSS no rechaza (p > 0.05) → I(0)
    - ADF no rechaza + KPSS rechaza → I(1)
    - Ambos rechazan → Resultado mixto (posible tendencia)
    - Ninguno rechaza → Resultado mixto
    """
    adf_rejects = adf_pval < 0.05
    kpss_rejects = kpss_pval < 0.05

    if adf_rejects and not kpss_rejects:
        return "I(0)"
    elif not adf_rejects and kpss_rejects:
        return "I(1)"
    elif adf_rejects and kpss_rejects:
        return "Mixto (tendencia)"
    else:
        return "Mixto"


def run_unit_root_battery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta ADF y KPSS en niveles y primeras diferencias para todas las variables.

    Returns:
        DataFrame con resultados de Tabla 4.4.
    """
    records = []

    for var in VARS_TO_TEST:
        if var not in df.columns or df[var].notna().sum() < 30:
            continue

        series = df[var].dropna()
        label = VARIABLE_LABELS.get(var, var)

        # ── En niveles ───────────────────────────────────────────────────
        adf_level = run_adf(series, regression="c")
        kpss_level = run_kpss(series, regression="c")
        classification_level = classify_integration_order(
            adf_level["adf_pval"], kpss_level["kpss_pval"]
        )

        records.append({
            "Variable": label,
            "Forma": "Nivel",
            "ADF stat": adf_level["adf_stat"],
            "ADF p-val": adf_level["adf_pval"],
            "ADF sig": format_significance(adf_level["adf_pval"]),
            "ADF lags": adf_level["adf_lags"],
            "KPSS stat": kpss_level["kpss_stat"],
            "KPSS p-val": kpss_level["kpss_pval"],
            "KPSS sig": format_significance(kpss_level["kpss_pval"]),
            "Clasificación": classification_level,
        })

        # ── En primeras diferencias ──────────────────────────────────────
        diff = series.diff().dropna()
        if len(diff) < 20:
            continue

        adf_diff = run_adf(diff, regression="c")
        kpss_diff = run_kpss(diff, regression="c")
        classification_diff = classify_integration_order(
            adf_diff["adf_pval"], kpss_diff["kpss_pval"]
        )

        records.append({
            "Variable": label,
            "Forma": "Δ (1ª dif.)",
            "ADF stat": adf_diff["adf_stat"],
            "ADF p-val": adf_diff["adf_pval"],
            "ADF sig": format_significance(adf_diff["adf_pval"]),
            "ADF lags": adf_diff["adf_lags"],
            "KPSS stat": kpss_diff["kpss_stat"],
            "KPSS p-val": kpss_diff["kpss_pval"],
            "KPSS sig": format_significance(kpss_diff["kpss_pval"]),
            "Clasificación": classification_diff,
        })

    result = pd.DataFrame(records)
    return result


def generate_unit_root_table(df: pd.DataFrame) -> pd.DataFrame:
    """Genera y guarda Tabla 4.4."""
    logger.info("Ejecutando tests de raíz unitaria (ADF + KPSS)...")
    results = run_unit_root_battery(df)

    save_table(
        results.set_index(["Variable", "Forma"]),
        "tab_4_04_unit_root",
        caption="Tests de raíz unitaria: ADF y KPSS en niveles y primeras diferencias",
        label="tab:unit_root",
        float_format="%.3f",
    )

    # Resumen
    levels = results[results["Forma"] == "Nivel"]
    i1_count = (levels["Clasificación"].str.contains("I\\(1\\)")).sum()
    logger.info(f"  Variables I(1) en niveles: {i1_count} de {len(levels)}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_unit_root_table(df)

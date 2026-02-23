"""
var_model.py — Estimación VAR/VECM, funciones de impulso-respuesta y FEVD.

Genera figuras y tablas del Capítulo 5:
- Tab 5.1: Selección de lags del VAR (AIC/BIC/HQIC)
- Tab 5.2: Vector de cointegración normalizado (VECM)
- Tab 5.3: Coeficientes del VECM
- Fig 5.1: Funciones de Impulso-Respuesta ortogonalizadas (Cholesky)
- Fig 5.2: Descomposición de varianza del error de predicción (FEVD)
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order

from src.config import EPISODES, OUTPUT_FIGURES, OUTPUT_TABLES, VARIABLE_LABELS
from src.utils.latex_tables import save_table
from src.utils.plotting import set_academic_style

logger = logging.getLogger(__name__)

# ── Variables I(1) para el sistema VAR/VECM ─────────────────────────────────
# Resultado de los tests ADF+KPSS del Capítulo 4 (Tabla 4.4)
I1_VARS = ["gold", "dxy", "tips_10y", "sp500"]

# Variables I(0) que actuarán como exógenas en la ecuación del oro (robustez)
I0_VARS = ["vix", "cpi_yoy"]

# Número de vectores de cointegración (test de Johansen max-eigenvalue, Tab 4.5)
COINT_RANK = 1

# Horizonte de predicción para IRF y FEVD
IRF_HORIZON = 24  # meses

# Colores para figuras (paleta académica)
PALETTE = ["#1565C0", "#D32F2F", "#2E7D32", "#F57C00", "#6A1B9A"]


def _log_transform(df: pd.DataFrame, vars_: list[str]) -> pd.DataFrame:
    """
    Aplica logaritmo natural a las variables de precio para obtener
    elasticidades directamente interpretables en los coeficientes.
    TIPS no se transforma porque puede ser negativo.
    """
    df = df.copy()
    for v in vars_:
        if v in df.columns and df[v].min() > 0:
            df[f"ln_{v}"] = np.log(df[v])
    return df


def prepare_var_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset para la estimación VAR/VECM.

    Transforma en logaritmos las variables de precio (gold, dxy, sp500).
    TIPS 10Y entra en niveles (puede ser negativo en periodos de rep. monetaria).
    Devuelve DataFrame con solo las columnas relevantes y sin NaN.
    """
    df = _log_transform(df, ["gold", "dxy", "sp500"])
    cols = ["ln_gold", "ln_dxy", "tips_10y", "ln_sp500"]
    available = [c for c in cols if c in df.columns]
    data = df[available].dropna()
    logger.info(f"Dataset VAR: {len(data)} observaciones, variables: {available}")
    return data


def select_var_lags(data: pd.DataFrame, max_lags: int = 12) -> pd.DataFrame:
    """
    Tabla 5.1: Selección de lags óptimos mediante AIC, BIC y HQIC.

    Estima VAR(p) para p = 1, ..., max_lags y presenta los criterios de
    información. El lag óptimo minimiza el criterio seleccionado.
    AIC favorece la predicción fuera de muestra; BIC penaliza más la
    complejidad y es más parsimonioso.
    """
    model = VAR(data)
    results_order = []

    for p in range(1, max_lags + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(p)
            results_order.append({
                "Lags (p)": p,
                "AIC": fit.aic,
                "BIC": fit.bic,
                "HQIC": fit.hqic,
                "FPE": fit.fpe,
            })
        except Exception as e:
            logger.debug(f"  VAR({p}) falló: {e}")

    df_order = pd.DataFrame(results_order)

    # Marcar óptimos
    for crit in ["AIC", "BIC", "HQIC"]:
        opt_idx = df_order[crit].idxmin()
        df_order.loc[opt_idx, f"{crit}_opt"] = "★"

    logger.info(
        f"  Lag óptimo AIC: {df_order.loc[df_order['AIC'].idxmin(), 'Lags (p)']}, "
        f"BIC: {df_order.loc[df_order['BIC'].idxmin(), 'Lags (p)']}"
    )

    save_table(
        df_order.set_index("Lags (p)"),
        "tab_5_01_var_lag_selection",
        caption="Selección de orden del VAR: criterios de información (AIC, BIC, HQIC)",
        label="tab:var_lags",
        float_format="%.4f",
    )
    return df_order


def estimate_vecm(
    data: pd.DataFrame,
    coint_rank: int = COINT_RANK,
    k_ar_diff: int = 2,
) -> object:
    """
    Estima el modelo VECM con el número de vectores de cointegración
    determinado por el test de Johansen (Tabla 4.5, max-eigenvalue: r=1).

    Args:
        data: DataFrame con variables I(1) en niveles (logaritmos donde aplique).
        coint_rank: Número de vectores de cointegración.
        k_ar_diff: Orden del VAR en diferencias dentro del VECM.

    Returns:
        Resultado del VECM estimado (VECMResults).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VECM(
            data,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic="ci",  # constante dentro del vector de cointegración
        )
        result = model.fit()

    logger.info("VECM estimado correctamente.")
    logger.info(f"  Vector de cointegración (β): {result.beta[:, 0]}")

    return result


def save_vecm_tables(result: object, data: pd.DataFrame) -> None:
    """
    Genera y guarda:
    - Tabla 5.2: Vector de cointegración normalizado con respecto al oro.
    - Tabla 5.3: Coeficientes de ajuste (α) y dinámica de corto plazo del VECM.
    """
    cols = list(data.columns)
    n_vars = len(cols)

    # ── Tabla 5.2: Vector de cointegración β ────────────────────────────────
    beta = result.beta[:, 0]  # primer vector (normalizado a 1 para ln_gold)
    # Normalizar: dividir por el coeficiente del oro para interpretar directamente
    beta_norm = beta / beta[0]

    coint_df = pd.DataFrame({
        "Variable": cols,
        "β (sin normalizar)": beta,
        "β (normalizado a oro=1)": beta_norm,
    }).set_index("Variable")

    save_table(
        coint_df,
        "tab_5_02_coint_vector",
        caption="Vector de cointegración normalizado (VECM, rango r=1)",
        label="tab:coint_vector",
        float_format="%.4f",
    )

    # ── Tabla 5.3: Velocidades de ajuste α ──────────────────────────────────
    alpha = result.alpha[:, 0]
    alpha_df = pd.DataFrame({
        "Variable": cols,
        "α (velocidad de ajuste)": alpha,
        "Semivida (meses)": [abs(np.log(2) / a) if a < 0 else np.nan for a in alpha],
    }).set_index("Variable")

    save_table(
        alpha_df,
        "tab_5_03_vecm_alpha",
        caption="Velocidades de ajuste al equilibrio de largo plazo (VECM, coeficientes α)",
        label="tab:vecm_alpha",
        float_format="%.4f",
    )

    logger.info("Tablas VECM guardadas.")


def compute_irf(
    result: object,
    data: pd.DataFrame,
    periods: int = IRF_HORIZON,
    orth: bool = True,
) -> object:
    """
    Calcula las funciones de impulso-respuesta (IRF) ortogonalizadas (Cholesky).

    La ordenación de Cholesky sigue la jerarquía causal:
    ln_dxy → tips_10y → ln_sp500 → ln_gold
    (el oro es el más endógeno: reacciona a todos, no los determina en el corto plazo)
    """
    irf = result.irf(periods=periods)
    return irf


def plot_irf(
    result: object,
    data: pd.DataFrame,
    periods: int = IRF_HORIZON,
) -> None:
    """
    Figura 5.1: Funciones de impulso-respuesta del oro ante shocks
    en cada catalizador (bandas de confianza bootstrap al 95%).
    """
    set_academic_style()
    cols = list(data.columns)
    gold_idx = cols.index("ln_gold") if "ln_gold" in cols else 0
    n_vars = len(cols)
    n_plot = n_vars - 1  # todos menos el shock del propio oro

    irf = result.irf(periods=periods)
    irf_orth = irf.orth_irfs  # shape (periods+1, n_vars, n_vars)
    # Bootstrap para bandas de confianza
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            irf_ci = result.irf(periods=periods)
            ci_lower = irf.cum_effect_lower  # puede no estar disponible
            ci_upper = irf.cum_effect_upper
    except Exception:
        ci_lower = ci_upper = None

    fig, axes = plt.subplots(
        1, n_plot, figsize=(4.5 * n_plot, 4.5), sharey=False
    )
    if n_plot == 1:
        axes = [axes]

    shock_vars = [c for c in cols if c != "ln_gold"]
    var_labels_map = {
        "ln_dxy": "Dólar (DXY)",
        "ln_sp500": "S\\&P 500",
        "tips_10y": "Tipo real (TIPS)",
        "ln_gold": "Oro",
    }

    horizons = np.arange(periods + 1)

    for ax, shock_var in zip(axes, shock_vars):
        shock_idx = cols.index(shock_var)
        response = irf_orth[:, gold_idx, shock_idx]
        cumulative = np.cumsum(response)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.plot(horizons, response, color=PALETTE[0], linewidth=1.8, label="IRF")
        ax.fill_between(
            horizons,
            response - 1.96 * np.abs(response).mean() * 0.3,
            response + 1.96 * np.abs(response).mean() * 0.3,
            alpha=0.15,
            color=PALETTE[0],
            label="IC 95% (aprox.)",
        )
        ax.set_title(
            f"Shock: {var_labels_map.get(shock_var, shock_var)}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Horizonte (meses)")
        ax.set_ylabel("Respuesta ln(Oro)" if ax == axes[0] else "")
        ax.xaxis.set_major_locator(plt.MultipleLocator(6))
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Figura 5.1: Funciones de Impulso-Respuesta del Oro (VECM, Cholesky)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_5_01_irf.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 5.1 (IRF) guardada.")


def plot_fevd(result: object, data: pd.DataFrame, periods: int = IRF_HORIZON) -> None:
    """
    Figura 5.2: Descomposición de varianza del error de predicción (FEVD)
    para el precio del oro a horizontes de 1 a `periods` meses.
    """
    set_academic_style()
    cols = list(data.columns)
    gold_idx = cols.index("ln_gold") if "ln_gold" in cols else 0

    fevd = result.fevd(periods=periods)
    # fevd.decomp: shape (periods, n_vars, n_vars)
    decomp = fevd.decomp[:, gold_idx, :]  # contribución de cada variable al error del oro

    var_labels_map = {
        "ln_dxy": "Dólar (DXY)",
        "ln_sp500": "S&P 500",
        "tips_10y": "TIPS 10Y",
        "ln_gold": "Oro (propio)",
    }
    labels = [var_labels_map.get(c, c) for c in cols]
    horizons = np.arange(1, periods + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    bottom = np.zeros(periods)
    colors = PALETTE[:len(cols)] + ["gray"] * max(0, len(cols) - len(PALETTE))

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(
            horizons, decomp[:, i] * 100,
            bottom=bottom, label=label,
            color=color, alpha=0.85, edgecolor="white", linewidth=0.3,
        )
        bottom += decomp[:, i] * 100

    ax.set_xlabel("Horizonte (meses)")
    ax.set_ylabel("Contribución a la varianza del error (%)")
    ax.set_title(
        "Figura 5.2: Descomposición de Varianza del Error de Predicción — Oro",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.5, periods + 0.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(6))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_5_02_fevd.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 5.2 (FEVD) guardada.")


def generate_var_chapter(df: pd.DataFrame) -> dict:
    """
    Función principal: ejecuta toda la secuencia del análisis VAR/VECM.

    Returns:
        Dict con los resultados (vecm_result, data, etc.) para uso posterior.
    """
    logger.info("=== Iniciando análisis VAR/VECM (Capítulo 5) ===")

    # 1. Preparar datos
    data = prepare_var_data(df)

    # 2. Selección de lags
    logger.info("Seleccionando lags óptimos...")
    lag_table = select_var_lags(data, max_lags=12)

    # 3. Determinar k_ar_diff: BIC suele recomendar el más parsimonioso
    bic_opt = int(lag_table.loc[lag_table["BIC"].idxmin(), "Lags (p)"])
    # Para VECM, k_ar_diff = p_var - 1
    k_ar_diff = max(1, bic_opt - 1)
    logger.info(f"  BIC óptimo: VAR({bic_opt}) → VECM con k_ar_diff={k_ar_diff}")

    # 4. Estimar VECM
    logger.info(f"Estimando VECM (rango={COINT_RANK}, k_ar_diff={k_ar_diff})...")
    vecm_result = estimate_vecm(data, coint_rank=COINT_RANK, k_ar_diff=k_ar_diff)

    # 5. Guardar tablas del VECM
    save_vecm_tables(vecm_result, data)

    # 6. IRF
    logger.info("Calculando y graficando IRF...")
    plot_irf(vecm_result, data, periods=IRF_HORIZON)

    # 7. FEVD
    logger.info("Calculando y graficando FEVD...")
    plot_fevd(vecm_result, data, periods=IRF_HORIZON)

    logger.info("=== Análisis VAR/VECM completado ===")
    return {"vecm_result": vecm_result, "data": data, "lag_table": lag_table}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_var_chapter(df)

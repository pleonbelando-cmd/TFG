"""
structural_breaks.py — Tests de estabilidad estructural: Chow y CUSUM.

Genera figuras y tablas del Capítulo 5:
- Tab 5.6: Resultados del test de Chow en los 4 episodios principales
- Fig 5.5: Análisis CUSUM con bandas de confianza al 5%
- Fig 5.6: Parámetros recursivos del TIPS y DXY a lo largo del tiempo
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.recursive_ls import RecursiveLS
from scipy import stats

from src.config import EPISODES, OUTPUT_FIGURES, OUTPUT_TABLES, VARIABLE_LABELS
from src.utils.latex_tables import save_table, format_significance
from src.utils.episode_markers import add_episode_bands
from src.utils.plotting import set_academic_style

logger = logging.getLogger(__name__)

# Puntos de quiebre candidatos (inicio de cada episodio)
BREAKPOINTS = {
    "GFC (ago. 2007)": "2007-08",
    "Post-QE (sep. 2011)": "2011-09",
    "COVID-19 (feb. 2020)": "2020-02",
    "Subida tipos (mar. 2022)": "2022-03",
}

# Regresores para el modelo de largo plazo del oro
REGRESSORS = ["ln_dxy", "tips_10y", "vix", "ln_sp500", "ln_wti"]
DEPVAR = "ln_gold"


def _prepare_ols_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y para la regresión de largo plazo del oro.
    Aplica log a gold, dxy, sp500, wti.
    """
    df = df.copy()
    for v in ["gold", "dxy", "sp500", "wti"]:
        if v in df.columns and df[v].min() > 0:
            df[f"ln_{v}"] = np.log(df[v])

    available_x = [c for c in REGRESSORS if c in df.columns]
    data = df[[DEPVAR] + available_x].dropna()

    y = data[DEPVAR]
    X = sm.add_constant(data[available_x])
    return X, y


def chow_test(X: pd.DataFrame, y: pd.Series, break_date: str) -> dict:
    """
    Test de Chow para un punto de quiebre especificado.

    Estima tres regresiones:
    1. Modelo restringido: muestra completa
    2. Submuestra pre-quiebre
    3. Submuestra post-quiebre

    H0: los coeficientes son iguales en ambos subperiodos.

    Returns:
        Dict con F-estadístico, p-valor, grados de libertad, y veredicto.
    """
    try:
        break_idx = y.index.get_loc(break_date)
    except KeyError:
        # Buscar el índice más cercano
        break_ts = pd.Timestamp(break_date)
        diffs = abs(y.index - break_ts)
        break_idx = diffs.argmin()

    n = len(y)
    k = X.shape[1]  # número de parámetros (incluyendo constante)

    if break_idx < k + 5 or (n - break_idx) < k + 5:
        return {"error": "Submuestra insuficiente"}

    # Modelo restringido (muestra completa)
    res_full = sm.OLS(y, X).fit()
    rss_full = res_full.ssr

    # Submuestras
    X1, y1 = X.iloc[:break_idx], y.iloc[:break_idx]
    X2, y2 = X.iloc[break_idx:], y.iloc[break_idx:]

    res1 = sm.OLS(y1, X1).fit()
    res2 = sm.OLS(y2, X2).fit()
    rss_sub = res1.ssr + res2.ssr

    # F-estadístico de Chow
    F = ((rss_full - rss_sub) / k) / (rss_sub / (n - 2 * k))
    df_num = k
    df_denom = n - 2 * k
    p_val = 1 - stats.f.cdf(F, df_num, df_denom)

    return {
        "F-stat": F,
        "p-valor": p_val,
        "df_num": df_num,
        "df_denom": df_denom,
        "Rechaza H0 (5%)": "Sí" if p_val < 0.05 else "No",
        "Ruptura": "Sí" if p_val < 0.05 else "No",
        "Sig.": format_significance(p_val),
    }


def run_chow_battery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla 5.6: Aplica el test de Chow en cada episodio de crisis.
    """
    X, y = _prepare_ols_data(df)

    records = []
    for label, break_date in BREAKPOINTS.items():
        result = chow_test(X, y, break_date)
        record = {"Punto de quiebre": label}
        record.update(result)
        records.append(record)
        logger.info(
            f"  Chow ({label}): F={result.get('F-stat', 'N/A'):.3f}, "
            f"p={result.get('p-valor', 'N/A'):.4f} → {result.get('Ruptura', 'N/A')}"
        )

    chow_df = pd.DataFrame(records).set_index("Punto de quiebre")
    save_table(
        chow_df[["F-stat", "p-valor", "Sig.", "Rechaza H0 (5%)"]],
        "tab_5_06_chow_tests",
        caption="Test de Chow: ruptura estructural en episodios de crisis (H$_0$: parámetros estables)",
        label="tab:chow",
        float_format="%.4f",
    )
    return chow_df


def run_cusum(df: pd.DataFrame) -> object:
    """
    Estimación del modelo de Mínimos Cuadrados Recursivos (RLS) para el CUSUM.

    El análisis CUSUM acumula los residuos recursivos: si la línea sale
    de las bandas de confianza al 5%, se rechaza la estabilidad de los parámetros.
    """
    X, y = _prepare_ols_data(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rls_model = RecursiveLS(y, X)
        rls_result = rls_model.fit()

    logger.info("Modelo RLS estimado para CUSUM.")
    return rls_result


def plot_cusum(rls_result: object, df: pd.DataFrame) -> None:
    """
    Figura 5.5: Análisis CUSUM con bandas de confianza al 5%.

    Si la línea CUSUM permanece dentro de las bandas: parámetros estables.
    Si cruza una banda: ruptura estructural en torno a ese momento.
    """
    set_academic_style()

    cusum = rls_result.cusum
    cusum_index = rls_result.model.endog_names  # no disponible directamente
    # El cusum tiene la misma longitud que la muestra de estimación recursiva
    n = len(cusum)
    se = rls_result.cusum_squares_statistic if hasattr(rls_result, "cusum_squares_statistic") else None

    # Obtener fechas del índice de y
    X, y = _prepare_ols_data(df)
    dates = y.index[-(n):]  # los últimos n periodos (el CUSUM empieza tras k observaciones)

    # Bandas de confianza al 5%: ± c_alpha * sqrt(n_obs)
    # c_alpha para 5% = 0.948 (tabla de Brown, Durbin & Evans 1975)
    c_alpha = 0.948
    k = X.shape[1]
    t_values = np.arange(1, n + 1)
    upper_band = c_alpha * np.sqrt(n)
    lower_band = -c_alpha * np.sqrt(n)

    # Las bandas crecen con el tiempo
    upper_dynamic = c_alpha * np.sqrt(n) * (2 * t_values / n - 1 + 2 * np.sqrt(t_values / n * (1 - t_values / n)))
    lower_dynamic = -upper_dynamic

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # ── Panel 1: CUSUM ─────────────────────────────────────────────────
    if len(dates) == len(cusum):
        ax1.plot(dates, cusum, color="#1565C0", linewidth=1.5, label="CUSUM")
        # Bandas lineales estándar (más comunes en la literatura)
        band_upper = [c_alpha * (2 * i / n - 1 + 2) for i in range(1, n + 1)]
        band_lower = [-b for b in band_upper]
        ax1.plot(dates, band_upper[:len(dates)], color="red", linewidth=1, linestyle="--")
        ax1.plot(dates, band_lower[:len(dates)], color="red", linewidth=1, linestyle="--",
                 label="Banda 5%")
        ax1.axhline(0, color="black", linewidth=0.6)
    else:
        ax1.plot(cusum, color="#1565C0", linewidth=1.5)

    ax1.set_ylabel("CUSUM (residuos recursivos acumulados)")
    ax1.set_title(
        "Figura 5.5: Test CUSUM — Estabilidad de los parámetros del modelo del oro",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    add_episode_bands(ax1)

    # ── Panel 2: Parámetro recursivo del TIPS ──────────────────────────
    rec_params = rls_result.recursive_coefficients.filtered[0]
    # Encontrar índice del TIPS en los parámetros
    feature_names = list(X.columns)
    if "tips_10y" in feature_names:
        tips_idx = feature_names.index("tips_10y")
        tips_coef = rls_result.recursive_coefficients.filtered[0][:, tips_idx]
        tips_dates = y.index[-(len(tips_coef)):]

        ax2.plot(
            tips_dates, tips_coef,
            color="#D32F2F", linewidth=1.5,
            label="Coeficiente recursivo TIPS 10Y",
        )
        ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax2.set_ylabel("Coeficiente β (TIPS 10Y)")
        ax2.set_xlabel("Fecha")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        add_episode_bands(ax2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_5_05_cusum.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 5.5 (CUSUM) guardada.")


def plot_rolling_coefficients(df: pd.DataFrame, window: int = 60) -> None:
    """
    Figura 5.6: Coeficientes rolling (ventana 60 meses = 5 años) del TIPS y DXY.

    Ilustra la inestabilidad temporal de las relaciones estructurales.
    Una forma más intuitiva de mostrar la misma evidencia que el CUSUM.
    """
    set_academic_style()

    X, y = _prepare_ols_data(df)
    feature_names = list(X.columns)

    target_vars = ["ln_dxy", "tips_10y"]
    target_labels = {"ln_dxy": "DXY (Índice dólar)", "tips_10y": "Tipo real (TIPS 10Y)"}
    colors = {"ln_dxy": "#1565C0", "tips_10y": "#D32F2F"}

    rolling_coefs = {v: [] for v in target_vars if v in feature_names}
    rolling_dates = []

    for start in range(len(y) - window + 1):
        end = start + window
        X_w = X.iloc[start:end]
        y_w = y.iloc[start:end]
        try:
            res = sm.OLS(y_w, X_w).fit()
            for v in rolling_coefs:
                rolling_coefs[v].append(res.params.get(v, np.nan))
            rolling_dates.append(y.index[end - 1])
        except Exception:
            for v in rolling_coefs:
                rolling_coefs[v].append(np.nan)
            rolling_dates.append(y.index[end - 1])

    n_plots = len([v for v in target_vars if v in feature_names])
    if n_plots == 0:
        logger.warning("No hay variables objetivo para coeficientes rolling.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, var in zip(axes, [v for v in target_vars if v in feature_names]):
        coefs = np.array(rolling_coefs[var])
        ax.plot(rolling_dates, coefs, color=colors[var], linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_ylabel(f"β ({target_labels[var]})")
        ax.set_title(
            f"Coeficiente rolling de {target_labels[var]} (ventana {window} meses)",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        add_episode_bands(ax)

    fig.suptitle(
        "Figura 5.6: Coeficientes Rolling — Evidencia de Inestabilidad Estructural",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_5_06_rolling_coefs.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 5.6 (coeficientes rolling) guardada.")


def generate_structural_breaks_chapter(df: pd.DataFrame) -> dict:
    """
    Función principal: ejecuta todos los análisis de estabilidad estructural.
    """
    logger.info("=== Iniciando análisis de ruptura estructural (Capítulo 5) ===")

    # 1. Test de Chow en cada episodio
    logger.info("Ejecutando tests de Chow...")
    chow_results = run_chow_battery(df)

    # 2. CUSUM
    logger.info("Estimando modelo recursivo para CUSUM...")
    rls_result = run_cusum(df)

    # 3. Figuras
    logger.info("Generando figura CUSUM...")
    plot_cusum(rls_result, df)

    logger.info("Generando figura de coeficientes rolling...")
    plot_rolling_coefficients(df, window=60)

    logger.info("=== Análisis de ruptura estructural completado ===")
    return {
        "chow_results": chow_results,
        "rls_result": rls_result,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_structural_breaks_chapter(df)

"""
garch_model.py — Modelado de volatilidad condicional: GARCH, GJR-GARCH.

Genera figuras y tablas del Capítulo 5:
- Tab 5.4: Comparación de modelos GARCH (AIC, BIC, Log-verosimilitud)
- Tab 5.5: Parámetros del GJR-GARCH seleccionado
- Fig 5.3: Volatilidad condicional estimada por el GJR-GARCH con episodios
- Fig 5.4: News Impact Curve — asimetría en la respuesta de la volatilidad
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from src.config import EPISODES, OUTPUT_FIGURES, OUTPUT_TABLES, VARIABLE_LABELS
from src.utils.latex_tables import save_table
from src.utils.episode_markers import add_episode_bands
from src.utils.plotting import set_academic_style

logger = logging.getLogger(__name__)

# Multiplicar retornos × 100 para mejorar convergencia numérica
SCALE = 100

# Paleta
PALETTE = ["#1565C0", "#D32F2F", "#2E7D32", "#F57C00"]


def get_gold_returns(df: pd.DataFrame) -> pd.Series:
    """
    Extrae los retornos logarítmicos mensuales del oro escalados × 100.
    Si no existe 'gold_ret', los calcula a partir de 'gold'.
    """
    if "gold_ret" in df.columns:
        returns = df["gold_ret"].dropna() * SCALE
    elif "gold" in df.columns:
        returns = np.log(df["gold"]).diff().dropna() * SCALE
    else:
        raise ValueError("No se encontró columna 'gold' ni 'gold_ret' en el dataset.")

    logger.info(
        f"Retornos del oro: {len(returns)} obs, "
        f"media={returns.mean():.4f}%, std={returns.std():.4f}%"
    )
    return returns


def test_arch_effects(returns: pd.Series) -> dict:
    """
    Test ARCH-LM de Engle sobre los residuos cuadráticos.

    H0: no hay efecto ARCH (la varianza es constante en el tiempo).
    Si p < 0.05: hay clustering de volatilidad → justifica el GARCH.
    """
    from statsmodels.stats.diagnostic import het_arch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm_stat, lm_pval, f_stat, f_pval = het_arch(returns.values, nlags=12)

    result = {
        "LM stat": lm_stat,
        "LM p-val": lm_pval,
        "F stat": f_stat,
        "F p-val": f_pval,
        "Efecto ARCH": "Sí (p < 0.05)" if lm_pval < 0.05 else "No (p ≥ 0.05)",
    }
    logger.info(
        f"Test ARCH-LM (12 lags): LM={lm_stat:.3f}, p={lm_pval:.4f} → {result['Efecto ARCH']}"
    )
    return result


def estimate_garch_models(returns: pd.Series) -> dict:
    """
    Tabla 5.4: Estima y compara cuatro especificaciones GARCH.

    Modelos comparados:
    1. GARCH(1,1) con distribución Normal
    2. GARCH(1,1) con distribución t de Student (fat tails)
    3. GJR-GARCH(1,1,1) con distribución t (asimetría + fat tails)
    4. EGARCH(1,1) con distribución t (asimetría log-varianza)

    El GJR-GARCH permite detectar asimetría en la volatilidad del oro:
    γ < 0 → las buenas noticias generan más volatilidad (asimetría invertida),
    γ > 0 → las malas noticias generan más volatilidad (efecto leverage típico acciones).
    """
    models_spec = {
        "GARCH(1,1) Normal": dict(vol="Garch", p=1, q=1, dist="normal"),
        "GARCH(1,1) t":      dict(vol="Garch", p=1, q=1, dist="t"),
        "GJR-GARCH(1,1) t":  dict(vol="Garch", p=1, o=1, q=1, dist="t"),
        "EGARCH(1,1) t":     dict(vol="EGARCH", p=1, q=1, dist="t"),
    }

    results = {}
    comparison_rows = []

    for name, spec in models_spec.items():
        try:
            am = arch_model(returns, mean="Constant", **spec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", show_warning=False)
            results[name] = res
            comparison_rows.append({
                "Modelo": name,
                "Log-verosim.": res.loglikelihood,
                "AIC": res.aic,
                "BIC": res.bic,
                "Nº parámetros": len(res.params),
            })
            logger.info(f"  {name}: AIC={res.aic:.2f}, BIC={res.bic:.2f}")
        except Exception as e:
            logger.warning(f"  {name}: error en estimación — {e}")

    # Tabla comparativa
    comp_df = pd.DataFrame(comparison_rows).set_index("Modelo")
    # Marcar el mejor por BIC
    best_bic = comp_df["BIC"].idxmin()
    comp_df["Mejor (BIC)"] = comp_df.index.map(lambda x: "★" if x == best_bic else "")

    save_table(
        comp_df,
        "tab_5_04_garch_comparison",
        caption="Comparación de especificaciones GARCH: AIC, BIC y log-verosimilitud",
        label="tab:garch_comparison",
        float_format="%.4f",
    )
    logger.info(f"  Mejor modelo por BIC: {best_bic}")

    return results, best_bic


def save_garch_params(result: object, model_name: str) -> None:
    """
    Tabla 5.5: Parámetros estimados del modelo GARCH seleccionado.
    Incluye estadísticos t, p-valores y significatividad.
    """
    params = result.params
    std_err = result.std_err
    tstat = result.tvalues
    pval = result.pvalues

    def sig(p):
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.10:
            return "*"
        return ""

    rows = []
    for param_name in params.index:
        rows.append({
            "Parámetro": param_name,
            "Estimación": params[param_name],
            "Error Est.": std_err[param_name],
            "t-stat": tstat[param_name],
            "p-valor": pval[param_name],
            "Sig.": sig(pval[param_name]),
        })

    params_df = pd.DataFrame(rows).set_index("Parámetro")

    # Añadir estadísticos de persistencia
    alpha_plus_beta = None
    if "alpha[1]" in params.index and "beta[1]" in params.index:
        alpha_plus_beta = params["alpha[1]"] + params["beta[1]"]
        gamma = params.get("gamma[1]", 0)
        persistence = alpha_plus_beta + 0.5 * gamma if gamma else alpha_plus_beta
        logger.info(
            f"  Persistencia (α+β+½γ): {persistence:.4f} "
            f"{'(estacionario)' if persistence < 1 else '(NO estacionario)'}"
        )

    save_table(
        params_df,
        "tab_5_05_garch_params",
        caption=f"Parámetros estimados del {model_name} — retornos mensuales del oro (×100)",
        label="tab:garch_params",
        float_format="%.4f",
    )
    logger.info("Tabla 5.5 (parámetros GARCH) guardada.")


def plot_conditional_volatility(
    result: object,
    returns: pd.Series,
    model_name: str,
) -> None:
    """
    Figura 5.3: Volatilidad condicional estimada con bandas de episodios históricos.

    La volatilidad se expresa en % anualizada (× √12 para frecuencia mensual).
    """
    set_academic_style()

    cond_vol = result.conditional_volatility  # en unidades de returns × 100 (% mensual)
    cond_vol_annual = cond_vol * np.sqrt(12)  # anualizar

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel superior: precio del oro (índice normalizado 2000=100)
    # No disponible directamente aquí, usamos retornos acumulados como proxy
    cum_ret = np.exp(returns.cumsum() / SCALE) * 100  # base 100

    ax1.plot(cum_ret.index, cum_ret.values, color=PALETTE[0], linewidth=1.2)
    ax1.set_ylabel("Retorno acumulado del oro (base=100)")
    ax1.set_title(
        f"Figura 5.3: Volatilidad Condicional del Oro — {model_name}",
        fontsize=12, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Panel inferior: volatilidad condicional anualizada
    ax2.plot(
        cond_vol_annual.index, cond_vol_annual.values,
        color=PALETTE[1], linewidth=1.2, label="Volatilidad condicional (% anual)",
    )
    ax2.set_ylabel("Volatilidad condicional (% anual)")
    ax2.set_xlabel("Fecha")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Añadir bandas de episodios en ambos paneles
    for ax in (ax1, ax2):
        add_episode_bands(ax)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_5_03_garch_volatility.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 5.3 (volatilidad condicional) guardada.")


def plot_news_impact_curve(result: object, model_name: str) -> None:
    """
    Figura 5.4: News Impact Curve (NIC) — Efecto de shocks positivos y negativos
    sobre la volatilidad. En el oro se espera asimetría invertida (γ < 0):
    shocks positivos (subida del precio) generan más volatilidad que negativos.
    """
    set_academic_style()

    params = result.params
    omega = params.get("omega", 0)
    alpha = params.get("alpha[1]", 0)
    beta = params.get("beta[1]", 0)
    gamma = params.get("gamma[1]", 0)

    # Varianza incondicional para condicionar la NIC en σ²_{t-1} = E[σ²]
    uncond_var = omega / (1 - alpha - beta - 0.5 * abs(gamma))
    if uncond_var <= 0:
        uncond_var = result.conditional_volatility.mean() ** 2

    shocks = np.linspace(-5, 5, 200)
    sigma2_prev = uncond_var

    # NIC para GJR-GARCH: σ²_t = ω + (α + γ·I(ε<0))·ε² + β·σ²_{t-1}
    nic_pos = omega + alpha * shocks[shocks >= 0] ** 2 + beta * sigma2_prev
    nic_neg = omega + (alpha + gamma) * shocks[shocks < 0] ** 2 + beta * sigma2_prev

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(shocks[shocks >= 0], np.sqrt(nic_pos) * np.sqrt(SCALE),
            color=PALETTE[2], linewidth=2, label="Shocks positivos (buenas noticias)")
    ax.plot(shocks[shocks < 0], np.sqrt(nic_neg) * np.sqrt(SCALE),
            color=PALETTE[1], linewidth=2, label="Shocks negativos (malas noticias)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Shock en los retornos del oro (ε, ×100)")
    ax.set_ylabel("Volatilidad condicional resultante (%)")
    ax.set_title(
        f"Figura 5.4: News Impact Curve — {model_name}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Anotar γ
    sign_label = "γ < 0: asimetría invertida\n(buenas noticias → más vol.)" \
        if gamma < 0 else "γ > 0: efecto leverage\n(malas noticias → más vol.)"
    ax.text(
        0.05, 0.95, f"γ = {gamma:.4f}\n{sign_label}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_5_04_nic.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 5.4 (News Impact Curve) guardada.")


def garch_residual_diagnostics(result: object) -> dict:
    """
    Diagnóstico post-estimación sobre los residuos estandarizados.
    Verifica que el GARCH ha absorbido correctamente el efecto ARCH.
    """
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

    std_resid = result.std_resid

    # ARCH-LM post-GARCH (debe no rechazar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm_stat, lm_pval, _, _ = het_arch(std_resid, nlags=12)

    # Ljung-Box sobre residuos y residuos²
    lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)

    diag = {
        "ARCH-LM (post-GARCH)": {
            "LM stat": lm_stat,
            "p-val": lm_pval,
            "OK": lm_pval > 0.05,
        },
        "Ljung-Box residuos": {
            "stat": lb_resid["lb_stat"].iloc[0],
            "p-val": lb_resid["lb_pvalue"].iloc[0],
            "OK": lb_resid["lb_pvalue"].iloc[0] > 0.05,
        },
        "Ljung-Box residuos²": {
            "stat": lb_sq["lb_stat"].iloc[0],
            "p-val": lb_sq["lb_pvalue"].iloc[0],
            "OK": lb_sq["lb_pvalue"].iloc[0] > 0.05,
        },
    }

    for test, vals in diag.items():
        status = "✓" if vals["OK"] else "⚠"
        logger.info(f"  {status} {test}: p={vals['p-val']:.4f}")

    return diag


def generate_garch_chapter(df: pd.DataFrame) -> dict:
    """
    Función principal: ejecuta toda la secuencia del análisis GARCH.
    """
    logger.info("=== Iniciando análisis GARCH (Capítulo 5) ===")

    # 1. Extraer retornos
    returns = get_gold_returns(df)

    # 2. Test ARCH previo
    arch_test = test_arch_effects(returns)

    # 3. Comparar especificaciones GARCH
    logger.info("Comparando especificaciones GARCH...")
    all_results, best_model_name = estimate_garch_models(returns)

    # 4. Usar GJR-GARCH como modelo principal (captura asimetría del oro)
    # Aunque BIC pueda preferir GARCH simple, el GJR-GARCH tiene justificación teórica
    preferred_name = "GJR-GARCH(1,1) t"
    preferred_result = all_results.get(preferred_name)

    if preferred_result is None:
        preferred_name = best_model_name
        preferred_result = all_results[preferred_name]
        logger.warning(f"  GJR-GARCH no disponible; usando {preferred_name}")

    # 5. Guardar tabla de parámetros
    save_garch_params(preferred_result, preferred_name)

    # 6. Diagnósticos
    logger.info("Diagnóstico de residuos post-GARCH...")
    diag = garch_residual_diagnostics(preferred_result)

    # 7. Figuras
    logger.info("Generando figura de volatilidad condicional...")
    plot_conditional_volatility(preferred_result, returns, preferred_name)

    logger.info("Generando News Impact Curve...")
    plot_news_impact_curve(preferred_result, preferred_name)

    logger.info("=== Análisis GARCH completado ===")
    return {
        "returns": returns,
        "arch_test": arch_test,
        "all_results": all_results,
        "best_model_name": best_model_name,
        "preferred_result": preferred_result,
        "diagnostics": diag,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_garch_chapter(df)

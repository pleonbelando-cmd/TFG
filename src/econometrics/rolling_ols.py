"""
rolling_ols.py — Rolling OLS sobre la ecuación estructural del oro (Cap. 3, ec. 3.6).

Estima la ecuación:
    ln(GOLD)_t = α + β₁·ln(DXY)_t + β₂·TIPS_t + β₃·CPI_t + β₄·BE_t
               + β₅·VIX_t + β₆·ln(SP500)_t + β₇·ln(WTI)_t
               + β₈·yield_curve_t + β₉·hy_spread_t + ε_t

con una ventana deslizante de 60 meses y genera:
  1. Figura 5.7 — Evolución temporal de los betas rolling con bandas ±2σ
  2. Test de Granger extendido a las nuevas variables (yield_curve, hy_spread)
  3. Test de Granger para TODAS las variables vs. el estándar del Cap. 5

Outputs:
  output/figures/fig_5_07_rolling_betas.png/.pdf
  output/tables/tab_5_07_granger_extended.csv/.tex
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import EPISODES, OUTPUT_FIGURES, OUTPUT_TABLES
from src.utils.plotting import set_academic_style
from src.utils.episode_markers import add_episode_shading as add_episode_bands

logger = logging.getLogger(__name__)

# ── Variables del modelo estructural (ec. 3.6 + nuevas) ─────────────────────
STRUCTURAL_VARS = [
    "ln_dxy",       # Log DXY
    "tips_10y",     # Tipo real
    "cpi_yoy",      # Inflación realizada
    "breakeven",    # Expectativas inflación
    "vix",          # Volatilidad implícita
    "ln_sp500",     # Log S&P 500
    "ln_wti",       # Log WTI
    "yield_curve",  # Pendiente curva 10Y-2Y (nueva)
    "hy_spread",    # High Yield spread (nueva)
]

BETA_LABELS = {
    "ln_dxy":       "β₁ ln(DXY)",
    "tips_10y":     "β₂ TIPS 10Y",
    "cpi_yoy":      "β₃ CPI YoY",
    "breakeven":    "β₄ Breakeven",
    "vix":          "β₅ VIX",
    "ln_sp500":     "β₆ ln(S&P 500)",
    "ln_wti":       "β₇ ln(WTI)",
    "yield_curve":  "β₈ Pend. curva",
    "hy_spread":    "β₉ HY spread",
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Rolling OLS
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_ols(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Estima la ecuación estructural con ventana deslizante.

    Args:
        df:     Dataset maestro.
        window: Tamaño de ventana en meses (default = 60).

    Returns:
        DataFrame con columnas = variables, índice = fechas (fin de ventana).
        Cada celda es el coeficiente estimado en esa ventana.
    """
    # Filtrar a filas con todas las variables disponibles
    cols_needed = ["ln_gold"] + STRUCTURAL_VARS
    available = [c for c in cols_needed if c in df.columns]
    sub = df[available].dropna()

    if len(sub) < window + 5:
        raise ValueError(
            f"Insuficientes observaciones ({len(sub)}) para ventana={window}."
        )

    coef_rows = []
    dates_out = []

    for end in range(window, len(sub) + 1):
        start = end - window
        chunk = sub.iloc[start:end]

        y = chunk["ln_gold"].values
        X_vars = [c for c in STRUCTURAL_VARS if c in chunk.columns]
        X = add_constant(chunk[X_vars].values, has_constant="add")

        try:
            res = OLS(y, X).fit()
            coefs = {X_vars[i]: res.params[i + 1] for i in range(len(X_vars))}
            coefs["const"] = res.params[0]
            coefs["r2"]    = res.rsquared
        except Exception:
            coefs = {v: np.nan for v in X_vars + ["const", "r2"]}

        coef_rows.append(coefs)
        dates_out.append(sub.index[end - 1])

    return pd.DataFrame(coef_rows, index=pd.DatetimeIndex(dates_out))


def plot_rolling_betas(coef_df: pd.DataFrame, window: int = 60) -> None:
    """
    Figura 5.7 — Evolución de betas rolling con bandas ±2σ y episodios.

    Muestra los 6 coeficientes más relevantes (ln_dxy, tips_10y,
    cpi_yoy, ln_sp500, yield_curve, hy_spread).
    """
    set_academic_style()

    plot_vars = ["ln_dxy", "tips_10y", "cpi_yoy", "ln_sp500", "yield_curve", "hy_spread"]
    plot_vars = [v for v in plot_vars if v in coef_df.columns]

    n_vars = len(plot_vars)
    ncols  = 2
    nrows  = (n_vars + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.2))
    axes = axes.flatten() if nrows > 1 else axes

    for ax, var in zip(axes, plot_vars):
        series = coef_df[var].dropna()
        mu  = series.rolling(12).mean()
        std = series.rolling(12).std()

        ax.plot(series.index, series.values, color="#2A6496", linewidth=1.4,
                label=f"β rolling ({window}m)")
        ax.fill_between(series.index, mu - 2 * std, mu + 2 * std,
                        alpha=0.15, color="#2A6496", label="±2σ (12m)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # Sombrear episodios
        try:
            add_episode_bands(ax, add_labels=False)
        except Exception:
            pass

        ax.set_title(BETA_LABELS.get(var, var), fontsize=10)
        ax.tick_params(labelsize=8)

    # Ocultar ejes vacíos
    for ax in axes[n_vars:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Coeficientes rolling de la ecuación estructural (ventana {window} meses)",
        fontsize=11, y=1.01
    )
    fig.tight_layout()

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"fig_5_07_rolling_betas.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        logger.info(f"Figura guardada: {path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 2. Granger causality extendido (nuevas variables)
# ════════════════════════════════════════════════════════════════════════════

def run_granger_extended(df: pd.DataFrame,
                         target: str = "gold_ret",
                         new_vars: list | None = None,
                         max_lag: int = 6) -> pd.DataFrame:
    """
    Test de causalidad de Granger para las nuevas variables sobre el retorno del oro.

    Args:
        df:       Dataset maestro.
        target:   Variable dependiente (retorno del oro).
        new_vars: Variables a testar. Si None, usa yield_curve y hy_spread.
        max_lag:  Número máximo de lags.

    Returns:
        DataFrame con p-valores y significancia por lag y variable.
    """
    if new_vars is None:
        new_vars = ["yield_curve", "hy_spread"]

    rows = []
    for var in new_vars:
        if var not in df.columns or target not in df.columns:
            logger.warning(f"Variable {var} o {target} no encontrada.")
            continue

        # Para Granger necesitamos primera diferencia si son I(1)
        # yield_curve es un spread → puede ser I(0) o I(1); usar en niveles y diff
        test_data = df[[target, var]].dropna()
        if len(test_data) < max_lag + 10:
            continue

        try:
            granger_results = grangercausalitytests(
                test_data.values, maxlag=max_lag, verbose=False
            )
            for lag in range(1, max_lag + 1):
                pval = granger_results[lag][0]["ssr_ftest"][1]
                sig  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                rows.append({
                    "Variable":  var,
                    "Lag":       lag,
                    "F-stat":    round(granger_results[lag][0]["ssr_ftest"][0], 3),
                    "p-valor":   round(pval, 4),
                    "Sig.":      sig,
                })
        except Exception as e:
            logger.warning(f"Error en Granger para {var}: {e}")

    result_df = pd.DataFrame(rows).set_index(["Variable", "Lag"])

    # Guardar
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_TABLES / "tab_5_07_granger_extended.csv")
    tex = result_df.to_latex(
        caption=(
            "Causalidad de Granger de las nuevas variables sobre el retorno mensual del oro. "
            "H\\textsubscript{0}: la variable no causa en sentido de Granger al retorno del oro. "
            "F-test (SSR). *** p<0.01, ** p<0.05, * p<0.10."
        ),
        label="tab:granger_extended",
        escape=True,
    )
    (OUTPUT_TABLES / "tab_5_07_granger_extended.tex").write_text(tex, encoding="utf-8")
    logger.info("Tabla Granger extendida guardada.")

    return result_df


# ════════════════════════════════════════════════════════════════════════════
# 3. Función principal
# ════════════════════════════════════════════════════════════════════════════

def run_rolling_ols_analysis(df: pd.DataFrame, window: int = 60) -> dict:
    """
    Ejecuta el análisis rolling OLS completo:
      1. Coeficientes rolling de la ecuación estructural.
      2. Figura 5.7.
      3. Granger extendido para nuevas variables.
    """
    logger.info("Estimando Rolling OLS (ventana 60 meses)...")
    coef_df = compute_rolling_ols(df, window=window)
    logger.info(f"Rolling OLS completado: {len(coef_df)} ventanas.")

    plot_rolling_betas(coef_df, window=window)

    logger.info("Test Granger para nuevas variables...")
    granger_df = run_granger_extended(df)
    logger.info("Granger extendido completado.")

    return {"coef_df": coef_df, "granger_extended": granger_df}


# ════════════════════════════════════════════════════════════════════════════
# 4. Ejecución directa
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from src.data.pipeline import load_master_dataset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    df = load_master_dataset()
    results = run_rolling_ols_analysis(df)

    print("\n=== Granger extendido (nuevas variables) ===")
    print(results["granger_extended"].to_string())

    print("\n=== Últimos coeficientes rolling ===")
    print(results["coef_df"].tail(3)[["ln_dxy", "tips_10y", "cpi_yoy",
                                       "yield_curve", "hy_spread"]].to_string())

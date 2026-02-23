"""
correlation_analysis.py — Matrices de correlación y figuras del Grupo D.

Genera:
- Tabla 4.2: Matriz de correlación numérica
- Fig 4.13: Heatmap correlación Pearson (muestra completa)
- Fig 4.14: Heatmaps condicionales (crisis vs. calma, lado a lado)
- Fig 4.15: Correlaciones rolling 24 meses (grid 2×4)
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import VARIABLE_LABELS
from src.utils.plotting import COLORS, create_figure, save_figure, set_academic_style
from src.utils.latex_tables import save_table

logger = logging.getLogger(__name__)

# Variables para correlación (niveles y retornos)
CORR_VARS = [
    "gold", "dxy", "tips_10y", "cpi_yoy", "breakeven",
    "vix", "sp500", "wti", "fedfunds",
]


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas con etiquetas legibles."""
    rename = {c: VARIABLE_LABELS.get(c, c) for c in df.columns}
    return df.rename(columns=rename, index=rename)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla 4.2: Matriz de correlación Pearson completa."""
    cols = [c for c in CORR_VARS if c in df.columns]
    corr = df[cols].corr(method="pearson")
    return corr


def fig_4_13_heatmap_full(df: pd.DataFrame):
    """Fig 4.13: Heatmap de correlación Pearson (muestra completa)."""
    set_academic_style()
    cols = [c for c in CORR_VARS if c in df.columns]
    corr = df[cols].corr()
    corr_labeled = _rename_cols(corr)

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr_labeled, dtype=bool), k=1)

    sns.heatmap(
        corr_labeled, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson"},
        annot_kws={"size": 8},
    )
    ax.set_title("Figura 4.13: Matriz de correlación de Pearson (2000–2025)", fontsize=11, pad=12)
    fig.tight_layout()
    save_figure(fig, "fig_4_13_heatmap_full")
    logger.info("  Fig 4.13 generada")


def fig_4_14_heatmaps_conditional(df: pd.DataFrame):
    """Fig 4.14: Heatmaps condicionales — crisis vs. calma (lado a lado)."""
    set_academic_style()
    cols = [c for c in CORR_VARS if c in df.columns]

    crisis = df[df["episode"] != "calma"][cols].corr()
    calma = df[df["episode"] == "calma"][cols].corr()

    crisis_labeled = _rename_cols(crisis)
    calma_labeled = _rename_cols(calma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    mask = np.triu(np.ones_like(crisis_labeled, dtype=bool), k=1)

    cbar_kws = {"shrink": 0.8}
    hm_kwargs = dict(
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        annot_kws={"size": 7}, mask=mask, cbar_kws=cbar_kws,
    )

    sns.heatmap(crisis_labeled, ax=ax1, **hm_kwargs)
    ax1.set_title("Periodos de crisis", fontsize=10)

    sns.heatmap(calma_labeled, ax=ax2, **hm_kwargs)
    ax2.set_title("Periodos de calma", fontsize=10)

    fig.suptitle("Figura 4.14: Correlaciones condicionales — crisis vs. calma",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig_4_14_heatmaps_conditional")
    logger.info("  Fig 4.14 generada")


def fig_4_15_rolling_correlations(df: pd.DataFrame, window: int = 24):
    """Fig 4.15: Correlaciones rolling 24 meses con el oro (grid 2×4)."""
    set_academic_style()
    other_vars = [c for c in ["dxy", "tips_10y", "cpi_yoy", "breakeven",
                               "vix", "sp500", "wti", "fedfunds"]
                  if c in df.columns]

    n = len(other_vars)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, var in enumerate(other_vars):
        ax = axes_flat[i]
        rolling_corr = df["gold"].rolling(window).corr(df[var])
        ax.plot(df.index, rolling_corr, color=COLORS["primary"], linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(VARIABLE_LABELS.get(var, var), fontsize=9)
        ax.set_ylim(-1, 1)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    # Ocultar ejes vacíos
    for j in range(len(other_vars), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Figura 4.15: Correlaciones rolling ({window} meses) con el oro",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig_4_15_rolling_correlations")
    logger.info("  Fig 4.15 generada")


def generate_correlation_analysis(df: pd.DataFrame) -> dict:
    """Genera todas las figuras y tablas de correlación."""
    logger.info("Generando análisis de correlación...")

    # Tabla 4.2
    corr = compute_correlation_matrix(df)
    corr_labeled = _rename_cols(corr)
    save_table(
        corr_labeled, "tab_4_02_correlation",
        caption="Matriz de correlación de Pearson (muestra completa, 2000--2025)",
        label="tab:correlation",
        float_format="%.3f",
    )

    # Figuras
    fig_4_13_heatmap_full(df)
    fig_4_14_heatmaps_conditional(df)
    fig_4_15_rolling_correlations(df)

    logger.info("Análisis de correlación completado")
    return {"correlation_matrix": corr}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_correlation_analysis(df)

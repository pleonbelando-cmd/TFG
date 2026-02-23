"""
visualizations.py — Todas las figuras académicas del Capítulo 4.

17 figuras organizadas en 5 grupos:
  A) Evolución temporal (6 figuras)
  B) Variables estructurales/sentimiento (3 figuras)
  C) Distribuciones (3 figuras)
  D) Correlaciones (3 figuras) → en correlation_analysis.py
  E) Diagnósticos (2 figuras)

Las figuras de correlación (Grupo D) están en correlation_analysis.py.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

from src.config import EPISODES, VARIABLE_LABELS
from src.utils.plotting import COLORS, create_figure, save_figure, set_academic_style
from src.utils.episode_markers import add_episode_shading, add_episode_legend

logger = logging.getLogger(__name__)


def _format_date_axis(ax: plt.Axes):
    """Formato estándar para ejes de fecha."""
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO A — Evolución temporal (Figuras 4.1-4.6)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_4_01_gold_price(df: pd.DataFrame):
    """Fig 4.1: Precio del oro 2000-2025 con episodios sombreados."""
    fig, ax = create_figure()
    ax.plot(df.index, df["gold"], color=COLORS["gold"], linewidth=1.8)
    add_episode_shading(ax)
    ax.set_title("Figura 4.1: Precio del oro (2000–2025)")
    ax.set_ylabel("USD / onza troy")
    ax.set_xlabel("")
    _format_date_axis(ax)
    add_episode_legend(ax, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig_4_01_gold_price")
    logger.info("  Fig 4.1 generada")


def fig_4_02_gold_vs_dxy(df: pd.DataFrame):
    """Fig 4.2: Oro vs DXY (eje dual)."""
    fig, ax1 = create_figure()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)
    ax2.plot(df.index, df["dxy"], color=COLORS["primary"], label="DXY", linewidth=1.2, alpha=0.8)

    add_episode_shading(ax1, add_labels=False)

    ax1.set_ylabel("Oro (USD/oz)", color=COLORS["gold"])
    ax2.set_ylabel("DXY", color=COLORS["primary"])
    ax1.set_title("Figura 4.2: Oro vs. índice del dólar (DXY)")
    _format_date_axis(ax1)

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    save_figure(fig, "fig_4_02_gold_vs_dxy")
    logger.info("  Fig 4.2 generada")


def fig_4_03_gold_vs_tips(df: pd.DataFrame):
    """Fig 4.3: Oro vs TIPS yield (eje TIPS invertido)."""
    fig, ax1 = create_figure()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)
    ax2.plot(df.index, df["tips_10y"], color=COLORS["secondary"], label="TIPS 10Y (%)", linewidth=1.2)
    ax2.invert_yaxis()  # Invertir: menor yield → arriba (relación inversa visual)

    add_episode_shading(ax1, add_labels=False)

    ax1.set_ylabel("Oro (USD/oz)", color=COLORS["gold"])
    ax2.set_ylabel("TIPS 10Y (%) — invertido", color=COLORS["secondary"])
    ax1.set_title("Figura 4.3: Oro vs. tipo real a 10 años (TIPS, eje invertido)")
    _format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    save_figure(fig, "fig_4_03_gold_vs_tips")
    logger.info("  Fig 4.3 generada")


def fig_4_04_gold_vs_inflation(df: pd.DataFrame):
    """Fig 4.4: Oro vs inflación (CPI YoY + Breakeven)."""
    fig, ax1 = create_figure()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)
    ax2.plot(df.index, df["cpi_yoy"], color=COLORS["secondary"], label="CPI YoY (%)",
             linewidth=1.0, alpha=0.8)
    ax2.plot(df.index, df["breakeven"], color=COLORS["tertiary"], label="Breakeven 10Y (%)",
             linewidth=1.0, alpha=0.8, linestyle="--")

    add_episode_shading(ax1, add_labels=False)

    ax1.set_ylabel("Oro (USD/oz)", color=COLORS["gold"])
    ax2.set_ylabel("Inflación (%)")
    ax1.set_title("Figura 4.4: Oro vs. inflación (CPI interanual y breakeven)")
    _format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    save_figure(fig, "fig_4_04_gold_vs_inflation")
    logger.info("  Fig 4.4 generada")


def fig_4_05_gold_vs_vix(df: pd.DataFrame):
    """Fig 4.5: Oro vs VIX (eje dual)."""
    fig, ax1 = create_figure()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)
    ax2.plot(df.index, df["vix"], color=COLORS["quaternary"], label="VIX", linewidth=1.0, alpha=0.8)

    add_episode_shading(ax1, add_labels=False)

    ax1.set_ylabel("Oro (USD/oz)", color=COLORS["gold"])
    ax2.set_ylabel("VIX", color=COLORS["quaternary"])
    ax1.set_title("Figura 4.5: Oro vs. índice de volatilidad (VIX)")
    _format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    save_figure(fig, "fig_4_05_gold_vs_vix")
    logger.info("  Fig 4.5 generada")


def fig_4_06_gold_vs_wti_sp500(df: pd.DataFrame):
    """Fig 4.6: Oro vs WTI + S&P 500 (panel 2 subplots)."""
    fig, (ax1, ax2) = create_figure(nrows=2, ncols=1, figsize=(10, 8))

    # Subplot 1: Oro vs WTI
    ax1a = ax1.twinx()
    ax1.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)
    ax1a.plot(df.index, df["wti"], color=COLORS["primary"], label="WTI (USD/bbl)", linewidth=1.0)
    add_episode_shading(ax1, add_labels=False)
    ax1.set_ylabel("Oro (USD/oz)")
    ax1a.set_ylabel("WTI (USD/bbl)")
    ax1.set_title("Oro vs. WTI")
    _format_date_axis(ax1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1a.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    ax1a.spines["right"].set_visible(True)

    # Subplot 2: Oro vs S&P 500
    ax2a = ax2.twinx()
    ax2.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)
    ax2a.plot(df.index, df["sp500"], color=COLORS["secondary"], label="S&P 500", linewidth=1.0)
    add_episode_shading(ax2, add_labels=False)
    ax2.set_ylabel("Oro (USD/oz)")
    ax2a.set_ylabel("S&P 500")
    ax2.set_title("Oro vs. S&P 500")
    _format_date_axis(ax2)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2a.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    ax2a.spines["right"].set_visible(True)

    fig.suptitle("Figura 4.6: Oro vs. materias primas y renta variable", y=1.01, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig_4_06_gold_vs_wti_sp500")
    logger.info("  Fig 4.6 generada")


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO B — Variables estructurales / sentimiento (Figuras 4.7-4.9)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_4_07_cb_reserves(df: pd.DataFrame):
    """Fig 4.7: Compras netas de bancos centrales (barras anuales)."""
    fig, ax = create_figure()

    if df["cb_reserves"].notna().any():
        # Agregar por año
        annual = df["cb_reserves"].resample("YE").sum().dropna()
        colors = [COLORS["tertiary"] if v >= 0 else COLORS["secondary"] for v in annual]
        ax.bar(annual.index.year, annual.values, color=colors, alpha=0.8, width=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Compras netas (toneladas)")
    else:
        ax.text(0.5, 0.5, "Datos no disponibles\n(ver data/manual/README.md)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    ax.set_title("Figura 4.7: Compras netas de oro por bancos centrales")
    ax.set_xlabel("")
    fig.tight_layout()
    save_figure(fig, "fig_4_07_cb_reserves")
    logger.info("  Fig 4.7 generada")


def fig_4_08_google_trends(df: pd.DataFrame):
    """Fig 4.8: Google Trends "gold price" vs precio del oro."""
    fig, ax1 = create_figure()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["gold"], color=COLORS["gold"], label="Oro (USD/oz)", linewidth=1.5)

    if df["google_trends"].notna().any():
        ax2.plot(df.index, df["google_trends"], color=COLORS["primary"],
                 label='Google Trends "gold price"', linewidth=1.0, alpha=0.7)
        ax2.set_ylabel("Google Trends (índice 0-100)", color=COLORS["primary"])
    else:
        ax2.text(0.5, 0.5, "Google Trends no disponible",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")

    add_episode_shading(ax1, add_labels=False)
    ax1.set_ylabel("Oro (USD/oz)", color=COLORS["gold"])
    ax1.set_title('Figura 4.8: Oro vs. Google Trends "gold price"')
    _format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax2.spines["right"].set_visible(True)
    fig.tight_layout()
    save_figure(fig, "fig_4_08_google_trends")
    logger.info("  Fig 4.8 generada")


def fig_4_09_etf_flows(df: pd.DataFrame):
    """Fig 4.9: ETF flows (acumulado + barras mensuales)."""
    fig, ax1 = create_figure()

    if df["etf_flows"].notna().any():
        flows = df["etf_flows"].dropna()
        ax2 = ax1.twinx()

        # Barras: flujos mensuales
        colors = [COLORS["tertiary"] if v >= 0 else COLORS["secondary"] for v in flows]
        ax1.bar(flows.index, flows.values, width=25, color=colors, alpha=0.5, label="Flujo mensual")
        ax1.set_ylabel("Flujo mensual (toneladas)")
        ax1.axhline(0, color="black", linewidth=0.5)

        # Línea: acumulado
        cumulative = flows.cumsum()
        ax2.plot(cumulative.index, cumulative.values, color=COLORS["gold"],
                 linewidth=1.5, label="Acumulado")
        ax2.set_ylabel("Acumulado (toneladas)", color=COLORS["gold"])
        ax2.spines["right"].set_visible(True)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.text(0.5, 0.5, "Datos ETF no disponibles\n(ver data/manual/README.md)",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=12, color="gray")

    ax1.set_title("Figura 4.9: Flujos de ETFs de oro")
    _format_date_axis(ax1)
    fig.tight_layout()
    save_figure(fig, "fig_4_09_etf_flows")
    logger.info("  Fig 4.9 generada")


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO C — Distribuciones (Figuras 4.10-4.12)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_4_10_gold_returns_distribution(df: pd.DataFrame):
    """Fig 4.10: Histograma + KDE de retornos del oro vs normal."""
    fig, ax = create_figure()

    ret = df["gold_ret"].dropna()

    # Histograma + KDE
    ax.hist(ret, bins=50, density=True, alpha=0.5, color=COLORS["gold"],
            edgecolor="white", linewidth=0.5, label="Retornos del oro")

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ret)
    x_range = np.linspace(ret.min(), ret.max(), 200)
    ax.plot(x_range, kde(x_range), color=COLORS["gold"], linewidth=2, label="KDE")

    # Normal teórica
    x_norm = np.linspace(ret.min(), ret.max(), 200)
    normal_pdf = stats.norm.pdf(x_norm, ret.mean(), ret.std())
    ax.plot(x_norm, normal_pdf, color=COLORS["secondary"], linewidth=1.5,
            linestyle="--", label=f"Normal (μ={ret.mean():.2f}, σ={ret.std():.2f})")

    ax.set_title("Figura 4.10: Distribución de retornos mensuales del oro")
    ax.set_xlabel("Retorno logarítmico mensual (%)")
    ax.set_ylabel("Densidad")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "fig_4_10_gold_returns_dist")
    logger.info("  Fig 4.10 generada")


def fig_4_11_qq_plot(df: pd.DataFrame):
    """Fig 4.11: QQ plot de retornos del oro vs distribución normal."""
    fig, ax = create_figure(figsize=(7, 7))

    ret = df["gold_ret"].dropna()
    stats.probplot(ret, dist="norm", plot=ax)

    ax.set_title("Figura 4.11: Q-Q plot de retornos del oro vs. normal")
    ax.get_lines()[0].set(color=COLORS["gold"], markersize=4)
    ax.get_lines()[1].set(color=COLORS["secondary"], linewidth=1.5)

    fig.tight_layout()
    save_figure(fig, "fig_4_11_qq_plot")
    logger.info("  Fig 4.11 generada")


def fig_4_12_boxplots(df: pd.DataFrame):
    """Fig 4.12: Boxplots comparativos de todas las variables (estandarizadas)."""
    fig, ax = create_figure(figsize=(12, 6))

    # Variables a incluir (estandarizadas para comparabilidad)
    vars_to_plot = [c for c in ["gold_ret", "dxy", "tips_10y", "cpi_yoy", "breakeven",
                                 "vix", "sp500_ret", "wti"]
                    if c in df.columns and df[c].notna().sum() > 10]

    data_standardized = []
    labels = []
    for col in vars_to_plot:
        s = df[col].dropna()
        standardized = (s - s.mean()) / s.std()
        data_standardized.append(standardized.values)
        labels.append(VARIABLE_LABELS.get(col, col))

    bp = ax.boxplot(data_standardized, labels=labels, patch_artist=True,
                    medianprops={"color": COLORS["secondary"], "linewidth": 1.5})

    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=COLORS["gold"] if i == 0 else COLORS["light_gray"], alpha=0.6)

    ax.set_title("Figura 4.12: Boxplots comparativos (variables estandarizadas)")
    ax.set_ylabel("Valores estandarizados (z-score)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_figure(fig, "fig_4_12_boxplots")
    logger.info("  Fig 4.12 generada")


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO E — Diagnósticos (Figuras 4.16-4.17)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_4_16_acf_pacf(df: pd.DataFrame):
    """Fig 4.16: ACF/PACF de retornos del oro."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, (ax1, ax2) = create_figure(nrows=2, ncols=1, figsize=(10, 7))

    ret = df["gold_ret"].dropna()
    plot_acf(ret, ax=ax1, lags=24, alpha=0.05, color=COLORS["primary"])
    ax1.set_title("Autocorrelación (ACF) de retornos del oro")

    plot_pacf(ret, ax=ax2, lags=24, alpha=0.05, method="ywm", color=COLORS["primary"])
    ax2.set_title("Autocorrelación parcial (PACF) de retornos del oro")

    fig.suptitle("Figura 4.16: ACF y PACF de retornos mensuales del oro", y=1.01)
    fig.tight_layout()
    save_figure(fig, "fig_4_16_acf_pacf")
    logger.info("  Fig 4.16 generada")


def fig_4_17_scatterplot_matrix(df: pd.DataFrame):
    """Fig 4.17: Scatterplot matrix (oro vs DXY, TIPS, VIX, SP500)."""
    import seaborn as sns
    set_academic_style()

    vars_scatter = ["gold", "dxy", "tips_10y", "vix", "sp500"]
    vars_available = [v for v in vars_scatter if v in df.columns]
    subset = df[vars_available].dropna()

    rename_map = {v: VARIABLE_LABELS.get(v, v) for v in vars_available}
    subset = subset.rename(columns=rename_map)

    g = sns.pairplot(
        subset,
        kind="scatter",
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 15, "color": COLORS["primary"]},
        diag_kws={"color": COLORS["gold"], "linewidth": 1.5},
        height=2,
    )
    g.figure.suptitle("Figura 4.17: Scatterplot matrix — Oro y catalizadores principales",
                       y=1.02, fontsize=11)
    save_figure(g.figure, "fig_4_17_scatterplot_matrix")
    logger.info("  Fig 4.17 generada")


# ═══════════════════════════════════════════════════════════════════════════════
# Generador de todas las figuras
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(df: pd.DataFrame):
    """Genera las 14 figuras de este módulo (3 de correlación están aparte)."""
    logger.info("Generando figuras del Capítulo 4...")

    # Grupo A
    fig_4_01_gold_price(df)
    fig_4_02_gold_vs_dxy(df)
    fig_4_03_gold_vs_tips(df)
    fig_4_04_gold_vs_inflation(df)
    fig_4_05_gold_vs_vix(df)
    fig_4_06_gold_vs_wti_sp500(df)

    # Grupo B
    fig_4_07_cb_reserves(df)
    fig_4_08_google_trends(df)
    fig_4_09_etf_flows(df)

    # Grupo C
    fig_4_10_gold_returns_distribution(df)
    fig_4_11_qq_plot(df)
    fig_4_12_boxplots(df)

    # Grupo E
    fig_4_16_acf_pacf(df)
    fig_4_17_scatterplot_matrix(df)

    logger.info("Figuras completadas (14 de 17; 3 de correlación generadas aparte)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.data.pipeline import load_master_dataset
    df = load_master_dataset()
    generate_all_figures(df)

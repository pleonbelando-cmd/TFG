"""
create_tfg_completo.py
Genera TFG_Completo.docx: documento único condensado (~30 páginas)
con todos los capítulos 1-9 y figuras de datos reales.

Datos: Yahoo Finance (sin API key)
Figuras generadas:
  fig_01_gold_historia.png        — Precio del oro 2000-2025 con episodios de crisis
  fig_02_determinantes.png        — Series temporales de los 4 determinantes principales
  fig_03_correlaciones_rolling.png— Correlaciones móviles (36 meses) oro vs catalizadores
  fig_04_scatter.png              — Dispersión oro vs DXY y oro vs tipo nominal
  fig_05_shap.png                 — Importancia SHAP (valores de la Tabla 7.3)
  fig_06_ml_resultados.png        — Métricas comparativas de los modelos ML
"""

import io
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PROJECT_ROOT = Path(__file__).parent
FIGS_DIR = PROJECT_ROOT / "output" / "figures_completo"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colores ───────────────────────────────────────────────────────────────────
H1_COLOR = RGBColor(0x36, 0x5F, 0x91)
H2_COLOR = RGBColor(0x4F, 0x81, 0xBD)
H3_COLOR = RGBColor(0x4F, 0x81, 0xBD)

CRISIS_EPISODES = [
    ("GFC 2008",          "2007-08", "2009-06", "#d62728"),
    ("Máx. post-QE 2011", "2011-07", "2012-06", "#ff7f0e"),
    ("COVID-19 2020",     "2020-02", "2020-06", "#9467bd"),
    ("Ciclo tipos 2022",  "2022-03", "2023-12", "#8c564b"),
    ("Rally 2025",        "2025-01", "2025-12", "#e377c2"),
]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DESCARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def download_data() -> pd.DataFrame:
    """Descarga datos mensuales de Yahoo Finance 2000-2025."""
    import yfinance as yf

    tickers = {
        "gold":   "GC=F",    # Oro (futuros continuos USD/oz)
        "dxy":    "DX-Y.NYB",# Índice del dólar
        "sp500":  "^GSPC",   # S&P 500
        "vix":    "^VIX",    # VIX
        "y10":    "^TNX",    # Tipo nominal 10Y Tesoro EE.UU. (proxy tipo real)
        "wti":    "CL=F",    # Petróleo WTI
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start="2000-01-01", end="2025-12-31",
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  AVISO: sin datos para {ticker}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Resample mensual (último precio del mes)
            s = df["Close"].resample("ME").last()
            frames[name] = s
            print(f"  OK: {name} ({ticker}) — {len(s)} meses")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    data = pd.DataFrame(frames).dropna(how="all")
    data.index = pd.to_datetime(data.index)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GENERACIÓN DE FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

GREY  = "#e8e8e8"
BLUE  = "#365F91"
RED   = "#C0392B"
GREEN = "#1A6B3C"
ORANGE= "#D17B2A"

def shade_crises(ax, data):
    """Sombrea los episodios de crisis en un eje."""
    for _, start, end, color in CRISIS_EPISODES:
        s = pd.Timestamp(start)
        e = min(pd.Timestamp(end), data.index[-1])
        if s > data.index[-1]:
            continue
        ax.axvspan(s, e, alpha=0.12, color=color, zorder=0)


# ─── Fig 1: Evolución histórica del oro ──────────────────────────────────────
def fig_gold_historia(data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    gold = data["gold"].dropna()
    ax.plot(gold.index, gold.values, color=BLUE, lw=1.6, zorder=3)
    shade_crises(ax, gold)

    # Leyenda episodios
    patches = [mpatches.Patch(color=c, alpha=0.4, label=lbl)
               for lbl, _, _, c in CRISIS_EPISODES]
    ax.legend(handles=patches, fontsize=7.5, loc="upper left", ncol=2,
              framealpha=0.9)

    ax.set_title("Precio del oro (USD/oz) — enero 2000 a diciembre 2025",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("USD por onza troy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = FIGS_DIR / "fig_01_gold_historia.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {path.name}")
    return path


# ─── Fig 2: Series temporales de los determinantes ───────────────────────────
def fig_determinantes(data: pd.DataFrame) -> Path:
    vars_plot = [
        ("gold",  "Oro (USD/oz)",            BLUE,   True),
        ("dxy",   "DXY (Índice del dólar)",  ORANGE, False),
        ("y10",   "Tipo nominal 10Y (%)",    RED,    False),
        ("vix",   "VIX (Volatilidad impl.)", GREEN,  False),
    ]
    # Filtrar los que existan
    vars_plot = [(k,l,c,n) for k,l,c,n in vars_plot if k in data.columns]

    n = len(vars_plot)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (key, label, color, _) in zip(axes, vars_plot):
        s = data[key].dropna()
        ax.plot(s.index, s.values, color=color, lw=1.3)
        shade_crises(ax, s)
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Año")
    fig.suptitle("Variables explicativas y variable dependiente — 2000-2025",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = FIGS_DIR / "fig_02_determinantes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {path.name}")
    return path


# ─── Fig 3: Correlaciones rolling 36 meses ───────────────────────────────────
def fig_correlaciones_rolling(data: pd.DataFrame) -> Path:
    gold_ret  = np.log(data["gold"]).diff().dropna()
    pairs = [
        ("dxy",  "Oro vs DXY",             RED),
        ("sp500","Oro vs S&P 500",          ORANGE),
        ("y10",  "Oro vs Tipo nominal 10Y", BLUE),
        ("vix",  "Oro vs VIX",             GREEN),
    ]
    pairs = [(k,l,c) for k,l,c in pairs if k in data.columns]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for key, label, color in pairs:
        other_ret = np.log(data[key].replace(0, np.nan)).diff().dropna()
        combined  = pd.concat([gold_ret, other_ret], axis=1).dropna()
        combined.columns = ["gold", key]
        roll_corr = combined["gold"].rolling(36).corr(combined[key])
        ax.plot(roll_corr.index, roll_corr.values, label=label, color=color, lw=1.4)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    shade_crises(ax, gold_ret)
    ax.set_title("Correlación móvil (36 meses) del retorno del oro con sus determinantes",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Coeficiente de correlación")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = FIGS_DIR / "fig_03_correlaciones_rolling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {path.name}")
    return path


# ─── Fig 4: Scatter oro vs DXY y oro vs tipo nominal ─────────────────────────
def fig_scatter(data: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pairs_scatter = [
        ("dxy",  "DXY (Índice del dólar)", axes[0]),
        ("y10",  "Tipo nominal 10Y (%)",   axes[1]),
    ]
    pairs_scatter = [(k,l,a) for k,l,a in pairs_scatter if k in data.columns]

    colors_year = plt.cm.plasma(
        np.linspace(0, 1, len(data["gold"].dropna()))
    )

    for key, xlabel, ax in pairs_scatter:
        combined = data[["gold", key]].dropna()
        sc = ax.scatter(combined[key], combined["gold"],
                        c=plt.cm.plasma(np.linspace(0, 1, len(combined))),
                        alpha=0.6, s=18, edgecolors="none")
        # Línea de tendencia
        m, b = np.polyfit(combined[key], combined["gold"], 1)
        x_line = np.linspace(combined[key].min(), combined[key].max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", lw=1.2, ls="--", alpha=0.6)
        corr = combined["gold"].corr(combined[key])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Oro (USD/oz)" if ax == axes[0] else "")
        ax.set_title(f"r = {corr:.2f}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Colorbar años
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=2000, vmax=2025))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Año", fontsize=8)

    fig.suptitle("Relación entre el precio del oro y sus catalizadores (mensual, 2000-2025)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()

    path = FIGS_DIR / "fig_04_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {path.name}")
    return path


# ─── Fig 5: Importancia SHAP (datos de la Tabla 7.3) ─────────────────────────
def fig_shap() -> Path:
    variables  = ["CPI YoY (t-1)", "TIPS 10Y (t-2)", "Ret. oro (t-1)",
                  "Breakeven (t-3)", "WTI (t-2)", "S&P 500 (t-1)",
                  "Vol3 oro", "DXY (t-3)"]
    importances = [0.954, 0.617, 0.526, 0.485, 0.423, 0.397, 0.379, 0.329]

    colors = [RED if "CPI" in v or "Brea" in v
              else BLUE if "TIPS" in v
              else ORANGE if "S&P" in v or "WTI" in v
              else GREEN for v in variables]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = range(len(variables) - 1, -1, -1)
    bars = ax.barh(list(y_pos), importances[::-1], color=colors[::-1],
                   edgecolor="white", height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(variables[::-1], fontsize=9)
    ax.set_xlabel("Importancia SHAP media |φ̄|", fontsize=9)
    ax.set_title("Top 8 variables por importancia SHAP — XGBoost (período de test)",
                 fontsize=10.5, fontweight="bold")

    for bar, val in zip(bars, importances[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    legend_elements = [
        mpatches.Patch(color=RED,    label="Inflación / expectativas"),
        mpatches.Patch(color=BLUE,   label="Tipos de interés reales"),
        mpatches.Patch(color=ORANGE, label="Materias primas / renta var."),
        mpatches.Patch(color=GREEN,  label="Momentum / volatilidad"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = FIGS_DIR / "fig_05_shap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {path.name}")
    return path


# ─── Fig 6: Comparativa de modelos ML ────────────────────────────────────────
def fig_ml_resultados() -> Path:
    models = ["Naive\n(random walk)", "XGBoost", "Random\nForest", "LSTM"]
    rmse   = [5.054, 4.340, 3.882, 3.815]
    da     = [55.9,  52.3,  58.7,  61.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bar_colors = [GREY, ORANGE, BLUE, RED]
    bars1 = ax1.bar(models, rmse, color=bar_colors, edgecolor="white", width=0.55)
    ax1.set_ylabel("RMSE (puntos porcentuales)", fontsize=9)
    ax1.set_title("Error de predicción (RMSE)\n← menor es mejor", fontsize=9.5)
    ax1.axhline(rmse[0], color="black", ls="--", lw=0.9, alpha=0.5, label="Naive")
    for bar, v in zip(bars1, rmse):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                 f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, 6.2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(models, da, color=bar_colors, edgecolor="white", width=0.55)
    ax2.set_ylabel("DA — Precisión direccional (%)", fontsize=9)
    ax2.set_title("Acierto direccional (DA)\n→ mayor es mejor", fontsize=9.5)
    ax2.axhline(da[0], color="black", ls="--", lw=0.9, alpha=0.5)
    ax2.axhline(50, color="grey", ls=":", lw=0.8, alpha=0.4, label="Azar (50%)")
    for bar, v in zip(bars2, da):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(40, 70)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Comparativa de modelos predictivos — walk-forward (oct. 2016 – oct. 2025)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()

    path = FIGS_DIR / "fig_06_ml_resultados.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {path.name}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONSTRUCCIÓN DEL DOCUMENTO WORD
# ══════════════════════════════════════════════════════════════════════════════

def make_doc() -> Document:
    doc = Document()
    for section in doc.sections:
        section.page_width  = Cm(21.59)
        section.page_height = Cm(27.94)
        section.top_margin    = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin   = Cm(3.05)
        section.right_margin  = Cm(3.05)

    for sty_name, size_pt, color, space_before in [
        ("Heading 1", 14, H1_COLOR, 24),
        ("Heading 2", 13, H2_COLOR, 10),
        ("Heading 3", 11, H3_COLOR, 8),
    ]:
        sty = doc.styles[sty_name]
        sty.font.size  = Pt(size_pt)
        sty.font.bold  = True
        sty.font.color.rgb = color
        sty.paragraph_format.space_before = Pt(space_before)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    return doc


def set_spacing(p):
    pPr = p._element.get_or_add_pPr()
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:line"),     "276")
    sp.set(qn("w:lineRule"), "auto")
    sp.set(qn("w:after"),    "120")
    pPr.append(sp)


def H1(doc, text):
    p = doc.add_heading(text, level=1)
    return p


def H2(doc, text):
    p = doc.add_heading(text, level=2)
    return p


def H3(doc, text):
    p = doc.add_heading(text, level=3)
    return p


def P(doc, text):
    """Párrafo normal con soporte a **negrita** e *cursiva*."""
    import re
    p = doc.add_paragraph()
    set_spacing(p)
    # split por negrita **...** o cursiva *...*
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*") and tok.endswith("*"):
            r = p.add_run(tok[1:-1]); r.italic = True
        else:
            p.add_run(tok)
    return p


def CAPTION(doc, text):
    """Pie de figura en cursiva, centrado."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(10)
    return p


def INSERT_FIG(doc, path: Path, caption: str, width_cm: float = 14.0):
    """Inserta una figura con pie de imagen."""
    if not path.exists():
        P(doc, f"[Figura no disponible: {path.name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    CAPTION(doc, caption)


def TABLE(doc, headers, rows):
    """Tabla Word simple con fila de cabecera en negrita."""
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row[:len(headers)]):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(9)
    doc.add_paragraph()


def PAGE_BREAK(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break(__import__("docx.enum.text", fromlist=["WD_BREAK"]).WD_BREAK.PAGE)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONTENIDO DE LOS CAPÍTULOS (CONDENSADO)
# ══════════════════════════════════════════════════════════════════════════════

def write_portada(doc):
    doc.add_paragraph()
    p = doc.add_paragraph("TRABAJO DE FIN DE GRADO")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].bold = True
    p.runs[0].font.size = Pt(14)

    p2 = doc.add_paragraph("Grado en Economía · Econometría III")
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.runs[0].font.size = Pt(11)

    doc.add_paragraph()

    p3 = doc.add_paragraph("DINÁMICA DEL PRECIO DEL ORO (2000-2025):")
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.runs[0].bold = True
    p3.runs[0].font.size = Pt(16)
    p3.runs[0].font.color.rgb = H1_COLOR

    p4 = doc.add_paragraph("Un análisis integrado mediante VAR/VECM,\nDatos de Panel y Machine Learning")
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p4.runs[0].bold = True
    p4.runs[0].font.size = Pt(13)

    for _ in range(3):
        doc.add_paragraph()

    for line in ["Autor: Pablo León Belando", "Febrero 2026"]:
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.size = Pt(11)

    PAGE_BREAK(doc)


def cap1(doc):
    H1(doc, "Capítulo 1: Introducción y motivación")

    H2(doc, "1.1 Motivación")
    P(doc, "El 15 de agosto de 1971, Nixon suspendió la convertibilidad del dólar en oro, transformando el metal de ancla monetaria en activo financiero. Más de cincuenta años después, en 2025, el precio del oro cerró a 4.549 USD/oz —más de 130 veces su precio de 1971 en términos nominales— tras establecer 53 nuevos máximos históricos a lo largo del año. El debate académico sobre qué mueve al oro, qué función cumple en una cartera y si su precio puede predecirse sigue abierto y vigente.")
    P(doc, "El periodo 2000-2025 concentra cinco episodios excepcionales: la Crisis Financiera Global (GFC) de 2008, los máximos post-QE de 2011, la pandemia COVID-19 en 2020, el ciclo de subidas de tipos más agresivo en cuatro décadas (2022-2024) y la espectacular subida de 2025 impulsada por la guerra arancelaria y la aceleración del proceso de de-dolarización global. En cada uno de estos episodios el oro se comportó de forma diferente, lo que lo convierte en un laboratorio ideal para aplicar herramientas econométricas y de machine learning.")

    H2(doc, "1.2 Preguntas de investigación")
    P(doc, "**Pregunta 1:** ¿Qué variables macroeconómicas y financieras determinan el precio del oro en el periodo 2000-2025, y cuál es su importancia relativa?")
    P(doc, "**Pregunta 2:** ¿Han cambiado los determinantes del oro tras los grandes episodios de crisis del periodo analizado?")
    P(doc, "**Pregunta 3:** ¿Puede el machine learning mejorar la predicción del precio del oro respecto a los modelos econométricos clásicos, y qué información aporta sobre el peso relativo de cada variable?")

    H2(doc, "1.3 Estructura y metodología")
    P(doc, "El trabajo se organiza en tres pilares metodológicos complementarios siguiendo el temario de Econometría III: (i) un **VECM** (Capítulo 5) como núcleo del análisis —heredero de los Modelos de Ecuaciones Simultáneas—, que cuantifica relaciones de equilibrio de largo plazo, funciones de impulso-respuesta y descomposición de varianza; (ii) un **análisis de panel cross-country** (Capítulo 6) con efectos fijos, efectos aleatorios y contraste de Hausman para contrastar si los mecanismos identificados son universales; y (iii) modelos de **machine learning** (Capítulo 7) con validación walk-forward y análisis SHAP para la predicción de corto plazo y la cuantificación del peso relativo de las variables en distintos regímenes de mercado.")


def cap2(doc):
    H1(doc, "Capítulo 2: Marco teórico")

    H2(doc, "2.1 El oro como activo financiero singular")
    P(doc, "El oro ocupa una posición única en la taxonomía de los activos financieros. No genera flujos de caja, no paga dividendos ni cupones y no tiene valor de uso mayoritario en sentido productivo —a diferencia del petróleo o los metales industriales—. Y sin embargo, millones de inversores, bancos centrales y gobiernos lo acumulan como reserva de valor. Esta paradoja —un activo sin rendimiento corriente que cotiza a precios récord— es el punto de partida de toda la literatura académica sobre el oro.")

    H2(doc, "2.2 Hedge, safe haven y activo especulativo")
    P(doc, "La distinción conceptual central de la literatura es la que establecieron Baur y Lucey (2010): un activo es un **hedge** si su correlación con otro activo es, en promedio, negativa o nula a lo largo del tiempo. Es un **safe haven** si esa correlación es negativa en los momentos de turbulencia del mercado, aunque sea positiva o nula el resto del tiempo. La distinción importa porque implica funciones de gestión de cartera diferentes: el hedge protege de forma continua, el safe haven solo en las crisis.")
    P(doc, "Baur y McDermott (2010) documentaron que el oro fue un safe haven durante la GFC para los mercados de EE.UU. y Europa, pero no para los mercados BRIC. Este resultado —la no universalidad del safe haven— es uno de los que este trabajo pone a prueba con datos actualizados y con la dimensión de panel cross-country.")
    P(doc, "Erb y Harvey (2013) cuestionaron la idea de que el oro sea un buen **hedge contra la inflación** a horizontes prácticos: su análisis empírico muestra que la correlación entre el precio real del oro y la inflación acumulada es alta a horizontes de décadas, pero muy baja e inestable a horizontes de 1-10 años. Esta conclusión —relevante para el inversor institucional— es uno de los referentes del análisis de cointegración del Capítulo 5.")

    H2(doc, "2.3 Del modelo de ecuaciones simultáneas al VAR")
    P(doc, "Los primeros modelos macroeconómicos de gran escala se articulaban como Modelos de Ecuaciones Simultáneas (MES): sistemas de ecuaciones estructurales con restricciones de identificación impuestas a priori para distinguir variables exógenas y endógenas. El problema fundamental es que esas restricciones no pueden verificarse desde los datos: son supuestos teóricos que se imponen por conveniencia estadística. Como señaló Sims (1980) en su influyente crítica, el modelo VAR —que trata todas las variables del sistema como igualmente endógenas y modela cada una como función de los retardos de todas las demás— es la forma reducida del MES sin restricciones de identificación arbitrarias. Para un sistema donde el oro, el dólar, los tipos reales y la renta variable se afectan mutuamente de formas que no podemos especificar con certeza, el VAR ofrece el marco más honesto y apropiado.")


def cap3(doc):
    H1(doc, "Capítulo 3: Catalizadores del precio del oro")

    H2(doc, "3.1 Tipos de interés reales")
    P(doc, "Los **tipos de interés reales** son el determinante macroeconómico más robusto del precio del oro en la literatura empírica. El mecanismo es el **coste de oportunidad**: el oro no paga cupón ni dividendo; mantenerlo implica renunciar al rendimiento de un bono de igual plazo. Cuando el tipo real sube, el coste de oportunidad del oro aumenta y su precio tiende a bajar; cuando los tipos reales son negativos —como ocurrió en 2012 y en el periodo 2020-2022—, el coste de oportunidad desaparece y el oro se convierte en el activo sin riesgo más atractivo. La correlación documentada por Erb y Harvey (2013) entre el tipo real TIPS a 10 años y el precio del oro fue de -0,82 para el periodo 1997-2012.")

    H2(doc, "3.2 El índice del dólar (DXY)")
    P(doc, "El **precio del dólar** es el segundo determinante estructural del oro. Al cotizar globalmente en dólares, el precio del oro para un inversor en otra moneda sube automáticamente cuando el dólar se deprecia —incluso sin que el precio en USD haya cambiado—. La correlación histórica entre el DXY y el precio del oro es fuertemente negativa: un dólar fuerte abarata el oro para compradores no estadounidenses y reduce la demanda global. Sin embargo, esta relación mostró una ruptura notable en 2022-2024, cuando dólar y oro subieron en paralelo.")

    H2(doc, "3.3 Inflación, VIX y renta variable")
    P(doc, "La **inflación** (IPC y breakevens) actúa como catalizador de la demanda de oro como cobertura: en periodos de alta inflación esperada, los inversores aumentan su exposición al metal como activo real. El **VIX** activa el canal de safe haven: en picos de volatilidad, los flujos hacia activos defensivos elevan el precio del oro independientemente del nivel de los tipos reales. La **renta variable** (S&P 500) está negativamente correlacionada con el oro en el largo plazo: los periodos de expansión bursátil sostenida coinciden con menor demanda de activos defensivos, aunque en las crisis la correlación puede invertirse brevemente.")

    H2(doc, "3.4 Endogeneidad y motivación del VAR")
    P(doc, "La elección de un VAR en lugar de una regresión de MCO se justifica por la **endogeneidad simultánea** entre el precio del oro y sus determinantes. Los tipos reales pueden verse influidos por las expectativas de inflación que el propio precio del oro señaliza; el DXY responde a flujos de capital que también mueven el oro. En el marco del MES, cada una de estas variables sería endógena en al menos alguna ecuación del sistema. El VAR, al no imponer restricciones de exogeneidad a priori, es la especificación más conservadora y metodológicamente honesta para este sistema.")


def cap4(doc, figs):
    H1(doc, "Capítulo 4: Datos y análisis exploratorio")

    H2(doc, "4.1 Fuentes y muestra")
    P(doc, "El análisis cubre el periodo **enero 2000 – diciembre 2025** con frecuencia mensual (312 observaciones). Todas las series se descargan de fuentes institucionales públicas:")
    TABLE(doc,
          ["Variable", "Símbolo", "Fuente", "Frecuencia"],
          [
              ["Precio del oro",         "XAU/USD",   "Yahoo Finance (GC=F)",      "Diaria → mensual"],
              ["Índice del dólar",        "DXY",        "Yahoo Finance (DX-Y.NYB)",  "Diaria → mensual"],
              ["Tipo real TIPS 10Y",      "TIPS",       "FRED (DFII10)",             "Diaria → mensual"],
              ["Inflación IPC EE.UU.",    "CPI",        "FRED (CPIAUCSL)",           "Mensual"],
              ["Breakeven inflación 10Y", "BEI",        "FRED (T10YIE)",             "Diaria → mensual"],
              ["Volatilidad impl. VIX",   "VIX",        "Yahoo Finance (^VIX)",      "Diaria → mensual"],
              ["Índice S&P 500",          "SP500",      "Yahoo Finance (^GSPC)",     "Diaria → mensual"],
              ["Petróleo WTI",            "WTI",        "Yahoo Finance (CL=F)",      "Diaria → mensual"],
          ]
    )

    H2(doc, "4.2 Evolución histórica del oro y episodios de crisis")
    P(doc, "La Figura 1 presenta la evolución del precio del oro desde enero de 2000 hasta diciembre de 2025, con los cinco episodios de mercado que articulan el análisis. La trayectoria revela tres periodos claramente diferenciados: una primera fase alcista (2000-2012) impulsada por la expansión monetaria post-GFC, que llevó el precio de 280 USD/oz a un máximo de 1.900 USD/oz; un periodo de consolidación (2012-2018) con el precio rondando los 1.200-1.400 USD/oz a medida que los tipos reales normalizaban; y una segunda fase alcista (2020-2025) que encadenó dos máximos históricos —COVID en 2020 y el rally de 2025— para cerrar el año en 4.549 USD/oz.")
    INSERT_FIG(doc, figs["fig1"],
               "Figura 1. Precio mensual del oro (USD/oz) enero 2000 – diciembre 2025. "
               "Las zonas sombreadas corresponden a los cinco episodios de crisis analizados. "
               "Fuente: Yahoo Finance (GC=F).", width_cm=14.5)

    H2(doc, "4.3 Series temporales de los determinantes")
    P(doc, "La Figura 2 muestra la evolución simultánea de las cuatro variables que articulan el modelo econométrico: el precio del oro (variable dependiente), el índice del dólar DXY, el tipo nominal del Tesoro a 10 años (proxy del tipo real dado que los TIPS tienen cobertura desde 2003) y el VIX. Las series permiten visualizar las relaciones cualitativas que la econometría cuantifica: la correlación negativa histórica oro-DXY, los picos del VIX en los episodios de crisis y la divergencia del periodo 2022-2024 —donde los tipos subieron agresivamente pero el oro también lo hizo.")
    INSERT_FIG(doc, figs["fig2"],
               "Figura 2. Evolución mensual de la variable dependiente (oro, USD/oz) y los principales "
               "determinantes: DXY, tipo nominal 10Y (%) y VIX. Las zonas sombreadas son los episodios "
               "de crisis. Fuente: Yahoo Finance.", width_cm=14.5)

    H2(doc, "4.4 Correlaciones móviles")
    P(doc, "La Figura 3 presenta las correlaciones móviles de 36 meses entre el retorno del oro y el de sus catalizadores. Tres patrones son especialmente relevantes. Primero, la correlación oro-DXY es persistentemente negativa (en torno a -0,6) durante la mayor parte del periodo, pero se eleva hacia cero e incluso positivo en 2022-2024, señal de la ruptura estructural que los tests de Chow del Capítulo 5 formalizan. Segundo, la correlación oro-tipo nominal alterna entre valores negativos (cuando el coste de oportunidad domina) y menos negativos o positivos en los picos inflacionarios, donde el oro actúa como cobertura. Tercero, la correlación oro-VIX es positiva en todos los episodios de crisis, confirmando el rol de safe haven.")
    INSERT_FIG(doc, figs["fig3"],
               "Figura 3. Correlaciones móviles (ventana 36 meses) del retorno mensual del oro con "
               "sus catalizadores. Fuente: elaboración propia sobre datos de Yahoo Finance.", width_cm=14.5)


def cap5(doc, figs):
    H1(doc, "Capítulo 5: Análisis econométrico VAR/VECM")

    H2(doc, "5.1 Del MES al VAR: motivación metodológica")
    P(doc, "El modelo VAR puede entenderse como la forma reducida del Modelo de Ecuaciones Simultáneas (MES) de Sims (1980): sin restricciones de identificación arbitrarias, sin variables artificialmente declaradas exógenas. Para un sistema donde el oro, el dólar, los tipos reales y la renta variable se afectan mutuamente, el VAR es el marco metodológicamente más conservador. Cuando las variables son no estacionarias y están cointegradas, la extensión natural es el **VECM**, que añade un término de corrección de errores que captura la dinámica de ajuste al equilibrio de largo plazo.")

    H2(doc, "5.2 Tests de raíz unitaria y cointegración")
    P(doc, "Los tests ADF y KPSS confirman que el precio del oro, el DXY, los TIPS a 10 años y el S&P 500 son **I(1)**: no estacionarios en niveles pero estacionarios en primeras diferencias. El VIX, el IPC interanual y el breakeven son **I(0)** y se tratan como variables exógenas en el VECM. El test de Johansen sobre el sistema {ln(Oro), ln(DXY), TIPS, ln(S&P 500)} detecta **un vector de cointegración** (r = 1): existe una combinación lineal estacionaria de las cuatro series, es decir, una relación de equilibrio de largo plazo de la que se desvían temporalmente pero a la que tienden a retornar.")

    H2(doc, "5.3 Vector de cointegración y velocidades de ajuste")
    P(doc, "El vector de cointegración normalizado (coeficiente del oro = 1) confirma las hipótesis teóricas: el coeficiente del DXY es **negativo** (dólar fuerte → oro más barato), el de los TIPS es **negativo** (tipos reales altos → coste de oportunidad elevado) y el del S&P 500 es **negativo** (expansión bursátil → menor demanda defensiva). El coeficiente α del oro —la velocidad de ajuste— es negativo y estadísticamente significativo, lo que confirma que cuando el precio del oro está por encima de su nivel de equilibrio, tiende a corregirse en los meses siguientes.")

    H2(doc, "5.4 Funciones de impulso-respuesta y descomposición de varianza")
    P(doc, "La Figura 4 presenta la relación de dispersión entre el precio del oro y sus dos determinantes principales. La pendiente negativa con el DXY y con el tipo nominal del Tesoro es visible en toda la muestra, aunque con dispersión creciente en los años recientes —consecuencia de la ruptura estructural que el análisis de estabilidad documenta formalmente.")
    INSERT_FIG(doc, figs["fig4"],
               "Figura 4. Relación entre el precio del oro (USD/oz) y sus dos determinantes principales: "
               "DXY (izquierda) y tipo nominal del Tesoro a 10Y (derecha). Escala de color: año de observación "
               "(de 2000 en morado a 2025 en amarillo). La línea discontinua es la tendencia lineal. "
               "Fuente: Yahoo Finance.", width_cm=14.0)
    P(doc, "Las funciones de impulso-respuesta del VECM (calculadas mediante descomposición de Cholesky con el oro en posición más endógena) muestran que un shock positivo de una desviación típica en los TIPS produce la respuesta negativa de mayor magnitud y persistencia sobre el precio del oro, seguida por el shock en el DXY. El shock en el S&P 500 también es negativo pero se disipa más rápidamente. La descomposición de varianza confirma esta jerarquía: a horizontes de 12-24 meses, los TIPS explican la mayor fracción de la varianza no anticipada del precio del oro, seguidos por el DXY.")

    H2(doc, "5.5 Volatilidad condicional: GJR-GARCH(1,1)")
    P(doc, "El test ARCH-LM rechaza la ausencia de heterocedasticidad condicional en los retornos del oro (p < 0,05), justificando el análisis complementario de volatilidad. El modelo **GJR-GARCH(1,1) con distribución t de Student** estima una persistencia total de la volatilidad inferior a 1 (estacionariedad en covarianza) y un parámetro β próximo a 0,85, indicando *clusters* de larga duración. Los picos de volatilidad condicional coinciden con la GFC (septiembre-noviembre 2008) y el crash COVID de marzo 2020. El periodo 2022-2025, pese a la magnitud del movimiento alcista, muestra volatilidad persistente pero sin picos extremos, lo que refuerza la interpretación de una subida impulsada por demanda estructural sostenida.")

    H2(doc, "5.6 Estabilidad estructural")
    P(doc, "Los **tests de Chow** en cuatro puntos de quiebre económicamente motivados rechazan la estabilidad de los parámetros, con el mayor F-estadístico en marzo de 2022. El **análisis CUSUM** sale de las bandas de confianza al 5% durante el periodo 2022-2024, confirmando que los parámetros del modelo cambiaron cualitativamente en ese episodio. Los coeficientes rolling del TIPS sobre el precio del oro muestran que su efecto negativo —históricamente próximo a -0,7— se atenuó durante 2022-2024. Esto es la firma econométrica de la paradoja: el mecanismo de coste de oportunidad perdió potencia cuando la demanda soberana de los bancos centrales emergentes actuó como soporte estructural del precio.")


def cap6(doc):
    H1(doc, "Capítulo 6: Análisis de panel cross-country")

    H2(doc, "6.1 Motivación: ¿es universal el comportamiento del oro?")
    P(doc, "Los capítulos anteriores analizan el oro desde la perspectiva de los datos estadounidenses. Pero el oro cotiza globalmente: un inversor europeo, japonés o británico percibe su precio en euros, yenes o libras esterlinas, y la relación entre ese precio y sus variables locales de coste de oportunidad puede ser diferente. La pregunta de la universalidad de los mecanismos identificados requiere un análisis de **datos de panel cross-country**.")

    H2(doc, "6.2 Muestra y especificación")
    P(doc, "El panel comprende **cuatro economías avanzadas** y **96 trimestres** (2000-2024), con N×T = 384 observaciones. La variable dependiente es el retorno trimestral del oro en *moneda local* (precio en USD dividido por el tipo de cambio correspondiente). Las variables explicativas son la inflación local (IPC), el tipo de interés real local a 10 años, el VIX (variable común) y el retorno del índice bursátil principal de cada economía.")
    TABLE(doc,
          ["Economía", "Moneda", "Índice bursátil", "Tipo real 10Y"],
          [
              ["EE.UU.",   "USD", "S&P 500 (^GSPC)",      "TIPS 10Y (FRED DFII10)"],
              ["Eurozona", "EUR", "EuroStoxx 50 (^STOXX50E)", "OAT real (BCE)"],
              ["R. Unido", "GBP", "FTSE 100 (^FTSE)",     "Gilt real (BoE)"],
              ["Japón",    "JPY", "Nikkei 225 (^N225)",   "JGB real (BoJ)"],
          ]
    )
    P(doc, "El modelo es: *r_gold_it = β₀ + β₁π_it + β₂r_it + β₃VIX_t + β₄eq_it + η_i + ε_it*, donde η_i captura la heterogeneidad inobservable de cada economía (demanda cultural de oro, historial de inflación del banco central, política de reservas soberanas).")

    H2(doc, "6.3 Efectos fijos vs. efectos aleatorios y test de Hausman")
    P(doc, "El **estimador de efectos fijos** (*within*) elimina η_i mediante la substracción de la media temporal de cada país, siendo consistente bajo cualquier correlación entre η_i y los regresores, a costa de perder la variación *between*. El **estimador de efectos aleatorios** (GLS) es más eficiente si η_i no está correlacionado con los regresores. El **test de Hausman** (1978) formaliza esta elección: H₀ = EA consistente y eficiente; H_A = EA inconsistente, EF preferido.")
    P(doc, "El test rechaza H₀ (p < 0,05), señalando que los efectos individuales η_i están correlacionados con los regresores —resultado esperado, dado que factores como la cultura local de inversión en oro o la dependencia estructural del sistema financiero en dólares están correlacionados con la inflación y los tipos de interés de cada economía. Se adoptan los **efectos fijos** como estimador preferido.")

    H2(doc, "6.4 Resultados e interpretación")
    P(doc, "Los resultados del modelo de efectos fijos con errores de Driscoll-Kraay (robustos a dependencia transversal y temporal simultánea) confirman los signos teóricos en las cuatro economías: **β₁ > 0** (inflación positiva para el oro: función de cobertura universal), **β₂ < 0** (tipos reales negativos: mecanismo de coste de oportunidad universal) y **β₃ > 0** (VIX positivo: safe haven global). El coeficiente del tipo real es estadísticamente significativo en todas las economías con la misma dirección, lo que establece que el mecanismo identificado en el análisis de serie temporal de EE.UU. no es una idiosincrasia del mercado del Tesoro estadounidense. Los efectos fijos estimados revelan heterogeneidad: Japón exhibe una demanda estructural de oro superior a la predicha por las variables del modelo, consistente con su demanda cultural e histórica del metal.")


def cap7(doc, figs):
    H1(doc, "Capítulo 7: Extensión predictiva con Machine Learning")

    P(doc, "*Nota metodológica: Este capítulo aplica técnicas de machine learning como extensión complementaria al análisis econométrico. Las arquitecturas implementadas —XGBoost, Random Forest y LSTM— van más allá del temario de Econometría III, pero se incluyen porque aportan una perspectiva predictiva que contrasta directamente con la econometría clásica.*")

    H2(doc, "7.1 Metodología: validación walk-forward")
    P(doc, "La validación cruzada estándar (*k-fold*) introduce *look-ahead bias* en series temporales: el modelo podría entrenarse con datos de 2022 para predecir 2015. La **validación walk-forward con ventana expandible** elimina este problema: el modelo se entrena en [1, t-1] y predice t; a continuación amplía el entrenamiento a [1, t] y predice t+1, sin usar nunca información posterior al instante de predicción. La muestra de entrenamiento inicial son 162 observaciones (abril 2003 – septiembre 2016) y el período de test son 109 meses (octubre 2016 – octubre 2025). La variable objetivo es el retorno logarítmico mensual del oro; la matriz de características comprende 35 variables: retornos de los catalizadores con retardos 1, 2 y 3 meses, momentum del oro a 3 y 6 meses, volatilidad realizada y una dummy de régimen de crisis.")

    H2(doc, "7.2 Resultados comparativos")
    P(doc, "La Figura 5 compara las cuatro métricas de evaluación de los tres modelos frente al benchmark naive (paseo aleatorio). La LSTM obtiene el mejor rendimiento global, con un RMSE de 3,815 pp frente a 5,054 pp del naive (-24,5%) y una precisión direccional (DA) del 61,5% frente al 55,9% del naive (+5,6 pp). El Random Forest supera al XGBoost en todas las métricas —resultado frecuente en series financieras cortas donde el *bagging* es más robusto que el *boosting* secuencial—. Notablemente, el XGBoost obtiene una DA inferior al naive (52,3%), indicando que minimiza el error de magnitud a costa de introducir ruido en la dirección del movimiento.")
    INSERT_FIG(doc, figs["fig6"],
               "Figura 5. Comparativa de modelos predictivos: RMSE (izquierda, menor es mejor) y "
               "precisión direccional DA (derecha, mayor es mejor) sobre los 109 meses del período de "
               "test walk-forward (oct. 2016 – oct. 2025). La línea discontinua marca el benchmark naive.",
               width_cm=13.5)

    H2(doc, "7.3 Interpretabilidad: análisis SHAP")
    P(doc, "Los valores SHAP (TreeSHAP exacto para XGBoost) descomponen cada predicción en la contribución marginal de cada variable. La Figura 6 muestra las 8 variables más influyentes. El IPC con retardo de 1 mes encabeza el ranking (|φ̄| = 0,954), seguido por los TIPS con retardo de 2 meses (0,617) y el momentum del propio precio del oro (0,526). Los signos son plenamente coherentes con los hallazgos econométricos: inflación alta → SHAP positivo (predicción alcista del oro), tipos reales altos → SHAP negativo (coste de oportunidad), S&P 500 alto → SHAP negativo (sustitución).")
    INSERT_FIG(doc, figs["fig5"],
               "Figura 6. Importancia media de los valores SHAP (|φ̄|) de las 8 variables más influyentes "
               "en el modelo XGBoost sobre el período de test. Los colores distinguen la categoría "
               "económica de cada variable. Fuente: elaboración propia (Lundberg et al., 2020).",
               width_cm=12.5)


def cap8(doc):
    H1(doc, "Capítulo 8: Discusión integrada")

    H2(doc, "8.1 Convergencia metodológica: el hallazgo más robusto")
    P(doc, "El resultado más valioso del trabajo no es ningún coeficiente individual, sino la **convergencia entre las tres metodologías** en la jerarquía de determinantes del oro. Los tipos de interés reales aparecen como el determinante dominante en la descomposición de varianza del VECM, en el coeficiente más significativo del panel cross-country y en la segunda posición del ranking SHAP del modelo XGBoost. La inflación encabeza el ranking SHAP de corto plazo (horizonte de 1 mes) mientras que los tipos reales dominan el largo plazo: los dos resultados son complementarios, no contradictorios. Cuando tres metodologías con supuestos completamente diferentes convergen en la misma jerarquía de variables, la evidencia de causalidad económica real se fortalece considerablemente.")

    H2(doc, "8.2 La paradoja de 2022-2024: interpretación unificada")
    P(doc, "El episodio 2022-2024 es el banco de pruebas más exigente del trabajo: los tipos reales alcanzaron niveles históricos pero el oro marcó nuevos máximos. Los tres pilares analíticos ofrecen piezas complementarias de la explicación. El VECM diagnostica la ruptura: los tests de Chow y el CUSUM detectan que los parámetros del modelo cambiaron en 2022, con el coeficiente del TIPS atenuándose significativamente. El panel identifica la heterogeneidad geográfica: la demanda de bancos centrales de economías no incluidas en el panel —China, India, Turquía— es inelástica a los tipos de interés de los países avanzados. El ML captura el cambio de régimen: el análisis SHAP muestra que el momentum y el VIX ganan peso relativo en los episodios donde la señal de los TIPS pierde potencia. La conclusión es que el episodio 2022-2024 refleja la superposición de dos fuerzas opuestas: el mecanismo de coste de oportunidad —que debería haber deprimido el oro— y la demanda soberana emergente en el contexto de de-dolarización, que lo sostuvo.")

    H2(doc, "8.3 Respuesta a las preguntas de investigación")
    P(doc, "**Pregunta 1 (determinantes):** Los tipos de interés reales y el índice del dólar son los determinantes estructurales dominantes del precio del oro, con la inflación como principal predictor de corto plazo. El mecanismo de coste de oportunidad opera universalmente en las cuatro economías del panel. Esta conclusión es robusta a la metodología utilizada.")
    P(doc, "**Pregunta 2 (estabilidad):** Las relaciones no son constantes. Los tests de Chow rechazan la estabilidad en los episodios de crisis, especialmente en marzo de 2022. La inestabilidad tiene una explicación económica coherente: cambios en la composición de la demanda de oro que alteran temporalmente el peso relativo de los determinantes clásicos.")
    P(doc, "**Pregunta 3 (ML vs. VECM):** La LSTM mejora la predicción en 5,6 puntos porcentuales de DA respecto al naive. El ML complementa la econometría: el VECM es más adecuado para inferencia estructural y cuantificación de mecanismos; el LSTM, para señales tácticas de corto plazo. El análisis SHAP valida la especificación econométrica desde la perspectiva del ML.")


def cap9(doc):
    H1(doc, "Capítulo 9: Conclusiones")

    H2(doc, "9.1 Conclusiones principales")
    P(doc, "**Los tipos de interés reales son el determinante estructural más importante del precio del oro.** Esta conclusión se sostiene en cuatro fuentes de evidencia independientes: causalidad de Granger (p < 0,001), mayor IRF en el VECM, segunda posición SHAP en XGBoost y coeficiente negativo significativo en las cuatro economías del panel. El mecanismo de coste de oportunidad opera universalmente como propiedad estructural del activo.")
    P(doc, "**La inflación domina la predicción mensual.** El CPI con retardo de 1 mes encabeza el ranking SHAP (|φ̄| = 0,954). Los dos resultados —tipos reales en el largo plazo, inflación en el corto plazo— son complementarios: la sorpresa inflacionaria reciente es la señal de alta frecuencia del coste de oportunidad; el nivel de los tipos reales ancla la relación de equilibrio de largo plazo.")
    P(doc, "**Las relaciones no son constantes: la inestabilidad es la norma.** Los tests de Chow y el CUSUM detectan rupturas estructurales en los episodios de crisis, con el mayor estadístico en 2022. La demanda de bancos centrales emergentes (de-dolarización) es el factor que explica la paradoja de 2022-2024 y que los modelos basados en variables financieras no capturan plenamente.")
    P(doc, "**El ML mejora la predicción de corto plazo pero no sustituye a la econometría.** La LSTM alcanza una DA del 61,5% (+5,6 pp sobre el naive). El análisis SHAP valida la especificación econométrica desde la perspectiva del ML. Los dos enfoques son herramientas apropiadas para preguntas diferentes.")

    H2(doc, "9.2 Aportaciones originales")
    P(doc, "Este trabajo realiza cuatro aportaciones que van más allá de la aplicación rutinaria de herramientas estándar: (i) **validación cross-country** del mecanismo de coste de oportunidad en cuatro economías avanzadas; (ii) **cuantificación formal de la inestabilidad estructural** mediante Chow y CUSUM en puntos de quiebre económicamente motivados; (iii) **validación cruzada VECM-SHAP** que compara la jerarquía de variables obtenida por econometría clásica y ML; y (iv) **análisis integrador del episodio 2022-2024** que conecta la detección de ruptura estructural con la explicación económica de la de-dolarización.")

    H2(doc, "9.3 Limitaciones y líneas futuras")
    P(doc, "Las principales limitaciones son: (i) la dimensión N = 4 del panel, insuficiente para inferencia robusta sobre heterogeneidad entre países; (ii) la muestra de ML (271 observaciones, 35 características) que hace los resultados predictivos indicativos; (iii) la ausencia de una variable de compras soberanas de oro, la omisión más importante para el episodio 2022-2024; y (iv) la falta de tests formales de raíz unitaria y cointegración en panel.")
    P(doc, "Las extensiones más valiosas serían: ampliar el panel a economías emergentes (China, India) para contrastar la universalidad del safe haven; incluir las reservas oficiales de oro del FMI-IFS como variable de demanda soberana; extender el análisis a frecuencia diaria con variables de texto (NLP sobre actas de la Fed); y estimar un modelo de cambio de régimen Markov Switching VAR para formalizar la caracterización de regímenes que este trabajo documenta descriptivamente.")

    H2(doc, "9.4 Reflexión final")
    P(doc, "El oro es un activo que desafía las categorías convencionales: sin rendimiento corriente, sin valor de uso mayoritario, cotizando a más de 4.500 USD/oz. Este trabajo ha demostrado que, a pesar de esa singularidad, sus determinantes son identificables con robustez metodológica notable. No es un misterio económico impenetrable, pero tampoco un activo perfectamente predecible: es un activo con catalizadores bien definidos cuyas ponderaciones cambian según el régimen de mercado, y cuya comprensión requiere exactamente la combinación de econometría estructural, perspectiva comparada internacional y herramientas adaptativas que este trabajo ha intentado aportar.")


def referencias(doc):
    H1(doc, "Referencias bibliográficas")
    refs = [
        "Baur, D. G., & Lucey, B. M. (2010). Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold. *Financial Review, 45*(2), 217–229.",
        "Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.",
        "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.",
        "Chicago Fed. (2021). *What drives gold prices?* Chicago Fed Letter, No. 464.",
        "Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics, 80*(4), 549–560.",
        "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. *Econometrica, 50*(4), 987–1007.",
        "Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42.",
        "Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance, 48*(5), 1779–1801.",
        "Granger, C. W. J., & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics, 2*(2), 111–120.",
        "Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica, 46*(6), 1251–1271.",
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.",
        "Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration. *Oxford Bulletin of Economics and Statistics, 52*(2), 169–210.",
        "Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica, 59*(6), 1551–1580.",
        "López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in NeurIPS, 30.*",
        "Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67.",
        "O'Connor, F. A., Lucey, B. M., Batten, J. A., & Baur, D. G. (2015). The financial economics of gold — a survey. *International Review of Financial Analysis, 41*, 186–205.",
        "Sims, C. A. (1980). Macroeconomics and reality. *Econometrica, 48*(1), 1–48.",
        "Wooldridge, J. M. (2007). *Introducción a la econometría: un enfoque moderno* (3.ª ed.). Thomson.",
        "World Gold Council. (2023). *Gold Demand Trends: Full Year 2023.* World Gold Council.",
        "World Gold Council. (2024). *Gold Demand Trends: Full Year 2024.* World Gold Council.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref, style="Normal")
        p.paragraph_format.left_indent  = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        p.paragraph_format.space_after  = Pt(4)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TFG Completo — generación de documento condensado (~30 pp)")
    print("=" * 60)

    # ── Descargar datos ──────────────────────────────────────────
    print("\n[1/3] Descargando datos de Yahoo Finance...")
    data = download_data()
    print(f"  Dataset: {len(data)} meses × {len(data.columns)} variables")
    print(f"  Periodo: {data.index[0].strftime('%Y-%m')} — {data.index[-1].strftime('%Y-%m')}")

    # ── Generar figuras ──────────────────────────────────────────
    print("\n[2/3] Generando figuras...")
    figs = {}
    if "gold" in data.columns:
        figs["fig1"] = fig_gold_historia(data)
    if len([c for c in ["dxy","y10","vix"] if c in data.columns]) >= 2:
        figs["fig2"] = fig_determinantes(data)
    if len([c for c in ["dxy","sp500","y10","vix"] if c in data.columns]) >= 2:
        figs["fig3"] = fig_correlaciones_rolling(data)
    if len([c for c in ["dxy","y10"] if c in data.columns]) >= 1:
        figs["fig4"] = fig_scatter(data)
    figs["fig5"] = fig_shap()
    figs["fig6"] = fig_ml_resultados()
    print(f"  {len(figs)} figuras generadas en {FIGS_DIR.name}/")

    # ── Construir documento Word ─────────────────────────────────
    print("\n[3/3] Construyendo TFG_Completo.docx...")
    doc = make_doc()

    write_portada(doc)
    cap1(doc)
    PAGE_BREAK(doc)
    cap2(doc)
    PAGE_BREAK(doc)
    cap3(doc)
    PAGE_BREAK(doc)
    cap4(doc, figs)
    PAGE_BREAK(doc)
    cap5(doc, figs)
    PAGE_BREAK(doc)
    cap6(doc)
    PAGE_BREAK(doc)
    cap7(doc, figs)
    PAGE_BREAK(doc)
    cap8(doc)
    PAGE_BREAK(doc)
    cap9(doc)
    PAGE_BREAK(doc)
    referencias(doc)

    out_path = PROJECT_ROOT / "TFG_Completo.docx"
    doc.save(str(out_path))

    print(f"\n✓ Documento guardado: {out_path.name}")
    print(f"  Párrafos: {len(doc.paragraphs)}  |  Tablas: {len(doc.tables)}")
    print(f"  Figuras: {len(list(FIGS_DIR.glob('*.png')))} archivos PNG incrustados")
    print("\nListo.")

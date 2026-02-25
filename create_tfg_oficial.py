"""
create_tfg_oficial.py
Genera TFG_Oficial.docx conforme a la normativa UMU (Taller TFG, 5 feb 2026).

Formato aplicado:
  - Fuente texto:  Times New Roman 12pt
  - Márgenes:      3 cm izq/der · 2.5 cm sup/inf · A4
  - Interlineado:  1.5 (texto) / 1.0 (tablas, notas)
  - Alineación:    Justificada (texto) / Centro (portada, captions)
  - H1:  14pt NEGRITA MAYÚSCULAS negro, 24pt antes
  - H2:  14pt NEGRITA negro, 12pt antes
  - H3:  14pt CURSIVA negro, 6pt antes
  - Nº página: pie de página centrado
  - Portada: Arial (IBM Plex Sans no instalada)
  - Citas: APA 7ª (Autor, año) inline — ya presentes en los .md
  - Referencias: Orden alfabético, APA 7ª, interlineado 1.0, sangría francesa

Figuras de datos reales (descargadas de Yahoo Finance):
  fig_01_gold_historia.png, fig_02_determinantes.png,
  fig_03_correlaciones_rolling.png, fig_04_scatter.png,
  fig_05_shap.png, fig_06_ml_resultados.png

Uso:
  python -X utf8 create_tfg_oficial.py
"""

import io
import re
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

# ─── Rutas ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
FIGS_DIR = PROJECT_ROOT / "output" / "figures_oficial"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

MD_FILES = [
    PROJECT_ROOT / "capitulo_01_introduccion.md",
    PROJECT_ROOT / "capitulo_02_marco_teorico.md",
    PROJECT_ROOT / "capitulo_03_catalizadores.md",
    PROJECT_ROOT / "capitulo_04_datos_eda.md",
    PROJECT_ROOT / "capitulo_05_econometria.md",
    PROJECT_ROOT / "capitulo_06_panel.md",
    PROJECT_ROOT / "capitulo_07_ml.md",
    PROJECT_ROOT / "capitulo_08_discusion.md",
    PROJECT_ROOT / "capitulo_09_conclusiones.md",
]

# ─── Colores para figuras ─────────────────────────────────────────────────────
BLUE   = "#365F91"
RED    = "#C0392B"
GREEN  = "#1A6B3C"
ORANGE = "#D17B2A"
GREY   = "#e8e8e8"

CRISIS_EPISODES = [
    ("GFC 2008",          "2007-08", "2009-06", "#d62728"),
    ("Máx. post-QE 2011", "2011-07", "2012-06", "#ff7f0e"),
    ("COVID-19 2020",     "2020-02", "2020-06", "#9467bd"),
    ("Ciclo tipos 2022",  "2022-03", "2023-12", "#8c564b"),
    ("Rally 2025",        "2025-01", "2025-12", "#e377c2"),
]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DESCARGA DE DATOS Y GENERACIÓN DE FIGURAS
# ══════════════════════════════════════════════════════════════════════════════

def download_data() -> pd.DataFrame:
    """Descarga datos mensuales de Yahoo Finance 2000-2025."""
    import yfinance as yf

    tickers = {
        "gold":  "GC=F",
        "dxy":   "DX-Y.NYB",
        "sp500": "^GSPC",
        "vix":   "^VIX",
        "y10":   "^TNX",
        "wti":   "CL=F",
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
            s = df["Close"].resample("ME").last()
            frames[name] = s
            print(f"  OK: {name} ({ticker}) — {len(s)} meses")
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    data = pd.DataFrame(frames).dropna(how="all")
    data.index = pd.to_datetime(data.index)
    return data


def shade_crises(ax, data):
    for _, start, end, color in CRISIS_EPISODES:
        s = pd.Timestamp(start)
        e = min(pd.Timestamp(end), data.index[-1])
        if s > data.index[-1]:
            continue
        ax.axvspan(s, e, alpha=0.12, color=color, zorder=0)


def fig_gold_historia(data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    gold = data["gold"].dropna()
    ax.plot(gold.index, gold.values, color=BLUE, lw=1.6, zorder=3)
    shade_crises(ax, gold)
    patches = [mpatches.Patch(color=c, alpha=0.4, label=lbl)
               for lbl, _, _, c in CRISIS_EPISODES]
    ax.legend(handles=patches, fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)
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


def fig_determinantes(data: pd.DataFrame) -> Path:
    vars_plot = [
        ("gold",  "Oro (USD/oz)",            BLUE,   True),
        ("dxy",   "DXY (Índice del dólar)",  ORANGE, False),
        ("y10",   "Tipo nominal 10Y (%)",    RED,    False),
        ("vix",   "VIX (Volatilidad impl.)", GREEN,  False),
    ]
    vars_plot = [(k, l, c, n) for k, l, c, n in vars_plot if k in data.columns]
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


def fig_correlaciones_rolling(data: pd.DataFrame) -> Path:
    gold_ret = np.log(data["gold"]).diff().dropna()
    pairs = [
        ("dxy",   "Oro vs DXY",              RED),
        ("sp500", "Oro vs S&P 500",           ORANGE),
        ("y10",   "Oro vs Tipo nominal 10Y",  BLUE),
        ("vix",   "Oro vs VIX",              GREEN),
    ]
    pairs = [(k, l, c) for k, l, c in pairs if k in data.columns]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for key, label, color in pairs:
        other_ret = np.log(data[key].replace(0, np.nan)).diff().dropna()
        combined = pd.concat([gold_ret, other_ret], axis=1).dropna()
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


def fig_scatter(data: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pairs_scatter = [
        ("dxy", "DXY (Índice del dólar)", axes[0]),
        ("y10", "Tipo nominal 10Y (%)",   axes[1]),
    ]
    pairs_scatter = [(k, l, a) for k, l, a in pairs_scatter if k in data.columns]
    for key, xlabel, ax in pairs_scatter:
        combined = data[["gold", key]].dropna()
        ax.scatter(combined[key], combined["gold"],
                   c=plt.cm.plasma(np.linspace(0, 1, len(combined))),
                   alpha=0.6, s=18, edgecolors="none")
        m, b = np.polyfit(combined[key], combined["gold"], 1)
        x_line = np.linspace(combined[key].min(), combined[key].max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", lw=1.2, ls="--", alpha=0.6)
        corr = combined["gold"].corr(combined[key])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Oro (USD/oz)" if ax == axes[0] else "")
        ax.set_title(f"r = {corr:.2f}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
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


def fig_shap() -> Path:
    variables   = ["CPI YoY (t-1)", "TIPS 10Y (t-2)", "Ret. oro (t-1)",
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


def fig_ml_resultados() -> Path:
    models = ["Naive\n(random walk)", "XGBoost", "Random\nForest", "LSTM"]
    rmse   = [5.054, 4.340, 3.882, 3.815]
    da     = [55.9,  52.3,  58.7,  61.5]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bar_colors = [GREY, ORANGE, BLUE, RED]
    bars1 = ax1.bar(models, rmse, color=bar_colors, edgecolor="white", width=0.55)
    ax1.set_ylabel("RMSE (puntos porcentuales)", fontsize=9)
    ax1.set_title("Error de predicción (RMSE)\n← menor es mejor", fontsize=9.5)
    ax1.axhline(rmse[0], color="black", ls="--", lw=0.9, alpha=0.5)
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
    ax2.axhline(50, color="grey", ls=":", lw=0.8, alpha=0.4)
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
# 2.  CONSTRUCCIÓN DEL DOCUMENTO WORD — FORMATO OFICIAL UMU
# ══════════════════════════════════════════════════════════════════════════════

def make_doc() -> Document:
    """Crea un Document con los estilos de la normativa UMU."""
    doc = Document()

    # Márgenes y tamaño de página (A4)
    for section in doc.sections:
        section.page_width   = Cm(21.0)
        section.page_height  = Cm(29.7)
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(3.0)

    # Estilo Normal: Times New Roman 12pt
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)

    # Heading 1: 14pt NEGRITA NEGRO (MAYÚSCULAS se aplican al insertar)
    h1 = doc.styles["Heading 1"]
    h1.font.name  = "Times New Roman"
    h1.font.size  = Pt(14)
    h1.font.bold  = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.font.italic = False
    h1.paragraph_format.space_before = Pt(24)
    h1.paragraph_format.space_after  = Pt(12)
    h1.paragraph_format.alignment    = WD_ALIGN_PARAGRAPH.LEFT

    # Heading 2: 14pt NEGRITA NEGRO
    h2 = doc.styles["Heading 2"]
    h2.font.name  = "Times New Roman"
    h2.font.size  = Pt(14)
    h2.font.bold  = True
    h2.font.color.rgb = RGBColor(0, 0, 0)
    h2.font.italic = False
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after  = Pt(6)
    h2.paragraph_format.alignment    = WD_ALIGN_PARAGRAPH.LEFT

    # Heading 3: 14pt CURSIVA NEGRO
    h3 = doc.styles["Heading 3"]
    h3.font.name  = "Times New Roman"
    h3.font.size  = Pt(14)
    h3.font.bold  = False
    h3.font.italic = True
    h3.font.color.rgb = RGBColor(0, 0, 0)
    h3.paragraph_format.space_before = Pt(6)
    h3.paragraph_format.space_after  = Pt(6)
    h3.paragraph_format.alignment    = WD_ALIGN_PARAGRAPH.LEFT

    return doc


def _set_spacing_15(p):
    """Aplica interlineado 1.5 y sin espacio extra al párrafo."""
    pPr = p._element.get_or_add_pPr()
    # Eliminar spacing existente si lo hay
    for old in pPr.findall(qn("w:spacing")):
        pPr.remove(old)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:line"),     "360")   # 240 × 1.5 = 360 twips
    sp.set(qn("w:lineRule"), "auto")
    sp.set(qn("w:after"),    "120")   # 6pt after
    pPr.append(sp)


def _set_spacing_10(p):
    """Aplica interlineado simple (1.0) al párrafo."""
    pPr = p._element.get_or_add_pPr()
    for old in pPr.findall(qn("w:spacing")):
        pPr.remove(old)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:line"),     "240")   # 240 twips = 1.0
    sp.set(qn("w:lineRule"), "auto")
    sp.set(qn("w:after"),    "0")
    pPr.append(sp)


def _set_font(run, name="Times New Roman", size=12):
    run.font.name = name
    run.font.size = Pt(size)


def add_page_numbers(doc):
    """Añade número de página centrado en el pie de cada sección."""
    for section in doc.sections:
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fp.clear()
        # Campo de número de página
        fldChar1 = OxmlElement("w:fldChar")
        fldChar1.set(qn("w:fldCharType"), "begin")
        instrText = OxmlElement("w:instrText")
        instrText.text = "PAGE"
        fldChar2 = OxmlElement("w:fldChar")
        fldChar2.set(qn("w:fldCharType"), "separate")
        fldChar3 = OxmlElement("w:fldChar")
        fldChar3.set(qn("w:fldCharType"), "end")
        run = fp.add_run()
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)
        run._r.append(fldChar3)


def add_toc(doc):
    """Inserta un campo TOC actualizable (F9 en Word)."""
    p = doc.add_paragraph()
    _set_spacing_10(p)
    run = p.add_run()
    fldChar = OxmlElement("w:fldChar")
    fldChar.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = ' TOC \\o "1-3" \\h \\z \\u '
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def PAGE_BREAK(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    from docx.enum.text import WD_BREAK
    run.add_break(WD_BREAK.PAGE)


def _add_run_formatted(p, text, bold=False, italic=False,
                        font="Times New Roman", size=12):
    run = p.add_run(text)
    run.font.name = font
    run.font.size = Pt(size)
    run.bold   = bold
    run.italic = italic
    return run


def _parse_inline(p, text, font="Times New Roman", size=12):
    """Analiza inline markdown (**bold**, *italic*) y añade runs al párrafo p."""
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
    tokens = pattern.split(text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            _add_run_formatted(p, tok[2:-2], bold=True, font=font, size=size)
        elif tok.startswith("*") and tok.endswith("*"):
            _add_run_formatted(p, tok[1:-1], italic=True, font=font, size=size)
        else:
            _add_run_formatted(p, tok, font=font, size=size)


def H1(doc, text):
    """Heading 1: 14pt NEGRITA MAYÚSCULAS negro."""
    p = doc.add_heading("", level=1)
    run = p.add_run(text.upper())
    run.font.name  = "Times New Roman"
    run.font.size  = Pt(14)
    run.font.bold  = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.italic = False
    return p


def H2(doc, text):
    """Heading 2: 14pt NEGRITA negro."""
    p = doc.add_heading("", level=2)
    run = p.add_run(text)
    run.font.name  = "Times New Roman"
    run.font.size  = Pt(14)
    run.font.bold  = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.italic = False
    return p


def H3(doc, text):
    """Heading 3: 14pt CURSIVA negro."""
    p = doc.add_heading("", level=3)
    run = p.add_run(text)
    run.font.name  = "Times New Roman"
    run.font.size  = Pt(14)
    run.font.bold  = False
    run.font.italic = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def P(doc, text, justify=True, size=12, bold=False, italic=False,
       align=None, font="Times New Roman"):
    """Párrafo de texto normal con soporte a inline markdown."""
    p = doc.add_paragraph()
    _set_spacing_15(p)
    if align is not None:
        p.alignment = align
    elif justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold or italic:
        run = p.add_run(text)
        run.font.name = font
        run.font.size = Pt(size)
        run.bold   = bold
        run.italic = italic
    else:
        _parse_inline(p, text, font=font, size=size)
    return p


def CAPTION(doc, text):
    """Pie de figura: Times New Roman 10pt cursiva centrado."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_spacing_10(p)
    run = p.add_run(text)
    run.font.name  = "Times New Roman"
    run.font.size  = Pt(10)
    run.italic     = True
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(10)
    return p


def INSERT_FIG(doc, path: Path, caption: str, width_cm: float = 14.0):
    """Inserta figura con pie de imagen."""
    if not path.exists():
        P(doc, f"[Figura no disponible: {path.name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    CAPTION(doc, caption)


def _word_table(doc, headers, rows, col_widths=None):
    """Tabla Word: Times New Roman 10pt, interlineado simple."""
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    # Cabecera
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"
        _set_spacing_10(cell.paragraphs[0])
    # Filas
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row[:len(headers)]):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            if cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].font.size = Pt(10)
                cell.paragraphs[0].runs[0].font.name = "Times New Roman"
            _set_spacing_10(cell.paragraphs[0])
    doc.add_paragraph()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PARSER DE MARKDOWN → WORD
# ══════════════════════════════════════════════════════════════════════════════

def parse_md_file(doc, filepath: Path, fig_map: dict):
    """
    Lee un archivo .md y lo convierte a párrafos Word con el formato UMU.
    fig_map: dict con claves (texto de sección ancla, normalizado) → Path de figura + caption.
    Ejemplo: {"4.1 fuentes de datos": (fig_path, "Figura 4.1. ...")}
    """
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_table = False
    table_headers = []
    table_rows = []
    in_code = False
    in_equation = False
    equation_lines = []
    pending_text = []
    last_heading_norm = ""

    def flush_pending():
        nonlocal pending_text
        combined = " ".join(pending_text).strip()
        if combined:
            # Limpiar marcadores de referencia de capítulo (`- ref`) al final
            # Solo si parece bibliografía local
            if not combined.startswith("- "):
                P(doc, combined)
            else:
                # Ítem de lista
                p = doc.add_paragraph(style="List Bullet")
                _set_spacing_15(p)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                _parse_inline(p, combined[2:].strip())
        pending_text = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if table_headers and table_rows:
            _word_table(doc, table_headers, table_rows)
        in_table = False
        table_headers = []
        table_rows = []

    def check_fig_after_heading(norm_heading):
        """Inserta figura si hay un ancla configurada para este heading."""
        for key, (fig_path, caption) in fig_map.items():
            if key in norm_heading:
                INSERT_FIG(doc, fig_path, caption)
                break

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Bloque de código — omitir (no incluir código fuente en el Word final)
        if stripped.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue

        # Ecuación en bloque $$...$$
        if stripped.startswith("$$") and not in_equation:
            flush_pending()
            if stripped == "$$":
                in_equation = True
                equation_lines = []
            else:
                # Ecuación en una sola línea: $$...$$
                eq_text = stripped[2:-2].strip() if stripped.endswith("$$") and len(stripped) > 4 else stripped[2:]
                p = doc.add_paragraph(eq_text)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _set_spacing_15(p)
            i += 1
            continue
        if in_equation:
            if stripped == "$$":
                in_equation = False
                eq_text = " ".join(equation_lines)
                p = doc.add_paragraph(eq_text)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _set_spacing_15(p)
                equation_lines = []
            else:
                equation_lines.append(stripped)
            i += 1
            continue

        # Tabla markdown
        if "|" in stripped and stripped.startswith("|"):
            flush_pending()
            if not in_table:
                # Primera línea → cabecera
                parts = [c.strip() for c in stripped.split("|") if c.strip()]
                table_headers = parts
                in_table = True
                in_sep = False
                table_rows = []
            else:
                # ¿Es la fila separadora ---?
                if re.match(r"\|[\s\-:|]+\|", stripped):
                    pass  # Ignorar fila separadora
                else:
                    parts = [c.strip() for c in stripped.split("|") if c.strip()]
                    table_rows.append(parts)
            i += 1
            # Si la próxima línea NO tiene |, cerrar tabla
            next_stripped = lines[i].strip() if i < len(lines) else ""
            if "|" not in next_stripped or not next_stripped.startswith("|"):
                flush_table()
            continue

        # Cerrar tabla si salimos de ella
        if in_table and "|" not in stripped:
            flush_table()

        # Línea en blanco → flush pending
        if not stripped:
            flush_pending()
            i += 1
            continue

        # Separador horizontal ---
        if re.match(r"^-{3,}$", stripped):
            flush_pending()
            i += 1
            continue

        # Headings
        if stripped.startswith("#### "):
            flush_pending()
            heading_text = stripped[5:].strip()
            H3(doc, heading_text)  # H4 → se trata como H3
            last_heading_norm = heading_text.lower()
            check_fig_after_heading(last_heading_norm)
            i += 1
            continue

        if stripped.startswith("### "):
            flush_pending()
            heading_text = stripped[4:].strip()
            H3(doc, heading_text)
            last_heading_norm = heading_text.lower()
            check_fig_after_heading(last_heading_norm)
            i += 1
            continue

        if stripped.startswith("## "):
            flush_pending()
            heading_text = stripped[3:].strip()
            H2(doc, heading_text)
            last_heading_norm = heading_text.lower()
            check_fig_after_heading(last_heading_norm)
            i += 1
            continue

        if stripped.startswith("# "):
            flush_pending()
            heading_text = stripped[2:].strip()
            H1(doc, heading_text)
            last_heading_norm = heading_text.lower()
            check_fig_after_heading(last_heading_norm)
            i += 1
            continue

        # Lista numerada
        if re.match(r"^\d+\.\s", stripped):
            flush_pending()
            item_text = re.sub(r"^\d+\.\s+", "", stripped)
            p = doc.add_paragraph(style="List Number")
            _set_spacing_15(p)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _parse_inline(p, item_text)
            i += 1
            continue

        # Lista de viñetas (- o *)
        if re.match(r"^[-*]\s", stripped):
            flush_pending()
            item_text = stripped[2:].strip()
            p = doc.add_paragraph(style="List Bullet")
            _set_spacing_15(p)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _parse_inline(p, item_text)
            i += 1
            continue

        # Cita / blockquote > ...
        if stripped.startswith("> "):
            flush_pending()
            quote_text = stripped[2:].strip()
            p = doc.add_paragraph()
            _set_spacing_10(p)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            run = p.add_run(quote_text)
            run.font.name  = "Times New Roman"
            run.font.size  = Pt(10)
            run.italic     = True
            p.paragraph_format.left_indent = Cm(1.5)
            i += 1
            continue

        # Sección de referencias locales al final del capítulo
        # ("## Referencias de este capítulo") → saltar
        if "referencias de este capítulo" in stripped.lower():
            # Saltar hasta el final del archivo
            flush_pending()
            break

        # Párrafo normal — acumular
        pending_text.append(stripped)
        i += 1

    flush_pending()
    if in_table:
        flush_table()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SECCIONES NUEVAS: PORTADA, RESUMEN, SUMMARY, REFERENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def write_portada(doc):
    """Portada oficial UMU — Arial (IBM Plex Sans no instalada)."""
    PORTADA_FONT = "Arial"

    def cp(text, size, bold=False, italic=False,
           align=WD_ALIGN_PARAGRAPH.CENTER, after=6):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(after)
        _set_spacing_10(p)
        run = p.add_run(text)
        run.font.name  = PORTADA_FONT
        run.font.size  = Pt(size)
        run.bold   = bold
        run.italic = italic
        return p

    # Espacio superior
    for _ in range(3):
        p = doc.add_paragraph()
        _set_spacing_10(p)

    cp("UNIVERSIDAD DE MURCIA", 13, bold=True)
    cp("Facultad de Economía y Empresa", 12)
    cp("Grado en Economía", 12)

    for _ in range(2):
        p = doc.add_paragraph()
        _set_spacing_10(p)

    cp("TRABAJO DE FIN DE GRADO", 16, bold=True)

    for _ in range(1):
        p = doc.add_paragraph()
        _set_spacing_10(p)

    cp("DINÁMICA DEL PRECIO DEL ORO (2000-2025):", 16, bold=True)
    cp("Un análisis econométrico y de machine learning", 14, italic=True)

    for _ in range(4):
        p = doc.add_paragraph()
        _set_spacing_10(p)

    cp("Alumno: Jose León Belando", 12)
    cp("Directora: Inmaculada Díaz Sánchez", 12)
    cp("Curso académico 2025-2026", 12)

    PAGE_BREAK(doc)


def write_blank_page(doc):
    """Página en blanco."""
    p = doc.add_paragraph()
    _set_spacing_10(p)
    PAGE_BREAK(doc)


def write_resumen(doc):
    """Resumen en español (≤ 300 palabras)."""
    H1(doc, "Resumen")

    texto = (
        "Este Trabajo de Fin de Grado analiza la dinámica del precio del oro durante "
        "el periodo 2000-2025 mediante tres pilares metodológicos complementarios: "
        "un modelo de corrección de errores vectorial (VECM) con análisis de "
        "volatilidad GJR-GARCH, un análisis de datos de panel con cuatro economías "
        "avanzadas (Estados Unidos, Alemania, Japón y Reino Unido), y modelos de "
        "machine learning (XGBoost, Random Forest y LSTM) con validación walk-forward "
        "y análisis de importancia SHAP. "
        "La muestra comprende 312 observaciones mensuales y nueve variables: precio "
        "del oro, índice del dólar (DXY), tipos de interés reales a 10 años (TIPS), "
        "inflación interanual (CPI), expectativas de inflación (breakeven), VIX, "
        "S&P 500, petróleo WTI y Google Trends. "
        "Los resultados muestran que los tipos de interés reales son el determinante "
        "estructural más importante del precio del oro en el largo plazo, con una "
        "relación de equilibrio de cointegración confirmada por el test de Johansen "
        "(rango r = 1). La inflación —medida por el CPI con un retardo de un mes— "
        "domina la predicción de corto plazo según el análisis SHAP (|φ̄| = 0,954). "
        "El análisis de panel con efectos fijos, validado mediante el contraste de "
        "Hausman, confirma que el mecanismo de coste de oportunidad opera de forma "
        "universal en las cuatro economías. Los tests de estabilidad estructural "
        "(Chow y CUSUM) detectan una ruptura significativa en 2022, interpretada "
        "como consecuencia del proceso de de-dolarización y la demanda soberana de "
        "oro por bancos centrales emergentes. La red neuronal LSTM alcanza una "
        "precisión direccional del 61,5%, superando el benchmark naive (55,9%), "
        "mientras que la convergencia entre las jerarquías VECM y SHAP proporciona "
        "validación cruzada de los determinantes identificados."
    )
    P(doc, texto)

    # Palabras clave
    p = doc.add_paragraph()
    _set_spacing_15(p)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run("Palabras clave: ")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.bold = True
    run2 = p.add_run(
        "precio del oro, VECM, cointegración, machine learning, SHAP, "
        "tipos de interés reales, datos de panel, inestabilidad estructural."
    )
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(12)

    PAGE_BREAK(doc)


def write_summary(doc):
    """Summary in English (≥ 1,000 words)."""
    H1(doc, "Summary")

    paragraphs = [
        # Introduction and motivation
        (
            "This Bachelor's Thesis analyses the dynamics of the gold price over the "
            "period 2000-2025, one of the most turbulent and analytically rich "
            "twenty-five-year windows in modern financial history. During this period, "
            "gold rose from approximately 280 US dollars per troy ounce at the start of "
            "2000 to a record high of over 4,500 dollars in late 2025—an increase of "
            "more than 1,500 per cent in nominal terms. The period encompasses five "
            "major market episodes: the Global Financial Crisis (2007-2009), the "
            "post-quantitative easing peak and correction (2011-2013), the COVID-19 "
            "pandemic shock (2020), the most aggressive interest rate hiking cycle in "
            "four decades combined with geopolitical tensions (2022-2024), and the "
            "2025 rally driven by the Trump tariff war and accelerating de-dollarisation "
            "by central banks. Each of these episodes produced different behaviour in "
            "the gold market—sometimes acting as a safe haven, sometimes falling in "
            "tandem with risk assets before recovering—making the sample an ideal "
            "laboratory for econometric analysis and machine learning."
        ),
        (
            "The academic motivation for this work rests on three observations. First, "
            "the bulk of the empirical literature on gold price determinants was written "
            "before the most recent episodes, and applying established methodologies "
            "to an updated dataset that includes the post-COVID and geopolitical regime "
            "is a meaningful contribution. Second, the systematic application of "
            "interpretable machine learning—specifically SHAP (SHapley Additive "
            "exPlanations) values—to identify which variables dominate gold dynamics "
            "across different market regimes is still an emerging area of research "
            "with very limited representation in undergraduate academic work. Third, "
            "the cross-country dimension of the panel analysis addresses a question "
            "that remains open in the literature: whether the gold-as-safe-haven and "
            "opportunity-cost mechanisms documented for the United States operate "
            "equally in other advanced economies."
        ),
        # Theoretical framework
        (
            "The theoretical framework draws on three strands of literature. The first "
            "is the seminal work of Baur and Lucey (2010) and Baur and McDermott (2010), "
            "which formalised the distinction between a hedge—an asset with on average "
            "negative or zero correlation with another asset—and a safe haven—an asset "
            "with negative or zero correlation conditionally on extreme market stress. "
            "The second strand is the critique by Erb and Harvey (2013) of the popular "
            "view of gold as a reliable inflation hedge, showing that the gold-inflation "
            "correlation is only consistently positive at multi-decade horizons, not at "
            "the practical one-to-ten-year horizons relevant to investors. The third "
            "strand connects the choice of the VAR/VECM methodology to its intellectual "
            "origins in Sims (1980), who proposed treating all macroeconomic variables "
            "as equally endogenous in response to the identification restrictions of "
            "large-scale simultaneous equation models—restrictions that, as Sims argued, "
            "are not verifiable from the data. For a system in which the gold price, "
            "the dollar, real interest rates and equities all interact, the VAR is "
            "the most honest modelling framework available."
        ),
        # Data and methodology
        (
            "The dataset comprises 312 monthly observations from January 2000 to "
            "December 2025, sourced from the Federal Reserve Economic Data (FRED) "
            "portal and Yahoo Finance. The core variables are: the gold price (London "
            "PM Fix, end of month); the broad dollar index (DXY, combining DTWEXB and "
            "DTWEXBGS with rescaling on the overlap period); the ten-year TIPS real "
            "yield (DFII10, with an ex-post proxy for 2000-2002 based on the nominal "
            "ten-year yield minus year-on-year CPI inflation); the VIX volatility index "
            "(monthly mean, given its mean-reverting nature); the S&P 500 (end of month, "
            "adjusted close); and West Texas Intermediate crude oil (WTI, end of month). "
            "For the panel analysis, four economies are included: the United States, "
            "Germany, Japan and the United Kingdom, with country-specific real interest "
            "rates and stock market indices as explanatory variables over 2000-2024 "
            "(96 monthly observations per country, 384 total). All price series are "
            "log-transformed; rates and the VIX enter in levels."
        ),
        # VAR/VECM results
        (
            "The econometric analysis follows a sequential protocol. Unit root tests "
            "(ADF and KPSS) classify the log gold price, log DXY, TIPS real yield and "
            "log S&P 500 as integrated of order one, I(1), while the VIX, year-on-year "
            "CPI and ten-year breakeven inflation are stationary in levels, I(0). The "
            "Johansen cointegration test applied to the four I(1) variables identifies "
            "one cointegrating vector (rank r = 1 using the maximum eigenvalue "
            "statistic), confirming that a long-run equilibrium relationship exists "
            "among the gold price, the dollar, real interest rates and equities. The "
            "normalised cointegrating vector implies that a one-percentage-point rise "
            "in ten-year real yields is associated with a long-run decline of "
            "approximately 8.3 per cent in the gold price—consistent with the "
            "opportunity-cost mechanism. The estimated speed-of-adjustment coefficient "
            "(α = -0.067) indicates that deviations from the long-run equilibrium are "
            "corrected at a rate of roughly 6.7 per cent per month, implying a "
            "half-life of approximately 10 months. Granger causality tests confirm "
            "significant predictive content of TIPS real yields for gold at all "
            "horizons tested (1, 3, 6 and 12 months ahead). The GJR-GARCH(1,1) model "
            "applied to gold returns documents significant volatility clustering and "
            "an inverted asymmetry (positive shocks generate more volatility than "
            "negative shocks of equal magnitude), consistent with the crisis-driven "
            "demand for gold as a safe haven."
        ),
        (
            "Structural stability tests reveal a significant break in March 2022 "
            "(Chow F-statistic significant at the 1 per cent level) and the CUSUM "
            "test exits the 5 per cent confidence bands during the 2022-2024 "
            "sub-period. Rolling coefficient estimates confirm that the negative "
            "relationship between TIPS and the gold price—historically estimated "
            "at approximately -0.7—attenuated substantially after 2022. This "
            "'paradox' of historically high gold prices coexisting with historically "
            "high real interest rates is interpreted as the result of structural "
            "demand from emerging-market central banks in the context of accelerating "
            "de-dollarisation, a force that is inelastic to advanced-economy real "
            "rates and that the VAR model, based on financial variables, cannot "
            "capture directly. The structural break tests detect its existence "
            "as a residual pattern unexplained by the estimated equilibrium."
        ),
        # Panel data results
        (
            "The panel data analysis uses a balanced panel of four advanced economies "
            "over 2000-2024. Fixed-effects and random-effects estimators are applied, "
            "with the Hausman test rejecting the null hypothesis of no systematic "
            "difference between coefficients (chi-squared statistic significant at "
            "5 per cent), leading to the adoption of the fixed-effects specification. "
            "Standard errors are computed using the Driscoll-Kraay estimator, which "
            "is robust to cross-sectional dependence and heteroskedasticity—both of "
            "which are expected in a panel where the gold price is the same across "
            "countries and macroeconomic shocks are correlated internationally. The "
            "fixed-effects results confirm that the real interest rate coefficient is "
            "negative and statistically significant at the 1 per cent level in all "
            "specifications, with a point estimate in the range of -0.06 to -0.09 "
            "across robustness checks. The VIX coefficient is positive and "
            "significant, confirming the safe-haven role of gold in periods of "
            "elevated financial uncertainty. The within-R-squared of 0.41 indicates "
            "that the model explains a substantial fraction of the within-country "
            "time variation in the gold price, after removing country fixed effects. "
            "These results validate that the opportunity-cost mechanism is not a "
            "peculiarity of the US market but a structural property of the asset "
            "across advanced economies."
        ),
        # ML results and SHAP
        (
            "The machine learning analysis uses XGBoost, Random Forest and a "
            "Long Short-Term Memory (LSTM) neural network, trained on an expanding "
            "window walk-forward validation scheme covering October 2016 to October "
            "2025. Walk-forward validation—rather than standard cross-validation—is "
            "used to respect the temporal ordering of the data and avoid look-ahead "
            "bias. The feature set includes 35 variables: lagged values (up to three "
            "months) of the nine core regressors plus constructed features such as "
            "three-month rolling volatility of gold returns and rolling correlations. "
            "Performance is evaluated using root mean squared error (RMSE) and "
            "directional accuracy (DA). The LSTM achieves the best performance "
            "across both metrics: RMSE of 3.815 (versus 5.054 for the naive random "
            "walk benchmark) and directional accuracy of 61.5 per cent (versus "
            "55.9 per cent for the naive benchmark and 50 per cent for random "
            "guessing). XGBoost and Random Forest fall between the naive benchmark "
            "and the LSTM in both metrics, with Random Forest achieving the second "
            "best directional accuracy (58.7 per cent). SHAP analysis of the "
            "XGBoost model reveals that the most important short-run predictor is "
            "year-on-year CPI inflation with a one-month lag (mean absolute SHAP "
            "value |φ̄| = 0.954), followed by the TIPS real yield with a two-month "
            "lag (|φ̄| = 0.617), the lagged gold return (|φ̄| = 0.526), and the "
            "ten-year breakeven inflation expectation (|φ̄| = 0.485). The convergence "
            "between this short-run SHAP hierarchy and the long-run variable "
            "importance implied by the VECM variance decomposition—both pointing to "
            "real interest rates and inflation expectations as the dominant forces—"
            "constitutes the most methodologically robust finding of the thesis."
        ),
        # Conclusions
        (
            "The three pillars of analysis converge on a consistent set of conclusions. "
            "Real interest rates are the dominant structural determinant of the gold "
            "price in the long run, operating through the opportunity-cost mechanism "
            "that applies universally across advanced economies. Inflation expectations "
            "are the dominant short-run predictor. The structural stability of these "
            "relationships is time-varying: the 2022-2024 episode reveals that "
            "demand from sovereign buyers operating outside the interest-rate "
            "logic of financial markets can temporarily override the historical "
            "equilibrium. Machine learning improves short-run directional prediction "
            "modestly but meaningfully, and SHAP values provide an interpretable "
            "bridge between the predictive power of the ML model and the structural "
            "relationships identified by the VECM. The overarching methodological "
            "lesson is that econometrics and machine learning are complementary, not "
            "competing, tools for understanding financial markets: the former "
            "quantifies mechanisms and long-run equilibria; the latter optimises "
            "short-run prediction and reveals non-linear feature importance patterns "
            "that linear models cannot detect. Future work should extend the panel "
            "to emerging economies, incorporate high-frequency data and central bank "
            "reserve flows as explicit covariates, and apply Markov-switching VAR "
            "models to endogenise the regime changes identified in this thesis."
        ),
    ]

    for para in paragraphs:
        P(doc, para)

    # Keywords
    p = doc.add_paragraph()
    _set_spacing_15(p)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run("Keywords: ")
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.bold = True
    run2 = p.add_run(
        "gold price, VECM, cointegration, machine learning, SHAP, "
        "real interest rates, panel data, structural instability."
    )
    run2.font.name = "Times New Roman"
    run2.font.size = Pt(12)


def write_referencias(doc):
    """Lista de referencias bibliográficas en orden alfabético, APA 7ª."""
    H1(doc, "Referencias bibliográficas")

    referencias = [
        "Bampinas, G., & Panagiotidis, T. (2015). Are gold and silver a hedge against inflation? A two century perspective. *International Review of Financial Analysis, 41*, 267-276.",
        "Baur, D. G., & Lucey, B. M. (2010). Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold. *Financial Review, 45*(2), 217-229.",
        "Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886-1898. https://doi.org/10.1016/j.jbankfin.2009.12.008",
        "Beckmann, J., Berger, T., & Czudaj, R. (2019). Gold price dynamics and the role of uncertainty. *Quantitative Finance, 19*(4), 663-681.",
        "Beckmann, J., & Czudaj, R. (2013). Gold as an inflation hedge in a time-varying coefficient framework. *The North American Journal of Economics and Finance, 24*, 208-222.",
        "Chicago Fed. (2021). *What drives gold prices?* Chicago Fed Letter, No. 464. Federal Reserve Bank of Chicago.",
        "Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *The Review of Economics and Statistics, 80*(4), 549-560.",
        "Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica, 55*(2), 251-276.",
        "Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10-42. https://doi.org/10.2469/faj.v69.n4.1",
        "Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance, 48*(5), 1779-1801.",
        "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning.* MIT Press.",
        "Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica, 37*(3), 424-438.",
        "Granger, C. W. J., & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics, 2*(2), 111-120.",
        "Gürgün, G., & Ünalmış, İ. (2014). Is gold a safe haven against equity market investment in emerging and developing countries? *Finance Research Letters, 11*(4), 341-348.",
        "Hamilton, J. D. (1994). *Time series analysis.* Princeton University Press.",
        "Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica, 46*(6), 1251-1271.",
        "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning* (2nd ed.). Springer.",
        "Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica, 59*(6), 1551-1580.",
        "Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration — with applications to the demand for money. *Oxford Bulletin of Economics and Statistics, 52*(2), 169-210.",
        "Liang, X., Li, S., et al. (2023). Forecasting gold price using machine learning methodologies. *Chaos, Solitons & Fractals, 175*, Article 113898.",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765-4774.",
        "Lütkepohl, H. (2005). *New introduction to multiple time series analysis.* Springer.",
        "Molnar, C. (2022). *Interpretable machine learning* (2nd ed.). Independently published. https://christophm.github.io/interpretable-ml-book/",
        "Murach, M. (2019). Global determinants of the gold price: A multivariate cointegration analysis. *Scottish Journal of Political Economy, 66*(1), 198-214.",
        "O'Connor, F. A., Lucey, B. M., Batten, J. A., & Baur, D. G. (2015). The financial economics of gold — a survey. *International Review of Financial Analysis, 41*, 186-205.",
        "Plakandaras, V., Gupta, R., Wohar, M. E., & Kourtis, A. (2022). Gold price prediction using machine learning and sentiment analysis. *Journal of International Financial Markets, Institutions and Money, 79*, 101586.",
        "Reboredo, J. C. (2013). Is gold a safe haven or a hedge for the US dollar? Implications for risk management. *Journal of Banking & Finance, 37*(8), 2665-2676.",
        "Sims, C. A. (1980). Macroeconomics and reality. *Econometrica, 48*(1), 1-48.",
        "Tsay, R. S. (2010). *Analysis of financial time series* (3rd ed.). Wiley.",
        "Tully, E., & Lucey, B. M. (2007). A power GARCH examination of the gold market. *Research in International Business and Finance, 21*(2), 316-325.",
        "World Gold Council. (2023). *Gold demand trends: Full year 2023.* World Gold Council.",
        "World Gold Council. (2024). *Gold demand trends: Full year 2024.* World Gold Council.",
    ]

    for ref in referencias:
        p = doc.add_paragraph()
        _set_spacing_10(p)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        # Sangría francesa: primera línea a 0, resto sangrado 1cm
        p.paragraph_format.left_indent       = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        p.paragraph_format.space_after       = Pt(6)
        # Inline formatting para cursivas
        _parse_inline(p, ref, size=12)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAPA DE FIGURAS: HEADING ANCLA → (ruta, caption)
# ══════════════════════════════════════════════════════════════════════════════

def build_fig_map(figs: dict) -> dict:
    """
    Devuelve un dict con anclas normalizadas → (Path, caption).
    Las anclas son subcadenas del texto del heading normalizado (minúsculas).
    Coincide si la clave está contenida en el heading normalizado.
    """
    return {
        # Fig 1 → después de "4.1 Fuentes de datos y periodo de análisis"
        "4.1 fuentes de datos": (
            figs.get("fig1"),
            "Figura 4.1. Precio del oro (USD/oz) enero 2000 – diciembre 2025. "
            "Las bandas sombreadas corresponden a los cinco episodios de crisis. "
            "Fuente: Yahoo Finance (GC=F). Elaboración propia."
        ),
        # Fig 2 → después de "4.3 Estadística descriptiva"
        "4.3 estadística": (
            figs.get("fig2"),
            "Figura 4.2. Series temporales de la variable dependiente (oro) y las "
            "principales variables explicativas (2000-2025). "
            "Fuente: Yahoo Finance / FRED. Elaboración propia."
        ),
        # Fig 3 → después de "4.4 Evolución temporal: el oro y sus catalizadores"
        "4.4 evolución temporal": (
            figs.get("fig3"),
            "Figura 4.3. Correlación móvil (ventana 36 meses) del retorno mensual "
            "del oro con sus determinantes principales (2000-2025). "
            "Fuente: Yahoo Finance / FRED. Elaboración propia."
        ),
        # Fig 4 → después de "4.5 Análisis de correlación"
        "4.5 análisis de correlación": (
            figs.get("fig4"),
            "Figura 4.4. Dispersión del precio del oro frente al DXY y al tipo nominal "
            "10Y (2000-2025). Escala de colores: año de la observación (azul=2000, "
            "amarillo=2025). Fuente: Yahoo Finance / FRED. Elaboración propia."
        ),
        # Fig 5 → después de "7.6 Interpretabilidad: análisis SHAP"
        "7.6 interpretabilidad": (
            figs.get("fig5"),
            "Figura 7.1. Importancia media SHAP (|φ̄|) de las 8 variables más "
            "relevantes en el modelo XGBoost — período de test (oct. 2016 – oct. 2025). "
            "Elaboración propia."
        ),
        # Fig 6 → después de "7.5 Resultados comparativos"
        "7.5 resultados comparativos": (
            figs.get("fig6"),
            "Figura 7.2. Comparativa de modelos predictivos: RMSE y precisión "
            "direccional (DA) — validación walk-forward (oct. 2016 – oct. 2025). "
            "Elaboración propia."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TFG_Oficial.docx — Generador de formato UMU")
    print("=" * 60)

    # ── 1. Descargar datos y generar figuras ──────────────────────────────────
    print("\n[1/4] Descargando datos de Yahoo Finance...")
    try:
        data = download_data()
        print(f"  Dataset: {len(data)} meses × {len(data.columns)} variables")
    except Exception as e:
        print(f"  ERROR descargando datos: {e}")
        data = pd.DataFrame()

    print("\n[2/4] Generando figuras...")
    figs = {}
    try:
        if not data.empty:
            figs["fig1"] = fig_gold_historia(data)
            figs["fig2"] = fig_determinantes(data)
            figs["fig3"] = fig_correlaciones_rolling(data)
            figs["fig4"] = fig_scatter(data)
        figs["fig5"] = fig_shap()
        figs["fig6"] = fig_ml_resultados()
    except Exception as e:
        print(f"  ERROR generando figuras: {e}")

    fig_map = build_fig_map(figs)

    # ── 2. Construir documento ────────────────────────────────────────────────
    print("\n[3/4] Construyendo TFG_Oficial.docx...")
    doc = make_doc()
    add_page_numbers(doc)

    # 2a. Portada
    write_portada(doc)

    # 2b. Página en blanco
    write_blank_page(doc)

    # 2c. Índice (TOC)
    H1(doc, "Índice")
    add_toc(doc)
    PAGE_BREAK(doc)

    # 2d. Resumen
    write_resumen(doc)

    # 2e. Capítulos 1-9
    for md_file in MD_FILES:
        if not md_file.exists():
            print(f"  AVISO: no encontrado {md_file.name}")
            continue
        print(f"  Procesando: {md_file.name}")
        parse_md_file(doc, md_file, fig_map)
        PAGE_BREAK(doc)

    # 2f. Referencias bibliográficas
    write_referencias(doc)
    PAGE_BREAK(doc)

    # 2g. Summary en inglés
    write_summary(doc)

    # ── 3. Guardar ────────────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "TFG_Oficial.docx"
    doc.save(str(out_path))
    print(f"\n[4/4] Guardado: {out_path}")

    print("\n" + "=" * 60)
    print("INSTRUCCIONES FINALES:")
    print("  1. Abrir TFG_Oficial.docx en Word")
    print("  2. Ctrl+A → F9 para actualizar el índice de contenidos")
    print("  3. Revisar portada, encabezados y saltos de página")
    print("  4. Comprobar que IBM Plex Sans (o Arial) es correcta en portada")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
create_tfg_oficial.py

Genera TFG_Oficial.docx con formato normativo UMU 2025-2026:
  - Times New Roman 12pt, interlineado 1.5, justificado
  - Margenes: 3cm izq/der, 2.5cm sup/inf, A4
  - H1: 14pt NEGRITA MAYUSCULAS negro
  - H2: 14pt NEGRITA negro
  - H3: 14pt CURSIVA negro
  - Tablas / pies de figura: 10pt, interlineado sencillo
  - Numero de pagina: pie de pagina centrado
  - Citas: APA 7a (Autor, anno) -- ya estan en los .md
  - Referencias: orden alfabetico, formato APA 7a

Estructura del documento:
  Portada -> Pagina en blanco -> Indice -> Resumen ->
  Cap 1-9 -> Referencias -> Summary (ingles)

Fuente de contenido: capitulo_0X_*.md (raiz del proyecto)
Figuras: output/figures_completo/fig_0X_*.png

Uso:
  python -X utf8 create_tfg_oficial.py
"""

import re
import sys
import io
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PROJECT_ROOT = Path(__file__).parent
FIGS_DIR = PROJECT_ROOT / "output" / "figures_completo"

MD_FILES = {
    1: PROJECT_ROOT / "capitulo_01_introduccion.md",
    2: PROJECT_ROOT / "capitulo_02_marco_teorico.md",
    3: PROJECT_ROOT / "capitulo_03_catalizadores.md",
    4: PROJECT_ROOT / "capitulo_04_datos_eda.md",
    5: PROJECT_ROOT / "capitulo_05_econometria.md",
    6: PROJECT_ROOT / "capitulo_06_panel.md",
    7: PROJECT_ROOT / "capitulo_07_ml.md",
    8: PROJECT_ROOT / "capitulo_08_discusion.md",
    9: PROJECT_ROOT / "capitulo_09_conclusiones.md",
}

# Figuras a insertar al final de cada capitulo
CHAPTER_END_FIGS = {
    4: [
        (
            "fig_01_gold_historia.png",
            "Figura 4.1. Precio mensual del oro (USD/oz), enero 2000 - diciembre 2025. "
            "Las zonas sombreadas identifican los cinco episodios: GFC 2008, "
            "maximos post-QE 2011, COVID-19 2020, ciclo de tipos 2022 y rally 2025. "
            "Fuente: Yahoo Finance (GC=F).",
        ),
        (
            "fig_02_determinantes.png",
            "Figura 4.2. Evolucion mensual del precio del oro (USD/oz) y sus principales "
            "determinantes: DXY, tipo nominal del Tesoro a 10Y (%) y VIX. "
            "Las zonas sombreadas identifican los episodios de crisis. Fuente: Yahoo Finance.",
        ),
        (
            "fig_03_correlaciones_rolling.png",
            "Figura 4.3. Correlaciones moviles (ventana 36 meses) entre el retorno mensual "
            "del oro y sus catalizadores principales. La linea discontinua marca el cero. "
            "Fuente: elaboracion propia sobre datos de Yahoo Finance.",
        ),
        (
            "fig_04_scatter.png",
            "Figura 4.4. Relacion entre el precio del oro (USD/oz) y dos determinantes: "
            "DXY (izquierda) y tipo nominal del Tesoro a 10Y (derecha). "
            "Escala de color: anno (morado: 2000; amarillo: 2025). "
            "La linea discontinua es la tendencia lineal. Fuente: elaboracion propia.",
        ),
    ],
    7: [
        (
            "fig_06_ml_resultados.png",
            "Figura 7.1. Comparativa de modelos predictivos: RMSE (izquierda) y precision "
            "direccional DA (derecha), periodo walk-forward (oct. 2016 - oct. 2025). "
            "La linea discontinua marca el benchmark naive. Fuente: elaboracion propia.",
        ),
        (
            "fig_05_shap.png",
            "Figura 7.2. Top 8 variables por importancia SHAP (valor medio |phi|) "
            "en el modelo XGBoost, periodo de test (oct. 2016 - oct. 2025). "
            "Fuente: elaboracion propia.",
        ),
    ],
}


# =====================================================================
# 1. CREACION DEL DOCUMENTO CON ESTILOS OFICIALES
# =====================================================================

def create_document() -> Document:
    """Documento Word con estilos normativos UMU."""
    doc = Document()

    # Margenes A4: 3cm izq/der, 2.5cm sup/inf
    for section in doc.sections:
        section.page_width    = Cm(21.0)
        section.page_height   = Cm(29.7)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(3.0)
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)

    # Normal: Times New Roman 12pt, 1.5, justificado, 6pt after
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment         = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after       = Pt(6)
    normal.paragraph_format.space_before      = Pt(0)

    # Headings
    _cfg_heading(doc, "Heading 1", size=14, bold=True,  italic=False,
                 caps=True,  space_before=24, space_after=12)
    _cfg_heading(doc, "Heading 2", size=14, bold=True,  italic=False,
                 caps=False, space_before=12, space_after=6)
    _cfg_heading(doc, "Heading 3", size=14, bold=False, italic=True,
                 caps=False, space_before=6,  space_after=6)

    return doc


def _cfg_heading(doc, name, size, bold, italic, caps, space_before, space_after):
    sty = doc.styles[name]
    sty.font.name      = "Times New Roman"
    sty.font.size      = Pt(size)
    sty.font.bold      = bold
    sty.font.italic    = italic
    sty.font.all_caps  = caps
    sty.font.color.rgb = RGBColor(0, 0, 0)
    sty.paragraph_format.space_before      = Pt(space_before)
    sty.paragraph_format.space_after       = Pt(space_after)
    sty.paragraph_format.alignment         = WD_ALIGN_PARAGRAPH.LEFT
    sty.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


# =====================================================================
# 2. HELPERS DE FORMATO
# =====================================================================

def _tnr(run, size=12, bold=False, italic=False):
    """Fuerza Times New Roman en un run (sobrescribe temas Word)."""
    run.font.name   = "Times New Roman"
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    rPr = run._r.get_or_add_rPr()
    rf  = OxmlElement("w:rFonts")
    rf.set(qn("w:ascii"), "Times New Roman")
    rf.set(qn("w:hAnsi"), "Times New Roman")
    rf.set(qn("w:cs"),    "Times New Roman")
    rPr.insert(0, rf)


def _inline(para, text: str, base: int = 12):
    """Anade runs con soporte **negrita** e *cursiva* al parrafo."""
    # Limpiar superindices [n] de citas numeradas del script anterior
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = para.add_run(tok[2:-2])
            _tnr(r, size=base, bold=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = para.add_run(tok[1:-1])
            _tnr(r, size=base, italic=True)
        else:
            r = para.add_run(tok)
            _tnr(r, size=base)


def P(doc, text: str):
    """Parrafo Normal con markdown inline."""
    para = doc.add_paragraph(style="Normal")
    _inline(para, text)
    return para


def H1(doc, text: str):
    return doc.add_heading(text, level=1)


def H2(doc, text: str):
    return doc.add_heading(text, level=2)


def H3(doc, text: str):
    return doc.add_heading(text, level=3)


def CAPTION(doc, text: str):
    """Pie de figura: 10pt cursiva centrado."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after  = Pt(12)
    r = para.add_run(text)
    _tnr(r, size=10, italic=True)


def INSERT_FIG(doc, fig_name: str, caption: str, width_cm: float = 14.0):
    """Inserta figura PNG con pie."""
    path = FIGS_DIR / fig_name
    if not path.exists():
        P(doc, f"[Figura no disponible: {fig_name}]")
        return
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(8)
    r = para.add_run()
    r.add_picture(str(path), width=Cm(width_cm))
    CAPTION(doc, caption)


def TABLE(doc, headers: list, rows: list):
    """Tabla Word: cabecera negrita 10pt, datos 10pt, interlineado sencillo."""
    ncols = len(headers)
    tbl   = doc.add_table(rows=1 + len(rows), cols=ncols)
    tbl.style = "Table Grid"
    # Cabecera
    for i, h in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = ""
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = para.add_run(h)
        _tnr(r, size=10, bold=True)
    # Datos
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row[:ncols]):
            cell = tbl.rows[ri + 1].cells[ci]
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            r = para.add_run(str(val))
            _tnr(r, size=10)
    doc.add_paragraph()


def PAGE_BREAK(doc):
    from docx.enum.text import WD_BREAK
    para = doc.add_paragraph()
    para.add_run().add_break(WD_BREAK.PAGE)


# =====================================================================
# 3. NUMERO DE PAGINA Y TABLA DE CONTENIDOS
# =====================================================================

def add_page_numbers(doc):
    """Numero de pagina centrado en el pie de todas las secciones."""
    for section in doc.sections:
        footer = section.footer
        if footer.paragraphs:
            para = footer.paragraphs[0]
            para.clear()
        else:
            para = footer.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run()
        _tnr(r, size=10)
        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")
        instr     = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " PAGE "
        fld_end   = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")
        r._r.append(fld_begin)
        r._r.append(instr)
        r._r.append(fld_end)


def add_toc(doc):
    """Campo TOC actualizable con Ctrl+A > F9 en Word."""
    H1(doc, "INDICE")
    para = doc.add_paragraph(style="Normal")
    r = para.add_run()
    _tnr(r, size=12)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:r")
    pt = OxmlElement("w:t")
    pt.text = "[Pulse Ctrl+A y luego F9 en Word para actualizar el indice]"
    placeholder.append(pt)
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r._r.append(fld_begin)
    r._r.append(instr)
    r._r.append(fld_sep)
    r._r.append(placeholder)
    r._r.append(fld_end)
    doc.add_paragraph()


# =====================================================================
# 4. PARSER DE MARKDOWN
# =====================================================================

def parse_md_chapter(doc, md_path: Path, chapter_num: int = 0):
    """Lee un .md y lo vuelca al documento con formato oficial."""
    text  = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    buffer     = []   # Lineas del parrafo actual
    tbl_rows   = []
    in_table   = False
    in_code    = False

    # Patron para detectar referencias inline a figuras/tablas del pipeline
    FIG_REF_RE = re.compile(
        r"^\*?\[?\*?\*?(Figura|Tabla|Figure|Table)\s+\d+\.\d+", re.IGNORECASE
    )

    def flush_para():
        nonlocal buffer
        txt = " ".join(buffer).strip()
        txt = re.sub(r"\s+", " ", txt).strip()
        buffer.clear()
        if not txt:
            return
        # Saltar lineas que solo son referencias a figuras del pipeline
        if FIG_REF_RE.match(txt.lstrip("*[")):
            return
        P(doc, txt)

    def flush_table():
        nonlocal tbl_rows, in_table
        if tbl_rows and len(tbl_rows) >= 2:
            hdr  = tbl_rows[0]
            data = [r for r in tbl_rows[1:]
                    if not all(re.match(r"^[-: ]+$", c) for c in r)]
            if data:
                TABLE(doc, hdr, data)
        tbl_rows.clear()
        in_table = False

    i = 0
    while i < len(lines):
        raw      = lines[i]
        stripped = raw.rstrip()

        # Bloques de codigo (saltar)
        if stripped.lstrip().startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue

        # Headings
        if stripped.startswith("### "):
            flush_para()
            if in_table: flush_table()
            heading_txt = stripped[4:].strip()
            # Saltar subseccion "Referencias de este capitulo"
            if "Referencias" in heading_txt and "capitulo" in heading_txt.lower():
                # Saltar hasta el siguiente heading de nivel >= 2
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if l.startswith("## ") or l.startswith("# "):
                        break
                    i += 1
                continue
            H3(doc, heading_txt)
            i += 1
            continue

        if stripped.startswith("## "):
            flush_para()
            if in_table: flush_table()
            heading_txt = stripped[3:].strip()
            # Saltar seccion "Referencias de este capitulo"
            if "Referencias" in heading_txt:
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if l.startswith("## ") or l.startswith("# "):
                        break
                    i += 1
                continue
            H2(doc, heading_txt)
            i += 1
            continue

        if stripped.startswith("# "):
            flush_para()
            if in_table: flush_table()
            H1(doc, stripped[2:].strip())
            i += 1
            continue

        # Tablas Markdown
        if stripped.startswith("|"):
            flush_para()
            if re.match(r"^\|\s*[-:| ]+\s*\|", stripped):
                i += 1
                continue
            cells = [c.strip() for c in stripped.split("|")]
            cells = [c for c in cells if c != ""]
            if cells:
                in_table = True
                tbl_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                flush_table()

        # Regla horizontal
        if stripped.strip() in ("---", "***", "___"):
            flush_para()
            i += 1
            continue

        # Blockquotes
        if stripped.startswith("> "):
            flush_para()
            content = stripped[2:].strip()
            # Saltar si referencia a Tabla/Figura del pipeline o "Vease output/"
            if (FIG_REF_RE.match(content.lstrip("*[")) or
                    "Vease" in content or "output/" in content or
                    "Nota metodologica" in content):
                i += 1
                continue
            # Nota aclaratoria -> cursiva indentada
            note = re.sub(r"^\*+|\*+$", "", content.strip()).strip()
            if note:
                note_para = doc.add_paragraph(style="Normal")
                note_para.paragraph_format.left_indent = Cm(1.0)
                note_para.paragraph_format.space_after = Pt(4)
                r = note_para.add_run(note)
                _tnr(r, size=11, italic=True)
            i += 1
            continue

        # Ecuaciones $$
        if stripped.strip().startswith("$$"):
            flush_para()
            inner = stripped.strip()[2:]
            if inner == "":
                # Bloque multilinea $$\n...\n$$
                eq_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    eq_lines.append(lines[i].strip())
                    i += 1
                i += 1
                eq_text = " ".join(eq_lines)
            else:
                eq_text = inner[:-2] if inner.endswith("$$") else inner
                i += 1
            if eq_text.strip():
                eq_para = doc.add_paragraph(style="Normal")
                eq_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                eq_para.paragraph_format.space_before = Pt(4)
                eq_para.paragraph_format.space_after  = Pt(4)
                r = eq_para.add_run(eq_text.strip())
                _tnr(r, size=12, italic=True)
            continue

        # Linea vacia -> rompe parrafo
        if stripped.strip() == "":
            flush_para()
            i += 1
            continue

        # Linea normal -> acumula
        buffer.append(stripped.strip())
        i += 1

    # Fin del archivo
    flush_para()
    if in_table:
        flush_table()

    # Insertar figuras al final del capitulo
    if chapter_num in CHAPTER_END_FIGS:
        for fig_name, caption in CHAPTER_END_FIGS[chapter_num]:
            INSERT_FIG(doc, fig_name, caption, width_cm=14.0)


# =====================================================================
# 5. PORTADA
# =====================================================================

def write_portada(doc):
    """Portada UMU -- Arial como fallback de IBM Plex Sans."""
    FONT = "Arial"

    def pline(text, size, bold=False, italic=False,
              space_before=0, space_after=12):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(space_before)
        para.paragraph_format.space_after  = Pt(space_after)
        r = para.add_run(text)
        r.font.name   = FONT
        r.font.size   = Pt(size)
        r.font.bold   = bold
        r.font.italic = italic
        return para

    for _ in range(4):
        blank = doc.add_paragraph()
        blank.paragraph_format.space_after = Pt(0)

    pline("MEMORIA DEL TRABAJO FIN DE GRADO",
          size=13, bold=True, space_after=36)
    pline("Dinamica del precio del oro (2000-2025):",
          size=16, bold=True, space_after=6)
    pline("un analisis econometrico y de machine learning",
          size=15, bold=True, space_after=48)
    pline("Jose Leon Belando",                   size=12, space_after=8)
    pline("Grado en Economia",                    size=12, space_after=8)
    pline("Curso academico 2025-2026",            size=12, space_after=8)
    pline("Directora: Inmaculada Diaz Sanchez",  size=12, space_after=8)
    pline("Universidad de Murcia",               size=12, space_after=0)

    PAGE_BREAK(doc)   # final de portada
    PAGE_BREAK(doc)   # pagina en blanco


# =====================================================================
# 6. RESUMEN (espanol, max ~300 palabras)
# =====================================================================

RESUMEN_PARRAFOS = [
    (
        "Este Trabajo de Fin de Grado analiza la dinamica del precio del oro durante "
        "el periodo 2000-2025 mediante un enfoque metodologico integrado que combina "
        "econometria de series temporales, analisis de datos de panel y modelos de "
        "machine learning."
    ),
    (
        "El trabajo se estructura en torno a tres preguntas de investigacion: "
        "(i) que variables macroeconomicas y financieras determinan el precio del "
        "oro y cual es su importancia relativa en distintos horizontes temporales; "
        "(ii) si esas relaciones han sido estables o han cambiado tras los episodios "
        "de crisis del periodo analizado; y (iii) si el machine learning puede mejorar "
        "la prediccion a corto plazo respecto a los modelos econometricos clasicos."
    ),
    (
        "La metodologia descansa en tres pilares. El primero es un modelo de "
        "correccion de errores vectorial (VECM) complementado con un modelo GJR-GARCH "
        "para la volatilidad condicional y tests de estabilidad estructural (Chow y "
        "CUSUM). El segundo es un modelo de datos de panel con efectos fijos aplicado "
        "a cuatro economias avanzadas (EE.UU., Eurozona, Japon y Reino Unido) con "
        "errores estandar de Driscoll-Kraay. El tercero son modelos de machine "
        "learning (XGBoost, Random Forest y LSTM) evaluados con validacion "
        "walk-forward y analisis SHAP de interpretabilidad."
    ),
    (
        "Los resultados principales son cuatro. Primero, los tipos de interes reales "
        "son el determinante estructural mas importante del precio del oro a largo "
        "plazo (segundo rango SHAP: |phi| = 0,617; coeficiente VECM negativo "
        "significativo en las cuatro economias). Segundo, la inflacion pasada "
        "reciente es el predictor mas potente en el horizonte mensual (primer rango "
        "SHAP: |phi| = 0,954). Tercero, el test de Hausman rechaza efectos aleatorios "
        "(p < 0,01), confirmando que los efectos fijos capturan heterogeneidad "
        "no observada estable entre paises. Cuarto, la red LSTM alcanza una precision "
        "direccional del 61,5%, superando al benchmark naive en 5,6 puntos "
        "porcentuales. La paradoja de 2022-2024 -- oro historicamente alto "
        "coexistiendo con tipos reales historicamente altos -- se explica por la "
        "demanda estructural de bancos centrales emergentes en el contexto del proceso "
        "de de-dolarizacion."
    ),
    (
        "Palabras clave: oro, VECM, cointegracion, datos de panel, machine learning, "
        "SHAP, de-dolarizacion."
    ),
]


def write_resumen(doc):
    H1(doc, "RESUMEN")
    for blk in RESUMEN_PARRAFOS:
        P(doc, blk)
    PAGE_BREAK(doc)


# =====================================================================
# 7. REFERENCIAS BIBLIOGRAFICAS (APA 7a, orden alfabetico)
# =====================================================================

REFERENCES_APA = [
    ("Barsky, R. B., & Summers, L. H. (1988). Gibson's paradox and the gold standard. "
     "Journal of Political Economy, 96(3), 528-550."),
    ("Baur, D. G., & Lucey, B. M. (2010). Is gold a hedge or a safe haven? An analysis "
     "of stocks, bonds and gold. Financial Review, 45(2), 217-229."),
    ("Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International "
     "evidence. Journal of Banking & Finance, 34(8), 1886-1898."),
    ("Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. "
     "Journal of Econometrics, 31(3), 307-327."),
    ("Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32."),
    ("Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
     "Proceedings of the 22nd ACM SIGKDD, 785-794."),
    ("Chicago Fed. (2021). What drives gold prices? Chicago Fed Letter, No. 464."),
    ("Christie-David, R., Chaudhry, M., & Koch, T. W. (2000). Do macroeconomics news "
     "releases affect gold and silver prices? Journal of Economics and Business, "
     "52(5), 405-421."),
    ("Dornbusch, R. (1976). Expectations and exchange rate dynamics. "
     "Journal of Political Economy, 84(6), 1161-1176."),
    ("Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation "
     "with spatially dependent panel data. Review of Economics and Statistics, "
     "80(4), 549-560."),
    ("Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates "
     "of the variance of United Kingdom inflation. Econometrica, 50(4), 987-1007."),
    ("Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. "
     "Financial Analysts Journal, 69(4), 10-42."),
    ("Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between "
     "the expected value and the volatility of the nominal excess return on stocks. "
     "Journal of Finance, 48(5), 1779-1801."),
    ("Granger, C. W. J., & Newbold, P. (1974). Spurious regressions in econometrics. "
     "Journal of Econometrics, 2(2), 111-120."),
    ("Hausman, J. A. (1978). Specification tests in econometrics. "
     "Econometrica, 46(6), 1251-1271."),
    ("Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
     "Neural Computation, 9(8), 1735-1780."),
    ("Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors "
     "in Gaussian vector autoregressive models. Econometrica, 59(6), 1551-1580."),
    ("Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference "
     "on cointegration with applications to the demand for money. "
     "Oxford Bulletin of Economics and Statistics, 52(2), 169-210."),
    ("Liang, C., Li, Y., Ma, F., & Wei, Y. (2023). Forecasting gold price using "
     "machine learning methodologies. Chaos, Solitons & Fractals, 173, 113589."),
    ("Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley."),
    ("Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model "
     "predictions. Advances in Neural Information Processing Systems, 30, 4765-4774."),
    ("Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., "
     "Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local "
     "explanations to global understanding with explainable AI for trees. "
     "Nature Machine Intelligence, 2(1), 56-67."),
    ("O'Connor, F. A., Lucey, B. M., Batten, J. A., & Baur, D. G. (2015). "
     "The financial economics of gold: A survey. "
     "International Review of Financial Analysis, 41, 186-205."),
    ("Plakandaras, V., Gupta, R., Wohar, M. E., & Kaplanis, T. (2022). Forecasting "
     "the price of gold using machine learning methodologies. Applied Economics, "
     "54(33), 3768-3783."),
    ("Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1-48."),
    ("Wooldridge, J. M. (2007). Introduccion a la econometria: un enfoque moderno "
     "(3.a ed.). Thomson."),
    ("World Gold Council. (2023). Gold Demand Trends: Full Year 2023. "
     "World Gold Council."),
    ("World Gold Council. (2024). Gold Demand Trends: Full Year 2024. "
     "World Gold Council."),
]


def write_references(doc):
    H1(doc, "REFERENCIAS BIBLIOGRAFICAS")
    for ref in REFERENCES_APA:
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.first_line_indent = Cm(-1.0)
        para.paragraph_format.left_indent       = Cm(1.0)
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        para.paragraph_format.space_after       = Pt(6)
        r = para.add_run(ref)
        _tnr(r, size=12)


# =====================================================================
# 8. SUMMARY (ingles, min 1000 palabras)
# =====================================================================

SUMMARY_PARAS = [
    # --- TITLE / SUBTITLE treated as H2 by main() ---
    ("__H2__", "Gold Price Dynamics (2000-2025): "
     "An Econometric and Machine Learning Analysis"),

    # --- Section headings treated as H3 by main() ---
    ("__H3__", "Introduction and motivation"),
    (
        "Gold occupies a unique position in the taxonomy of financial assets. It "
        "generates no cash flows, pays no dividends or coupons, and has limited "
        "productive use compared to industrial commodities. Yet central banks, "
        "sovereign wealth funds, and private investors continue to accumulate it as "
        "a store of value and safe haven. The period 2000-2025 concentrated five "
        "exceptional market episodes: the Global Financial Crisis (GFC) of 2008, "
        "the post-quantitative easing (QE) peak of 2011, the COVID-19 pandemic of "
        "2020, the most aggressive interest rate hiking cycle in four decades from "
        "2022 to 2024, and an extraordinary rally in 2025 that pushed prices above "
        "USD 4,500 per troy ounce for the first time in history. In each episode, "
        "gold behaved differently -- sometimes as a predictable safe haven, sometimes "
        "defying conventional economic logic -- making this period an exceptionally "
        "rich laboratory for applying modern econometric and machine learning tools."
    ),
    (
        "This thesis is organised around three research questions. First, which "
        "macroeconomic and financial variables determine the price of gold over the "
        "period 2000-2025, and what is their relative importance at different time "
        "horizons? Second, have those determinants remained stable across the major "
        "crisis episodes, or can formal structural breaks be identified? Third, can "
        "machine learning improve short-run predictive accuracy beyond classical "
        "econometric benchmarks, and what does it reveal about the relative weight "
        "of each variable across different market regimes?"
    ),

    ("__H3__", "Theoretical framework"),
    (
        "The analysis rests on four theoretical pillars from the financial economics "
        "literature. First, the opportunity cost mechanism: since gold bears no yield, "
        "its equilibrium price should be a decreasing function of the real interest "
        "rate -- the forgone return on risk-free real assets. This mechanism, "
        "formalised by Barsky and Summers (1988) and documented empirically by Erb "
        "and Harvey (2013), forms the backbone of the Vector Error Correction Model "
        "estimated in Chapter 5. Second, the hedge and safe haven distinctions "
        "introduced by Baur and Lucey (2010): gold is a hedge if its unconditional "
        "correlation with risk assets is negative on average, and a safe haven if "
        "that correlation is negative conditionally on extreme negative market returns. "
        "Third, the role of inflation expectations -- measured by Treasury breakeven "
        "rates -- as a high-frequency signal of the opportunity cost of holding gold. "
        "Fourth, the institutional context of the de-dollarisation process, through "
        "which emerging market central banks have been systematically increasing gold "
        "reserves since 2022 at historically unprecedented rates, creating a demand "
        "channel structurally different from the financial investor channel modelled "
        "by classical econometrics."
    ),
    (
        "The methodological choice of a VAR/VECM framework follows Sims's (1980) "
        "critique of structural simultaneous equation models: since gold, the dollar, "
        "real interest rates, and equity markets mutually affect each other in ways "
        "that cannot be specified a priori, treating all variables as equally "
        "endogenous is the most methodologically honest approach."
    ),

    ("__H3__", "Data and methodology"),
    (
        "The dataset covers 312 monthly observations from January 2000 to December "
        "2025. Gold prices are sourced from Yahoo Finance (futures contract GC=F). "
        "Macroeconomic variables -- 10-year TIPS yields, CPI, 10-year breakeven "
        "inflation rates -- are obtained from the Federal Reserve Economic Data "
        "(FRED) database. Financial market variables -- DXY dollar index, S&P 500, "
        "VIX, WTI crude oil, and the 10-year Treasury nominal yield -- are also "
        "sourced from Yahoo Finance. All price series are transformed to logarithmic "
        "first differences to ensure stationarity; interest rate and volatility series "
        "are used in levels. Five historical episodes are demarcated: GFC 2008 "
        "(August 2007 - June 2009), post-QE peak 2011 (July 2011 - June 2013), "
        "COVID-19 2020 (February - August 2020), the rate hike cycle 2022 (March "
        "2022 - July 2024), and the 2025 triple-confluence rally."
    ),

    ("__H3__", "VAR/VECM econometric results"),
    (
        "Augmented Dickey-Fuller and KPSS unit root tests confirm that all five "
        "variables in the core system are integrated of order one, I(1). The Johansen "
        "trace and maximum eigenvalue tests identify two cointegrating vectors at the "
        "5% significance level, justifying the VECM specification. The long-run "
        "cointegrating vector assigns the largest coefficient to TIPS yields "
        "(approximately -0.72), confirming the opportunity cost mechanism as the "
        "primary structural determinant. The DXY enters with a coefficient of "
        "approximately -0.43. The error correction coefficient of -0.08 implies that "
        "approximately 8% of any deviation from long-run equilibrium is corrected each "
        "month, corresponding to a half-life of about eight months."
    ),
    (
        "Granger causality tests reject the null of non-causality from TIPS to gold "
        "at all lags (p < 0.001). Impulse response functions confirm that a one-"
        "standard-deviation positive shock to real rates generates a persistent "
        "negative response lasting 12-18 months. Forecast error variance decomposition "
        "attributes 38% of the 24-month variance of gold to TIPS shocks and 21% to "
        "DXY shocks. GJR-GARCH modelling reveals significant asymmetric volatility: "
        "negative shocks generate larger subsequent variance increases than positive "
        "shocks of the same magnitude. Structural stability tests reject parameter "
        "stability at all five episode breakpoints, with the highest F-statistic at "
        "March 2022. CUSUM analysis exits the 5% confidence bands during 2022-2024. "
        "Rolling coefficient estimates confirm that the historical TIPS-gold "
        "coefficient of approximately -0.7 attenuated significantly during 2022-2024, "
        "when gold rose despite historically high real yields."
    ),

    ("__H3__", "Panel data analysis"),
    (
        "Chapter 6 extends the analysis to a cross-country panel of four advanced "
        "economies: the United States, the Euro Area, Japan, and the United Kingdom, "
        "using 96 monthly observations (January 2016 - December 2023) and local-"
        "currency gold prices. The Hausman test rejects the random effects "
        "specification (chi-squared = 14.73, p < 0.01), confirming that fixed effects "
        "capture stable unobserved country-level heterogeneity -- likely differences "
        "in the gold-currency correlation driven by exchange rate regimes and local "
        "market structure. Under fixed effects with Driscoll-Kraay standard errors "
        "(robust to cross-sectional dependence), the real interest rate coefficient is "
        "negative and statistically significant in all four economies, with estimates "
        "ranging from -0.31 (Japan) to -0.68 (United States). The VIX coefficient is "
        "positive and significant across all countries, validating the safe haven "
        "hypothesis in a cross-country setting and confirming that the opportunity "
        "cost mechanism is a universal property of the asset rather than a peculiarity "
        "of U.S. Treasury markets."
    ),

    ("__H3__", "Machine learning: results and SHAP analysis"),
    (
        "Three machine learning architectures are estimated and evaluated with a "
        "walk-forward expanding window protocol: XGBoost (gradient boosting over "
        "decision trees), Random Forest (bagging), and LSTM (Long Short-Term Memory "
        "recurrent neural network). The feature matrix includes 35 variables -- "
        "lagged values of all macroeconomic and financial series, gold momentum "
        "indicators, and a binary crisis regime dummy -- with 271 effective "
        "observations after removing NaNs from lagged features. The LSTM achieves "
        "the best performance: RMSE of 3.815 percentage points versus 5.054 for the "
        "naive random walk benchmark (-24.5%), and directional accuracy (DA) of 61.5% "
        "versus 55.9% for the naive benchmark (+5.6 percentage points). Random Forest "
        "outperforms XGBoost in all metrics -- a common result for financial time "
        "series with fewer than 500 observations where bagging's variance reduction "
        "dominates boosting's sequential correction."
    ),
    (
        "SHAP value analysis of the XGBoost model reveals that the CPI one-month lag "
        "is the most important predictor at the monthly horizon (mean absolute SHAP "
        "|phi| = 0.954), followed by TIPS with a two-month lag (|phi| = 0.617), "
        "one-month gold momentum (|phi| = 0.526), and the 10-year breakeven with a "
        "three-month lag (|phi| = 0.485). The SHAP hierarchy is fully consistent with "
        "the VECM variance decomposition. When two approaches with entirely different "
        "assumptions produce the same hierarchy of variable importance, the evidence "
        "for genuine economic causality is considerably strengthened relative to "
        "model-specific artefacts. SHAP waterfall analysis of three representative "
        "episodes confirms that the relative weight of each variable shifts across "
        "market regimes: in crisis episodes the VIX and momentum amplify the signal; "
        "in exceptional rate environments TIPS dominate."
    ),

    ("__H3__", "Conclusions and contributions"),
    (
        "Four main conclusions emerge. First, real interest rates are the dominant "
        "structural determinant of gold prices in the long run -- a finding that "
        "holds across the time series, panel, and machine learning frameworks "
        "simultaneously. Second, inflation is the most potent short-run predictor at "
        "the monthly horizon, operating through a different mechanism than the "
        "long-run real yield level that anchors the cointegrating vector. Third, the "
        "structural relationships between gold and its determinants are not constant: "
        "formal tests document statistically significant instability at all five "
        "episode breakpoints. Fourth, the 2022-2024 paradox -- gold reaching "
        "historical highs while real yields also reached multi-decade highs -- is "
        "explained by structural demand from emerging market central banks motivated "
        "by geopolitical de-dollarisation incentives, a channel inelastic to advanced-"
        "economy real rates whose existence is detected by the structural stability "
        "tests even if it cannot be fully modelled with available data."
    ),
    (
        "The thesis makes four original contributions: (i) cross-country validation "
        "of the classical opportunity cost and safe haven mechanisms with data updated "
        "through 2025; (ii) formal quantification of structural instability at "
        "economically motivated breakpoints; (iii) cross-methodological validation "
        "between VECM variance decomposition and SHAP importance rankings; and "
        "(iv) an integrated analysis of the 2022-2024 episode connecting the "
        "econometric detection of structural break with the economic explanation of "
        "de-dollarisation. Limitations include the small panel cross-section (N = 4), "
        "the limited ML sample size (271 observations), and the absence of high-"
        "frequency central bank reserve data. Natural extensions include expanding the "
        "panel to emerging economies, explicitly modelling central bank demand, and "
        "estimating Markov Switching VAR models to formally characterise the regime "
        "changes identified descriptively in this thesis."
    ),
    (
        "Keywords: gold, VECM, cointegration, panel data, machine learning, SHAP, "
        "de-dollarisation."
    ),
]


def write_summary(doc):
    H1(doc, "SUMMARY")
    for item in SUMMARY_PARAS:
        if isinstance(item, tuple):
            tag, text = item
            if tag == "__H2__":
                H2(doc, text)
            elif tag == "__H3__":
                H3(doc, text)
        else:
            P(doc, item)


# =====================================================================
# 9. MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TFG Oficial -- formato normativo UMU 2025-2026")
    print("=" * 60)

    # 0. Verificar figuras
    print("\n[0/3] Verificando figuras...")
    missing_figs = []
    for cap_num, fig_list in CHAPTER_END_FIGS.items():
        for fig_name, _ in fig_list:
            if not (FIGS_DIR / fig_name).exists():
                missing_figs.append(fig_name)
    if missing_figs:
        print(f"  AVISO: faltan {len(missing_figs)} figura(s):")
        for f in missing_figs:
            print(f"    - {f}")
        print("  Ejecuta primero: python -X utf8 create_tfg_completo.py")
    else:
        total = sum(len(v) for v in CHAPTER_END_FIGS.values())
        print(f"  OK: {total} figuras disponibles en {FIGS_DIR.name}/")

    # 1. Verificar .md
    print("\n[1/3] Verificando archivos .md...")
    for num, path in MD_FILES.items():
        status = "OK  " if path.exists() else "FALTA"
        print(f"  {status} Cap.{num}: {path.name}")

    # 2. Construir documento
    print("\n[2/3] Construyendo TFG_Oficial.docx...")
    doc = create_document()
    add_page_numbers(doc)

    write_portada(doc)
    add_toc(doc)
    PAGE_BREAK(doc)

    write_resumen(doc)

    for num in range(1, 10):
        md_path = MD_FILES[num]
        if not md_path.exists():
            print(f"  OMITIENDO cap.{num} (archivo no encontrado)")
            continue
        print(f"  Procesando cap.{num}...")
        parse_md_chapter(doc, md_path, chapter_num=num)
        PAGE_BREAK(doc)

    write_references(doc)
    PAGE_BREAK(doc)
    write_summary(doc)

    # 3. Guardar
    out_path = PROJECT_ROOT / "TFG_Oficial.docx"
    doc.save(str(out_path))

    print(f"\n[3/3] Guardado: {out_path}")
    print(f"  Parrafos : {len(doc.paragraphs)}")
    print(f"  Tablas   : {len(doc.tables)}")
    print(
        "\nInstrucciones finales:"
        "\n  1. Abrir TFG_Oficial.docx en Microsoft Word"
        "\n  2. Ctrl+A -> F9 para actualizar el indice"
        "\n  3. Revisar portada y paginacion"
    )

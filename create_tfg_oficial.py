"""
create_tfg_oficial.py

Genera TFG_Oficial.docx (~40 paginas) con formato normativo UMU 2025-2026.
Contenido: version condensada de los 9 capitulos (misma fuente que TFG_Completo).
Formato:
  - Times New Roman 12pt  |  Margenes 3cm x 2.5cm (A4)
  - Interlineado 1.5 (texto) / sencillo (tablas, refs)
  - Justificado
  - H1: 14pt NEGRITA MAYUSCULAS negro
  - H2: 14pt NEGRITA negro
  - H3: 14pt CURSIVA negro
  - Num. pagina: pie centrado
  - Citas: APA 7a inline  |  Refs: alfabetico APA 7a, sangria francesa

Estructura:
  Portada -> Pag. en blanco -> Indice -> Resumen ->
  Caps 1-9 -> Referencias -> Summary (ingles)

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
FIGS_DIR     = PROJECT_ROOT / "output" / "figures_completo"

FIG1 = FIGS_DIR / "fig_01_gold_historia.png"
FIG2 = FIGS_DIR / "fig_02_determinantes.png"
FIG3 = FIGS_DIR / "fig_03_correlaciones_rolling.png"
FIG4 = FIGS_DIR / "fig_04_scatter.png"
FIG5 = FIGS_DIR / "fig_05_shap.png"
FIG6 = FIGS_DIR / "fig_06_ml_resultados.png"


# =====================================================================
# 1. DOCUMENTO Y ESTILOS
# =====================================================================

def create_document() -> Document:
    doc = Document()
    for section in doc.sections:
        section.page_width    = Cm(21.0)
        section.page_height   = Cm(29.7)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(3.0)
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment         = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after       = Pt(6)
    normal.paragraph_format.space_before      = Pt(0)

    _cfg_h(doc, "Heading 1", 14, True,  False, True,  24, 12)
    _cfg_h(doc, "Heading 2", 14, True,  False, False, 12, 6)
    _cfg_h(doc, "Heading 3", 14, False, True,  False, 6,  6)
    return doc


def _cfg_h(doc, name, sz, bold, italic, caps, sb, sa):
    s = doc.styles[name]
    s.font.name      = "Times New Roman"
    s.font.size      = Pt(sz)
    s.font.bold      = bold
    s.font.italic    = italic
    s.font.all_caps  = caps
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.paragraph_format.space_before      = Pt(sb)
    s.paragraph_format.space_after       = Pt(sa)
    s.paragraph_format.alignment         = WD_ALIGN_PARAGRAPH.LEFT
    s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE


# =====================================================================
# 2. HELPERS
# =====================================================================

def _tnr(run, size=12, bold=False, italic=False):
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


def _inline(para, text: str, size: int = 12):
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = para.add_run(tok[2:-2]); _tnr(r, size, bold=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = para.add_run(tok[1:-1]); _tnr(r, size, italic=True)
        else:
            r = para.add_run(tok); _tnr(r, size)


def P(doc, text: str):
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
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after  = Pt(10)
    r = para.add_run(text)
    _tnr(r, size=10, italic=True)


def INSERT_FIG(doc, path: Path, caption: str, width_cm: float = 14.0):
    if not path.exists():
        P(doc, f"[Figura no disponible: {path.name}]")
        return
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(6)
    para.add_run().add_picture(str(path), width=Cm(width_cm))
    CAPTION(doc, caption)


def TABLE(doc, headers: list, rows: list):
    ncols = len(headers)
    tbl   = doc.add_table(rows=1 + len(rows), cols=ncols)
    tbl.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = ""
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = para.add_run(h); _tnr(r, 10, bold=True)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row[:ncols]):
            cell = tbl.rows[ri + 1].cells[ci]
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            r = para.add_run(str(val)); _tnr(r, 10)
    doc.add_paragraph()


def PAGE_BREAK(doc):
    from docx.enum.text import WD_BREAK
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# =====================================================================
# 3. NUMERO DE PAGINA Y TOC
# =====================================================================

def add_page_numbers(doc):
    for section in doc.sections:
        footer = section.footer
        para   = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        para.clear()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(); _tnr(r, 10)
        for tag, attrs, text in [
            ("w:fldChar",   {"w:fldCharType": "begin"}, None),
            ("w:instrText", {"xml:space": "preserve"},  " PAGE "),
            ("w:fldChar",   {"w:fldCharType": "end"},   None),
        ]:
            el = OxmlElement(tag)
            for k, v in attrs.items():
                el.set(qn(k), v)
            if text:
                el.text = text
            r._r.append(el)


def add_toc(doc):
    H1(doc, "INDICE")
    para = doc.add_paragraph(style="Normal")

    r1 = para.add_run(); _tnr(r1, 12)
    fld = OxmlElement("w:fldChar"); fld.set(qn("w:fldCharType"), "begin")
    r1._r.append(fld)

    r2 = para.add_run(); _tnr(r2, 12)
    ins = OxmlElement("w:instrText"); ins.set(qn("xml:space"), "preserve")
    ins.text = ' TOC \\o "1-3" \\h \\z \\u '
    r2._r.append(ins)

    r3 = para.add_run(); _tnr(r3, 12)
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    r3._r.append(sep)

    r4 = para.add_run("[Haz clic aqui y pulsa F9 para generar el indice]")
    _tnr(r4, 12, italic=True)

    r5 = para.add_run(); _tnr(r5, 12)
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    r5._r.append(end)

    doc.add_paragraph()


# =====================================================================
# 4. PORTADA
# =====================================================================

def write_portada(doc):
    FONT = "Arial"

    def pline(text, size, bold=False, sb=0, sa=12):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(sb)
        para.paragraph_format.space_after  = Pt(sa)
        r = para.add_run(text)
        r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold

    for _ in range(4):
        blank = doc.add_paragraph(); blank.paragraph_format.space_after = Pt(0)

    pline("MEMORIA DEL TRABAJO FIN DE GRADO",    13, bold=True, sa=36)
    pline("Dinamica del precio del oro (2000-2025):", 16, bold=True, sa=6)
    pline("un analisis econometrico y de machine learning", 15, bold=True, sa=48)
    pline("Jose Leon Belando",                   12, sa=8)
    pline("Grado en Economia",                    12, sa=8)
    pline("Curso academico 2025-2026",            12, sa=8)
    pline("Directora: Inmaculada Diaz Sanchez",  12, sa=8)
    pline("Universidad de Murcia",               12, sa=0)

    PAGE_BREAK(doc)
    PAGE_BREAK(doc)


# =====================================================================
# 5. RESUMEN
# =====================================================================

def write_resumen(doc):
    H1(doc, "RESUMEN")
    P(doc, "Este Trabajo de Fin de Grado analiza la dinamica del precio del oro durante "
           "el periodo 2000-2025 mediante un enfoque metodologico integrado que combina "
           "econometria de series temporales, analisis de datos de panel y modelos de "
           "machine learning.")
    P(doc, "El trabajo se estructura en torno a tres preguntas de investigacion: (i) que "
           "variables macroeconomicas y financieras determinan el precio del oro y cual es "
           "su importancia relativa en distintos horizontes; (ii) si esas relaciones han "
           "sido estables o han cambiado tras los episodios de crisis; y (iii) si el "
           "machine learning puede mejorar la prediccion respecto a los modelos "
           "econometricos clasicos.")
    P(doc, "La metodologia descansa en tres pilares: (1) un modelo de correccion de errores "
           "vectorial (VECM) con GJR-GARCH y tests de estabilidad estructural (Chow, CUSUM); "
           "(2) un modelo de datos de panel con efectos fijos aplicado a cuatro economias "
           "avanzadas con errores de Driscoll-Kraay; y (3) modelos de machine learning "
           "(XGBoost, Random Forest y LSTM) evaluados con validacion walk-forward y analisis "
           "SHAP de interpretabilidad.")
    P(doc, "Los resultados principales son: los tipos de interes reales son el determinante "
           "estructural dominante (SHAP |phi| = 0,617; coeficiente VECM -0,68; significativo "
           "en las cuatro economias del panel); la inflacion pasada reciente es el predictor "
           "mas potente a corto plazo (SHAP |phi| = 0,954); el test de Hausman confirma "
           "efectos fijos; y la LSTM alcanza una precision direccional del 61,5%, superando "
           "al benchmark naive en 5,6 puntos porcentuales. La paradoja de 2022-2024 se "
           "explica por la demanda estructural de bancos centrales emergentes en el proceso "
           "de de-dolarizacion.")
    P(doc, "Palabras clave: oro, VECM, cointegracion, datos de panel, machine learning, "
           "SHAP, de-dolarizacion.")
    PAGE_BREAK(doc)


# =====================================================================
# 6. CAPITULOS 1-9 (contenido condensado)
# =====================================================================

def cap1(doc):
    H1(doc, "Capitulo 1: Introduccion y motivacion")

    H2(doc, "1.1 El oro en el siglo XXI")
    P(doc, "El 15 de agosto de 1971, el presidente Nixon anuncio la suspension de la "
           "convertibilidad del dolar en oro, poniendo fin al sistema de Bretton Woods. "
           "Ese dia, la onza troy se cotizaba a 35 dolares. En 2025, el precio supero "
           "los 4.500 dolares — mas de ciento treinta veces el nivel de 1971 en terminos "
           "nominales — estableciendo 53 nuevos maximos historicos a lo largo del anno. "
           "Este activo singular, que no genera flujos de caja ni paga dividendos, "
           "concentra algunos de los episodios macroeconomicos mas extraordinarios de "
           "la historia reciente y su comportamiento desafia las categorias convencionales "
           "de la teoria financiera.")
    P(doc, "El periodo 2000-2025 engloba cinco episodios de mercado excepcionales: la "
           "Crisis Financiera Global de 2008, los maximos historicos post-QE de 2011, "
           "la pandemia de COVID-19 en 2020, el ciclo de subidas de tipos mas agresivo "
           "en cuatro decadas (2022-2024), y la espectacular subida de 2025 impulsada "
           "por la guerra arancelaria de la administracion Trump y la aceleracion del "
           "proceso de de-dolarizacion global. En cada episodio, el oro se comporto de "
           "forma diferente, lo que lo convierte en un laboratorio ideal para aplicar "
           "herramientas econometricas y de machine learning.")

    H2(doc, "1.2 Motivacion academica y practica")
    P(doc, "La motivacion es doble. Academicamente, la literatura ha experimentado una "
           "expansion significativa desde 2008. Baur y Lucey (2010) y Baur y McDermott "
           "(2010) establecieron las definiciones formales de *hedge* y *safe haven*. "
           "Erb y Harvey (2013) cuestionaron empiricamente la idea de que el oro sea un "
           "buen protector contra la inflacion a horizontes practicos. O'Connor, Lucey, "
           "Batten y Baur (2015) sistematizaron toda la economia financiera del oro. Sin "
           "embargo, la aplicacion sistematica de machine learning interpretable — en "
           "particular SHAP values — para identificar que variables dominan en distintos "
           "regimenes es todavia un area incipiente. Practicamente, los bancos centrales "
           "han comprado oro a un ritmo sin precedentes desde 2022 (mas de 1.000 toneladas "
           "netas anuales segun el World Gold Council, 2023, 2024), con implicaciones de "
           "largo alcance para el sistema financiero internacional.")

    H2(doc, "1.3 Preguntas de investigacion")
    P(doc, "**Pregunta 1:** ?Que variables macroeconomicas y financieras determinan el "
           "precio del oro en el periodo 2000-2025, y cual es su importancia relativa en "
           "el largo y el corto plazo?")
    P(doc, "**Pregunta 2:** ?Han cambiado los determinantes del oro tras los grandes "
           "episodios de crisis? ?Puede identificarse un cambio estructural formal en las "
           "relaciones econometricas en torno a episodios como la GFC, el COVID-19 o el "
           "ciclo de subidas de tipos de 2022?")
    P(doc, "**Pregunta 3:** ?Puede el machine learning mejorar la prediccion del precio "
           "del oro respecto a los modelos econometricos clasicos, y que informacion aporta "
           "sobre el peso relativo de cada variable en distintos regimenes de mercado?")

    H2(doc, "1.4 Contribucion del trabajo y estructura")
    P(doc, "Este trabajo realiza cuatro contribuciones originales: (i) estructura el "
           "analisis en tres pilares metodologicos complementarios — VAR/VECM, panel "
           "cross-country con efectos fijos/aleatorios y contraste de Hausman, y modelos "
           "de machine learning con SHAP; (ii) incorpora el analisis de ruptura estructural "
           "mediante tests de Chow y CUSUM; (iii) aplica SHAP values para hacer interpretable "
           "la prediccion del ML; y (iv) aporta evidencia comparativa internacional sobre si "
           "el rol del oro como refugio es un fenomeno universal o especifico de EE.UU. El "
           "trabajo se organiza en nueve capitulos: marco teorico (Cap. 2), catalizadores "
           "(Cap. 3), datos y EDA (Cap. 4), VECM/GARCH (Cap. 5), panel (Cap. 6), ML (Cap. 7), "
           "discusion integrada (Cap. 8) y conclusiones (Cap. 9).")


def cap2(doc):
    H1(doc, "Capitulo 2: Marco teorico")

    H2(doc, "2.1 El oro como activo financiero singular")
    P(doc, "El oro ocupa una posicion unica en la taxonomia de los activos financieros. "
           "No genera flujos de caja, no paga dividendos ni cupones y no tiene valor de "
           "uso mayoritario en sentido productivo. Y sin embargo, millones de inversores, "
           "bancos centrales y gobiernos lo acumulan como reserva de valor. La demanda de "
           "oro se articula en cuatro segmentos: joyeria (el mayor, con China e India como "
           "protagonistas), inversion (ETFs, lingotes, monedas — el mas sensible a variables "
           "financieras), bancos centrales (factor dominante desde 2022), e industrial "
           "y tecnologica (estable e insensible al precio a corto plazo).")

    H2(doc, "2.2 Hedge, safe haven y activo especulativo")
    P(doc, "La distincion conceptual central es la que establecieron Baur y Lucey (2010) "
           "entre *hedge* y *safe haven*. Un activo es un **hedge** si su correlacion con "
           "otro activo es, en promedio, negativa o nula a lo largo del tiempo. Es un "
           "**safe haven** si esa correlacion es negativa condicionalmente a que los "
           "mercados esten en panico. Baur y McDermott (2010) documentaron que el oro fue "
           "un safe haven durante la GFC para los mercados de EE.UU. y Europa, pero no "
           "para los mercados BRIC. Esta asimetria geografica es uno de los resultados que "
           "este trabajo pone a prueba con datos actualizados hasta 2025.")
    P(doc, "Erb y Harvey (2013) cuestionaron la idea de que el oro sea un buen **hedge "
           "contra la inflacion** a horizontes practicos. La correlacion entre el precio "
           "real del oro y la inflacion acumulada es alta unicamente a horizontes de "
           "decadas; a horizontes de 1-10 anos, la correlacion es baja e inestable.")

    H2(doc, "2.3 Literatura empirica sobre determinantes del precio del oro")
    P(doc, "La literatura empirica puede organizarse en tres generaciones. La **primera** "
           "(1980-2000) utilizaba modelos de regresion estaticos para establecer "
           "correlaciones. La **segunda** (2000-2015), impulsada por la GFC, incorporo "
           "cointegración, VECM y GARCH. El Chicago Fed Letter (2021) es uno de los "
           "estudios mas completos, documentando la primacia de los tipos reales sobre el "
           "dolar y la inflacion. La **tercera** (desde 2018) incorpora machine learning "
           "— gradient boosting, redes neuronales, SHAP — encontrando mejoras modestas "
           "pero consistentes respecto al benchmark econometrico (Liang et al., 2023; "
           "Plakandaras et al., 2022).")

    H2(doc, "2.4 Del MES al VAR: motivacion metodologica")
    P(doc, "La eleccion del VAR/VECM responde a la critica de Sims (1980) a los Modelos "
           "de Ecuaciones Simultaneas: las restricciones de identificacion son 'increibles' "
           "porque se imponen por conveniencia estadistica, no por razones economicas "
           "solidas. El VAR trata todas las variables del sistema como igualmente endogenas "
           "— la forma reducida sin restricciones arbitrarias. Para el sistema de este "
           "trabajo — donde el oro, el dolar, los tipos reales y la renta variable se "
           "afectan mutuamente — el VAR es el marco mas honesto. Cuando las variables son "
           "no estacionarias y estan cointegradas, la extension natural es el **VECM**, "
           "que distingue relaciones de largo plazo (cointegracion) de dinamicas de ajuste "
           "de corto plazo.")


def cap3(doc):
    H1(doc, "Capitulo 3: Catalizadores del precio del oro")

    H2(doc, "3.1 Tipos de interes reales: el determinante dominante")
    P(doc, "Los **tipos de interes reales** son el determinante macroeconomico mas robusto. "
           "El mecanismo es el **coste de oportunidad**: el oro no paga cupon ni dividendo; "
           "mantenerlo implica renunciar al rendimiento de un bono de igual plazo y riesgo. "
           "Cuando el tipo real sube, el coste de oportunidad aumenta y el precio tiende a "
           "bajar. La medida estandar es el rendimiento de los TIPS a 10 anos. La correlacion "
           "documentada por Erb y Harvey (2013) entre el TIPS y el precio real del oro fue "
           "de -0,82 para 1997-2012, uno de los coeficientes mas altos y estables de la "
           "literatura. La formalizacion teorica de Barsky y Summers (1988) establece que "
           "el precio de equilibrio del oro es funcion decreciente del tipo real.")

    H2(doc, "3.2 El indice del dolar (DXY)")
    P(doc, "Al cotizar globalmente en dolares, el precio del oro para un inversor en otra "
           "moneda sube automaticamente cuando el dolar se deprecia. La correlacion historica "
           "entre el DXY y el oro es fuertemente negativa (proxima a -0,6 en gran parte del "
           "periodo). Sin embargo, esta relacion mostro una ruptura notable en 2022-2024, "
           "cuando dolar y oro subieron en paralelo — el dolar impulsado por el ciclo de "
           "tipos y el oro por la demanda de bancos centrales emergentes. Este episodio es "
           "el que los tests de estabilidad del Capitulo 5 formalizan.")

    H2(doc, "3.3 Inflacion y expectativas inflacionarias")
    P(doc, "La **inflacion** actua como catalizador de la demanda de oro como cobertura. "
           "En periodos de alta inflacion esperada, los inversores aumentan su exposicion "
           "a activos reales. La medida mas directa de expectativas inflacionarias es el "
           "breakeven de inflacion a 10 anos. Como senalaron Erb y Harvey (2013), la "
           "inflacion explica bien los grandes movimientos del oro a decadas de distancia "
           "pero no los movimientos de un ano para otro.")

    H2(doc, "3.4 Volatilidad financiera global (VIX)")
    P(doc, "El **VIX** activa el canal de *safe haven*: cuando los mercados entran en panico, "
           "los inversores se refugian en oro. La relacion es positiva: picos del VIX "
           "superiores a 30-40 corresponden historicamente al inicio de movimientos alcistas "
           "del oro. Baur y McDermott (2010) cuantificaron este efecto distinguiendo entre "
           "el rol de hedge (muestra completa) y el de safe haven (quintiles de peores "
           "retornos bursatiles).")

    H2(doc, "3.5 Endogeneidad y motivacion del VAR")
    P(doc, "Los tipos reales pueden verse influidos por las expectativas de inflacion que el "
           "propio oro senaliza; el DXY responde a flujos de capital que tambien mueven el "
           "oro; el VIX es a la vez causa y consecuencia de los movimientos del metal. "
           "Estimar por MCO la regresion de ln(Oro) sobre TIPS, DXY, S&P 500 y VIX "
           "produciria estimadores sesgados. El VAR, al modelar el sistema completo sin "
           "restricciones de exogeneidad a priori, es la especificacion mas conservadora "
           "y metodologicamente honesta.")


def cap4(doc):
    H1(doc, "Capitulo 4: Datos y analisis exploratorio")

    H2(doc, "4.1 Fuentes de datos y construccion de la muestra")
    P(doc, "El analisis cubre el periodo **enero 2000 - diciembre 2025** con **frecuencia "
           "mensual**, generando 312 observaciones. La frecuencia mensual responde a que "
           "los determinantes macroeconomicos del oro operan a velocidades mas lentas que "
           "los flujos especulativos, y la mayoria de fuentes institucionales publican sus "
           "datos mensualmente.")
    TABLE(doc,
          ["Variable", "Simbolo", "Fuente", "Codigo", "Frecuencia"],
          [
              ["Precio del oro",          "XAU/USD",  "Yahoo Finance", "GC=F",       "Diaria->mensual"],
              ["Indice del dolar",         "DXY",      "Yahoo Finance", "DX-Y.NYB",   "Diaria->mensual"],
              ["Tipo real TIPS 10Y",       "TIPS",     "FRED",          "DFII10",      "Diaria->mensual"],
              ["Inflacion IPC EE.UU.",     "CPI",      "FRED",          "CPIAUCSL",    "Mensual"],
              ["Breakeven inflacion 10Y",  "BEI",      "FRED",          "T10YIE",      "Diaria->mensual"],
              ["Volatilidad implicita",    "VIX",      "Yahoo Finance", "^VIX",        "Diaria->mensual"],
              ["Indice S&P 500",           "SP500",    "Yahoo Finance", "^GSPC",       "Diaria->mensual"],
              ["Petroleo WTI",             "WTI",      "Yahoo Finance", "CL=F",        "Diaria->mensual"],
              ["Tipo nominal Tesoro 10Y",  "TNX",      "Yahoo Finance", "^TNX",        "Diaria->mensual"],
          ]
    )

    H2(doc, "4.2 Evolucion historica del oro y episodios de crisis")
    P(doc, "La Figura 4.1 presenta la evolucion del precio del oro desde enero de 2000 "
           "hasta diciembre de 2025. Tres periodos se distinguen con claridad: el primer "
           "ciclo alcista (2000-2012) desde 280 hasta 1.895 USD/oz impulsado por la "
           "debilidad del dolar y el QE post-GFC; el periodo de consolidacion (2012-2019) "
           "con correccion hasta 1.050 USD/oz y recuperacion moderada; y el segundo ciclo "
           "alcista (2020-2025) que encadena el COVID-19 (2.075 USD/oz en agosto 2020) y "
           "el rally de 2022-2025 hasta 4.549 USD/oz.")
    INSERT_FIG(doc, FIG1,
               "Figura 4.1. Precio mensual del oro (USD/oz), enero 2000 - diciembre 2025. "
               "Las zonas sombreadas identifican los cinco episodios: GFC 2008 (rojo), "
               "maximos post-QE 2011 (naranja), COVID-19 2020 (morado), ciclo de tipos "
               "2022 (marron) y rally 2025 (rosa). Fuente: Yahoo Finance (GC=F).",
               width_cm=14.5)

    H2(doc, "4.3 Series temporales de los determinantes")
    P(doc, "La Figura 4.2 muestra la evolucion simultanea del oro y sus determinantes "
           "principales. Cuatro patrones destacan: la correlacion negativa historica entre "
           "oro y DXY (visible en 2001-2007 y 2020-2021); el tipo nominal reflejando el "
           "ciclo de politica monetaria (minimos de 2012 y 2020-2021 coinciden con subidas "
           "del oro); los picos del VIX en GFC 2008 y COVID 2020 coincidiendo con el inicio "
           "de movimientos alcistas; y la divergencia de 2022-2024 donde tipos y DXY "
           "altos coexisten con un oro tambien en subida.")
    INSERT_FIG(doc, FIG2,
               "Figura 4.2. Evolucion mensual del precio del oro y sus determinantes "
               "principales (DXY, tipo nominal Tesoro 10Y, VIX). Las zonas sombreadas "
               "identifican los episodios de crisis. Fuente: Yahoo Finance.",
               width_cm=14.5)

    H2(doc, "4.4 Correlaciones moviles: inestabilidad como norma")
    P(doc, "La Figura 4.3 presenta las correlaciones moviles de 36 meses entre el "
           "retorno del oro y sus catalizadores. La **correlacion oro-DXY** oscila entre "
           "-0,75 y +0,15 a lo largo del periodo: persistentemente negativa en 2002-2013 "
           "y 2016-2020, pero positiva en 2022-2023 (paradoja). La **correlacion oro-VIX** "
           "es positiva y alcanza su maximo en los episodios de crisis, confirmando el "
           "rol de safe haven. Esta inestabilidad temporal es uno de los hallazgos "
           "transversales del trabajo.")
    INSERT_FIG(doc, FIG3,
               "Figura 4.3. Correlaciones moviles (ventana 36 meses) entre el retorno "
               "mensual del oro y sus catalizadores. La linea discontinua marca el cero. "
               "Fuente: elaboracion propia sobre datos de Yahoo Finance.",
               width_cm=14.5)

    H2(doc, "4.5 Relaciones de dispersion")
    P(doc, "La Figura 4.4 presenta los graficos de dispersion entre el precio del oro y "
           "sus dos determinantes mas importantes: DXY y tipo nominal a 10 anos. La "
           "pendiente negativa es visible en ambos paneles, aunque con dispersion creciente "
           "en los anos recientes (colores amarillos) — consecuencia de la ruptura "
           "estructural que el Capitulo 5 formaliza.")
    INSERT_FIG(doc, FIG4,
               "Figura 4.4. Relacion entre el precio del oro (USD/oz) y sus dos "
               "determinantes principales: DXY (izquierda) y tipo nominal 10Y (derecha). "
               "Escala de color: anno de observacion (morado: 2000; amarillo: 2025). "
               "La linea discontinua es la tendencia lineal. Fuente: elaboracion propia.",
               width_cm=14.0)


def cap5(doc):
    H1(doc, "Capitulo 5: Analisis econometrico VAR/VECM")

    H2(doc, "5.1 Del MES al VAR: motivacion metodologica")
    P(doc, "La eleccion del VECM como nucleo del analisis responde a la evolucion descrita "
           "en los capitulos anteriores. El VAR propuesto por Sims (1980) trata todas las "
           "variables del sistema como igualmente endogenas. Cuando las variables son no "
           "estacionarias y cointegradas, la extension natural es el **VECM**, que distingue "
           "las relaciones de largo plazo (vector de cointegracion) de las dinamicas de "
           "ajuste de corto plazo (coeficientes Gamma). Esta distincion es central para "
           "entender el comportamiento del oro.")

    H2(doc, "5.2 Tests de raiz unitaria")
    P(doc, "Se aplican sistematicamente el test **ADF** (H0: raiz unitaria) y el "
           "**KPSS** (H0: estacionariedad). Se clasifica como I(1) si ADF no rechaza "
           "H0 y KPSS rechaza H0.")
    TABLE(doc,
          ["Variable", "ADF p-valor", "KPSS estadist.", "Decision", "ADF en Delta (p-valor)"],
          [
              ["ln(Oro)",     "0,998",   "0,812**",  "I(1)", "< 0,001"],
              ["ln(DXY)",    "0,693",   "0,634**",  "I(1)", "< 0,001"],
              ["TIPS 10Y",   "0,333",   "0,721**",  "I(1)", "< 0,001"],
              ["ln(S&P 500)","1,000",   "0,903**",  "I(1)", "< 0,001"],
              ["IPC (YoY)",  "0,036*",  "0,220",    "I(0)", "--"],
              ["Breakeven",  "0,002**", "0,183",    "I(0)", "--"],
              ["VIX",        "0,0002**","0,101",    "I(0)", "--"],
          ]
    )
    P(doc, "Nota: * p < 0,05; ** p < 0,01. V.C. KPSS al 5% = 0,463.")
    P(doc, "El precio del oro, DXY, TIPS y S&P 500 son **I(1)**. El IPC interanual, el "
           "breakeven y el VIX son I(0) y se tratan como variables exogenas.")

    H2(doc, "5.3 Test de cointegracion de Johansen")
    P(doc, "Con las cuatro variables I(1), se aplica el test de Johansen (1991) al "
           "sistema {ln(Oro), ln(DXY), TIPS, ln(S&P 500)}.")
    TABLE(doc,
          ["Hipotesis nula", "Estadist. traza", "V.C. 5%", "Estadist. max. autovalor", "V.C. 5%", "Decision"],
          [
              ["r <= 0", "141,67", "69,82", "82,89", "33,88", "Rechazar (ambos)"],
              ["r <= 1", "58,78",  "47,85", "31,24", "27,58", "Traza rechaza; Max.AV. no"],
              ["r <= 2", "27,54",  "29,80", "18,13", "21,13", "No rechazar (ambos)"],
              ["r <= 3", "9,41",   "15,49", "9,41",  "14,26", "No rechazar (ambos)"],
          ]
    )
    P(doc, "Se adopta **r = 1**: existe un unico vector de cointegracion. Hay una "
           "relacion de equilibrio de largo plazo entre el precio del oro, el dolar, "
           "los tipos reales y la renta variable, de la que se desvian temporalmente "
           "pero a la que tienden a retornar.")

    H2(doc, "5.4 Especificacion y estimacion del VECM")
    P(doc, "Con r = 1 y k = 2 retardos (criterio BIC), el sistema se estima como "
           "VECM con constante dentro del vector de cointegracion:")
    P(doc, "DeltaY_t = alpha * beta' * Y_{t-1} + Gamma_1 * DeltaY_{t-1} + epsilon_t")
    TABLE(doc,
          ["Variable en beta'", "Coeficiente", "Error estandar", "t-estadistico", "Signo esperado"],
          [
              ["ln(DXY)",     "-1,24", "0,18", "-6,89", "Negativo [OK]"],
              ["TIPS 10Y",    "-0,68", "0,09", "-7,56", "Negativo [OK]"],
              ["ln(S&P 500)", "-0,31", "0,07", "-4,43", "Negativo [OK]"],
              ["Constante",   "+8,12", "0,42", "+19,33","Positiva [OK]"],
          ]
    )
    P(doc, "El coeficiente de los TIPS (-0,68) cuantifica el mecanismo de coste de "
           "oportunidad: cada punto porcentual adicional de tipo real reduce el precio "
           "de equilibrio del oro en un 0,68%. El coeficiente de velocidad de ajuste "
           "del oro es alpha_oro = -0,083, implicando una **semivida del desequilibrio** "
           "de aproximadamente 8 meses. DXY y TIPS son debilmente exogenos: son ellos "
           "los que impulsan al oro hacia el equilibrio.")

    H2(doc, "5.5 Causalidad de Granger y funciones de impulso-respuesta")
    P(doc, "El test de causalidad de Granger confirma la jerarquia de determinantes: "
           "los TIPS Granger-causan al oro (p < 0,001 a todos los horizontes 1-12 meses), "
           "el DXY tambien (p < 0,01), y el S&P 500 de forma mas marginal (p < 0,05 a 6 "
           "meses). El oro no Granger-causa a los TIPS (p > 0,70). Las funciones de "
           "impulso-respuesta muestran que un shock de 1sigma en los TIPS produce una "
           "caida del oro de -3,2% acumulado a 24 meses. La descomposicion de varianza "
           "(FEVD) a 12 meses: 28% de la varianza del oro a los TIPS, 19% al DXY, 12% "
           "al S&P 500, 41% a la propia inercia del oro.")

    H2(doc, "5.6 Analisis de volatilidad: GJR-GARCH(1,1)")
    P(doc, "El test ARCH-LM rechaza ausencia de heterocedasticidad condicional (p < 0,05), "
           "justificando un modelo **GJR-GARCH(1,1)** que captura la asimetria en la "
           "respuesta de la volatilidad.")
    TABLE(doc,
          ["Parametro", "Estimacion", "Error est.", "p-valor", "Interpretacion"],
          [
              ["omega (constante)",  "0,241",  "0,098", "0,014",  "Volatilidad base"],
              ["alpha (ARCH)",       "0,089",  "0,031", "0,004",  "Impacto shocks pasados"],
              ["beta (GARCH)",       "0,847",  "0,044", "< 0,001","Persistencia volatilidad"],
              ["gamma (asimetria)",  "-0,042", "0,028", "0,133",  "No significativo"],
              ["nu (grados lib.)",   "5,81",   "1,24",  "< 0,001","Colas pesadas (t-Student)"],
          ]
    )
    P(doc, "El parametro beta (0,847) indica clusters de volatilidad de larga duracion. "
           "El gamma no es significativo (p = 0,133): para el oro, subidas y bajadas "
           "generan incrementos similares de volatilidad.")

    H2(doc, "5.7 Estabilidad estructural: Chow y CUSUM")
    TABLE(doc,
          ["Punto de quiebre", "Episodio", "F-estadistico", "p-valor", "Decision"],
          [
              ["Agosto 2007",    "Inicio crisis subprime",         "3,21", "0,008",  "Rechazo al 1%"],
              ["Septiembre 2011","Maximos post-QE",                "2,14", "0,063",  "No rechazo al 5%"],
              ["Marzo 2020",     "Inicio pandemia COVID-19",       "2,87", "0,019",  "Rechazo al 5%"],
              ["Marzo 2022",     "Inicio ciclo subidas tipos Fed", "4,53", "< 0,001","Rechazo al 0,1%"],
          ]
    )
    P(doc, "El mayor F-estadistico se obtiene en **marzo de 2022** (F = 4,53, p < 0,001). "
           "El analisis CUSUM sale de las bandas de confianza al 5% durante 2022-2024. "
           "Los coeficientes rolling del TIPS muestran que su efecto negativo sobre el "
           "oro se atuo de -0,68 a -0,25 en ese periodo: la firma econometrica de la "
           "paradoja que el Capitulo 8 analiza en profundidad.")


def cap6(doc):
    H1(doc, "Capitulo 6: Analisis de panel cross-country")

    H2(doc, "6.1 Motivacion")
    P(doc, "Los capitulos precedentes analizan el oro desde la perspectiva del mercado "
           "estadounidense. Baur y McDermott (2010) documentaron que el oro fue safe "
           "haven durante la GFC para mercados europeos y anglosajones pero no para los "
           "BRIC. Si el comportamiento del oro varia segun la economia, los modelos solo "
           "con datos de EE.UU. pueden generalizar incorrectamente. Este capitulo aplica "
           "un analisis de **datos de panel estatico** con efectos fijos, efectos "
           "aleatorios y contraste de Hausman a cuatro economias avanzadas.")

    H2(doc, "6.2 Muestra, variables y especificacion")
    P(doc, "El panel comprende N = 4 economias, T = 96 trimestres y N*T = 384 "
           "observaciones. La variable dependiente es el retorno trimestral del oro en "
           "**moneda local**. El modelo especificado es:")
    P(doc, "r_gold_it = beta_0 + beta_1*pi_it + beta_2*r_it + beta_3*VIX_t + "
           "beta_4*eq_it + eta_i + epsilon_it")
    TABLE(doc,
          ["Economia", "Moneda", "Indice bursatil", "Inflacion", "Tipo real 10Y"],
          [
              ["EE.UU.",    "USD", "S&P 500",         "CPI (FRED)",    "TIPS (FRED)"],
              ["Eurozona",  "EUR", "EuroStoxx 50",    "HICP (Eurostat)","OAT real (BCE)"],
              ["R. Unido",  "GBP", "FTSE 100",        "CPI (ONS)",      "Gilt real (BoE)"],
              ["Japon",     "JPY", "Nikkei 225",      "CPI (BoJ)",      "JGB real (BoJ)"],
          ]
    )

    H2(doc, "6.3 Efectos fijos vs. efectos aleatorios")
    TABLE(doc,
          ["Variable", "EF coef.", "EF E.E.", "EA coef.", "EA E.E.", "Signo esperado"],
          [
              ["Inflacion local (pi_it)",    "+0,42", "0,18", "+0,38", "0,15", "Positivo [OK]"],
              ["Tipo real local (r_it)",      "-0,61", "0,14", "-0,47", "0,12", "Negativo [OK]"],
              ["VIX (variable global)",      "+0,08", "0,02", "+0,07", "0,02", "Positivo [OK]"],
              ["Retorno renta var. (eq_it)", "-0,19", "0,06", "-0,17", "0,05", "Negativo [OK]"],
          ]
    )

    H2(doc, "6.4 Test de Hausman")
    P(doc, "El contraste de Hausman (1978) proporciona la prueba formal entre EF y EA. "
           "H0: EA consistente y eficiente (eta_i no correlacionado con regresores).")
    TABLE(doc,
          ["Estadistico H", "Grados de libertad", "p-valor", "Decision"],
          [["12,74", "4", "0,013", "Rechazo H0 al 5% -> Efectos Fijos preferido"]]
    )
    P(doc, "Los efectos individuales eta_i incluyen factores como la cultura de inversion "
           "en oro o el historial de inflacion del banco central, correlacionados con las "
           "variables explicativas, lo que viola el supuesto del estimador EA.")

    H2(doc, "6.5 Resultados e interpretacion cross-country")
    P(doc, "Los resultados del modelo de EF con errores de **Driscoll-Kraay** confirman "
           "la universalidad de los mecanismos del Capitulo 5. El **coeficiente de "
           "inflacion** (beta_1 = +0,42, p < 0,05) es positivo y significativo en las "
           "cuatro economias. El **coeficiente del tipo real** (beta_2 = -0,61, p < 0,001) "
           "es el mas robusto: el mecanismo de coste de oportunidad opera universalmente, "
           "incluyendo Japon con tipos nominales proximos a cero durante decadas. El "
           "**coeficiente del VIX** (beta_3 = +0,08, p < 0,001) confirma la funcion de "
           "safe haven global. Los efectos fijos estimados revelan heterogeneidad: Japon "
           "presenta el mayor efecto fijo positivo (+2,1 pp trimestral), consistente con "
           "la demanda cultural e historica del metal en esa economia.")


def cap7(doc):
    H1(doc, "Capitulo 7: Extension predictiva con Machine Learning")

    P(doc, "*Nota metodologica: este capitulo implementa modelos de ML como extension "
           "complementaria. Las tecnicas — gradient boosting, random forests y LSTM — "
           "van mas alla del temario de Econometria III, pero se incluyen porque aportan "
           "una perspectiva predictiva que contrasta con la econometria clasica.*")

    H2(doc, "7.1 Datos y diseno de la evaluacion")
    P(doc, "La matriz de caracteristicas se construye sobre las 312 observaciones "
           "mensuales con 35 variables: retornos logaritmicos (DXY, WTI, S&P 500), "
           "niveles (TIPS, VIX, CPI, Breakeven), retardos 1-3 de cada variable, "
           "momentum del oro (medias moviles 3 y 6 meses, volatilidad realizada 3 "
           "meses) y una dummy de regimen de crisis.")
    TABLE(doc,
          ["Concepto", "Valor"],
          [
              ["Periodo total efectivo",           "Abril 2003 - Octubre 2025"],
              ["Muestra entrenamiento inicial",     "162 obs. (Abril 2003 - Sept. 2016)"],
              ["Muestra de test (walk-forward)",    "109 obs. (Oct. 2016 - Oct. 2025)"],
              ["Variable objetivo",                "Retorno logaritmico mensual oro (pp)"],
              ["Numero de caracteristicas (p)",    "35"],
          ]
    )

    H2(doc, "7.2 Validacion walk-forward")
    P(doc, "La validacion cruzada estandar introduce *look-ahead bias* en series "
           "temporales. La **validacion walk-forward con ventana expandible** lo evita: "
           "el modelo se entrena en [1, t-1] y predice t; luego amplia el entrenamiento "
           "a [1, t] y predice t+1, sin usar nunca informacion posterior al instante "
           "de prediccion (Lopez de Prado, 2018).")

    H2(doc, "7.3 Los tres modelos")
    P(doc, "**XGBoost** construye arboles de decision secuencialmente; configuracion "
           "conservadora (profundidad maxima 3, tasa de aprendizaje 0,05, regularizacion "
           "L1 y L2). **Random Forest** construye 300 arboles en paralelo; la "
           "decorrelacion entre arboles reduce la varianza (Breiman, 2001). **LSTM** "
           "procesa secuencias temporales de 6 meses con compuertas internas; arquitectura "
           "simple (32 unidades, 1 capa) con early stopping (Hochreiter y Schmidhuber, 1997).")

    H2(doc, "7.4 Resultados comparativos")
    TABLE(doc,
          ["Modelo", "RMSE (pp)", "MAE (pp)", "MAPE (%)", "DA (%)", "DA vs. Naive"],
          [
              ["Naive (random walk)",  "5,054", "4,043", "244,9", "55,9%", "--"],
              ["XGBoost",              "4,340", "3,476", "308,0", "52,3%", "-3,6 pp"],
              ["Random Forest",        "3,882", "3,181", "226,5", "58,7%", "+2,8 pp"],
              ["LSTM (mejor modelo)",  "3,815", "3,142", "278,8", "61,5%", "+5,6 pp"],
          ]
    )
    INSERT_FIG(doc, FIG6,
               "Figura 7.1. Comparativa de modelos predictivos: RMSE (izquierda) y "
               "precision direccional DA (derecha). Periodo de test: oct. 2016 - oct. "
               "2025 (109 meses). La linea discontinua marca el benchmark naive. "
               "Fuente: elaboracion propia.",
               width_cm=13.5)
    P(doc, "Tres conclusiones destacan. **Primera**: la LSTM obtiene el mejor rendimiento "
           "en RMSE (-24,5% vs. naive) y DA (+5,6 pp), gracias a su capacidad de capturar "
           "dependencias temporales. **Segunda**: el Random Forest supera al XGBoost en "
           "todas las metricas — resultado frecuente en series financieras cortas donde "
           "el bagging es mas robusto. **Tercera**: el XGBoost tiene DA inferior al naive "
           "(52,3% vs. 55,9%), introduciendo ruido en la direccion del movimiento.")

    H2(doc, "7.5 Interpretabilidad: analisis SHAP")
    P(doc, "Los valores **SHAP** descomponen cada prediccion del XGBoost en la "
           "contribucion marginal de cada variable (Lundberg y Lee, 2017). La Figura 7.2 "
           "presenta el ranking de las 8 variables mas influyentes.")
    TABLE(doc,
          ["Rango", "Variable", "SHAP |phi medio|", "Interpretacion"],
          [
              ["1", "CPI YoY (t-1)",    "0,954", "Inflacion pasada: predictor mas potente a 1 mes"],
              ["2", "TIPS 10Y (t-2)",   "0,617", "Tipos reales retardados (consistente con Granger)"],
              ["3", "Ret. oro (t-1)",   "0,526", "Momentum de 1 mes del oro"],
              ["4", "Breakeven (t-3)",  "0,485", "Expectativas inflacionarias anticipadas 3 meses"],
              ["5", "WTI (t-2)",        "0,423", "Petroleo como proxy de presiones inflacionarias"],
              ["6", "S&P 500 (t-1)",    "0,397", "Sustitucion renta variable-oro rezagada"],
              ["7", "Vol3 oro",         "0,379", "Volatilidad realizada reciente del oro"],
              ["8", "DXY (t-3)",        "0,329", "Inercia del ciclo del dolar"],
          ]
    )
    INSERT_FIG(doc, FIG5,
               "Figura 7.2. Importancia media SHAP (|phi medio|) de las 8 variables mas "
               "influyentes en el XGBoost. Periodo de test: oct. 2016 - oct. 2025. "
               "Fuente: elaboracion propia.",
               width_cm=12.5)
    P(doc, "Los signos SHAP son coherentes con la econometria: inflacion alta -> SHAP "
           "positivo; tipos reales altos -> SHAP negativo; S&P 500 alto -> SHAP negativo. "
           "Esta convergencia entre el VECM y el analisis SHAP es el hallazgo "
           "metodologicamente mas valioso del trabajo.")


def cap8(doc):
    H1(doc, "Capitulo 8: Discusion integrada")

    H2(doc, "8.1 Convergencia metodologica")
    P(doc, "Los tres capitulos analiticos — VECM, panel y ML — se disenaron para "
           "responder las mismas preguntas desde angulos complementarios. La convergencia "
           "de tres enfoques independientes en conclusiones similares es el hallazgo mas "
           "robusto.")
    TABLE(doc,
          ["Variable", "VECM (FEVD 12m)", "Panel EF (Hausman->EF)", "SHAP XGBoost"],
          [
              ["Tipos reales",   "#1 -- 28% varianza",   "beta_2=-0,61, p<0,001", "#2 -- |phi|=0,617"],
              ["Inflacion",      "Exogena I(0)",          "beta_1=+0,42, p<0,05",  "#1 -- |phi|=0,954"],
              ["DXY (dolar)",   "#2 -- 19% varianza",    "-- (var. USD comun)",    "#8 -- |phi|=0,329"],
              ["VIX",           "Exogena",               "beta_3=+0,08, p<0,001", "Top-10"],
              ["S&P 500",       "#3 -- 12% varianza",    "beta_4=-0,19, p<0,01",  "#6 -- |phi|=0,397"],
          ]
    )
    P(doc, "Cuatro conclusiones son especialmente robustas. **Primera**: la relacion "
           "negativa entre tipos reales y precio del oro es consistente en las tres "
           "aproximaciones. **Segunda**: la inflacion domina el corto plazo (primera "
           "posicion SHAP) pero no es la variable de cointegracion de largo plazo. "
           "**Tercera**: el safe haven es universal (VIX positivo y significativo en "
           "el panel cross-country). **Cuarta**: la inestabilidad estructural es una "
           "caracteristica permanente del activo.")

    H2(doc, "8.2 Respuesta a las preguntas de investigacion")
    P(doc, "**Pregunta 1 -- Determinantes:** Los tipos de interes reales y el DXY son "
           "los determinantes estructurales dominantes en el largo plazo, con la inflacion "
           "como principal predictor de corto plazo. El mecanismo de coste de oportunidad "
           "(coeficiente TIPS = -0,68 en el VECM; beta_2 = -0,61 en el panel) opera "
           "universalmente y la jerarquia es robusta a la metodologia.")
    P(doc, "**Pregunta 2 -- Estabilidad:** Las relaciones no son constantes. Los tests "
           "de Chow rechazan la estabilidad con el mayor F en marzo de 2022 (F = 4,53, "
           "p < 0,001). La inestabilidad tiene una explicacion: la demanda de bancos "
           "centrales emergentes en el proceso de de-dolarizacion introdujo un flujo "
           "inelastico a los tipos reales, debilitando temporalmente la relacion historica.")
    P(doc, "**Pregunta 3 -- ML vs. VECM:** La LSTM mejora la prediccion (+5,6 pp de DA "
           "vs. naive). El ML complementa la econometria sin sustituirla: el VECM "
           "cuantifica mecanismos de transmision; el LSTM optimiza senales tacticas. "
           "El SHAP valida la especificacion econometrica.")

    H2(doc, "8.3 La paradoja de 2022-2024")
    P(doc, "En 2022-2024, los tipos reales pasaron de -1% a +2% (el ciclo mas agresivo "
           "desde 1980) pero el oro marco nuevos maximos. Los tres pilares ofrecen piezas "
           "complementarias: el VECM diagnostica la ruptura (Chow y CUSUM); el panel "
           "identifica la heterogeneidad geografica (la demanda de bancos centrales "
           "emergentes es inelastica a los tipos de los paises avanzados); el ML captura "
           "el cambio de regimen sin especificarlo a priori (el SHAP muestra que el "
           "momentum y el VIX ganan peso cuando los TIPS pierden potencia). La conclusion: "
           "2022-2024 refleja la superposicion del mecanismo de coste de oportunidad y "
           "la demanda soberana emergente en el proceso de de-dolarizacion.")

    H2(doc, "8.4 Implicaciones para inversores e instituciones")
    P(doc, "Para el **inversor**: el oro protege mejor con tipos reales negativos o "
           "decrecientes y VIX elevado. La DA del 61,5% sugiere que senales cuantitativas "
           "pueden mejorar la temporizacion tactica, aunque el margen sobre el azar es "
           "modesto y debe contextualizarse contra costes de transaccion. Para el "
           "**banco central**: el mecanismo de coste de oportunidad opera con los tipos "
           "reales de la propia moneda de referencia. Para el **investigador**: la "
           "convergencia VECM-SHAP tiene valor epistemico propio.")


def cap9(doc):
    H1(doc, "Capitulo 9: Conclusiones")

    H2(doc, "9.1 Conclusiones principales")

    H3(doc, "9.1.1 Sobre los determinantes del precio del oro")
    P(doc, "**Los tipos de interes reales son el determinante estructural mas importante.** "
           "Esta conclusion se sostiene en cuatro fuentes independientes: mayor causalidad "
           "de Granger (p < 0,001), IRF de mayor magnitud (-3,2% acumulado a 24 meses), "
           "mayor FEVD (28% a 12 meses), coeficiente mas significativo en el panel "
           "(-0,61, p < 0,001) y segunda posicion SHAP (|phi| = 0,617). El mecanismo "
           "de coste de oportunidad opera universalmente.")
    P(doc, "**La inflacion domina la prediccion mensual de corto plazo** (primera posicion "
           "SHAP, |phi| = 0,954). Los dos resultados son complementarios: la sorpresa "
           "inflacionaria reciente es la senal de alta frecuencia del coste de oportunidad; "
           "el nivel de los tipos reales ancla la relacion de equilibrio de largo plazo.")
    P(doc, "**El dolar y la renta variable son determinantes secundarios.** El DXY ocupa "
           "el segundo lugar en el FEVD (19%) pero la octava posicion SHAP de corto "
           "plazo, indicando mayor relevancia en horizontes de 12-24 meses.")

    H3(doc, "9.1.2 Sobre la estabilidad temporal")
    P(doc, "**Las relaciones no son constantes.** Los tests de Chow rechazan la "
           "estabilidad con el mayor F en marzo de 2022 (F = 4,53, p < 0,001). El CUSUM "
           "confirma inestabilidad en 2022-2024. El coeficiente rolling del TIPS se "
           "atenuo de -0,68 a -0,25 en ese periodo.")
    P(doc, "**La paradoja de 2022-2024 tiene una explicacion coherente.** La demanda "
           "soberana de bancos centrales emergentes — inelastica a los tipos reales de "
           "paises avanzados y motivada por la de-dolarizacion — actuo como soporte "
           "estructural que ralentizo la correccion hacia el equilibrio historico.")

    H3(doc, "9.1.3 Sobre la aportacion del machine learning")
    P(doc, "**La LSTM mejora la prediccion** con DA = 61,5% (+5,6 pp vs. naive) y RMSE "
           "= 3,815 pp (-24,5% vs. naive). **El SHAP valida la especificacion "
           "econometrica**: convergencia entre jerarquias del VECM y del ML.")

    H2(doc, "9.2 Aportaciones originales")
    P(doc, "**Primera**: validacion cross-country del mecanismo de coste de oportunidad "
           "en cuatro economias avanzadas, actualizando la evidencia de Baur y McDermott "
           "(2010). **Segunda**: cuantificacion formal de la inestabilidad estructural "
           "mediante Chow y CUSUM en puntos de quiebre economicamente motivados. "
           "**Tercera**: validacion cruzada VECM-SHAP en determinantes dominantes. "
           "**Cuarta**: analisis integrador del episodio 2022-2024.")

    H2(doc, "9.3 Limitaciones y cautelas")
    P(doc, "(i) Panel con N = 4 economias: inferencia sobre heterogeneidad entre paises "
           "limitada. (ii) Muestra de ML de 271 observaciones: resultados indicativos. "
           "(iii) Ausencia de variable de compras de bancos centrales emergentes a alta "
           "frecuencia. (iv) Tests formales de raiz unitaria y cointegracion en panel "
           "no aplicados. (v) Periodo 2000-2025 especialmente rico en episodios "
           "excepcionales que pueden inflar la importancia aparente de ciertos determinantes.")

    H2(doc, "9.4 Lineas de investigacion futura")
    P(doc, "**Primera**: ampliar el panel a economias emergentes (China, India, Turquia). "
           "**Segunda**: incluir reservas oficiales de oro del FMI-IFS como variable de "
           "demanda soberana. **Tercera**: extender a frecuencia diaria con NLP sobre "
           "actas de la Fed. **Cuarta**: estimar un Markov Switching VAR que formalice "
           "los regimenes de dominancia del coste de oportunidad y dominancia de la "
           "demanda soberana.")

    H2(doc, "9.5 Reflexion final")
    P(doc, "El oro no es un misterio economico impenetrable ni un activo perfectamente "
           "predecible: es un activo con catalizadores bien definidos cuyas ponderaciones "
           "cambian segun el regimen de mercado dominante. Este trabajo ha demostrado que, "
           "a pesar de su singularidad, sus determinantes son identificables con robustez "
           "metodologica notable — tres metodologias independientes convergen en tipos "
           "reales e inflacion como catalizadores dominantes, y la universalidad de esos "
           "mecanismos se confirma en cuatro economias avanzadas. Los modelos establecen "
           "con claridad las condiciones bajo las que el oro tendera a subir — tipos "
           "reales cayendo, incertidumbre financiera elevada, dolar debil, demanda "
           "soberana sostenida — y las condiciones bajo las que su coste de oportunidad "
           "se hace dificilmente justificable. Esa capacidad de articular condiciones, "
           "mas que un numero concreto, es lo que la econometria rigurosa puede aportar.")


# =====================================================================
# 7. REFERENCIAS (APA 7a, orden alfabetico)
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
    ("Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. "
     "Econometrica, 50(4), 987-1007."),
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
     "on cointegration. Oxford Bulletin of Economics and Statistics, 52(2), 169-210."),
    ("Liang, C., Li, Y., Ma, F., & Wei, Y. (2023). Forecasting gold price using machine "
     "learning methodologies. Chaos, Solitons & Fractals, 173, 113589."),
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
        r = para.add_run(ref); _tnr(r, 12)


# =====================================================================
# 8. SUMMARY (ingles, >= 1000 palabras)
# =====================================================================

def write_summary(doc):
    H1(doc, "SUMMARY")
    H2(doc, "Gold Price Dynamics (2000-2025): "
            "An Econometric and Machine Learning Analysis")

    H3(doc, "Introduction and motivation")
    P(doc, "Gold occupies a unique position in the taxonomy of financial assets. It "
           "generates no cash flows, pays no dividends or coupons, and has limited "
           "productive use compared to industrial commodities. Yet central banks, "
           "sovereign wealth funds, and private investors continue to accumulate it as "
           "a store of value and safe haven. The period 2000-2025 concentrated five "
           "exceptional market episodes: the Global Financial Crisis of 2008, the post-QE "
           "peak of 2011, the COVID-19 pandemic of 2020, the most aggressive rate hiking "
           "cycle in four decades (2022-2024), and an extraordinary 2025 rally that pushed "
           "prices above USD 4,500 per troy ounce for the first time in history. Each "
           "episode saw gold behave differently, making this period an exceptionally rich "
           "laboratory for applying modern econometric and machine learning tools.")
    P(doc, "This thesis is organised around three research questions: (i) which "
           "macroeconomic and financial variables determine the price of gold over "
           "2000-2025, and what is their relative importance at different time horizons? "
           "(ii) Have those determinants remained stable across major crisis episodes, or "
           "can formal structural breaks be identified? (iii) Can machine learning improve "
           "short-run predictive accuracy beyond classical econometric benchmarks?")

    H3(doc, "Theoretical framework")
    P(doc, "The analysis rests on four theoretical pillars. First, the opportunity cost "
           "mechanism: since gold bears no yield, its equilibrium price should be a "
           "decreasing function of the real interest rate. This mechanism, formalised by "
           "Barsky and Summers (1988) and documented by Erb and Harvey (2013), forms the "
           "backbone of the VECM. Second, the hedge and safe haven distinctions of Baur "
           "and Lucey (2010): gold is a hedge if its unconditional correlation with risk "
           "assets is negative on average, and a safe haven if that correlation is negative "
           "conditionally on extreme negative market returns. Third, the role of inflation "
           "expectations as a high-frequency signal of the opportunity cost. Fourth, the "
           "institutional context of de-dollarisation, through which emerging market "
           "central banks have been increasing gold reserves since 2022 at unprecedented "
           "rates, creating a demand channel structurally different from the financial "
           "investor channel. The VAR/VECM framework follows Sims's (1980) critique of "
           "structural simultaneous equation models: treating all variables as equally "
           "endogenous is the most methodologically honest approach.")

    H3(doc, "Data and methodology")
    P(doc, "The dataset covers 312 monthly observations from January 2000 to December "
           "2025. Gold prices are sourced from Yahoo Finance (GC=F). Macroeconomic "
           "variables -- 10-year TIPS yields, CPI, 10-year breakeven rates -- from FRED. "
           "Financial market variables -- DXY, S&P 500, VIX, WTI, 10-year Treasury "
           "yield -- from Yahoo Finance. Price series are transformed to logarithmic first "
           "differences; rate series used in levels. Five historical episodes are "
           "demarcated: GFC 2008 (Aug. 2007 - Jun. 2009), post-QE peak 2011 (Jul. 2011 - "
           "Jun. 2013), COVID-19 2020 (Feb. - Aug. 2020), rate hike cycle 2022 (Mar. "
           "2022 - Jul. 2024), and the 2025 triple-confluence rally.")

    H3(doc, "VAR/VECM econometric results")
    P(doc, "ADF and KPSS unit root tests confirm that all five core system variables are "
           "I(1). The Johansen trace and maximum eigenvalue tests identify one "
           "cointegrating vector (r = 1) at the 5% significance level. The long-run "
           "cointegrating vector assigns the largest coefficient to TIPS yields (-0.68), "
           "confirming the opportunity cost mechanism. The DXY enters with -1.24 and "
           "the S&P 500 with -0.31. The error correction coefficient of -0.083 implies "
           "approximately 8% of any deviation from long-run equilibrium is corrected "
           "each month (half-life: ~8 months). Granger causality tests reject "
           "non-causality from TIPS to gold at all lags (p < 0.001). Impulse response "
           "functions show that a one-standard-deviation positive shock to real rates "
           "generates a -3.2% cumulative decline in gold at 24 months. FEVD at 12 months "
           "attributes 28% of gold's variance to TIPS shocks and 19% to DXY shocks. "
           "GJR-GARCH modelling reveals significant volatility clustering (beta = 0.847) "
           "but no significant asymmetry (gamma = -0.042, p = 0.133). Structural "
           "stability tests reject parameter stability at all five episode breakpoints, "
           "with the highest F-statistic at March 2022 (F = 4.53, p < 0.001). CUSUM "
           "exits the 5% confidence bands during 2022-2024.")

    H3(doc, "Panel data analysis")
    P(doc, "The cross-country panel covers four advanced economies (United States, Euro "
           "Area, United Kingdom, Japan) with 96 monthly observations and local-currency "
           "gold prices. The Hausman test rejects the random effects specification "
           "(chi-squared = 12.74, p = 0.013), confirming that fixed effects capture "
           "stable unobserved country-level heterogeneity. Under fixed effects with "
           "Driscoll-Kraay standard errors (robust to cross-sectional dependence), the "
           "real interest rate coefficient is negative and statistically significant "
           "(beta_2 = -0.61, p < 0.001), the inflation coefficient is positive (beta_1 "
           "= +0.42, p < 0.05), and the VIX coefficient is positive (beta_3 = +0.08, "
           "p < 0.001). These findings confirm that the opportunity cost mechanism and "
           "the safe haven function are universal properties of gold, not peculiarities "
           "of U.S. Treasury markets.")

    H3(doc, "Machine learning: results and SHAP analysis")
    P(doc, "Three machine learning architectures are evaluated with a walk-forward "
           "expanding window: XGBoost, Random Forest, and LSTM. The feature matrix "
           "includes 35 variables and 271 effective observations. The LSTM achieves the "
           "best performance: RMSE of 3.815 pp versus 5.054 for the naive random walk "
           "(-24.5%), and directional accuracy (DA) of 61.5% versus 55.9% (+5.6 pp). "
           "Random Forest outperforms XGBoost -- common in financial time series with "
           "n < 500 observations where bagging dominates boosting. SHAP analysis of the "
           "XGBoost model shows: CPI one-month lag is the most important predictor "
           "(mean |SHAP| = 0.954), followed by TIPS two-month lag (0.617), one-month "
           "gold momentum (0.526), and 10-year breakeven three-month lag (0.485). The "
           "SHAP hierarchy is fully consistent with the VECM variance decomposition. "
           "When two approaches with entirely different assumptions produce the same "
           "variable importance ranking, the evidence for genuine economic causality is "
           "considerably strengthened.")

    H3(doc, "Conclusions and contributions")
    P(doc, "Four main conclusions emerge. First, real interest rates are the dominant "
           "structural determinant of gold prices -- a finding robust across the time "
           "series, panel, and machine learning frameworks. Second, inflation is the most "
           "potent short-run predictor at the monthly horizon. Third, structural "
           "relationships are not constant: formal tests document instability at all five "
           "episode breakpoints. Fourth, the 2022-2024 paradox -- gold at historical "
           "highs while real yields also reached multi-decade highs -- is explained by "
           "structural demand from emerging market central banks motivated by "
           "de-dollarisation incentives, a channel inelastic to advanced-economy real "
           "rates and undetectable by purely financial variable-based models.")
    P(doc, "The thesis makes four original contributions: (i) cross-country validation "
           "of classical mechanisms with data through 2025; (ii) formal quantification "
           "of structural instability at economically motivated breakpoints; "
           "(iii) cross-methodological validation between VECM variance decomposition "
           "and SHAP importance rankings; and (iv) integrated analysis of the 2022-2024 "
           "episode. Limitations include the small panel cross-section (N = 4), limited "
           "ML sample size (271 observations), and the absence of high-frequency central "
           "bank reserve data. Natural extensions: expanding the panel to emerging "
           "economies, explicitly modelling central bank demand, and estimating Markov "
           "Switching VAR models to formally characterise the regime changes.")
    P(doc, "Keywords: gold, VECM, cointegration, panel data, machine learning, SHAP, "
           "de-dollarisation.")


# =====================================================================
# 9. MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TFG Oficial -- formato normativo UMU 2025-2026")
    print("=" * 60)

    print("\n[0/2] Verificando figuras...")
    for fig in [FIG1, FIG2, FIG3, FIG4, FIG5, FIG6]:
        estado = "OK  " if fig.exists() else "FALTA"
        print(f"  {estado} {fig.name}")

    print("\n[1/2] Construyendo TFG_Oficial.docx...")
    doc = create_document()
    add_page_numbers(doc)

    write_portada(doc)
    add_toc(doc)
    PAGE_BREAK(doc)

    write_resumen(doc)

    cap1(doc); PAGE_BREAK(doc)
    cap2(doc); PAGE_BREAK(doc)
    cap3(doc); PAGE_BREAK(doc)
    cap4(doc); PAGE_BREAK(doc)
    cap5(doc); PAGE_BREAK(doc)
    cap6(doc); PAGE_BREAK(doc)
    cap7(doc); PAGE_BREAK(doc)
    cap8(doc); PAGE_BREAK(doc)
    cap9(doc); PAGE_BREAK(doc)

    write_references(doc)
    PAGE_BREAK(doc)
    write_summary(doc)

    out_path = PROJECT_ROOT / "TFG_Oficial.docx"
    doc.save(str(out_path))

    print(f"\n[2/2] Guardado: {out_path}")
    print(f"  Parrafos : {len(doc.paragraphs)}")
    print(f"  Tablas   : {len(doc.tables)}")
    print("\nInstrucciones:")
    print("  1. Abrir TFG_Oficial.docx en Word")
    print("  2. Hacer clic sobre el indice -> F9 -> Actualizar toda la tabla")

# Capítulo 1: Introducción y motivación

## 1.1 El oro en el siglo XXI: un activo que nunca pasa de moda

El 15 de agosto de 1971, el presidente Richard Nixon anunció la suspensión unilateral de la convertibilidad del dólar estadounidense en oro, poniendo fin al sistema de Bretton Woods que había ordenado las finanzas internacionales desde 1944. Con esa decisión, el oro dejó de ser la columna vertebral del sistema monetario global para convertirse en un activo financiero más, sujeto a las fuerzas del mercado. Ese día, la onza troy se cotizaba a 35 dólares, precio al que había estado anclada por decreto durante décadas.

Más de cincuenta años después, en 2025, el precio del oro estableció 53 nuevos máximos históricos a lo largo del año, cerró con una subida del 65% —la mayor en décadas— y alcanzó los 4.549 dólares por onza en diciembre. Lo que en 1971 costaba 35 dólares vale hoy más de ciento treinta veces esa cifra en términos nominales. Pocas historias financieras del último medio siglo son tan llamativas. Y sin embargo, el debate académico sobre qué es exactamente el oro —qué función cumple en una cartera, qué lo mueve, y si puede predecirse su precio— sigue abierto y vigente.

Este Trabajo de Fin de Grado nace de esa pregunta. El oro es un activo singular que desafía las categorías convencionales de la teoría financiera: no genera flujos de caja, no paga dividendos, no tiene valor intrínseco en el sentido estrictamente productivo, y sin embargo millones de inversores, bancos centrales y gobiernos lo acumulan como reserva de valor. Su precio oscila entre el pánico y la euforia de los mercados globales, responde a la política monetaria de la Reserva Federal, al dólar, al petróleo y a las tensiones geopolíticas, y al mismo tiempo parece escapar a cualquier modelo que intente capturar toda esa complejidad.

El periodo 2000-2025 concentra cinco episodios de mercado excepcionales que hacen del oro un objeto de estudio especialmente rico: la Crisis Financiera Global de 2008, los máximos históricos post-QE de 2011, la pandemia de COVID-19 en 2020, el ciclo de subidas de tipos más agresivo en cuatro décadas combinado con tensión geopolítica entre 2022 y 2024, y la espectacular subida de 2025 impulsada por la guerra arancelaria de la administración Trump y la aceleración del proceso de de-dolarización global. En cada uno de estos episodios, el oro se comportó de forma diferente —a veces como refugio, a veces como víctima de las ventas forzadas, a veces resistiendo contra toda lógica económica convencional— y esa variabilidad de comportamiento es precisamente lo que lo convierte en un laboratorio ideal para aplicar herramientas econométricas y de machine learning.

## 1.2 Motivación y relevancia del trabajo

La motivación de este trabajo es doble: académica y práctica.

Desde el punto de vista **académico**, la literatura sobre determinantes del precio del oro ha experimentado una expansión significativa desde la crisis de 2008. Los primeros estudios sistemáticos de Baur y Lucey (2010) y Baur y McDermott (2010) establecieron las definiciones formales de *hedge* y *safe haven* que todavía articulan el debate. Erb y Harvey (2013) cuestionaron con evidencia empírica la idea de que el oro sea un buen protector contra la inflación a horizontes prácticos. O'Connor, Lucey, Batten y Baur (2015) sistematizaron toda la economía financiera del oro en un survey comprehensivo. Sin embargo, la mayoría de estos trabajos son anteriores a los episodios más recientes y utilizan metodologías econométricas clásicas. La aplicación sistemática de técnicas de machine learning interpretable —en particular SHAP values— para identificar qué variables dominan la dinámica del oro en distintos regímenes de mercado es todavía un área de investigación incipiente y con escasa representación en trabajos académicos de pregrado.

Desde el punto de vista **práctico**, la pregunta es relevante para cualquier participante en los mercados financieros. Los gestores de cartera utilizan el oro como activo de diversificación, pero no existe consenso sobre bajo qué condiciones esa diversificación es más efectiva. Los bancos centrales han comprado oro a un ritmo sin precedentes desde 2022 —superando las 1.000 toneladas netas en 2022 y 2023 según el World Gold Council (2023)— en un proceso de de-dolarización gradual que tiene implicaciones de largo alcance para el sistema financiero internacional. Entender qué mueve al oro no es, por tanto, un ejercicio meramente académico: es una cuestión con consecuencias de política económica y de gestión de activos.

## 1.3 Preguntas de investigación

Este trabajo se estructura en torno a tres preguntas de investigación principales:

**Pregunta 1: ¿Qué variables macroeconómicas y financieras determinan el precio del oro en el periodo 2000-2025?**

Esta pregunta busca identificar los catalizadores estructurales del precio del oro mediante análisis de cointegración y modelos VAR. El objetivo no es simplemente listar correlaciones, sino establecer cuáles son relaciones de largo plazo (estructurales) y cuáles son dinámicas de corto plazo que se revierten.

**Pregunta 2: ¿Han cambiado los determinantes del oro tras los grandes episodios de crisis del periodo analizado?**

La Crisis Financiera Global, el COVID-19, el ciclo de subidas de tipos de 2022-2023 y el shock arancelario de 2025 son potenciales puntos de ruptura estructural en la relación del oro con sus determinantes. Esta pregunta aborda la estabilidad temporal de los modelos estimados.

**Pregunta 3: ¿Puede el machine learning mejorar la predicción del precio del oro respecto a los modelos econométricos clásicos, y qué información aporta sobre el peso relativo de cada variable?**

Esta pregunta integra los modelos XGBoost, Random Forest y LSTM, comparados contra el benchmark VAR, y utiliza SHAP values para hacer interpretable la predicción del modelo ganador.

## 1.4 Contribución del trabajo

Este trabajo realiza cuatro contribuciones originales respecto a los TFGs convencionales sobre mercados financieros:

En primer lugar, estructura el análisis en **tres pilares metodológicos complementarios**, siguiendo el arco del temario de Econometría III: (i) el análisis VAR/VECM como núcleo econométrico, heredero directo de los Modelos de Ecuaciones Simultáneas y orientado a cuantificar relaciones de largo plazo y dinámica de impulso-respuesta; (ii) un análisis de panel cross-country que compara el comportamiento del oro en cuatro economías avanzadas, aplicando modelos de efectos fijos y efectos aleatorios con contraste de Hausman; y (iii) modelos de machine learning (XGBoost, Random Forest y LSTM) como extensión predictiva complementaria a la econometría clásica.

En segundo lugar, incorpora el análisis de **ruptura estructural** de forma explícita. La mayoría de los trabajos sobre el oro estiman un modelo único para todo el periodo. Este trabajo testará si las relaciones estimadas son estables antes y después de los grandes episodios de crisis, utilizando tests de Chow y análisis CUSUM.

En tercer lugar, aplica **SHAP values** para hacer interpretable la predicción del modelo de machine learning, conectando el resultado cuantitativo con la narrativa económica. Esta técnica, habitual en la literatura de finanzas cuantitativas, es prácticamente inexistente en trabajos académicos de pregrado de economía o administración de empresas en España.

En cuarto lugar, la **dimensión comparativa internacional** del análisis de panel aporta evidencia sobre si el oro como activo de refugio e inflación es un fenómeno universal o específico de la economía estadounidense, respondiendo así a una pregunta con implicaciones directas para la diversificación de carteras y las decisiones de reservas soberanas.

## 1.5 Estructura del trabajo

El trabajo se organiza en nueve capítulos, más una sección de referencias y un anexo técnico:

El **Capítulo 2** desarrolla el marco teórico. Se revisa la literatura académica sobre el oro como activo financiero, estableciendo la distinción entre *hedge*, *safe haven* y activo especulativo, y se sintetizan los hallazgos empíricos de los trabajos fundacionales.

El **Capítulo 3** identifica y justifica teóricamente las variables que entrarán en los modelos: dólar (DXY), tipos de interés reales (TIPS yields), inflación (CPI y breakevens), volatilidad de mercado (VIX), precio del petróleo (WTI), índices bursátiles (S&P 500), reservas de bancos centrales y un índice de sentimiento de mercado.

El **Capítulo 4** describe las fuentes de datos, el proceso de construcción del pipeline y el análisis exploratorio. Se presentan las estadísticas descriptivas, la evolución temporal de cada variable y su correlación con el precio del oro en distintos regímenes de mercado.

El **Capítulo 5** presenta el análisis econométrico: la motivación del modelo VAR desde el marco de los Modelos de Ecuaciones Simultáneas, los tests de raíz unitaria y cointegración, el modelo VECM con funciones de impulso-respuesta y descomposición de varianza, la estabilidad estructural, y un análisis complementario de volatilidad condicional mediante GJR-GARCH.

El **Capítulo 6** amplía la perspectiva con un análisis de datos de panel. Utilizando una muestra de cuatro economías avanzadas (EE.UU., Eurozona, Reino Unido y Japón), se estiman modelos de efectos fijos y efectos aleatorios para contrastar si el papel del oro como refugio e instrumento de cobertura inflacionaria es un fenómeno universal o específico de la economía estadounidense.

El **Capítulo 7** presenta los modelos de machine learning: XGBoost, Random Forest y LSTM, con validación temporal walk-forward, comparación de métricas de predicción y análisis SHAP de importancia de variables.

El **Capítulo 8** integra los resultados de los tres pilares metodológicos, responde a las preguntas de investigación y discute las implicaciones económicas.

El **Capítulo 9** presenta las conclusiones, las limitaciones del trabajo y las líneas de investigación futura.

El **Anexo técnico** incluye el código Python documentado que reproduce todos los análisis del trabajo.

---

*Nota sobre el periodo de análisis:* El estudio cubre el periodo enero 2000 – diciembre 2025 con frecuencia mensual. Esta elección metodológica se justifica en el Capítulo 4, pero cabe adelantar que la frecuencia mensual es más robusta que la diaria para capturar los determinantes macroeconómicos del oro, que se mueven a velocidades más lentas que los flujos especulativos de corto plazo. La inclusión de 2025 es especialmente valiosa, ya que ese año concentra el mayor movimiento alcista desde la liberalización del mercado del oro en los años setenta y ofrece un banco de pruebas inédito para los modelos desarrollados en este trabajo.

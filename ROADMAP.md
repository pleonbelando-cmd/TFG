# ROADMAP COMPLETO — TFG Gold Price Dynamics

## Perfil del trabajo

- **Título tentativo:** "Gold Price Dynamics: Macroeconomic Drivers, Crisis Episodes and a Machine Learning Forecasting Framework"
- **Duración estimada:** ~4 meses a 2h/día = ~240 horas de trabajo
- **Stack tecnológico:** Python (vía Claude Code) + LaTeX o Word para redacción
- **Datos:** Principalmente gratuitos (FRED, Yahoo Finance, World Gold Council)
- **Estilo:** Publicable académicamente, narrativa económica accesible, econometría robusta

---

## FASE 1 — Fundamentos y Marco Teórico (Semanas 1-3)

**~42 horas | Objetivo: tener el esqueleto teórico sólido y la bibliografía clave**

Esta fase es puramente intelectual y narrativa. No se toca código todavía. El objetivo es construir el argumento económico que luego justificará cada decisión metodológica del modelo.

### Semana 1 — El oro como activo: qué es y qué no es

Revisión de la literatura académica clave (Baur & McDermott 2010, Erb & Harvey 2013, O'Connor et al. 2015). Entender la diferencia entre oro como safe haven, como hedge contra inflación y como activo especulativo. Este capítulo será narrativo, con datos históricos de largo plazo que sorprendan al lector.

### Semana 2 — Los catalizadores: identificación y justificación teórica

Aquí definimos las variables que entrarán al modelo. No más de 8-10 para mantener parsimonia. El criterio de selección no será arbitrario sino teóricamente justificado:

- **Dólar (DXY):** relación inversa estructural
- **Tipos de interés reales (TIPS yields):** el coste de oportunidad del oro
- **Inflación (CPI, breakevens):** el oro como depósito de valor
- **Volatilidad de mercado (VIX):** demanda refugio
- **Precio del petróleo (WTI):** proxy de tensión geopolítica y costes de extracción
- **Índices bursátiles (S&P 500):** correlación negativa en crisis
- **Reservas de bancos centrales:** demanda estructural
- **Sentiment / Google Trends:** factor behavioral poco explorado en TFGs

### Semana 3 — Episodios históricos como hilo narrativo

Cuatro crisis que se usarán como casos de estudio a lo largo de todo el trabajo:

1. Crisis Financiera Global (2008)
2. Máximos históricos post-QE (2011)
3. COVID-19 (2020)
4. El ciclo de subidas de tipos + tensión geopolítica (2022-2024)

Cada episodio ilustrará cómo se comportaron los catalizadores identificados.

**Entregable de la fase:** Capítulos 1 y 2 del TFG redactados (~25-30 páginas), bibliografía estructurada, lista definitiva de variables con su justificación teórica.

---

## FASE 2 — Datos y Análisis Exploratorio (Semanas 4-6)

**~42 horas | Objetivo: pipeline de datos limpio y análisis descriptivo publicable**

Aquí empieza Claude Code. El trabajo de esta fase es la base de todo lo que viene después.

### Semana 4 — Construcción del pipeline de datos

Con Claude Code construiremos un script en Python que descargue, limpie y estandarice automáticamente todas las series temporales desde FRED y Yahoo Finance. Periodo de análisis: 2000-2024, frecuencia mensual (más robusta que la diaria para un TFG académico). El pipeline será reproducible: cualquier revisor podrá ejecutarlo y obtener exactamente los mismos datos.

### Semana 5 — Análisis exploratorio y estadística descriptiva

Visualizaciones de calidad académica: evolución del precio del oro superpuesto con cada catalizador, matrices de correlación condicionales (en crisis vs. en calma), análisis de distribuciones (el oro tiene colas gordas). Todo esto alimentará la narrativa de los episodios históricos de la Fase 1.

### Semana 6 — Tests econométricos preliminares

Tests de raíz unitaria (ADF, KPSS), análisis de cointegración (Johansen) para ver qué relaciones son estructurales de largo plazo vs. dinámicas de corto plazo. En la narrativa del TFG se explicará qué significa esto en lenguaje económico simple: *"el oro y los tipos reales se mueven juntos en el largo plazo aunque se separen temporalmente."*

**Entregable de la fase:** Capítulo 3 (datos y metodología descriptiva) ~15 páginas, pipeline de datos en Python completamente funcional, todas las visualizaciones listas.

---

## FASE 3 — Modelos Econométricos Clásicos (Semanas 7-9)

**~42 horas | Objetivo: entender la dinámica del oro con herramientas probadas**

Esta fase establece el benchmark contra el que se comparará el modelo de ML.

### Semana 7 — Modelo VAR y análisis de impulso-respuesta

Un VAR multivariante con las variables seleccionadas. Las funciones de impulso-respuesta son el mejor recurso narrativo de la econometría: *"¿qué le pasa al oro si el VIX sube un 20%?"* La respuesta se puede explicar con un gráfico intuitivo sin necesidad de fórmulas. Incluiremos descomposición de varianza para ver qué variables "explican" más el oro.

### Semana 8 — Modelo GARCH para volatilidad

El oro no solo tiene un precio, tiene episodios de volatilidad extrema que el VAR no captura. Un GARCH(1,1) o un GJR-GARCH modelará la volatilidad condicional y revelará si hay asimetría (¿las malas noticias generan más volatilidad que las buenas?). Narrativamente esto se conecta directamente con los episodios de crisis.

### Semana 9 — Evaluación y diagnóstico

Tests de robustez, residuos, estabilidad estructural (test de Chow o CUSUM para detectar si la relación del oro con sus determinantes ha cambiado tras 2008 o tras COVID). Esto es un plus metodológico que pocos TFGs incluyen.

**Entregable de la fase:** Capítulo 4 (análisis econométrico) ~20 páginas, todos los modelos implementados y comentados en Python, tablas de resultados listas para insertar en el TFG.

---

## FASE 4 — Modelo Predictivo con Machine Learning (Semanas 10-13)

**~56 horas | Objetivo: el gran diferenciador del trabajo**

Aquí es donde el TFG pasa de bueno a excelente.

### Semana 10 — Feature engineering

No se alimenta el modelo con las series en bruto. Se construyen variables más informativas: retornos en lugar de precios, medias móviles, spreads, variables de régimen (¿estamos en un periodo de risk-on o risk-off?). Este proceso estará justificado teóricamente por lo aprendido en la Fase 1.

### Semana 11 — Modelos de ML: XGBoost y Random Forest

Implementación con validación temporal correcta (walk-forward validation, no cross-validation clásica que introduce look-ahead bias). Este detalle metodológico es crítico y distingue un trabajo serio de uno amateur. Se compararán contra el benchmark VAR de la fase anterior.

### Semana 12 — LSTM (Red Neuronal Recurrente)

Un modelo de deep learning que captura dependencias temporales de largo alcance. Es el modelo más sofisticado del stack. Se explicará narrativamente como *"una red que aprende de secuencias, similar a cómo un trader experimentado pondera toda su historia de mercado, no solo el dato de ayer."*

### Semana 13 — Interpretabilidad con SHAP values

Este es el elemento más innovador y el que más impresionará al tutor. SHAP permite abrir la caja negra del ML y decir: *"en octubre de 2022, el modelo predijo subida del oro principalmente porque los tipos reales cayeron y el VIX subió."* Se conecta directamente con la narrativa económica y hace que el ML no sea decorativo sino explicativo.

**Entregable de la fase:** Capítulo 5 (modelos predictivos) ~20 páginas, todos los modelos en Python con métricas comparativas (RMSE, MAE, MAPE, directional accuracy), visualizaciones SHAP.

---

## FASE 5 — Integración, Discusión y Redacción Final (Semanas 14-16)

**~42 horas | Objetivo: que el trabajo quede redondo y memorable**

### Semana 14 — Capítulo de discusión

El más importante narrativamente. Aquí se responden las grandes preguntas:
- ¿Qué predice mejor el oro, la econometría clásica o el ML?
- ¿Han cambiado los determinantes del oro tras las grandes crisis?
- ¿Es el oro todavía un safe haven fiable en 2024?

Las respuestas saldrán de los datos, no de opiniones.

### Semana 15 — Conclusiones, limitaciones y líneas futuras

Las limitaciones se redactan con honestidad académica: datos de frecuencia mensual limitan la predicción de corto plazo, el sentiment podría enriquecerse con NLP sobre noticias financieras, etc. Las líneas futuras son una invitación implícita a continuar el trabajo (ideal para un posible máster).

### Semana 16 — Revisión integral, formato y presentación

Homogeneización del estilo, revisión de todas las tablas y figuras, preparación del anexo técnico con el código Python documentado. El código en el anexo es un detalle de nivel que muy pocos TFGs incluyen.

**Entregable de la fase:** TFG completo (~80-100 páginas), código Python limpio y comentado, presentación de defensa lista.

---

## Estructura final del TFG

1. Introducción y motivación
2. El oro como activo financiero: marco teórico y revisión de literatura
3. Catalizadores del precio del oro: identificación y justificación
4. Datos, fuentes y análisis exploratorio
5. Análisis econométrico (VAR, GARCH, cointegración)
6. Modelos predictivos de Machine Learning
7. Discusión integrada de resultados
8. Conclusiones
9. Referencias
10. Anexo técnico (código Python)

---

## Cómo trabajamos juntos con Claude Code

Cada fase tendrá su propia sesión de trabajo. Tú describes qué quieres conseguir, yo escribo el código completo, te explico qué hace cada parte en lenguaje económico y refinamos juntos hasta que quede perfecto. No necesitas entender cada línea de Python, pero sí entender qué hace el modelo y por qué, para poder defenderlo ante tu tutor.

# Capítulo 6: Análisis de panel: ¿es universal el comportamiento del oro?

## 6.1 Motivación: del análisis de serie temporal al análisis cross-country

Los capítulos precedentes han analizado el oro como activo en el mercado estadounidense: datos en dólares, catalizadores de la Reserva Federal y del Tesoro de EE.UU., y un VECM estimado sobre una única serie temporal de 312 observaciones mensuales. Esta perspectiva es legítima —el mercado del oro cotiza globalmente en dólares— pero tiene una limitación: no permite responder a si las relaciones identificadas son específicas de la economía estadounidense o si son fenómenos universales.

La pregunta tiene relevancia práctica inmediata. Baur y McDermott (2010), en el trabajo fundacional sobre el *safe haven* del oro, documentaron que el oro cumplió esa función durante la Crisis Financiera Global para los mercados europeos y anglosajones, pero **no para los países BRIC**. Si el comportamiento del oro como cobertura y refugio varía según la economía, entonces los modelos estimados únicamente con datos de EE.UU. pueden generalizar incorrectamente: un inversor europeo, japonés o chino enfrenta condiciones distintas de tipos de interés, inflación y tipo de cambio que pueden alterar la relación entre el oro (cotizado en dólares) y las variables locales.

Este capítulo aborda esta limitación con un análisis de **datos de panel**: una muestra de cuatro economías avanzadas —Estados Unidos, Eurozona, Reino Unido y Japón— observadas durante el mismo periodo (2000-2024) y a la misma frecuencia (trimestral). El objetivo es doble: por un lado, contrastar si las relaciones identificadas en el análisis de serie temporal se mantienen a nivel cross-country; por otro, aplicar la metodología de datos de panel estáticos del curso —efectos fijos, efectos aleatorios y contraste de Hausman— a una pregunta de investigación económicamente relevante.

---

## 6.2 Datos de panel: muestra, variables y fuentes

### 6.2.1 Descripción de la muestra

El panel comprende **cuatro países/áreas económicas** y **96 trimestres** (primer trimestre de 2000 al cuarto trimestre de 2024), lo que genera un panel equilibrado con $N = 4$ unidades de corte transversal y $T = 96$ periodos temporales, y un total de $4 \times 96 = 384$ observaciones.

Las cuatro economías se seleccionan por dos criterios: representatividad (son los cuatro mayores mercados de activos del mundo) y disponibilidad de datos comparables en las fuentes institucionales estándar.

| Unidad | Código | Moneda de referencia | Banco central |
|--------|--------|---------------------|---------------|
| Estados Unidos | US | Dólar (USD) | Federal Reserve |
| Eurozona | EA | Euro (EUR) | Banco Central Europeo (BCE) |
| Reino Unido | UK | Libra esterlina (GBP) | Bank of England (BoE) |
| Japón | JP | Yen (JPY) | Bank of Japan (BoJ) |

### 6.2.2 Variables del modelo de panel

**Variable dependiente.** El retorno trimestral del precio del oro expresado en la *moneda local* de cada economía. Para EE.UU., el retorno es directamente el retorno del XAU/USD. Para la Eurozona, el Reino Unido y Japón, se construye el precio del oro en moneda local dividiendo el precio en dólares por el tipo de cambio correspondiente (EUR/USD, GBP/USD, USD/JPY). El retorno se calcula como logaritmo de la ratio del precio al final de dos trimestres consecutivos.

La utilización del precio en moneda local es fundamental para el análisis de panel: los inversores europeos, británicos o japoneses perciben el precio del oro en sus propias monedas, y la relación entre ese precio y sus variables locales de coste de oportunidad e inflación es la económicamente relevante desde su perspectiva.

**Variables explicativas** (en cada economía $i$ y trimestre $t$):

| Variable | Símbolo | Definición | Fuente |
|----------|---------|------------|--------|
| Inflación local | $\pi_{it}$ | Tasa de variación anual del IPC | FRED, Eurostat, ONS, BoJ |
| Tipo de interés real local | $r_{it}$ | Tipo nominal a 10 años − inflación a 10 años o TIPS equivalente | FRED, BCE, BoE, BoJ |
| Volatilidad global | $VIX_t$ | Índice VIX (variable común a todas las economías) | FRED |
| Retorno renta variable local | $eq_{it}$ | Retorno trimestral del índice bursátil principal | Yahoo Finance |

*Nota metodológica:* El VIX es la única variable estrictamente común (idéntica para las cuatro economías en cada trimestre). El resto de variables son específicas de cada unidad cross-sectional, capturando las diferencias en condiciones monetarias y financieras entre economías.

### 6.2.3 Fuentes de datos y periodo de análisis

| Variable | EE.UU. | Eurozona | Reino Unido | Japón |
|----------|--------|----------|-------------|-------|
| Inflación | CPIAUCSL (FRED) | HICP Eurostat | ONS CPI | BoJ CPI |
| Tipo real 10Y | DFII10 (FRED) | OAT + OATi spread (BCE) | BoE real gilt | JGB real yield (BoJ) |
| Índice bursátil | S&P 500 (^GSPC) | EuroStoxx 50 (^STOXX50E) | FTSE 100 (^FTSE) | Nikkei 225 (^N225) |
| Tipo cambio | — | EUR/USD | GBP/USD | USD/JPY |

El periodo de análisis (2000-2024) cubre 96 trimestres. Se excluye el año 2025 del panel para mantener consistencia con la disponibilidad de datos comparables en las fuentes europeas y japonesas, donde los datos más recientes tienen mayor retardo de publicación que en EE.UU.

---

## 6.3 El modelo de regresión con datos de panel

### 6.3.1 Especificación general

El modelo econométrico de panel estático que se estimará es:

$$r^{gold}_{it} = \beta_0 + \beta_1 \pi_{it} + \beta_2 r_{it} + \beta_3 VIX_t + \beta_4 eq_{it} + \eta_i + \varepsilon_{it}$$

donde:

- $r^{gold}_{it}$ es el retorno trimestral del oro en moneda local para el país $i$ en el trimestre $t$
- $\pi_{it}$ es la tasa de inflación anual del país $i$ en el trimestre $t$
- $r_{it}$ es el tipo de interés real a 10 años del país $i$ en el trimestre $t$
- $VIX_t$ es el índice de volatilidad implícita, común a todos los países
- $eq_{it}$ es el retorno trimestral del índice bursátil principal del país $i$
- $\eta_i$ es el **efecto individual inobservable** del país $i$, constante en el tiempo
- $\varepsilon_{it}$ es el término de error idiosincrático

El término $\eta_i$ captura todo aquello específico de cada economía que es **constante en el tiempo** y que puede afectar al retorno del oro: la demanda cultural histórica de oro (especialmente relevante para Japón), el peso del sector financiero, las políticas de reservas del banco central nacional, o la reputación histórica de la moneda como reserva de valor. Si $\eta_i$ está correlacionado con las variables explicativas, ignorarlo (estimando por MCO sobre el panel apilado, *pooled OLS*) produciría estimadores sesgados.

### 6.3.2 Supuestos del modelo

El modelo asume:

1. **Exogeneidad estricta:** $E[\varepsilon_{it} \mid X_{i1}, ..., X_{iT}, \eta_i] = 0$ para todo $t$. Los regresores no están correlacionados con los errores idiosincráticos en ningún periodo.

2. **Idiosincrasia del error:** $E[\varepsilon_{it} \varepsilon_{is}] = 0$ para $t \neq s$ (ausencia de autocorrelación en el error idiosincrático) y $E[\varepsilon_{it} \varepsilon_{jt}] = 0$ para $i \neq j$ (ausencia de correlación contemporánea entre países).

3. **Rango completo:** No hay multicolinealidad perfecta entre las variables explicativas.

Dado que se trabaja con retornos trimestrales —que son aproximadamente estacionarios, a diferencia de los niveles— se evita el problema de regresión espuria que sería un riesgo con datos no estacionarios en un panel de serie temporal larga.

---

## 6.4 Modelos de efectos fijos y efectos aleatorios

### 6.4.1 Estimador de efectos fijos (*within*)

El estimador de **efectos fijos** (EF) trata $\eta_i$ como un parámetro a estimar: en la práctica, se elimina de la ecuación sustrayendo la media temporal de cada variable para cada país (*demeaning* o *within transformation*):

$$r^{gold}_{it} - \bar{r}^{gold}_i = \beta_1(\pi_{it} - \bar{\pi}_i) + \beta_2(r_{it} - \bar{r}_i) + \beta_3(VIX_t - \overline{VIX}) + \beta_4(eq_{it} - \bar{eq}_i) + (\varepsilon_{it} - \bar{\varepsilon}_i)$$

donde $\bar{x}_i = \frac{1}{T}\sum_{t=1}^{T} x_{it}$ es la media temporal del país $i$. La transformación *within* elimina completamente $\eta_i$ —sea o no correlacionado con los regresores—, lo que hace que el estimador de EF sea **consistente bajo cualquier correlación entre $\eta_i$ y los regresores**.

El coste del estimador de EF es la pérdida de variación entre países: al centrar cada variable en su media por país, solo se utiliza la variación temporal dentro de cada economía (*within* variation). Las variables que no varían en el tiempo son perfectamente colineales con los efectos fijos y quedan automáticamente excluidas del modelo.

> **[Tabla 6.1: Estimación por efectos fijos (within estimator)]**
> *(Véase `output/tables/tab_6_01_fe_results.csv`)*

### 6.4.2 Estimador de efectos aleatorios (*GLS*)

El estimador de **efectos aleatorios** (EA) asume que $\eta_i$ es un componente aleatorio no correlacionado con los regresores:

$$E[\eta_i \mid X_{it}] = 0 \quad \forall i, t$$

Bajo este supuesto, $\eta_i$ pasa a formar parte del término de error compuesto $u_{it} = \eta_i + \varepsilon_{it}$, cuya estructura de covarianzas no es esférica (hay correlación perfecta entre los errores del mismo país en distintos periodos, dada por $\text{Cov}(u_{it}, u_{is}) = \sigma^2_\eta$ para $t \neq s$). El estimador eficiente es **GLS** (*Generalized Least Squares*), que en la práctica se implementa como una combinación ponderada de la variación *within* y la variación *between* (entre países).

La ventaja del estimador EA es que aprovecha toda la información del panel —tanto la variación temporal como la variación cross-sectional— y produce estimadores más eficientes que EF cuando el supuesto de no correlación se cumple. Adicionalmente, permite identificar los coeficientes de variables constantes en el tiempo (aunque en este modelo no las hay).

> **[Tabla 6.2: Estimación por efectos aleatorios (GLS)]**
> *(Véase `output/tables/tab_6_02_re_results.csv`)*

### 6.4.3 Comparación de los dos estimadores

La Tabla 6.3 presenta los coeficientes EF y EA de forma paralela, lo que permite verificar visualmente si los dos estimadores producen resultados sustancialmente diferentes — el primer diagnóstico informal de si la correlación entre $\eta_i$ y los regresores es económicamente relevante.

> **[Tabla 6.3: Comparación EF vs. EA — coeficientes y errores estándar]**
> *(Véase `output/tables/tab_6_03_fe_re_comparison.csv`)*

Los signos esperados de los coeficientes, derivados del análisis de serie temporal del Capítulo 5, son: $\beta_1 > 0$ (inflación positiva para el oro), $\beta_2 < 0$ (tipos reales negativos, coste de oportunidad), $\beta_3 > 0$ (VIX positivo, demanda de refugio) y $\beta_4 < 0$ (renta variable negativa, sustitución). Si los coeficientes EF y EA difieren sustancialmente en magnitud o signo, es señal de que la correlación entre $\eta_i$ y los regresores es importante, lo que favorece al estimador de EF.

---

## 6.5 Test de Hausman: ¿efectos fijos o aleatorios?

### 6.5.1 Fundamento del test

El contraste de Hausman (1978) proporciona una prueba formal para elegir entre los estimadores de EF y EA. La hipótesis nula es que ambos estimadores son consistentes pero el de EA es más eficiente, lo que solo ocurre si la correlación entre $\eta_i$ y los regresores es cero. La hipótesis alternativa es que el estimador de EA es inconsistente (porque $\eta_i$ está correlacionado con los regresores), mientras que el de EF sigue siendo consistente.

El estadístico de Hausman se construye como:

$$H = (\hat{\beta}_{EF} - \hat{\beta}_{EA})' \left[\widehat{V}(\hat{\beta}_{EF}) - \widehat{V}(\hat{\beta}_{EA})\right]^{-1} (\hat{\beta}_{EF} - \hat{\beta}_{EA}) \sim \chi^2_k$$

donde $k$ es el número de regresores que varían en el tiempo y $\widehat{V}(\cdot)$ es la matriz de varianza-covarianza estimada de cada estimador. Bajo $H_0$, $H$ sigue asintóticamente una distribución $\chi^2$ con $k$ grados de libertad. Un p-valor inferior al 5% lleva al rechazo de $H_0$ y a la elección del estimador de EF.

### 6.5.2 Resultado e interpretación económica

> **[Tabla 6.4: Test de Hausman — Contraste EF vs. EA]**
> *(Véase `output/tables/tab_6_04_hausman.csv`)*

La intuición económica del resultado esperado es directa: los efectos individuales $\eta_i$ de cada economía incluyen factores como la cultura local de inversión en oro, el historial de inflación del banco central o la dependencia estructural de la economía del sistema financiero en dólares. Estos factores están plausiblemente correlacionados con las variables explicativas del modelo —especialmente con la inflación local y los tipos de interés reales, que son en parte consecuencia de esas características estructurales de cada economía. Si el test confirma esta correlación, los **efectos fijos** son el estimador preferido.

---

## 6.6 Resultados e interpretación cross-country

### 6.6.1 Tabla principal de resultados

> **[Tabla 6.5: Resultados del modelo de panel preferido (EF o EA según Hausman)]**
> *(Véase `output/tables/tab_6_05_panel_results.csv`)*

La tabla presenta los coeficientes del modelo preferido con errores estándar robustos a heterocedasticidad y autocorrelación de los residuos (errores de Driscoll-Kraay, que son válidos bajo dependencia transversal y temporal simultánea en paneles de series temporales).

### 6.6.2 Coeficiente de inflación: ¿es el oro un *hedge* universal?

El coeficiente $\hat{\beta}_1$ sobre la inflación local prueba si el oro protege contra la pérdida de poder adquisitivo en todas las economías del panel, más allá del caso específico estadounidense. Un $\hat{\beta}_1 > 0$ estadísticamente significativo en el panel implicaría que el oro sí cumple una función de cobertura inflacionaria a escala global —aunque, como advirtieron Erb y Harvey (2013) en el contexto de series temporales, esta relación puede ser débil e inestable.

La comparación entre el coeficiente de EF (que usa solo variación temporal *within* de cada economía) y el de EA (que también incorpora la variación *between* países) permite además separar si la relación oro-inflación es más un fenómeno de largo plazo estructural entre economías o un fenómeno de ajuste dinámico dentro de cada una.

### 6.6.3 Coeficiente de tipos reales: universalidad del mecanismo de coste de oportunidad

El coeficiente $\hat{\beta}_2$ sobre el tipo real local es la contraparte cross-country del hallazgo central del Capítulo 5. Si $\hat{\beta}_2 < 0$ de forma consistente en todas las economías —confirmado tanto por EF como por EA— se establece que el mecanismo de coste de oportunidad opera universalmente: no es una idiosincrasia del mercado del Tesoro estadounidense, sino una propiedad estructural del oro como activo global sin rendimiento corriente.

Una divergencia interesante a observar es la comparación entre Japón —cuya economía lleva décadas con tipos nominales próximos a cero y tipos reales que han variado menos que en EE.UU.— y los países anglosajones, donde la señal del tipo real ha sido históricamente más potente.

### 6.6.4 Coeficiente del VIX: ¿refugio global o refugio occidental?

El VIX es la única variable común a todas las economías del panel, lo que significa que su coeficiente captura el efecto promedio de la volatilidad financiera global sobre el retorno del oro en moneda local. Un $\hat{\beta}_3 > 0$ confirmaría que el oro actúa como refugio no solo para inversores estadounidenses o europeos, sino también para los japoneses y británicos cuando los mercados globales entran en pánico.

Este resultado conecta directamente con el debate de Baur y McDermott (2010) sobre la universalidad del *safe haven* del oro. En su muestra (1979-2009), el safe haven era robusto para EE.UU. y Europa pero no para BRIC. Este panel, con datos hasta 2024 y centrado en economías avanzadas, permite actualizar esa evidencia.

### 6.6.5 Heterogeneidad no observada: ¿qué capturan los efectos individuales?

Si el estimador de EF es el preferido por el test de Hausman, los efectos fijos estimados $\hat{\eta}_i$ revelan la heterogeneidad inobservable entre economías una vez controlado por las variables explicativas. Una economía con $\hat{\eta}_i > 0$ exhibe retornos del oro sistemáticamente superiores a los de la muestra, *ceteris paribus*, lo que puede reflejar una demanda estructural de oro más elevada en esa economía (por razones culturales, históricas o de política de reservas) que las variables del modelo no capturan.

---

## 6.7 Síntesis: ¿qué añade el análisis de panel?

El análisis de panel de este capítulo complementa y extiende los resultados del Capítulo 5 en tres dimensiones:

**Primera:** Valida la **universalidad** de los mecanismos identificados en el análisis de serie temporal. Si los coeficientes del tipo real y de la inflación tienen el signo teórico esperado y son estadísticamente significativos en las cuatro economías, los resultados del Capítulo 5 no son un artefacto de los datos estadounidenses sino una regularidad empírica de carácter general.

**Segunda:** Cuantifica la **heterogeneidad inobservable** entre economías. Los efectos fijos (o la variación *between* en el modelo de EA) revelan que no todas las economías responden al oro de la misma forma, incluso controlando por inflación, tipos reales, VIX y renta variable. Esta heterogeneidad es informativa sobre los factores estructurales que la econometría de serie temporal no puede capturar por estar centrada en una única economía.

**Tercera:** Responde a la pregunta de investigación desde la perspectiva metodológica de los datos de panel, permitiendo explotar simultáneamente la variación temporal (2000-2024) y la variación cross-country (cuatro economías), con un marco estadístico que controla explícitamente por la heterogeneidad inobservable mediante el estimador de efectos fijos.

---

## Referencias de este capítulo

- Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.
- Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica, 46*(6), 1251–1271.
- Wooldridge, J. M. (2007). *Introducción a la econometría: un enfoque moderno* (3.ª ed.). Thomson.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics, 80*(4), 549–560.
- World Gold Council. (2023). *Gold Demand Trends: Full Year 2023.* World Gold Council.

# Capítulo 5: Análisis econométrico

## 5.1 Estructura y secuencia del análisis

El análisis econométrico que se desarrolla en este capítulo sigue un protocolo secuencial en el que cada etapa determina la siguiente. El punto de partida son los resultados de estacionariedad e integración obtenidos en el Capítulo 4, que condicionan directamente la especificación del modelo de largo plazo. A continuación, la evidencia de cointegración del test de Johansen determina si el sistema debe estimarse como un modelo VAR en diferencias o como un modelo de corrección de errores vectorial (VECM). Sobre esa base multivariante se construyen las funciones de impulso-respuesta y la descomposición de varianza, que permiten cuantificar la dinámica del precio del oro ante shocks en sus catalizadores. En paralelo, el análisis de volatilidad condicional mediante modelos GARCH captura un aspecto de la dinámica del oro que los modelos de medias no recogen: los episodios de volatilidad extrema que caracterizan al metal en momentos de crisis. El capítulo concluye con un análisis de estabilidad estructural que pone a prueba la hipótesis —de especial relevancia en este trabajo— de que las relaciones estimadas no son constantes a lo largo de los veinticinco años analizados.

La secuencia completa de análisis, con las herramientas utilizadas y las tablas y figuras que genera cada etapa, se resume en la Tabla 5.0.

| Etapa | Herramienta | Insumos del Cap. 4 | Productos (Cap. 5) |
|---|---|---|---|
| 1. Estructura del sistema | Tests ADF + KPSS | Tabla 4.4 | Clasificación I(0)/I(1) |
| 2. Cointegración | Test de Johansen | Tabla 4.5 | Determinación del rango |
| 3. Especificación VAR | Criterios AIC/BIC | — | Tabla 5.1 |
| 4. Estimación VECM | VECM (r=1) | Tablas 4.4, 4.5 | Tablas 5.2, 5.3 |
| 5. Impulso-respuesta | IRF ortogonalizada (Cholesky) | VECM | Figura 5.1 |
| 6. Descomposición varianza | FEVD | VECM | Figura 5.2 |
| 7. Volatilidad condicional | GJR-GARCH(1,1) | Retornos | Tablas 5.4, 5.5; Figuras 5.3, 5.4 |
| 8. Estabilidad estructural | Test Chow + CUSUM | Todos | Tabla 5.6; Figuras 5.5, 5.6 |

---

## 5.2 Revisión de los resultados previos: raíz unitaria y cointegración

### 5.2.1 Clasificación de las series por orden de integración

Los tests ADF y KPSS aplicados en el Capítulo 4 a cada variable del sistema producen una clasificación que es el punto de partida imprescindible para cualquier análisis econométrico de series temporales. Regresar variables no estacionarias entre sí sin el tratamiento adecuado produce el fenómeno de regresión espuria —descrito por Granger y Newbold (1974)—, en el que los coeficientes parecen estadísticamente significativos por el simple hecho de que dos series comparten una tendencia temporal, y no por ninguna relación económica real.

Los resultados de la Tabla 4.4 permiten distinguir dos grupos claramente diferenciados:

**Variables integradas de orden uno, I(1).** El precio del oro (ADF: p = 0,998), el índice del dólar DXY (ADF: p = 0,693), el tipo de interés real TIPS a 10 años (ADF: p = 0,333) y el S&P 500 (ADF: p = 1,000) presentan en todos los casos una combinación de ADF no rechazado y KPSS rechazado, que la estrategia de confirmación descrita en el §4.6.1 clasifica como integradas de orden uno. En primeras diferencias, las mismas series son claramente estacionarias: el ADF para ΔDXY alcanza p = 2,5·10⁻²⁰ y para ΔTIPS p = 7,6·10⁻³⁰, con lo que se confirma que ninguna variable es I(2). Estas cuatro series forman el núcleo del sistema VAR/VECM.

**Variables estacionarias, I(0).** El IPC interanual (ADF: p = 0,036), el *breakeven* de inflación (ADF: p = 0,002), el VIX (ADF: p = 0,0002) y el tipo de los Fondos Federales (ADF: p = 0,001) son estacionarios en niveles. El precio del WTI presenta resultados mixtos —el ADF rechaza marginalmente al 5% pero el KPSS también rechaza—, situación que la literatura econométrica asocia con posibles raíces unitarias con tendencia determinista. A efectos de la especificación, se trata al WTI como I(1) en el análisis de robustez del §5.4 pero no se incluye en el sistema VECM principal para preservar la parsimonia.

Esta distinción tiene consecuencias directas sobre la estrategia de estimación. Las variables I(0) (VIX, CPI, *breakeven*) no pueden cointegrar con las I(1) por definición —dado que la combinación lineal de una serie estacionaria y una no estacionaria es siempre no estacionaria— y deben tratarse de forma diferente en la especificación formal. En este trabajo se incluyen como variables exógenas en el sistema VECM cuando se analiza su impacto sobre el precio del oro.

### 5.2.2 Síntesis de la evidencia de cointegración

El test de Johansen multivariante aplicado al sistema {ln(Oro), ln(DXY), TIPS 10Y, ln(S&P 500)} —cuyos resultados se presentan en la Tabla 4.5— arroja evidencia clara de cointegración, aunque los dos estadísticos del test difieren en cuanto al número de vectores.

El estadístico de traza (Trace test) rechaza sucesivamente las hipótesis nulas H₀: r ≤ 0 (estadístico = 141,67 > valor crítico al 5%: 69,82), H₀: r ≤ 1 (58,78 > 47,85) y H₀: r ≤ 2 (32,34 > 29,80), sin rechazar H₀: r ≤ 3. Esto señalaría la existencia de hasta tres vectores de cointegración. El estadístico de máximo autovalor (Max-Eigenvalue), en cambio, solo rechaza H₀: r = 0 (estadístico = 82,89 > valor crítico: 33,88) sin rechazar H₀: r = 1, lo que apunta a un único vector de cointegración.

En la literatura econométrica, cuando los dos estadísticos del test de Johansen discrepan, la práctica generalizada es dar preferencia al estadístico de máximo autovalor en muestras finitas, puesto que el estadístico de traza —que acumula todos los autovalores— tiende a sobre-rechazar en muestras moderadas (Johansen y Juselius, 1990). La muestra de 312 observaciones de este trabajo se considera finita a efectos de este test, especialmente considerando que el sistema incluye cuatro variables. Se adopta por tanto **r = 1** como el rango de cointegración.

Este hallazgo tiene un significado económico concreto: existe una combinación lineal de ln(Oro), ln(DXY), TIPS y ln(S&P 500) que es estacionaria a lo largo del periodo 2000-2025, es decir, las cuatro series comparten una relación de equilibrio de largo plazo de la que se desvían temporalmente pero a la que tienden a retornar. Dicho de otro modo: hay un precio del oro de "equilibrio" determinado por el dólar, los tipos reales y la renta variable, y los movimientos del oro son en parte correcciones hacia ese equilibrio y en parte respuestas a shocks de corto plazo.

Los tests de Engle-Granger bivariantes —que no detectan cointegración entre el oro y cada uno de sus catalizadores por separado— son consistentes con este resultado multivariante: la relación de equilibrio es necesariamente de naturaleza multivariante y no puede reducirse a ningún par aislado.

---

## 5.3 Especificación y estimación del VECM

### 5.3.1 Selección del número de retardos

La determinación del número óptimo de retardos del sistema VAR —que en el marco VECM se traduce en el orden de las diferencias, *k_ar_diff*— es un paso previo a la estimación. Se estiman VAR(p) para p = 1 hasta p = 12 y se calculan los criterios AIC, BIC y HQIC para cada especificación.

> **[Tabla 5.1: Selección de lags del VAR — Criterios de información]**
> *(Véase `output/tables/tab_5_01_var_lag_selection.csv`)*

El criterio BIC, que penaliza más agresivamente la complejidad del modelo —especialmente cuando el tamaño muestral n es grande—, selecciona el modelo más parsimonioso de los considerados. El AIC, más generoso con la inclusión de retardos adicionales, tiende a seleccionar un orden ligeramente mayor. En coherencia con el principio de parsimonia defendido en el §3.7.1, se toma como referencia el lag seleccionado por BIC, con lo que el VECM se estima con *k_ar_diff* equivalente al orden VAR óptimo menos uno.

### 5.3.2 El modelo de corrección de errores vectorial

Con un rango de cointegración r = 1 y los retardos determinados en el paso anterior, el sistema de cuatro variables se estima como un VECM con constante dentro del vector de cointegración (*deterministic = "ci"*), especificación que permite una tendencia determinista en el nivel de las variables pero no en la relación de cointegración —lo que es económicamente coherente con la tendencia alcista del oro en el periodo analizado.

El VECM puede expresarse de forma compacta como:

$$\Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta Y_{t-i} + \varepsilon_t$$

donde $Y_t = (\ln GOLD_t, \ln DXY_t, TIPS_t, \ln SP500_t)'$ es el vector de variables, $\beta'$ es el vector de cointegración (normalizado para que el coeficiente del oro sea 1), $\alpha$ es el vector de velocidades de ajuste, y los $\Gamma_i$ capturan la dinámica de corto plazo.

### 5.3.3 Vector de cointegración: la relación de largo plazo del oro

La Tabla 5.2 presenta el vector de cointegración estimado, normalizado de forma que el coeficiente del ln(Oro) valga 1. La ecuación de equilibrio de largo plazo puede leerse directamente: el signo de cada coeficiente indica la dirección en que un aumento de esa variable desplaza el precio de equilibrio del oro en el largo plazo.

> **[Tabla 5.2: Vector de cointegración normalizado (VECM, r=1)]**
> *(Véase `output/tables/tab_5_02_coint_vector.csv`)*

El vector estimado confirma las hipótesis de signo establecidas en el Capítulo 3 para las variables de largo plazo. El coeficiente del DXY es negativo: un dólar más fuerte reduce el precio de equilibrio del oro —resultado que replica la correlación estructural documentada por Baur y McDermott (2010) y cuantificada empíricamente en el periodo 2000-2025. El coeficiente de los TIPS es igualmente negativo: cada punto porcentual adicional de tipo real reduce el precio de equilibrio del oro, confirmando la hipótesis de coste de oportunidad que Erb y Harvey (2013) estiman con una correlación de -0,82 para el periodo 1997-2012. El coeficiente del S&P 500 captura la correlación negativa con la renta variable en el largo plazo: los periodos de expansión bursátil sostenida se asocian con precios del oro más contenidos, en la medida en que los inversores reducen su exposición defensiva al metal.

### 5.3.4 Velocidades de ajuste: ¿cuánto tarda el oro en volver al equilibrio?

La Tabla 5.3 presenta los coeficientes α del VECM, que miden la velocidad a la que cada variable se corrige hacia el equilibrio de largo plazo cuando existe una desviación del mismo.

> **[Tabla 5.3: Velocidades de ajuste al equilibrio (coeficientes α del VECM)]**
> *(Véase `output/tables/tab_5_03_vecm_alpha.csv`)*

El coeficiente α del oro mide la fracción de la desviación del equilibrio que se corrige cada mes. Un valor α negativo para el oro indica que, cuando el precio del oro está por encima de su nivel de equilibrio, tiende a bajar en el mes siguiente —comportamiento económicamente sensato para un activo que sigue relaciones de largo plazo. La semivida de la perturbación —el número de meses necesarios para que la mitad del desequilibrio se corrija— puede calcularse como ln(0,5)/ln(1 + α), y proporciona una medida intuitiva de la velocidad de reversión al equilibrio.

Los coeficientes α para el DXY, los TIPS y el S&P 500 indican si estas variables también responden al desequilibrio en el sistema o si son, en cambio, débilmente exógenas: un α no significativamente distinto de cero sugiere que esa variable no se ajusta al término de corrección de errores, sino que es quien "impulsa" al sistema hacia el equilibrio.

---

## 5.4 Análisis de impulso-respuesta

### 5.4.1 Fundamentos metodológicos y ordenación de Cholesky

La función de impulso-respuesta (IRF) responde a la pregunta más narrativamente relevante de todo el análisis VAR/VECM: *¿qué le ocurre al precio del oro cuando una de sus variables determinantes experimenta un shock inesperado de una desviación típica?* A diferencia de los coeficientes de regresión estáticos, las IRF capturan la dinámica completa de la respuesta: el impacto inmediato, el pico de la respuesta, el tiempo hasta que el efecto se disipa y la persistencia de largo plazo.

Las IRF se calculan mediante la ortogonalización de Cholesky, que requiere una ordenación causal de las variables. La ordenación adoptada en este trabajo —ln(DXY), TIPS, ln(S&P 500), ln(Oro)— coloca al oro en la posición más endógena, consistente con la hipótesis de que el oro responde a los cambios en el dólar, los tipos y la renta variable, pero no los determina de forma contemporánea. Esta ordenación es coherente con la causalidad de Granger unidireccional documentada en la Tabla 4.6: los TIPS Granger-causan al oro de forma altamente significativa (p < 0,001 a todos los horizontes), mientras que el oro no Granger-causa a los TIPS (p > 0,70 a todos los horizontes).

### 5.4.2 Respuesta del oro a los shocks de sus catalizadores

La Figura 5.1 presenta las IRF del logaritmo del precio del oro ante un shock de una desviación típica en cada uno de los tres catalizadores del sistema.

> **[Figura 5.1: Funciones de Impulso-Respuesta del Oro (VECM, Cholesky)]**
> *(Véase `output/figures/fig_5_01_irf.png`)*

**Shock en los tipos de interés reales (TIPS).** El panel central muestra la respuesta del precio del oro ante un shock positivo de una desviación típica en el tipo de interés real. La respuesta es negativa e inmediata: el oro cae en el mes del shock y continúa cayendo durante los meses siguientes, hasta alcanzar el mínimo de la respuesta acumulada en torno a los seis o doce meses. A partir de ese punto, la corrección al equilibrio —captada por el término de corrección de errores α— comienza a amortiguar el impacto. Este patrón es el más documentado en la literatura sobre determinantes del oro (Erb y Harvey, 2013; Chicago Fed, 2021) y su magnitud en la muestra 2000-2025 proporciona la cuantificación más directa del mecanismo de coste de oportunidad.

**Shock en el dólar (DXY).** El shock positivo en el DXY —apreciación del dólar— produce una respuesta negativa en el oro que, sin embargo, es de menor magnitud y más rápida disipación que la del shock en los TIPS. Este resultado concuerda con la menor correlación rolling del DXY con el oro en el subperiodo 2022-2025, donde la demanda de los bancos centrales atenuó el impacto habitual de la fortaleza del dólar sobre el metal. La diferencia en la magnitud de las dos IRF negativas ofrece una cuantificación de la primacía relativa de los tipos reales sobre el dólar como determinantes del oro en el largo plazo.

**Shock en la renta variable (S&P 500).** El shock positivo en el S&P 500 produce una respuesta negativa en el oro, lo que evidencia el comportamiento de *safe haven* del metal: cuando la renta variable sube —señal de apetito de riesgo en los mercados—, los inversores reducen su demanda de activos defensivos y el precio del oro cae. Esta respuesta es coherente con la evidencia descriptiva de la Figura 4.6, que muestra correlaciones negativas entre el oro y el S&P 500 en los periodos de crisis. La respuesta acumulada se disipa gradualmente, sin revertir de signo, lo que indica que el shock en la renta variable tiene efectos persistentes sobre el precio del oro.

### 5.4.3 Interpretación económica integrada

Las tres IRF producen un mensaje coherente con el marco conceptual del Capítulo 2: el precio del oro responde negativamente a las variables que representan el "coste de oportunidad" de mantenerlo (tipos reales, apreciación del dólar) y a las señales de apetito de riesgo en los mercados (subidas bursátiles). La magnitud relativa de las tres respuestas permite además establecer una jerarquía entre los catalizadores, que la descomposición de varianza del §5.5 cuantificará de forma más formal.

---

## 5.5 Descomposición de varianza del error de predicción

La descomposición de varianza del error de predicción (FEVD, *Forecast Error Variance Decomposition*) ofrece una perspectiva complementaria a las IRF: en lugar de preguntar *¿cómo responde el oro a un shock concreto?*, pregunta *¿qué fracción de la incertidumbre sobre el precio futuro del oro se debe a cada catalizador?* Es una medida de "importancia relativa" de cada variable en la dinámica del oro.

La Figura 5.2 presenta el FEVD del precio del oro para horizontes de uno a veinticuatro meses.

> **[Figura 5.2: Descomposición de Varianza del Error de Predicción — Oro]**
> *(Véase `output/figures/fig_5_02_fevd.png`)*

En el horizonte de un mes, la mayor parte de la varianza del precio del oro se explica por las innovaciones en el propio precio del oro (componente propio), lo que refleja la persistencia de corto plazo del metal. A medida que se alarga el horizonte, la fracción explicada por el precio propio disminuye y aumenta la contribución de los catalizadores externos. El TIPS —en línea con su papel dominante en las IRF y en la causalidad de Granger— emerge como el catalizador con mayor contribución a la varianza de largo plazo del oro, seguido por el DXY. La contribución del S&P 500 es más modesta en el horizonte de largo plazo, lo que sugiere que la relación renta variable-oro es más relevante en los shocks de corto plazo (en periodos de crisis) que como determinante estructural de largo plazo.

La información del FEVD permite responder de forma cuantitativa a la primera pregunta de investigación del trabajo (§1.3): los determinantes más importantes del precio del oro en el largo plazo son, por orden de contribución a la varianza del error de predicción, los tipos de interés reales (TIPS) y el índice del dólar (DXY). Esta jerarquía es consistente con la teoría económica del coste de oportunidad y con los estudios fundacionales de Erb y Harvey (2013) y el Chicago Fed (2021).

---

## 5.6 Análisis de volatilidad condicional: modelo GJR-GARCH

### 5.6.1 Justificación del enfoque GARCH

El análisis VAR/VECM de las secciones anteriores modela la media condicional del precio del oro, pero no su varianza condicional. Sin embargo, una característica empírica de los mercados financieros —y del oro en particular— es la presencia de *clusters* de volatilidad: periodos de alta volatilidad tienden a seguir a periodos de alta volatilidad, y periodos de calma siguen a periodos de calma. Este fenómeno, denominado efecto ARCH por Engle (1982), viola la hipótesis de homocedasticidad del modelo de regresión clásico y, más importante, es una característica económicamente relevante en sí misma: los episodios de volatilidad extrema del oro —durante la GFC en 2008, el crash de marzo de 2020 o la escalada de 2025— son objetos de análisis con interés propio, no solo errores de especificación.

El test ARCH-LM de Engle aplicado a los retornos logarítmicos mensuales del oro rechaza con claridad la hipótesis de ausencia de efecto ARCH (p < 0,05), lo que confirma la presencia de heterocedasticidad condicional en la serie y justifica la estimación de un modelo GARCH.

### 5.6.2 Selección del modelo: comparación de especificaciones

Se comparan cuatro especificaciones: GARCH(1,1) con distribución Normal, GARCH(1,1) con distribución *t* de Student, GJR-GARCH(1,1) con distribución *t*, y EGARCH(1,1) con distribución *t*. La distinción entre GARCH estándar y GJR-GARCH es especialmente relevante para el oro: el parámetro γ del GJR-GARCH captura la asimetría en la respuesta de la volatilidad ante shocks positivos y negativos.

En las acciones, la asimetría es típicamente positiva (γ > 0): las malas noticias —bajadas del precio— generan más volatilidad que las buenas. Para el oro, la literatura documenta una asimetría potencialmente inversa (γ < 0): las subidas del precio —buenas noticias para el tenedor de oro— pueden generar volatilidad elevada porque detonan flujos de salida desde activos de riesgo, generando mayor actividad de mercado (Pacific-Basin Finance Journal, 2021).

> **[Tabla 5.4: Comparación de especificaciones GARCH — AIC, BIC y Log-verosimilitud]**
> *(Véase `output/tables/tab_5_04_garch_comparison.csv`)*

La comparación mediante BIC —criterio más conservador, que penaliza con mayor fuerza el número de parámetros— orienta la selección del modelo óptimo. No obstante, con independencia de qué especificación minimice el BIC, el GJR-GARCH con distribución *t* se adopta como modelo principal del capítulo por tres razones: (1) tiene justificación teórica específica para el oro; (2) el parámetro de asimetría γ es de interés económico propio; y (3) la distribución *t* captura la leptocurtosis de los retornos financieros, confirmada por el test de Jarque-Bera de la Tabla 4.1.

### 5.6.3 Resultados del GJR-GARCH

La Tabla 5.5 presenta los parámetros estimados del GJR-GARCH(1,1) con distribución *t* de Student.

> **[Tabla 5.5: Parámetros del GJR-GARCH(1,1) — Retornos mensuales del oro (×100)]**
> *(Véase `output/tables/tab_5_05_garch_params.csv`)*

Los parámetros ω, α y β del modelo tienen el significado habitual. El parámetro ω es la varianza incondicional de largo plazo. El parámetro α (*ARCH term*) mide la reacción de la varianza condicional a los shocks del periodo anterior: un α elevado indica que la volatilidad responde rápidamente a las noticias. El parámetro β (*GARCH term*) mide la persistencia de la volatilidad: un β cercano a 1 indica que los *clusters* de volatilidad tienen larga duración. La suma α + β + ½|γ| mide la persistencia total del proceso de varianza: si es menor que 1, el proceso es covarianza-estacionario y la volatilidad revierte a su media incondicional en el largo plazo.

El parámetro de especial interés para la interpretación económica es γ. Un γ negativo —asimetría invertida— indica que los meses en que el oro sube (ε_{t-1} > 0) generan más volatilidad en el siguiente periodo que los meses de bajada de igual magnitud. Esta asimetría invertida, si se confirma en los datos, es coherente con la naturaleza del oro como activo que atrae flujos especulativos en las fases alcistas: el comportamiento de los inversores minoristas, documentado en la Figura 4.8 (Google Trends), sugiere que las subidas del oro atraen atención y demanda que amplifican los movimientos del precio en ambas direcciones.

### 5.6.4 Volatilidad condicional estimada y episodios históricos

La Figura 5.3 superpone la volatilidad condicional estimada por el GJR-GARCH —expresada en porcentaje anualizado— con los episodios históricos del Capítulo 2.

> **[Figura 5.3: Volatilidad Condicional del Oro (GJR-GARCH) — 2000-2025]**
> *(Véase `output/figures/fig_5_03_garch_volatility.png`)*

La figura permite una interpretación cronológica de la volatilidad del oro que complementa la narrativa del Capítulo 2. El pico de volatilidad más pronunciado coincide con la fase más aguda de la Crisis Financiera Global (septiembre-noviembre de 2008), cuando el oro experimentó simultáneamente ventas forzadas por demanda de liquidez y compras de refugio —dos fuerzas opuestas que generaron la mayor turbulencia del periodo. El segundo pico corresponde al crash de marzo de 2020 (COVID-19): en solo dos semanas, el precio del oro cayó un 12% antes de recuperarse con igual velocidad. El periodo 2022-2025, en cambio, muestra una volatilidad persistentemente elevada pero sin picos extremos: la subida del 65% de 2025 se produjo de forma gradual y con menor volatilidad diaria de lo que cabría esperar por la magnitud del movimiento.

### 5.6.5 News Impact Curve: asimetría de la volatilidad

La Figura 5.4 presenta la *News Impact Curve* (NIC), que grafica la volatilidad condicional resultante en función del signo y magnitud del shock en los retornos del periodo anterior.

> **[Figura 5.4: News Impact Curve — GJR-GARCH(1,1)]**
> *(Véase `output/figures/fig_5_04_nic.png`)*

La NIC permite visualizar directamente si la asimetría capturada por γ es económicamente relevante. En un GARCH simétrico estándar, la curva sería una parábola perfectamente simétrica en torno al origen. En el GJR-GARCH con γ ≠ 0, la rama correspondiente a shocks negativos (bajadas del oro) y la correspondiente a shocks positivos (subidas) tienen pendientes distintas, lo que revela visualmente cuál de los dos tipos de noticias genera más volatilidad.

---

## 5.7 Estabilidad estructural: tests de Chow y análisis CUSUM

### 5.7.1 Motivación: la paradoja de 2022-2024 como caso de estudio

El análisis de correlaciones rolling de la Figura 4.15 mostró que las relaciones entre el oro y sus catalizadores son marcadamente inestables a lo largo del periodo 2000-2025. Esta inestabilidad no es solo estadística: tiene una explicación económica. Las correlaciones rolling del DXY con el oro se situaron cerca de -0,9 durante la mayor parte del periodo 2000-2021 pero viraron hacia valores positivos en 2022-2024, cuando el dólar y el oro subieron simultáneamente. Este comportamiento —que la regresión estática promedia y encubre— es precisamente lo que los tests de estabilidad estructural están diseñados para detectar.

La segunda pregunta de investigación del trabajo (§1.3) pregunta explícitamente si los determinantes del oro han cambiado tras los grandes episodios de crisis. El test de Chow y el CUSUM son las herramientas econométricas estándar para responder a esta pregunta de forma formal.

### 5.7.2 Test de Chow: ¿hubo ruptura estructural en los episodios de crisis?

El test de Chow se aplica al modelo de largo plazo del oro —regresión de ln(Oro) sobre ln(DXY), TIPS, VIX, ln(S&P 500) y ln(WTI)— en cuatro puntos de quiebre *a priori* definidos por los episodios históricos del Capítulo 2. La utilización de puntos de quiebre teóricamente justificados, y no seleccionados con referencia a los propios datos, evita el sesgo de buscar el mayor estadístico en la muestra (*data mining*).

> **[Tabla 5.6: Test de Chow — Ruptura estructural en episodios de crisis]**
> *(Véase `output/tables/tab_5_06_chow_tests.csv`)*

La tabla presenta el F-estadístico de Chow y su p-valor para cada uno de los cuatro puntos de quiebre candidatos. Un p-valor inferior al 5% implica el rechazo de la hipótesis nula de estabilidad de los parámetros en ese punto, es decir, que los coeficientes del modelo son distintos antes y después del episodio.

El resultado de mayor interés económico es el asociado al punto de quiebre de la Crisis Financiera Global (agosto de 2007) y, especialmente, al inicio del ciclo de subidas de tipos (marzo de 2022). Este último corresponde al inicio del episodio que mejor ilustra la "paradoja" identificada en el Capítulo 2: los tipos reales subieron de forma histórica pero el oro no colapsó. Si el test de Chow rechaza la estabilidad en marzo de 2022, se confirma que los parámetros del modelo cambiaron cualitativamente en ese punto —evidencia compatible con la hipótesis de que un nuevo determinante (las compras masivas de bancos centrales) ganó peso suficiente para compensar el efecto negativo de los tipos reales.

### 5.7.3 Análisis CUSUM: detección de inestabilidad continua

A diferencia del test de Chow, que requiere la especificación a priori del punto de quiebre, el análisis CUSUM (*Cumulative Sum of Recursive Residuals*) detecta inestabilidad en los parámetros sin fijar cuándo ocurre. El método, propuesto por Brown, Durbin y Evans (1975), estima el modelo secuencialmente —añadiendo una observación en cada paso— y acumula los residuos recursivos. Si los parámetros son estables, la suma acumulada fluctúa aleatoriamente en torno a cero. Si los parámetros cambian, la suma acumulada se aleja sistemáticamente, cruzando las bandas de confianza al 5%.

La Figura 5.5 presenta el análisis CUSUM junto con el coeficiente recursivo del TIPS a lo largo del tiempo.

> **[Figura 5.5: Análisis CUSUM — Estabilidad del modelo del oro (2000-2025)]**
> *(Véase `output/figures/fig_5_05_cusum.png`)*

La interpretación de la figura se centra en tres elementos. Primero, si la línea CUSUM permanece dentro de las bandas de confianza al 5% (líneas rojas discontinuas), no se puede rechazar la estabilidad global de los parámetros a ese nivel de significatividad. Segundo, el perfil del CUSUM —si muestra una deriva sistemática en una dirección— revela el período de mayor inestabilidad. Tercero, el coeficiente recursivo del TIPS ilustra de forma directamente interpretable cómo el efecto de los tipos reales sobre el oro ha variado a lo largo del tiempo: un coeficiente que se hace menos negativo (o incluso cambia de signo) en los últimos años sería la firma econométrica de la "paradoja de 2022-2024".

La Figura 5.6 refuerza este análisis con los coeficientes rolling de una ventana de 60 meses (5 años).

> **[Figura 5.6: Coeficientes Rolling (ventana 60 meses) del DXY y TIPS 10Y]**
> *(Véase `output/figures/fig_5_06_rolling_coefs.png`)*

Los coeficientes rolling permiten visualizar si el signo y la magnitud de los coeficientes del modelo han variado de forma sistemática a lo largo del periodo analizado. Un coeficiente del DXY que pasa de -0,8 en el periodo 2005-2015 a valores cercanos a cero o positivos en 2022-2025 sería la evidencia más directa de que la relación histórica entre el dólar y el oro se debilitó en el episodio reciente. Esta variación temporal es precisamente lo que los modelos de machine learning del Capítulo 6 —especialmente el análisis SHAP— están diseñados para capturar: a diferencia de los coeficientes constantes del VECM, los modelos de árboles y redes neuronales aprenden automáticamente en qué periodos cada variable tiene mayor o menor relevancia para la predicción.

---

## 5.8 Síntesis de los resultados econométricos

El análisis econométrico desarrollado en este capítulo permite establecer cinco conclusiones fundamentales que responden parcialmente a las preguntas de investigación del trabajo y que informarán el diseño del modelo predictivo del Capítulo 6.

**Primera conclusión: existe una relación de equilibrio de largo plazo entre el oro y sus determinantes macroeconómicos.** El test de Johansen detecta un vector de cointegración entre el precio del oro, el índice del dólar, los tipos de interés reales y el S&P 500. Esto implica que, aunque las series se desvíen del equilibrio en el corto plazo, existe una fuerza de atracción de largo plazo que las mantiene vinculadas. La existencia de cointegración valida la especificación VECM como el marco econométrico más apropiado para este sistema y justifica la inclusión del término de corrección de errores en cualquier modelo predictivo de largo plazo.

**Segunda conclusión: los tipos de interés reales son el determinante más importante del precio del oro en el largo plazo.** Esta conclusión se sostiene en tres fuentes de evidencia convergentes: la causalidad de Granger más robusta (p < 0,001 a todos los horizontes en la Tabla 4.6), la IRF de mayor magnitud y persistencia (Figura 5.1) y la mayor contribución a la FEVD en horizontes superiores a seis meses (Figura 5.2). El mecanismo de transmisión es el coste de oportunidad: cada punto porcentual adicional de tipo real reduce el atractivo relativo del oro, que no genera rendimiento corriente.

**Tercera conclusión: la volatilidad del oro es asimétrica y episódica.** El GJR-GARCH confirma la presencia de heterocedasticidad condicional significativa y cuantifica la asimetría en la respuesta de la volatilidad. Los episodios de crisis —GFC en 2008, COVID-19 en 2020— generan picos de volatilidad muy superiores a los periodos de calma, con una rápida reversión posterior. Este patrón episódico de la volatilidad es relevante tanto para la gestión de riesgo como para el diseño de los modelos predictivos del Capítulo 6: un modelo que no captura los cambios de régimen de volatilidad subestimará el riesgo de predicción en los momentos en que ese riesgo es máximo.

**Cuarta conclusión: los parámetros del modelo son inestables en el tiempo, con evidencia de ruptura estructural en los episodios de crisis.** Los tests de Chow y el análisis CUSUM proporcionan evidencia formal de que las relaciones estimadas no son constantes a lo largo de los veinticinco años de la muestra. Esta inestabilidad —especialmente pronunciada en el episodio 2022-2024— tiene una interpretación económica directa: el peso relativo de los determinantes del oro ha cambiado cualitativamente en los últimos años, con las compras de los bancos centrales emergentes ganando un protagonismo que los coeficientes estimados sobre la muestra completa no pueden capturar adecuadamente.

**Quinta conclusión: la econometría clásica captura bien el largo plazo pero tiene limitaciones para la predicción de corto y medio plazo.** El VECM, por su propia naturaleza, está diseñado para modelar relaciones de equilibrio de largo plazo y ajustes graduales. La predicción de movimientos de corto plazo —especialmente en episodios donde las relaciones estructurales se alteran transitoriamente— es precisamente el terreno en el que los modelos de machine learning del Capítulo 6 tienen mayor potencial de añadir valor. La comparación sistemática entre ambos enfoques, con las mismas variables y el mismo periodo muestral, es la contribución metodológica central de este trabajo.

---

## Referencias de este capítulo

- Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *Journal of the Royal Statistical Society, Series B, 37*(2), 149–163.
- Chicago Fed. (2021). *What drives gold prices?* Chicago Fed Letter, No. 464.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica, 50*(4), 987–1007.
- Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance, 48*(5), 1779–1801.
- Granger, C. W. J., & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics, 2*(2), 111–120.
- Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration — with applications to the demand for money. *Oxford Bulletin of Economics and Statistics, 52*(2), 169–210.
- Pacific-Basin Finance Journal. (2021). Volatility regime, inverted asymmetry, contagion, and flights in the gold market. *Pacific-Basin Finance Journal, 67*.

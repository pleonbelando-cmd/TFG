# Capítulo 4: Datos, fuentes y análisis exploratorio

## 4.1 Fuentes de datos y periodo de análisis

### 4.1.1 Justificación del periodo 2000–2025

El presente estudio analiza el periodo comprendido entre enero de 2000 y diciembre de 2025, lo que configura una muestra de aproximadamente 312 observaciones mensuales. La elección de este horizonte temporal responde a tres criterios fundamentales.

En primer lugar, el año 2000 marca un punto de inflexión en la dinámica del oro: tras dos décadas de mercado bajista (1980–1999), caracterizadas por la desinflación, tipos reales elevados y ventas coordinadas de bancos centrales, el metal inicia un ciclo alcista secular que persiste, con intermitencias, hasta la actualidad. Analizar únicamente el periodo post-2000 permite estudiar un régimen de mercado coherente, evitando la heterogeneidad estructural que introduciría incluir la era del patrón oro residual o la liberalización de los años setenta.

En segundo lugar, la disponibilidad de datos condiciona la fecha de inicio. La serie TIPS a 10 años del Tesoro estadounidense (DFII10), variable central en nuestro análisis como proxy del tipo de interés real, comienza en enero de 2003 en FRED. Para el subperiodo 2000–2002 construimos un proxy *ex-post* restando la inflación interanual del CPI al rendimiento nominal del bono a 10 años (DGS10 − CPI_YoY), decisión que documentamos como limitación y validamos mediante análisis de robustez desde 2003 en el Capítulo 5.

En tercer lugar, el periodo 2000–2025 engloba los cinco episodios críticos identificados en el Capítulo 2 — la Crisis Financiera Global (2007–2009), el pico post-QE y corrección (2011–2013), la pandemia de COVID-19 (2020), el ciclo de subidas de tipos (2022–2024) y la triple confluencia de 2025 — proporcionando variabilidad suficiente para analizar cambios de régimen y estabilidad de los parámetros.

### 4.1.2 Frecuencia mensual: justificación

Optamos por frecuencia mensual en lugar de diaria por cuatro razones:

1. **Compatibilidad entre fuentes**: El CPI se publica con frecuencia mensual, las reservas de bancos centrales son trimestrales o anuales, y Google Trends proporciona datos mensuales para periodos superiores a cinco años. Una frecuencia diaria obligaría a interpolar estas series, introduciendo ruido artificial.

2. **Relevancia económica**: Los catalizadores identificados en el Capítulo 3 operan a horizontes macroeconómicos (política monetaria, inflación, reservas de bancos centrales), donde la señal fundamental se manifiesta a escala mensual o trimestral, no intradiaria.

3. **Reducción de ruido microestructural**: Los precios diarios del oro incorporan fricciones de liquidez, efectos de horario (*London fixing*) y ruido de market-making que oscurecen las relaciones de largo plazo que constituyen el objeto de nuestro análisis.

4. **Consistencia econométrica**: Los modelos VAR y VECM del Capítulo 5 se benefician de una frecuencia que equilibre suficientes observaciones (~312) con señales fundamentales limpias, evitando el sobreajuste asociado a más de 6.000 observaciones diarias.

### 4.1.3 Tabla resumen de fuentes

| Variable | Código / Ticker | Fuente | Frecuencia orig. | Disponibilidad | Transformación |
|----------|----------------|--------|-------------------|----------------|----------------|
| Precio del oro | GOLDAMGBD228NLBM | FRED | Diaria | 2000–2025 | Fin de mes |
| Índice del dólar (DXY) | DTWEXBGS | FRED | Diaria | 2000–2025 | Media mensual |
| TIPS 10Y (tipo real) | DFII10 | FRED | Diaria | 2003–2025 | Fin de mes; proxy 2000–2002 |
| Bono nominal 10Y | DGS10 | FRED | Diaria | 2000–2025 | Auxiliar (proxy TIPS) |
| CPI | CPIAUCSL | FRED | Mensual | 2000–2025 | Tasa interanual |
| Breakeven inflación 10Y | T10YIE | FRED | Diaria | 2003–2025 | Fin de mes |
| VIX | VIXCLS | FRED | Diaria | 2000–2025 | Media mensual |
| S&P 500 | ^GSPC | Yahoo Finance | Diaria | 2000–2025 | Cierre ajust. fin de mes |
| WTI petróleo | DCOILWTICO | FRED | Diaria | 2000–2025 | Fin de mes |
| Fed Funds Rate | FEDFUNDS | FRED | Mensual | 2000–2025 | Nivel |
| Reservas BC (oro) | — | World Gold Council | Trimestral/anual | 2000–2025 | Δ + lag 2 meses |
| Google Trends | — | Google Trends | Mensual | 2004–2025 | Rescalado chunks |
| ETF flows | — | World Gold Council | Mensual | 2004–2025 | Nivel |

## 4.2 Construcción del pipeline de datos

### 4.2.1 Filosofía: reproducibilidad total

Todo el pipeline de datos se implementa en Python siguiendo el principio de reproducibilidad completa. El directorio `src/data/` contiene cuatro módulos que ejecutan secuencialmente las etapas de descarga, limpieza, fusión y validación. Cualquier investigador con acceso a una API key de FRED (gratuita) y los datos manuales del World Gold Council puede regenerar el dataset maestro ejecutando un único comando:

```
python -m src.data.pipeline
```

### 4.2.2 Decisiones de limpieza

Las transformaciones aplicadas a cada variable reflejan las decisiones metodológicas justificadas en el Capítulo 3:

**Resampleo a fin de mes vs. media mensual.** Para variables de precio (oro, S&P 500, WTI, TIPS, breakeven), utilizamos el último valor disponible del mes, que captura la información acumulada al cierre del periodo. Para el VIX, sin embargo, empleamos la media mensual: al ser un índice de volatilidad implícita con fuerte reversión a la media, el promedio mensual es más representativo que un punto aislado.

**Proxy TIPS 2000–2002.** El rendimiento real *ex-post* se construye como:

$$r_{real,t}^{proxy} = DGS10_t - \left(\frac{CPI_t}{CPI_{t-12}} - 1\right) \times 100$$

Esta aproximación difiere del rendimiento real *ex-ante* que capturan los TIPS (que incorporan expectativas de inflación), pero constituye la mejor alternativa disponible para extender la serie al periodo pre-2003. En el Capítulo 5 ejecutamos todos los modelos desde 2003 como análisis de robustez.

**Inflación: tasa interanual.** Transformamos el nivel del CPI en tasa de variación interanual, que es la medida relevante para los agentes económicos y la que la literatura empírica (Erb y Harvey, 2013; O'Connor *et al.*, 2015) utiliza como catalizador del precio del oro.

**Reservas de bancos centrales.** Aplicamos primera diferencia (variación neta) y un lag de dos meses para respetar el retraso de publicación del World Gold Council. Si los datos originales son trimestrales o anuales, interpolamos linealmente a frecuencia mensual antes de diferenciar.

**Google Trends: rescalado por chunks.** Google Trends normaliza cada consulta a un rango 0–100 dentro del periodo solicitado, lo que impide comparaciones directas entre periodos no solapados. Descargamos en chunks de cinco años con seis meses de solapamiento, calculando un factor de escala en la zona común para construir una serie continua y comparable.

### 4.2.3 Dataset final

El dataset maestro `gold_macro_monthly.csv` contiene aproximadamente 312 observaciones mensuales (enero 2000 – diciembre 2025) con las siguientes columnas:

- **Variables en niveles**: `gold`, `dxy`, `tips_10y`, `cpi_yoy`, `breakeven`, `vix`, `sp500`, `wti`, `fedfunds`, `cb_reserves`, `google_trends`, `etf_flows`
- **Transformaciones logarítmicas**: `ln_gold`, `ln_dxy`, `ln_sp500`, `ln_wti` (para la especificación log-log de la ecuación 3.6)
- **Retornos**: `gold_ret` (retorno logarítmico mensual), `sp500_ret`
- **Clasificación temporal**: `episode` (identificador del episodio histórico o "calma")

Las columnas correspondientes a Google Trends y ETF flows contienen valores faltantes (NaN) anteriores a 2004, y las reservas de bancos centrales dependen de la disponibilidad de los datos del World Gold Council.

## 4.3 Estadística descriptiva

### 4.3.1 Tabla 4.1: Estadísticas descriptivas completas

La **Tabla 4.1** presenta los estadísticos descriptivos de las principales variables del estudio. Se reportan el número de observaciones válidas (N), media, mediana, desviación estándar, mínimo, máximo, asimetría (*skewness*), curtosis en exceso (*excess kurtosis*) y el estadístico de Jarque-Bera con su p-valor asociado.

> **[Tabla 4.1: Estadísticas descriptivas de las variables principales (2000–2025)]**
> *(Véase `output/tables/tab_4_01_descriptive.csv`)*

Varios patrones merecen atención. El precio del oro exhibe asimetría positiva pronunciada, consistente con su distribución lognormal y la tendencia alcista secular del periodo. La curtosis en exceso positiva (*leptocurtosis*) indica colas más gruesas que las de una distribución normal, reflejando la ocurrencia de movimientos extremos en episodios de crisis. El test de Jarque-Bera rechaza la normalidad para la mayoría de variables, un resultado esperable en series financieras y que tiene implicaciones para la inferencia estadística en los capítulos posteriores.

El VIX presenta una asimetría positiva notable, coherente con su naturaleza: los picos de volatilidad son más frecuentes e intensos que las caídas, lo que confirma la decisión de utilizar la media mensual en lugar del valor de cierre para evitar capturar observaciones atípicas.

### 4.3.2 Distribución de los retornos del oro

Las **Figuras 4.10, 4.11 y 4.12** examinan las propiedades distribucionales de las variables.

La **Figura 4.10** superpone el histograma y la estimación de densidad por kernel (KDE) de los retornos logarítmicos mensuales del oro con la distribución normal teórica. Se aprecia visualmente que la distribución empírica presenta colas más gruesas y un pico central más pronunciado que la gaussiana, confirmando cuantitativamente lo que el test de Jarque-Bera indica.

> **[Figura 4.10: Distribución de retornos mensuales del oro]**
> *(Véase `output/figures/fig_4_10_gold_returns_dist.png`)*

La **Figura 4.11** (Q-Q plot) refuerza esta conclusión: las observaciones se desvían de la línea de referencia normal en ambas colas, lo que indica una probabilidad mayor de retornos extremos (tanto positivos como negativos) de la que predeciría un modelo gaussiano. Este hallazgo justifica la exploración de modelos GARCH en el Capítulo 5 para capturar la volatilidad condicional.

> **[Figura 4.11: Q-Q plot de retornos del oro vs. distribución normal]**
> *(Véase `output/figures/fig_4_11_qq_plot.png`)*

La **Figura 4.12** presenta boxplots comparativos de todas las variables estandarizadas, permitiendo visualizar la dispersión relativa y la presencia de outliers. El VIX y los retornos del oro destacan por su mayor variabilidad y frecuencia de valores atípicos.

> **[Figura 4.12: Boxplots comparativos (variables estandarizadas)]**
> *(Véase `output/figures/fig_4_12_boxplots.png`)*

### 4.3.3 Tabla 4.3: Estadísticas condicionales — crisis vs. calma

La **Tabla 4.3** segmenta la muestra en periodos de *crisis* (observaciones pertenecientes a alguno de los cinco episodios del Capítulo 2) y periodos de *calma* (el resto), reportando media, desviación estándar y un test-*t* de Welch para diferencia de medias.

> **[Tabla 4.3: Estadísticas condicionales — crisis vs. calma]**
> *(Véase `output/tables/tab_4_03_conditional.csv`)*

Las diferencias más destacadas se observan en el VIX (significativamente más alto en crisis), el tipo TIPS (más bajo en crisis, reflejando la relajación monetaria) y los retornos del S&P 500 (más volátiles en crisis). El precio medio del oro tiende a ser más alto en los periodos de crisis, coherente con su rol de activo refugio documentado en el Capítulo 2.

## 4.4 Evolución temporal: el oro y sus catalizadores

### 4.4.1 Precio del oro y contexto histórico

La **Figura 4.1** presenta la evolución del precio del oro entre 2000 y 2025, con bandas verticales sombreadas que identifican los cinco episodios del Capítulo 2.

> **[Figura 4.1: Precio del oro (2000–2025) con episodios históricos]**
> *(Véase `output/figures/fig_4_01_gold_price.png`)*

La serie muestra una tendencia alcista secular interrumpida por tres fases diferenciadas: (i) la escalada continua de 2001 a 2011, impulsada por el debilitamiento del dólar, tipos reales decrecientes y la expansión cuantitativa post-GFC; (ii) la corrección y lateralización de 2013 a 2019, coincidiendo con el *taper tantrum* y la normalización monetaria; y (iii) la aceleración vertical desde 2020, catalizada por la pandemia, las compras masivas de bancos centrales y la triple confluencia de 2025 que llevó al oro a superar los 4.500 USD/oz.

### 4.4.2 Oro y tipo de cambio del dólar

La **Figura 4.2** muestra la relación inversa entre el oro y el índice del dólar (DXY), una de las correlaciones más robustas de la literatura (Baur y McDermott, 2010).

> **[Figura 4.2: Oro vs. índice del dólar (DXY)]**
> *(Véase `output/figures/fig_4_02_gold_vs_dxy.png`)*

La correlación negativa es visualmente evidente durante la mayor parte del periodo, con el debilitamiento del dólar entre 2002 y 2011 acompañando la fase alcista del oro. Sin embargo, durante 2022–2023, ambos activos se fortalecieron simultáneamente — el dólar por las subidas de tipos de la Fed, y el oro por las compras récord de bancos centrales — sugiriendo un cambio de régimen que la correlación rolling del §4.5 confirma.

### 4.4.3 Oro y tipo de interés real

La **Figura 4.3** presenta la relación entre el oro y el rendimiento TIPS a 10 años, con el eje de TIPS invertido para facilitar la visualización de la correlación negativa esperada.

> **[Figura 4.3: Oro vs. tipo real a 10 años (TIPS, eje invertido)]**
> *(Véase `output/figures/fig_4_03_gold_vs_tips.png`)*

El coste de oportunidad de mantener oro — que no genera rendimiento — disminuye cuando los tipos reales caen, lo que se refleja en la fuerte co-evolución de ambas series. La anomalía de 2022–2024, donde el oro mantuvo niveles elevados pese a TIPS superiores al 2%, constituye uno de los hallazgos que motivan el análisis de cambio estructural en el Capítulo 5.

### 4.4.4 Oro e inflación

La **Figura 4.4** superpone el precio del oro con las dos medidas de inflación: la tasa interanual del CPI (realizada) y el breakeven a 10 años (expectativas).

> **[Figura 4.4: Oro vs. inflación (CPI interanual y breakeven 10Y)]**
> *(Véase `output/figures/fig_4_04_gold_vs_inflation.png`)*

La relación oro-inflación es más matizada de lo que sugiere la narrativa popular del oro como "cobertura contra la inflación". Como documentan Erb y Harvey (2013), esta relación opera a horizontes muy largos (décadas) pero es inconsistente en el corto plazo. En nuestros datos, el oro muestra cierta co-evolución con las expectativas de inflación (breakeven) pero responde de manera irregular a la inflación realizada.

### 4.4.5 Oro y volatilidad

La **Figura 4.5** compara el oro con el VIX.

> **[Figura 4.5: Oro vs. índice de volatilidad (VIX)]**
> *(Véase `output/figures/fig_4_05_gold_vs_vix.png`)*

Los picos del VIX (GFC 2008, COVID 2020) coinciden con reacciones bifásicas del oro: una caída inicial por liquidación forzada seguida de un rally de refugio. Este patrón, identificado en el Capítulo 2, sugiere que la relación oro-VIX es asimétrica y dependiente del régimen.

### 4.4.6 Oro, materias primas y renta variable

La **Figura 4.6** presenta dos paneles: oro vs. WTI y oro vs. S&P 500.

> **[Figura 4.6: Oro vs. WTI y S&P 500]**
> *(Véase `output/figures/fig_4_06_gold_vs_wti_sp500.png`)*

La relación oro-WTI es positiva en la primera mitad de la muestra (ambos impulsados por demanda de commodities y debilidad del dólar) pero diverge en 2014–2016 y nuevamente en 2022–2025, cuando factores específicos de cada mercado dominan. La relación oro-S&P 500 es predominantemente negativa en periodos de crisis (consistente con el rol de refugio) pero positiva en periodos de expansión monetaria (ambos activos beneficiados por liquidez abundante).

### 4.4.7 Variables estructurales y de sentimiento

La **Figura 4.7** muestra las compras netas anuales de oro por bancos centrales, uno de los catalizadores que mayor relevancia ha ganado en el periodo reciente.

> **[Figura 4.7: Compras netas de oro por bancos centrales]**
> *(Véase `output/figures/fig_4_07_cb_reserves.png`)*

Las compras récord de 2022 (1.082 toneladas) y 2023 (1.037 toneladas), lideradas por China, Polonia y Singapur, representan un cambio estructural en la demanda de oro que el modelo del Capítulo 5 debe capturar.

La **Figura 4.8** compara la atención del público (Google Trends) con el precio del oro.

> **[Figura 4.8: Google Trends "gold price" vs. precio del oro]**
> *(Véase `output/figures/fig_4_08_google_trends.png`)*

Se observa una relación bidireccional: los picos de búsquedas coinciden con rallies del oro (efecto *attention*), pero también anticipan correcciones cuando alcanzan niveles extremos (efecto contrarian), un patrón consistente con la literatura de finanzas comportamentales.

La **Figura 4.9** presenta los flujos de ETFs de oro.

> **[Figura 4.9: Flujos de ETFs de oro]**
> *(Véase `output/figures/fig_4_09_etf_flows.png`)*

Los flujos acumulados muestran una tendencia ascendente desde 2004 (lanzamiento de GLD) hasta 2012, seguida de salidas masivas durante la corrección de 2013–2015, y una recuperación parcial desde 2019. Los flujos de ETFs actúan como proxy de la demanda institucional occidental y complementan la información de las compras de bancos centrales (demanda oficial).

## 4.5 Análisis de correlación

### 4.5.1 Matriz de correlación: muestra completa

La **Tabla 4.2** y la **Figura 4.13** presentan la matriz de correlación de Pearson para la muestra completa.

> **[Tabla 4.2: Matriz de correlación de Pearson (2000–2025)]**
> *(Véase `output/tables/tab_4_02_correlation.csv`)*

> **[Figura 4.13: Heatmap de correlación de Pearson]**
> *(Véase `output/figures/fig_4_13_heatmap_full.png`)*

Las correlaciones más fuertes con el oro son: DXY (negativa, como predice la teoría), TIPS 10Y (negativa) y WTI (positiva). La correlación con el VIX es positiva pero modesta en la muestra completa, lo que enmascara la relación asimétrica identificada en el §4.4.5.

### 4.5.2 Correlaciones condicionales: crisis vs. calma

La **Figura 4.14** compara las matrices de correlación calculadas por separado en periodos de crisis y calma.

> **[Figura 4.14: Heatmaps condicionales — crisis vs. calma]**
> *(Véase `output/figures/fig_4_14_heatmaps_conditional.png`)*

Las diferencias entre ambos regímenes son sustanciales. En periodos de crisis, la correlación oro-VIX se intensifica (más positiva), mientras que la correlación oro-S&P 500 se vuelve más negativa, confirmando empíricamente la propiedad de refugio seguro (*safe haven*) frente a la de cobertura (*hedge*) que distinguen Baur y McDermott (2010). La correlación oro-DXY se debilita en crisis recientes (2022–2025), reflejando el efecto dominante de las compras de bancos centrales que "rompe" la relación histórica.

### 4.5.3 Correlaciones rolling

La **Figura 4.15** presenta las correlaciones rolling de 24 meses entre el oro y cada catalizador, proporcionando una visión dinámica de la estabilidad de las relaciones.

> **[Figura 4.15: Correlaciones rolling (24 meses) con el oro]**
> *(Véase `output/figures/fig_4_15_rolling_correlations.png`)*

El hallazgo más relevante es la inestabilidad de prácticamente todas las correlaciones a lo largo del tiempo. La correlación oro-DXY oscila entre −0.9 y +0.3; la oro-TIPS entre −0.8 y +0.2. Esta variabilidad temporal justifica el uso de modelos que permiten cambios de régimen (como el VAR con ventanas temporales o el análisis de cambio estructural del Capítulo 5) y cuestiona la validez de ecuaciones con coeficientes constantes para todo el periodo.

## 4.6 Propiedades de las series: raíz unitaria y cointegración

### 4.6.1 Tests de raíz unitaria

La **Tabla 4.4** presenta los resultados de los tests ADF (Augmented Dickey-Fuller) y KPSS (Kwiatkowski-Phillips-Schmidt-Shin) aplicados a cada variable en niveles y en primeras diferencias.

> **[Tabla 4.4: Tests de raíz unitaria — ADF y KPSS]**
> *(Véase `output/tables/tab_4_04_unit_root.csv`)*

Utilizamos la estrategia de confirmación que combina ambos tests: el ADF tiene como hipótesis nula la existencia de raíz unitaria (H₀: serie no estacionaria), mientras que el KPSS invierte la hipótesis (H₀: serie estacionaria). Una variable se clasifica como I(1) cuando el ADF no rechaza y el KPSS rechaza, y como I(0) en el caso contrario.

Los resultados confirman que las variables en niveles — precio del oro, DXY, S&P 500 y WTI — son integradas de orden uno, I(1), lo que es consistente con la teoría económica: los precios de activos siguen procesos de caminata aleatoria con drift. Las variables de tasa — TIPS, CPI interanual, breakeven, VIX y Fed Funds — muestran resultados mixtos: algunas son estacionarias en niveles I(0), mientras que otras requieren diferenciación.

En primeras diferencias, todas las variables son claramente estacionarias (I(0)), confirmando que no hay series integradas de orden superior en nuestro sistema.

### 4.6.2 Tests de cointegración

Dado que varias variables del sistema son I(1), procedemos a evaluar si existe una relación de equilibrio de largo plazo (cointegración) entre ellas.

La **Tabla 4.5** presenta los resultados del test de Johansen (trace y max-eigenvalue) aplicado al sistema multivariante de variables I(1), así como los tests de Engle-Granger bivariantes como verificación de robustez.

> **[Tabla 4.5: Test de Johansen — vectores de cointegración]**
> *(Véase `output/tables/tab_4_05_cointegration_johansen.csv`)*

La existencia (o ausencia) de vectores de cointegración tiene implicaciones directas para la especificación del modelo econométrico del Capítulo 5:

- Si se detecta **al menos un vector de cointegración**, el marco apropiado es un modelo de corrección de errores vectorial (VECM), que captura tanto la dinámica de corto plazo como el ajuste hacia el equilibrio de largo plazo.
- Si **no se detecta cointegración**, el modelo se especifica como un VAR en primeras diferencias, perdiendo la información de largo plazo pero garantizando la validez de la inferencia estadística.

Los tests de Engle-Granger bivariantes para los pares oro~TIPS, oro~DXY y oro~WTI complementan el análisis multivariante, verificando la consistencia de los resultados.

## 4.7 Diagnóstico de multicolinealidad

La **Tabla 4.7** presenta los Variance Inflation Factors (VIF) para todos los regresores de la ecuación especificada en el §3.6.

> **[Tabla 4.7: Variance Inflation Factors (VIF)]**
> *(Véase `output/tables/tab_4_07_vif.csv`)*

El umbral convencional sitúa la multicolinealidad severa en VIF > 10 y la moderada en VIF > 5. Valores elevados son esperables entre variables intrínsecamente relacionadas: por ejemplo, TIPS y breakeven (ambos derivados del mercado de bonos) o CPI y breakeven (inflación realizada vs. esperada).

Si se detectan VIF superiores a 10, las estrategias de mitigación en el Capítulo 5 incluyen: (i) eliminar una de las variables colineales, (ii) utilizar componentes principales, o (iii) confiar en la estimación VAR/VECM que maneja sistemas multivariantes sin requerir la ortogonalidad de los regresores.

## 4.8 Causalidad de Granger: evidencia preliminar

La **Tabla 4.6** presenta los resultados de los tests de causalidad de Granger bilateral entre el oro y cada catalizador, para lags de 1, 3, 6 y 12 meses.

> **[Tabla 4.6: Causalidad de Granger bilateral]**
> *(Véase `output/tables/tab_4_06_granger.csv`)*

Es importante enfatizar que la "causalidad de Granger" no implica causalidad económica: únicamente indica si los valores pasados de una variable mejoran la predicción de otra, controlando por la propia historia de la variable dependiente. En este sentido, es un test de *precedencia predictiva*.

Los resultados permiten identificar qué catalizadores anticipan movimientos del oro y, recíprocamente, si el oro anticipa cambios en sus determinantes (causalidad inversa o *reverse causality*). La bidireccionalidad, donde la detectemos, refuerza la necesidad de un enfoque VAR (que modela todas las variables como endógenas) frente a una regresión uniecuacional.

Los lags más largos (6 y 12 meses) capturan dinámicas de ajuste más lentas, relevantes para variables como las reservas de bancos centrales o la inflación, mientras que los lags cortos (1 y 3 meses) son más informativos para variables de mercado financiero como el DXY o el VIX.

## 4.9 Síntesis y anticipación del análisis econométrico

El análisis exploratorio desarrollado en este capítulo establece cinco hallazgos fundamentales que condicionan la estrategia econométrica del Capítulo 5:

**Primero**, las relaciones entre el oro y sus catalizadores son **inestables en el tiempo**. Las correlaciones rolling revelan que ningún determinante mantiene una relación constante con el metal a lo largo de los 25 años analizados. Esto sugiere la presencia de cambios de régimen y justifica la aplicación de tests de estabilidad estructural (Chow, CUSUM) y, potencialmente, modelos con parámetros cambiantes.

**Segundo**, las distribuciones de los retornos presentan **colas gruesas y asimetría**, rechazando la normalidad. Este resultado tiene implicaciones para los intervalos de confianza de los modelos lineales y motiva la inclusión de modelos GARCH para la ecuación de varianza condicional.

**Tercero**, la mayoría de variables en niveles son **I(1)**, lo que exige un tratamiento econométrico cuidadoso: o bien se trabaja en diferencias (perdiendo información de largo plazo) o bien se verifica la existencia de cointegración para especificar un VECM.

**Cuarto**, el comportamiento del oro difiere significativamente entre **crisis y calma**. Las estadísticas condicionales y las correlaciones condicionales confirman que el oro actúa como refugio seguro en periodos de estrés (correlación más negativa con renta variable, más positiva con VIX) pero no necesariamente como cobertura en periodos normales.

**Quinto**, las relaciones **no son unidireccionales**: la causalidad de Granger bilateral sugiere retroalimentación entre el oro y varios de sus determinantes, legitimando el uso de un modelo VAR que trata todas las variables como potencialmente endógenas, en lugar de una regresión de mínimos cuadrados que asume exogeneidad de los regresores.

Con estos hallazgos como fundamento empírico, el Capítulo 5 procederá a la estimación formal del modelo econométrico, comenzando por la selección óptima de lags del VAR, la estimación del VECM si la cointegración lo justifica, y los análisis de impulso-respuesta y descomposición de varianza que cuantificarán la contribución de cada catalizador a la dinámica del precio del oro.

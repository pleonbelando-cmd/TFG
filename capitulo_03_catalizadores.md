# Capítulo 3: Catalizadores del precio del oro: identificación y justificación teórica

## 3.1 El problema de la selección de variables

Cualquier modelo econométrico o de machine learning que intente explicar el precio del oro enfrenta de inmediato un problema de selección: ¿qué variables incluir y cuáles dejar fuera? La tentación es incluir todo lo que tenga correlación estadística con el oro —y la lista sería interminable—, pero un modelo con demasiadas variables pierde parsimonia, gana ruido y se vuelve difícil de interpretar y de defender.

Este capítulo establece el conjunto de catalizadores del modelo siguiendo un principio doble: cada variable incluida debe tener (1) una **justificación teórica sólida** que preceda al análisis de los datos —es decir, hay una razón económica *a priori* por la que esa variable debería afectar al precio del oro— y (2) **disponibilidad empírica** en series temporales mensuales para el periodo 2000-2025 a través de fuentes gratuitas y reproducibles. Las variables que solo satisfacen uno de los dos criterios no se incluyen en el modelo principal, aunque pueden aparecer como alternativas en el análisis de robustez.

El resultado es un conjunto de **ocho catalizadores** organizados en tres bloques funcionales: variables financieras y monetarias (las que determinan el coste de oportunidad del oro), variables de mercado y riesgo (las que capturan la demanda de refugio), y variables de demanda estructural y sentimiento (las que reflejan flujos físicos y de inversión). Al final del capítulo se presenta la especificación formal del modelo que será estimado en los Capítulos 5 y 6.

---

## 3.2 Bloque I — Variables financieras y monetarias

### 3.2.1 Índice del dólar estadounidense (DXY)

**Justificación teórica.** El oro se cotiza universalmente en dólares estadounidenses. Esta convención, heredada del sistema de Bretton Woods, genera una relación de doble causalidad entre la divisa americana y el metal: cuando el dólar se deprecia, el poder adquisitivo de los inversores no estadounidenses aumenta automáticamente en términos de su divisa local, lo que reduce el precio efectivo del oro para ellos y estimula la demanda. Simultáneamente, un dólar débil suele reflejar condiciones monetarias laxas en Estados Unidos —tipos bajos, expansión cuantitativa— que reducen el coste de oportunidad de mantener oro. Ambos canales operan en la misma dirección, reforzando la correlación inversa estructural.

**Evidencia empírica.** Beckers y Soenen (1984) fueron los primeros en documentar sistemáticamente la relación inversa entre el dólar y el oro. Investigación más reciente, utilizando datos mensuales del periodo 1976-2017, estima que un aumento del 1% en el valor del dólar deprime el precio del oro en un 3,09% en el régimen típico de mercado (IMI Working Paper, 2020). A lo largo de la última década, el coeficiente de correlación entre el oro y el DXY se ha situado en torno a -0,7, lo que lo convierte en uno de los predictores lineales más potentes del conjunto (CME Group, 2025).

Sin embargo, la relación no es estrictamente lineal ni siempre estable. Durante episodios de stress extremo —como el crash de marzo de 2020— ambos activos suben simultáneamente porque la demanda de liquidez en dólares y la demanda de refugio en oro operan a la vez. Este comportamiento no lineal justifica la utilización de modelos VAR con regímenes y de machine learning que pueden capturar estas discontinuidades.

**Variable en el modelo.** Se utilizará el *US Dollar Index* (DXY), publicado por el ICE (Intercontinental Exchange), disponible en FRED (código `DTWEXBGS`, tipo de cambio efectivo nominal amplio) con frecuencia mensual desde 1973. La variable entra en el modelo en niveles logarítmicos, con signo esperado negativo: $\beta_{DXY} < 0$.

---

### 3.2.2 Tipos de interés reales (TIPS yield a 10 años)

**Justificación teórica.** Esta es, junto con el dólar, la variable más importante del modelo. El oro es un activo que no genera rendimientos: no paga intereses, dividendos ni cupones. Por tanto, mantener oro tiene un **coste de oportunidad** que es precisamente el rendimiento real que se podría obtener invirtiendo en activos alternativos seguros —en la práctica, los bonos del Tesoro estadounidense indexados a la inflación (TIPS). Cuando los tipos reales suben, el coste de oportunidad de tener oro aumenta, lo que debería presionar el precio a la baja. Cuando los tipos reales caen —o se vuelven negativos, como ocurrió en 2020-2021— ese coste desaparece y el oro se vuelve relativamente más atractivo.

Este mecanismo está respaldado por la teoría del valor presente: el oro puede modelarse como un activo de "duración infinita" que produce un único flujo de caja implícito —su valor de liquidación futuro— descontado a la tasa real libre de riesgo. Un descenso en esa tasa aumenta el valor presente del activo, exactamente igual que con cualquier bono de larga duración.

**Evidencia empírica.** Erb y Harvey (2013) documentan una correlación de **-0,82** entre los tipos reales a 10 años y el precio real del oro para el periodo 1997-2012, la más alta de todas las variables candidatas. El Chicago Fed (2021) confirma que esta relación inversa se sostiene en los tres horizontes temporales analizados: niveles anuales, innovaciones trimestrales y diferencias diarias. El dato más ilustrativo: entre 2001 y 2012, una caída de 400 puntos básicos en el tipo real a 10 años se acompañó de una multiplicación por cinco del precio real del oro.

El episodio de 2022-2023 matiza esta relación sin invalidarla: los tipos reales pasaron de -1% a +2,5% en menos de dieciocho meses, pero el oro no colapsó como habría predicho el modelo. Esto no refuta el mecanismo —el coste de oportunidad sí ejerció presión bajista— sino que ilustra que otras variables (demanda de bancos centrales, tensión geopolítica) operaron en sentido contrario con suficiente fuerza como para compensarlo. Este tipo de dinámica compensatoria es precisamente lo que el modelo multivariante y el análisis SHAP del Capítulo 6 están diseñados para descomponer.

**Variable en el modelo.** Se utilizará el rendimiento del TIPS a 10 años (código FRED: `DFII10`), disponible mensualmente desde enero de 2003. Para el periodo 2000-2002, se imputará el tipo real construido como la diferencia entre el bono nominal a 10 años (`DGS10`) y el breakeven de inflación (`T10YIE`), siguiendo la metodología estándar de la literatura. El signo esperado es negativo: $\beta_{TIPS} < 0$.

---

### 3.2.3 Inflación: IPC y *breakeven* de inflación a 10 años

**Justificación teórica.** La inflación afecta al oro por dos canales distintos que conviene separar conceptualmente. El primero es el **canal de reserva de valor**: si los inversores anticipan que el poder adquisitivo del dinero va a deteriorarse, demandan activos que no puedan ser inflados por los bancos centrales, y el oro —cuya oferta crece apenas un 1,5% anual por la minería— es el candidato histórico por excelencia. El segundo canal es el **canal de tipos reales**: la inflación, al reducir el rendimiento real de los bonos nominales, baja el coste de oportunidad del oro de forma indirecta. Ambos canales predicen una relación positiva entre inflación y precio del oro.

Sin embargo, como documentan Erb y Harvey (2013), esta relación es mucho más débil e inestable en la práctica de lo que sugiere la narrativa popular. La razón es que los mercados son prospectivos: lo que importa no es la inflación observada hoy, sino la inflación *esperada* para el futuro. El IPC mide la inflación pasada; los *breakevens* de inflación —el diferencial de rendimiento entre bonos nominales y TIPS— miden la inflación que el mercado espera en el futuro. Para un modelo predictivo, los *breakevens* son informativamente superiores al IPC.

**Variables en el modelo.** Se incluirán **dos proxies de inflación**:

- **IPC general de EE.UU.** (código FRED: `CPIAUCSL`), transformado en tasa de variación anual para obtener estacionariedad. Captura la inflación realizada y su efecto sobre los tipos reales ex-post.

- ***Breakeven* de inflación a 10 años** (código FRED: `T10YIE`), que mide la inflación esperada por el mercado y es la variable más relevante para la demanda de cobertura prospectiva. Disponible mensualmente desde enero de 2003.

Ambas variables entran con signo esperado positivo: $\beta_{CPI} > 0$, $\beta_{BE} > 0$. En caso de multicolinealidad severa entre ambas —esperada, dado que miden fenómenos relacionados— se optará por incluir únicamente el *breakeven* en el modelo principal y el IPC como variable de robustez.

---

## 3.3 Bloque II — Variables de mercado y riesgo

### 3.3.1 Índice de volatilidad implícita del mercado (VIX)

**Justificación teórica.** El VIX, publicado por el CBOE (*Chicago Board Options Exchange*), mide la volatilidad implícita del S&P 500 a 30 días derivada de los precios de las opciones. Es el indicador más utilizado en la literatura financiera como proxy del miedo o la incertidumbre del mercado, y coloquialmente se le conoce como el "índice del miedo". Su relevancia para el precio del oro tiene dos canales:

El primero es el **canal de demanda de refugio**: cuando la volatilidad financiera se dispara, los inversores buscan activos que preserven valor independientemente de lo que ocurra en los mercados de renta variable. El oro, como documentan Baur y McDermott (2010), actúa como *safe haven* en estos episodios, lo que genera una correlación positiva entre VIX y precio del oro en periodos de stress.

El segundo es el **canal de apetito de riesgo**: en condiciones de calma —VIX bajo— los inversores prefieren activos de mayor rentabilidad y reducen su exposición a oro. En condiciones de pánico —VIX alto— la demanda de refugio eleva el precio del metal independientemente de sus fundamentales económicos.

**Evidencia empírica.** La investigación académica confirma que el VIX impacta positivamente en el precio del oro en el corto plazo, y que esta relación es más pronunciada en los percentiles extremos de la distribución del VIX —exactamente el tipo de relación no lineal que los modelos de machine learning capturan mejor que la econometría lineal clásica. Un estudio de Claremont College (Firth, 2019) documenta que durante los episodios de VIX superior a 30 (crisis aguda), la correlación semanal entre VIX y precio del oro se vuelve positiva y estadísticamente significativa, mientras que en entornos de VIX bajo la correlación es prácticamente nula.

**Variable en el modelo.** Se utilizará el VIX en niveles (código FRED: `VIXCLS`), con frecuencia mensual calculada como media de los valores diarios del mes. El signo esperado es positivo: $\beta_{VIX} > 0$, con la advertencia de que este efecto puede ser no lineal y dependiente del régimen de mercado.

---

### 3.3.2 Índice bursátil S&P 500

**Justificación teórica.** La renta variable estadounidense —representada por el S&P 500— entra en el modelo como proxy del ciclo de apetito de riesgo global. La teoría predice una relación negativa entre el S&P 500 y el oro en el largo plazo: cuando los mercados de acciones prosperan, los inversores reducen su exposición defensiva al oro en favor de activos con mayor potencial de retorno. Cuando las bolsas caen severamente, ocurre el proceso inverso.

Sin embargo, esta relación es mucho más compleja que una simple correlación negativa. En periodos de expansión económica moderada, el S&P 500 y el oro pueden subir simultáneamente, porque el crecimiento genera tanto beneficios empresariales como expectativas inflacionarias. La correlación negativa se materializa principalmente en los episodios de crisis, cuando el oro actúa como *safe haven* y los inversores rotan desde renta variable hacia activos de refugio.

**Variable en el modelo.** Se utilizará el S&P 500 en logaritmos (descargado de Yahoo Finance, código `^GSPC`), transformado en retorno mensual para la especificación dinámica. El signo esperado es negativo —una subida bursátil reduce la demanda de oro— aunque con la precaución de que esta relación es especialmente inestable en el tiempo: $\beta_{SP500} < 0$.

---

### 3.3.3 Precio del petróleo (WTI)

**Justificación teórica.** La relación entre el oro y el petróleo opera a través de tres canales distintos, lo que hace del WTI una variable especialmente rica para el análisis:

**Canal de costes de extracción**: El petróleo es un insumo fundamental en la minería del oro —para alimentar la maquinaria, el transporte y el procesado del mineral—. Un incremento sostenido en el precio del petróleo eleva los costes de producción de las minas de oro, reduciendo la oferta e impulsando el precio al alza. Este canal opera lentamente y es más visible a horizontes de varios trimestres.

**Canal de tensión geopolítica**: El petróleo y el oro reaccionan simultáneamente a los episodios de inestabilidad geopolítica. Las guerras, las sanciones y los conflictos en regiones productoras de crudo (Oriente Medio, Rusia) elevan el precio del petróleo y, a la vez, disparan la demanda de oro como refugio. En este sentido, el WTI actúa como proxy de la tensión geopolítica global, capturando información que los índices de riesgo geopolítico explícitos (como el GPR de Caldara y Iacoviello) también recogen pero que es más difícil de obtener en series largas y homogéneas.

**Canal macroeconómico**: El petróleo caro genera inflación, y la inflación —como se ha visto— es positiva para el oro. Este tercer canal introduce cierta multicolinealidad entre WTI e IPC que deberá gestionarse en la especificación econométrica.

**Evidencia empírica.** El ratio oro/petróleo ha sido objeto de investigación académica como indicador de tensión geopolítica: ratios superiores a 25 onzas de oro por barril de WTI han coincidido históricamente con episodios de crisis severos (Cogent Research, vía la literatura sobre cointegración oro-petróleo). El signo esperado de la relación es positivo: $\beta_{WTI} > 0$.

**Variable en el modelo.** Se utilizará el precio del crudo WTI (código FRED: `DCOILWTICO`), en logaritmos, con frecuencia mensual.

---

## 3.4 Bloque III — Demanda estructural y sentimiento

### 3.4.1 Compras netas de bancos centrales

**Justificación teórica.** Los bancos centrales son el mayor tenedor institucional de oro del mundo, con reservas totales superiores a las 36.000 toneladas. Su comportamiento como compradores o vendedores netos tiene efectos directos sobre el precio del metal, independientemente de los fundamentos financieros descritos en los dos bloques anteriores. Cuando los bancos centrales compran oro masivamente —como ha ocurrido desde 2022— generan un exceso de demanda estructural que puede sostener el precio incluso en entornos desfavorables para el oro (tipos reales altos, dólar fuerte).

La aceleración de las compras a partir de 2022 no es un fenómeno coyuntural: responde a un proceso de recomposición estratégica de reservas vinculado a la de-dolarización. La congelación de los activos del Banco Central de Rusia por parte de Occidente tras la invasión de Ucrania en febrero de 2022 demostró que las reservas en monedas extranjeras —incluido el dólar— son vulnerables a sanciones. Esta realización aceleró la demanda de oro soberano como activo sin riesgo de contraparte, cuyo valor no puede ser congelado por ningún gobierno extranjero. Según el World Gold Council (2025), el 73% de los bancos centrales tiene previsto reducir sus reservas en dólares en los próximos años, lo que anticipa una demanda estructural sostenida de oro.

**Limitaciones metodológicas y alternativa.** La inclusión de las reservas de bancos centrales como variable cuantitativa en el modelo econométrico presenta dos problemas prácticos relevantes:

Primero, los datos de compras netas mensuales se publican con un **retardo de uno a dos meses** por el World Gold Council, lo que introduce un sesgo de sincroneidad si no se gestiona correctamente. En el modelo, se utilizará la variable con un retardo de dos meses para respetar la secuencia temporal de publicación.

Segundo, la variable es inherentemente una **variable de stock** (toneladas totales de reservas) cuya primera diferencia (variación mensual) tiene escasa variabilidad en periodos de compras moderadas, lo que reduce su poder estadístico. Para el periodo 2000-2009, cuando los bancos centrales eran vendedores netos (Acuerdos del Oro del Banco Central), la señal es de signo opuesto.

Como **alternativa de robustez**, se propone sustituir las reservas de bancos centrales por un proxy de la demanda física de los dos mayores consumidores de oro del mundo: **India y China**. Esta demanda puede aproximarse mediante los retornos del índice bursátil de la India (Nifty 50) y del índice Shanghai Composite como proxies del ciclo económico doméstico, o mediante el tipo de cambio USD/INR y USD/CNY, que afectan al precio efectivo del oro en esas economías. Ambas especificaciones alternativas se estimarán en el análisis de robustez del Capítulo 5.

**Variable en el modelo principal.** Variación mensual de las reservas de oro de bancos centrales (en toneladas), publicadas por el World Gold Council con retardo de dos meses. Signo esperado positivo: $\beta_{CBR} > 0$.

---

### 3.4.2 Sentimiento de mercado: Google Trends y flujos de ETFs de oro

El sentimiento del inversor —la disposición subjetiva hacia el riesgo y hacia activos específicos— es un factor que la teoría financiera convencional tiende a minimizar, pero que la evidencia empírica señala como relevante incluso en mercados relativamente eficientes. Para el oro, el sentimiento tiene especial importancia porque una fracción significativa de la demanda de inversión —a diferencia de la demanda industrial o joyera— es de naturaleza especulativa o de protección psicológica, y responde a narrativas de mercado más que a cálculos de valoración fundamentales.

Se incluyen dos proxies de sentimiento que capturan dimensiones distintas:

#### 3.4.2.a Google Trends — búsquedas del término "gold"

**Justificación teórica.** El volumen de búsquedas del término "gold" en Google es un indicador de la **atención del inversor minorista** hacia el metal. Da Gao y Jagannathan (2011) introdujeron este enfoque para mercados de renta variable, y Dergiades et al. (2022) lo extendieron a los mercados de materias primas. Para el oro específicamente, Pierdzioch et al. (2015) documentan que el índice de búsquedas de Google impacta de forma positiva y significativa sobre los retornos del metal, y que su inclusión mejora la capacidad predictiva de los modelos respecto al paseo aleatorio.

La intuición económica es directa: cuando los inversores minoristas buscan masivamente información sobre el oro —normalmente en momentos de crisis o de narrativas mediáticas sobre inflación o guerra— parte de esa atención se materializa en demanda real a través de ETFs, monedas de inversión o lingotes. La búsqueda precede a la compra.

**Limitaciones.** Los datos de Google Trends están disponibles desde enero de 2004, lo que deja un hueco en el periodo 2000-2003. Para esos años se imputará cero (ausencia de señal diferencial) o se excluirá la variable del modelo estimado sobre el periodo completo, optando por una especificación alternativa sobre el subperiodo 2004-2025 donde Google Trends sí está disponible.

**Variable en el modelo.** Índice mensual de búsquedas del término "gold" en Google Trends para la región mundial, normalizado en la escala 0-100 propia de la plataforma. Signo esperado positivo: $\beta_{GT} > 0$.

#### 3.4.2.b Flujos netos hacia ETFs de oro (GLD/IAU)

**Justificación teórica.** Los ETFs de oro físico —encabezados por el SPDR Gold Shares (GLD), lanzado en noviembre de 2004— democratizaron el acceso a la inversión en oro y se convirtieron rápidamente en el mayor vehículo de inversión en el metal a escala global. Sus flujos netos mensuales son un indicador de la **demanda de inversión institucional y minorista sofisticada**, cualitativamente distinta de las búsquedas de Google y más directamente vinculada a transacciones reales de mercado.

El mecanismo es mecánico: cuando los flujos hacia los ETFs son positivos (entradas netas), el gestor del ETF debe comprar oro físico en el mercado para respaldar las nuevas participaciones emitidas, lo que ejerce presión directa sobre el precio. La magnitud de este efecto ha crecido con el tiempo: en el tercer trimestre de 2025, los ETFs de oro registraron entradas récord de 26.000 millones de dólares en un solo trimestre (World Gold Council, 2025), lo que ilustra la capacidad de este factor para mover el mercado.

**Evidencia empírica.** La evidencia académica sobre la causalidad es mixta: Kittsley (SPDR, 2009) encontró que los flujos de GLD no predicen el precio del oro en regresiones de corto plazo, sugiriendo que tanto el flujo como el precio responden a los mismos factores subyacentes. Sin embargo, investigaciones más recientes apuntan a una causalidad bidireccional: los flujos de ETF sí tienen poder predictivo marginal sobre el precio cuando se controla por las variables macroeconómicas. Este matiz hace especialmente interesante su inclusión en un modelo multivariante.

**Variable en el modelo.** Flujo neto mensual hacia ETFs de oro globales (en toneladas), publicado mensualmente por el World Gold Council. Disponible desde 2004. Signo esperado positivo: $\beta_{ETF} > 0$.

**Nota sobre la colinealidad entre las dos variables de sentimiento.** Google Trends y los flujos de ETF miden fenómenos relacionados pero no idénticos: el primero captura atención difusa del público general; el segundo, decisiones de compra de inversores activos. Es esperable cierta correlación entre ambas, especialmente en picos de pánico o euforia, pero también divergencias: en episodios como el COVID-19 de 2020, las búsquedas de "gold" dispararon antes que los flujos de ETF, que reaccionaron con retardo. En la estimación econométrica se comprobará la correlación entre ambas y, si supera el umbral de 0,8, se optará por incluir una sola de las dos en el modelo principal, reservando la otra para el análisis de robustez.

---

## 3.5 Resumen del conjunto de variables

La tabla siguiente sintetiza los ocho catalizadores seleccionados, su fuente, disponibilidad temporal y el signo teórico esperado de su coeficiente en la relación con el precio del oro:

| Variable | Descripción | Fuente | Disponible desde | Signo esperado |
|---|---|---|---|---|
| **DXY** | Índice del dólar (tipo cambio efectivo nominal) | FRED (`DTWEXBGS`) | 1973 | Negativo |
| **TIPS** | Tipo de interés real a 10 años (TIPS yield) | FRED (`DFII10`) | 2003 (imputado 2000-02) | Negativo |
| **CPI** | Tasa de inflación anual EE.UU. | FRED (`CPIAUCSL`) | 1947 | Positivo |
| **BE** | *Breakeven* de inflación a 10 años | FRED (`T10YIE`) | 2003 | Positivo |
| **VIX** | Índice de volatilidad implícita (CBOE) | FRED (`VIXCLS`) | 1990 | Positivo |
| **SP500** | Retorno mensual del S&P 500 | Yahoo Finance (`^GSPC`) | 1928 | Negativo |
| **WTI** | Precio del crudo West Texas Intermediate | FRED (`DCOILWTICO`) | 1986 | Positivo |
| **CBR** | Compras netas de bancos centrales (con retardo 2m) | World Gold Council | 2000 | Positivo |
| **GT** | Google Trends "gold" (índice 0-100) | Google Trends | 2004 | Positivo |
| **ETF** | Flujos netos mensuales ETFs de oro (ton.) | World Gold Council | 2004 | Positivo |

*Nota: CPI y BE son proxies alternativos de inflación; se incluirán ambos en la especificación inicial y se eliminará el menos informativo si la multicolinealidad es severa (VIF > 10). GT y ETF son proxies alternativos de sentimiento; se tratará igualmente.*

---

## 3.6 Especificación formal del modelo

Con las variables identificadas, el modelo que se estimará en los Capítulos 5 y 6 puede expresarse en su forma reducida general como:

$$\ln(GOLD_t) = \alpha + \beta_1 \ln(DXY_t) + \beta_2 \cdot TIPS_t + \beta_3 \cdot CPI_t + \beta_4 \cdot BE_t + \beta_5 \cdot VIX_t + \beta_6 \ln(SP500_t) + \beta_7 \ln(WTI_t) + \beta_8 \cdot CBR_{t-2} + \beta_9 \cdot SENT_t + \varepsilon_t$$

donde:

- $GOLD_t$ es el precio de cierre mensual del oro (XAU/USD) en la LBMA
- $\ln(\cdot)$ denota logaritmo natural, aplicado a las variables de precio para interpretar los coeficientes como elasticidades
- $TIPS_t$ y $VIX_t$ entran en niveles (no en logaritmos) por su naturaleza de tipos o índices que pueden tomar valores negativos o cercanos a cero
- $CBR_{t-2}$ denota la variación de reservas de bancos centrales retardada dos meses
- $SENT_t$ representa la variable de sentimiento seleccionada (Google Trends o flujos de ETF, o ambas si la colinealidad lo permite)
- $\varepsilon_t$ es el término de error, cuyas propiedades (estacionariedad, homocedasticidad, ausencia de autocorrelación) serán verificadas en el Capítulo 5

**Transformaciones para la estimación dinámica.** La ecuación anterior es una representación estática de largo plazo. Para el análisis VAR del Capítulo 5, las variables no estacionarias (aquellas con raíz unitaria, que se identificarán mediante los tests ADF y KPSS del Capítulo 4) se transformarán en primeras diferencias, a menos que exista una relación de cointegración que permita estimar un modelo VEC (*Vector Error Correction*). Para los modelos de machine learning del Capítulo 6, las variables se transformarán en retornos mensuales y z-scores para garantizar la comparabilidad de escalas.

**Hipótesis de signo.** La hipótesis nula para cada variable es que su coeficiente es cero (no hay relación con el precio del oro). La hipótesis alternativa es el signo indicado en la tabla anterior, derivado de la justificación teórica de las secciones precedentes. Cualquier coeficiente con signo opuesto al teórico esperado requerirá una explicación económica en el Capítulo 7, ya que podría indicar la presencia de colinealidad, un régimen de mercado atípico o una relación estructuralmente inestable.

---

## 3.7 Fundamentos econométricos de la especificación: parsimonia, multicolinealidad y condiciones del error

### 3.7.1 Por qué ocho variables y no más: parsimonia y grados de libertad

La selección de un número reducido de variables explicativas no es una concesión a la simplicidad: es una exigencia de la teoría econométrica. Este apartado justifica formalmente por qué el modelo se limita a ocho catalizadores principales.

**El principio de parsimonia.** En econometría, el principio de parsimonia —equivalente a la navaja de Occam en filosofía— establece que, ante dos modelos con igual capacidad explicativa, debe preferirse el más simple. Su fundamento estadístico es preciso: añadir variables a un modelo mejora mecánicamente el coeficiente de determinación $R^2$ aunque esas variables no tengan ninguna relación real con la variable dependiente. El coeficiente ajustado $\bar{R}^2$ corrige esta distorsión penalizando cada variable adicional:

$$\bar{R}^2 = 1 - \frac{SS_{res}/(n-k-1)}{SS_{tot}/(n-1)}$$

donde $n$ es el número de observaciones y $k$ el número de regresores. Si una variable añadida no reduce $SS_{res}$ en proporción suficiente al grado de libertad consumido, $\bar{R}^2$ disminuye, señalando que el modelo empeoró su eficiencia a pesar de mejorar el ajuste bruto.

Los criterios de información de Akaike (AIC) y Bayesiano (BIC) formalizan esta penalización de forma más rigurosa:

$$AIC = 2k - 2\ln(\hat{L}) \qquad BIC = k\ln(n) - 2\ln(\hat{L})$$

donde $\hat{L}$ es el valor máximo de la función de verosimilitud del modelo. El BIC penaliza la complejidad más agresivamente que el AIC —especialmente cuando $n$ es grande— y es el criterio más conservador de los dos. En la selección de modelos VAR del Capítulo 5, ambos criterios se calcularán sistemáticamente para comparar especificaciones alternativas.

**Grados de libertad y ratio observaciones/variables.** El periodo de análisis 2000-2025 con frecuencia mensual proporciona $n = 312$ observaciones. Con ocho regresores principales ($k = 8$), el ratio $n/k \approx 39$, muy por encima del umbral mínimo recomendado de 10-20 observaciones por regresor que establece la literatura econométrica (Greene, 2018). Este ratio asegura que los estimadores MCO son suficientemente precisos y que los tests de hipótesis tienen potencia estadística adecuada.

Si en lugar de ocho variables se incluyeran veinte —como podría ocurrir si se añadieran todos los catalizadores candidatos sin criterio de selección— el ratio caería a $n/k \approx 15$, próximo al límite inferior, y los errores estándar de los coeficientes se inflarían notablemente, reduciendo la capacidad del modelo para distinguir efectos reales de ruido muestral.

**Sobreajuste (*overfitting*).** El sobreajuste es la patología de los modelos con demasiadas variables: el modelo aprende el ruido de la muestra de entrenamiento y pierde capacidad predictiva en datos nuevos. En el contexto de este trabajo, el riesgo es especialmente relevante para los modelos de machine learning del Capítulo 6, donde la flexibilidad del modelo es mayor. La validación temporal *walk-forward* que se empleará en ese capítulo está precisamente diseñada para detectar y penalizar el sobreajuste fuera de muestra. Para los modelos econométricos del Capítulo 5, el riesgo es menor —la econometría lineal tiene menor capacidad de memorizar el ruido— pero el principio de parsimonia sigue siendo un criterio de calidad metodológica relevante.

---

### 3.7.2 Multicolinealidad: definición, consecuencias y diagnóstico

La multicolinealidad es la situación en que dos o más variables explicativas están correlacionadas entre sí. Es un problema casi inevitable cuando se trabaja con variables macroeconómicas, porque muchas de ellas comparten determinantes comunes: el dólar, la inflación, los tipos de interés y el crecimiento económico están interrelacionados por definición.

**Multicolinealidad perfecta e imperfecta.** La multicolinealidad perfecta —una variable es combinación lineal exacta de otras— hace que la matriz $X'X$ sea singular y que los estimadores MCO no existan. En la práctica, esto no ocurre salvo por errores de especificación (incluir el mismo dato dos veces o una variable y su complemento). La multicolinealidad imperfecta, en cambio, es habitual: las variables están correlacionadas pero no son combinaciones lineales exactas. Sus consecuencias, aunque menos dramáticas, son econométricamente relevantes.

**Consecuencias de la multicolinealidad imperfecta.** Bajo multicolinealidad, los estimadores MCO siguen siendo insesgados y consistentes — el teorema de Gauss-Markov no se viola —, pero su varianza aumenta. En la forma matricial, la varianza del vector de estimadores es:

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (X'X)^{-1}$$

Cuando las columnas de $X$ están altamente correlacionadas, la matriz $X'X$ se aproxima a la singularidad, su inversa tiene valores propios muy pequeños, y las varianzas de $\hat{\beta}_j$ se inflan. El resultado práctico es que los errores estándar aumentan, los estadísticos $t$ se reducen, y variables que en realidad son significativas pueden parecer insignificantes. En situaciones extremas, los coeficientes pueden incluso cambiar de signo respecto a lo esperado teóricamente.

**Diagnóstico: tres herramientas.** Se utilizarán tres herramientas complementarias para diagnosticar la multicolinealidad en el modelo:

**(1) Matriz de correlaciones.** La inspección de la matriz de correlaciones entre las variables explicativas es el diagnóstico más inmediato. Un coeficiente de correlación $|r_{ij}| > 0{,}8$ entre dos variables $X_i$ y $X_j$ es señal de multicolinealidad potencialmente problemática y requiere atención. Se esperan correlaciones moderadas entre, por ejemplo, TIPS e IPC (ambos ligados a la política monetaria) y entre el DXY y el WTI (ambos ligados al ciclo global de materias primas).

**(2) Factor de Inflación de la Varianza (VIF).** El VIF para el regresor $j$ se define como:

$$VIF_j = \frac{1}{1 - R_j^2}$$

donde $R_j^2$ es el coeficiente de determinación de la regresión auxiliar de $X_j$ sobre el resto de regresores. Intuitivamente, un $VIF_j$ elevado indica que $X_j$ está bien "explicada" por las demás variables —es decir, que aporta poca información marginal—. El umbral convencional es $VIF_j > 10$, que corresponde a $R_j^2 > 0{,}9$. Si alguna variable supera ese umbral, se estudiará su eliminación o su sustitución por la alternativa prevista en las secciones anteriores (por ejemplo, eliminar el IPC si el *breakeven* recoge la misma información, o eliminar Google Trends si los flujos de ETF tienen VIF más bajo).

**(3) Número de condición de la matriz $X'X$.** El número de condición $\kappa$ es el cociente entre el mayor y el menor valor propio de la matriz $X'X$ normalizada. Un $\kappa < 30$ indica ausencia de multicolinealidad problemática; $30 < \kappa < 100$ sugiere multicolinealidad moderada; $\kappa > 100$ indica multicolinealidad severa que requiere intervención (Belsley, Kuh y Welsch, 1980).

**Tratamiento.** Si el diagnóstico detecta multicolinealidad severa, se seguirá el siguiente protocolo por orden de preferencia: (1) eliminar la variable con mayor VIF que tenga justificación teórica más débil; (2) sustituir las variables colineales por su diferencia o su ratio (por ejemplo, el *spread* entre tipos nominales y reales en lugar de incluir ambos por separado); (3) en el contexto de los modelos de machine learning del Capítulo 6, aplicar regularización L2 (*Ridge*) o L1 (*LASSO*), que penalizan automáticamente la complejidad y gestionan la multicolinealidad sin eliminar variables.

---

### 3.7.3 Condiciones del término de error: hipótesis clásicas y extensiones para series temporales

El modelo de regresión lineal múltiple produce estimadores con propiedades estadísticas deseables —insesgadez, eficiencia y consistencia— solo si el término de error $\varepsilon_t$ cumple un conjunto de condiciones. En el contexto de series temporales financieras, estas condiciones deben extenderse respecto a las del modelo de regresión clásico de corte transversal.

#### Condición 1: Media condicional cero — $E[\varepsilon_t \mid X_t] = 0$

Esta condición exige que los regresores sean exógenos: no deben estar correlacionados con el término de error. En un modelo de precios financieros, esto implica que ninguna de las variables explicativas incorpora información que el mercado ya ha usado para fijar el precio del oro en el mismo instante $t$.

Con datos mensuales, esta condición es más fácilmente satisfecha que con datos diarios: las variables macroeconómicas (IPC, TIPS, reservas de bancos centrales) son publicadas con retardo por las agencias estadísticas, de modo que en el momento en que se observan ya son información del pasado. La variable de reservas de bancos centrales, que entra retardada dos meses, satisface esta condición por construcción.

Para los activos financieros cotizados en tiempo real (DXY, VIX, S&P 500), la endogeneidad potencial es más difícil de descartar: el precio del oro puede afectar al VIX o al DXY simultáneamente, creando el sesgo de simultaneidad que formalizan los Modelos de Ecuaciones Simultáneas (MES). En un MES, la ecuación del precio del oro sería una entre varias ecuaciones interrelacionadas, y estimar cada una por MCO por separado produciría estimadores sesgados e inconsistentes —el problema clásico de la endogeneidad de los regresores. El estimador de Variables Instrumentales (IV/MC2E) es la solución uniecuacional estándar cuando se puede identificar un instrumento válido, pero en la práctica encontrar instrumentos externos fuertes y exógenos para variables financieras de alta frecuencia es extremadamente difícil.

La alternativa —y la que se adoptará en el Capítulo 5— es el modelo **VAR**, que puede entenderse como el MES en su forma reducida: todas las variables se tratan simétricamente como endógenas, se modela cada una en función de sus propios retardos y de los retardos del resto, y se evita imponer restricciones de exogeneidad a priori que podrían ser incorrectas. Como señaló Sims (1980) al proponer el VAR precisamente como alternativa a los grandes modelos macroeconómicos de ecuaciones simultáneas, "si se sospecha que las restricciones de identificación no son válidas, el VAR proporciona un marco consistente sin necesidad de imponerlas". En nuestro caso, esta lógica se aplica directamente.

#### Condición 2: Homocedasticidad — $\text{Var}(\varepsilon_t \mid X_t) = \sigma^2$

Esta condición exige que la varianza del error sea constante en el tiempo. Es la condición más frecuentemente violada en series financieras: los mercados alternán periodos de calma y de turbulencia, generando **heterocedasticidad condicional** — la varianza del error es alta en crisis y baja en periodos tranquilos —. En el caso del oro, esta heterocedasticidad es especialmente pronunciada en los cinco episodios de crisis descritos en el Capítulo 2.

Su violación no invalida los estimadores MCO —que siguen siendo insesgados y consistentes— pero los hace ineficientes: ya no son los de mínima varianza, y los errores estándar convencionales están sesgados, lo que hace que los tests $t$ y $F$ sean incorrectos.

*Detección:* Se aplicará el test de Breusch-Pagan (1979) y el test de White (1980), ambos basados en la regresión de los residuos al cuadrado sobre las variables explicativas y sus cuadrados e interacciones.

*Corrección:* Si se detecta heterocedasticidad, se utilizarán **errores estándar robustos de Newey-West** (*Heteroscedasticity and Autocorrelation Consistent*, HAC), que son válidos bajo formas arbitrarias de heterocedasticidad y autocorrelación. Adicionalmente, el análisis de volatilidad del Capítulo 5 modelará la heterocedasticidad condicional explícitamente mediante un modelo GARCH, convirtiendo lo que en la regresión es un problema en el objeto de análisis en sí mismo.

#### Condición 3: Ausencia de autocorrelación — $\text{Cov}(\varepsilon_t, \varepsilon_s \mid X) = 0$ para $t \neq s$

Esta condición exige que los errores no estén correlacionados entre sí en el tiempo. En series temporales financieras mensuales, la autocorrelación de los residuos es casi universal: el precio del oro hoy depende parcialmente del precio del oro el mes pasado, y esa dependencia dinámica, si no está recogida por los regresores, queda en el término de error.

Al igual que la heterocedasticidad, la autocorrelación no sesga los estimadores MCO pero sí infla o deflacta los errores estándar, haciendo que los tests de significatividad sean incorrectos.

*Detección:* Se aplicará el test de Durbin-Watson (para autocorrelación de orden 1) y el test de Breusch-Godfrey (para autocorrelación de órdenes superiores, más apropiado cuando el modelo incluye variables retardadas).

*Corrección:* De nuevo, los errores estándar de Newey-West son la solución más directa. En el marco VAR del Capítulo 5, la autocorrelación se trata endógenamente incluyendo los retardos óptimos de cada variable (determinados mediante AIC y BIC), de modo que los residuos del VAR correctamente especificado deberían ser ruido blanco.

#### Condición 4: Normalidad del error — $\varepsilon_t \sim N(0, \sigma^2)$

La normalidad del término de error no es necesaria para que los estimadores MCO sean insesgados, eficientes o consistentes —estas propiedades se derivan exclusivamente de las condiciones anteriores—. Sin embargo, sí es necesaria para que los tests $t$ y $F$ sean exactos en muestras finitas. Con $n = 312$ observaciones, el teorema central del límite garantiza que los estimadores MCO son aproximadamente normales incluso si los errores no lo son, por lo que la condición de normalidad tiene relevancia práctica limitada en este trabajo.

No obstante, se realizará el test de Jarque-Bera sobre los residuos como diagnóstico informativo. Las series financieras tipicamente exhiben **exceso de curtosis** (*fat tails*) — las distribuciones de los retornos tienen más masa en las colas que una normal —, lo que el modelo GARCH del Capítulo 5 modelará explícitamente mediante distribuciones $t$ de Student o GED (*Generalized Error Distribution*).

#### Condición 5 (específica de series temporales): Estacionariedad

La condición más crítica en el contexto de este trabajo es la **estacionariedad** de las series. Una serie temporal es estacionaria (en sentido débil) si su media, varianza y estructura de covarianzas son constantes en el tiempo. La mayoría de las variables macroeconómicas y financieras son **no estacionarias**: tienen tendencia, y su varianza crece con el tiempo.

Regresar una variable no estacionaria sobre otra no estacionaria sin los cuidados adecuados produce la **regresión espuria**: los coeficientes parecen significativos y el $R^2$ es alto, pero la relación es estadísticamente espuria — fruto de tendencias compartidas, no de causalidad económica —. Yule (1926) fue el primero en documentar este fenómeno, y Granger y Newbold (1974) lo formalizaron en el contexto de series económicas.

La solución depende de la estructura de las series:

- Si las series son integradas de orden 1 —$I(1)$, es decir, estacionarias en primeras diferencias— y **no están cointegradas**, se estimará el modelo en primeras diferencias, perdiendo información de largo plazo pero evitando la regresión espuria.

- Si las series $I(1)$ **están cointegradas** —existe una combinación lineal estacionaria entre ellas, interpretable como una relación de largo plazo—, se estimará un modelo de corrección de error vectorial (VECM), que captura tanto la dinámica de corto plazo (en diferencias) como el ajuste al equilibrio de largo plazo. El análisis de Johansen del Capítulo 5 determinará cuántas relaciones de cointegración existen entre las variables del modelo.

- Si las series son estacionarias $I(0)$, se estimará directamente el modelo en niveles.

Los tests de Dickey-Fuller Aumentado (ADF) y KPSS —cuyas hipótesis nulas son opuestas, lo que permite una diagnosis cruzada más robusta— se aplicarán sistemáticamente a cada serie en el Capítulo 4, y sus resultados determinarán la transformación adecuada para cada variable antes de la estimación.

---

### 3.7.4 Síntesis: condiciones esperadas y protocolo de actuación

La siguiente tabla resume las cinco condiciones del término de error, la probabilidad esperada de violación dado el tipo de datos utilizados, el test de diagnóstico que se aplicará y la corrección prevista:

| Condición | Violación esperada | Test de diagnóstico | Corrección prevista |
|---|---|---|---|
| Media condicional cero | Baja (datos mensuales, VAR) | Especificación general | Modelo VAR (endogeneidad conjunta) |
| Homocedasticidad | Alta (finanzas, crisis) | Breusch-Pagan, White | Errores HAC Newey-West + GARCH |
| Ausencia de autocorrelación | Alta (series temporales) | Durbin-Watson, Breusch-Godfrey | Errores HAC + retardos óptimos en VAR |
| Normalidad | Moderada (*fat tails*) | Jarque-Bera | Distribución $t$ o GED en GARCH |
| Estacionariedad | Alta (variables macro) | ADF, KPSS | Diferenciación o VECM si hay cointegración |

El protocolo de diagnóstico y corrección se aplicará secuencialmente en el Capítulo 5, comenzando por los tests de raíz unitaria, siguiendo por la especificación del VAR/VECM, y concluyendo con la verificación de los residuos. Un modelo cuyos residuos superen todos los diagnósticos indicados en la tabla —o cuyas violaciones hayan sido explícitamente corregidas— proporciona inferencia estadística válida y resultados econométricamente defendibles.

---

## Referencias de este capítulo

- Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.
- Beckers, S., & Soenen, L. (1984). Gold: More attractive to non-US than to US investors? *Journal of Business Finance & Accounting, 11*(1), 107–112.
- Da, Z., Engelberg, J., & Gao, P. (2011). In search of attention. *Journal of Finance, 66*(5), 1461–1499.
- Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42.
- Firth, S. (2019). *A golden opportunity: An analysis of gold and the VIX as safe haven assets.* CMC Senior Theses, Claremont McKenna College.
- Federal Reserve Bank of Chicago. (2021). What drives gold prices? *Chicago Fed Letter, 464.*
- IMI Working Paper. (2020). *Nonlinear dynamics of gold and the dollar.* Renmin University of China.
- Pierdzioch, C., Risse, M., & Rohloff, S. (2015). On the efficiency of the gold market: Results of a real-time forecasting approach. *International Review of Financial Analysis, 41*, 243–251.
- World Gold Council. (2023). *Gold Demand Trends: Full Year 2023.*
- World Gold Council. (2025). *Gold Demand Trends: Full Year 2025.*
- World Gold Council. (2025). *Gold ETF: Holdings and Flows — November 2025.*
- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity.* Wiley.
- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica, 47*(5), 1287–1294.
- Granger, C. W. J., & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics, 2*(2), 111–120.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroscedasticity and autocorrelation consistent covariance matrix. *Econometrica, 55*(3), 703–708.
- White, H. (1980). A heteroscedasticity-consistent covariance matrix estimator and a direct test for heteroscedasticity. *Econometrica, 48*(4), 817–838.

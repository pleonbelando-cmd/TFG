# Capítulo 9: Conclusiones

## 9.1 Conclusiones principales

Este trabajo ha analizado la dinámica del precio del oro durante el periodo 2000-2025 mediante tres pilares metodológicos complementarios: un modelo de corrección de errores vectorial (VECM) con análisis de volatilidad GJR-GARCH, un análisis de datos de panel cross-country con cuatro economías avanzadas, y modelos de machine learning (XGBoost, Random Forest y LSTM) con validación walk-forward y análisis SHAP. Las conclusiones que se presentan a continuación se organizan en torno a las tres preguntas de investigación formuladas en el §1.3.

### 9.1.1 Sobre los determinantes del precio del oro

**Los tipos de interés reales son el determinante estructural más importante del precio del oro en el largo plazo.** Esta conclusión, que replica y actualiza los resultados de Erb y Harvey (2013) y el Chicago Fed (2021), se sostiene en cuatro fuentes de evidencia independientes: la mayor causalidad de Granger (p < 0,001 a todos los horizontes), la IRF de mayor magnitud y persistencia en el VECM, la segunda posición en el ranking SHAP del modelo XGBoost (|φ̄| = 0,617), y el coeficiente negativo estadísticamente significativo en todas las economías del panel con errores de Driscoll-Kraay. El mecanismo económico subyacente —el coste de oportunidad de mantener un activo sin rendimiento corriente— opera de forma universal: no es una idiosincrasia del mercado del Tesoro estadounidense, sino una propiedad estructural del activo.

**La inflación es el predictor más potente en el horizonte de un mes.** En el análisis SHAP de corto plazo, el CPI con retardo de un mes encabeza el ranking de variables (|φ̄| = 0,954), por delante de los TIPS. Esta aparente discrepancia con el resultado anterior se resuelve al reconocer que los dos hallazgos operan en horizontes diferentes: la sorpresa inflacionaria reciente es la señal de más alta frecuencia del coste de oportunidad del oro y domina la predicción mensual; el nivel de los tipos reales ancla la relación de equilibrio de largo plazo captada por el vector de cointegración.

**El dólar y la renta variable son determinantes secundarios.** El índice DXY ocupa el segundo lugar en la descomposición de varianza del VECM pero queda relegado a la octava posición en el ranking SHAP de corto plazo, lo que indica que su relevancia es mayor en horizontes estructurales de 12-24 meses que en la predicción mensual. El S&P 500 actúa principalmente como indicador de apetito de riesgo —con efecto negativo sobre el oro en los episodios de expansión bursátil sostenida— pero su contribución a la varianza de largo plazo es más modesta.

### 9.1.2 Sobre la estabilidad temporal de los determinantes

**Las relaciones entre el oro y sus determinantes no son constantes en el tiempo.** Los tests de Chow rechazan la estabilidad de los parámetros en los puntos de quiebre asociados a los principales episodios de crisis, con el estadístico F más elevado en marzo de 2022. El análisis CUSUM sale de las bandas de confianza al 5% durante el periodo 2022-2024. Los coeficientes rolling confirman que el efecto de los TIPS sobre el oro —históricamente estimado en torno a -0,7— se atenuó significativamente en ese episodio.

**La "paradoja de 2022-2024" tiene una explicación económica coherente.** El oro subió a máximos históricos mientras los tipos reales alcanzaban sus niveles más altos en décadas. Los tres pilares analíticos del trabajo convergen en la misma interpretación: la demanda estructural de los bancos centrales emergentes —en el contexto del proceso de de-dolarización documentado por el World Gold Council (2023, 2024)— actuó como una fuerza de soporte que ralentizó la corrección hacia el equilibrio histórico sin eliminar el mecanismo de coste de oportunidad. Esta demanda, inelástica a los tipos de interés reales de los países avanzados y motivada por incentivos geopolíticos, es el factor que los modelos econométricos basados en variables financieras no pueden capturar plenamente pero cuya existencia los tests de estabilidad estructural sí permiten detectar.

### 9.1.3 Sobre la aportación del machine learning a la predicción

**La LSTM mejora la predicción de corto plazo respecto al benchmark naive.** Con una precisión direccional (DA) del 61,5% frente al 55,9% del paseo aleatorio y un RMSE de 3,815 pp frente a 5,054 pp del naive, la red neuronal recurrente demuestra que existe información predictiva explotable en la secuencia histórica de los determinantes del oro, más allá de la inercia del proceso. Esta mejora, aunque modesta en términos absolutos, es económicamente relevante para decisiones de asignación táctica de activos.

**El análisis SHAP valida la especificación econométrica desde la perspectiva del ML.** La convergencia entre las jerarquías de variables del VECM y del análisis SHAP —tipos reales e inflación como determinantes dominantes, seguidos por el momentum del oro, el VIX y la renta variable— es el hallazgo metodológicamente más valioso del trabajo. Cuando dos enfoques con supuestos completamente diferentes producen la misma jerarquía de importancia, la evidencia de causalidad económica real se fortalece considerablemente frente a la posibilidad de que los resultados sean artefactos de los supuestos del modelo.

**El ML complementa, no sustituye, a la econometría.** El VECM identifica relaciones estructurales de largo plazo e impone restricciones de identificación económicamente justificadas; el LSTM maximiza la precisión de predicción sin restricciones estructurales. Los dos modelos son herramientas apropiadas para preguntas diferentes: el VECM para cuantificar mecanismos de transmisión y velocidades de ajuste al equilibrio; el LSTM para optimizar señales tácticas de corto plazo. Su complementariedad —no su competencia— es la conclusión metodológica central de este trabajo.

---

## 9.2 Aportaciones originales

Este trabajo realiza cuatro aportaciones que van más allá de la aplicación rutinaria de herramientas estándar a datos conocidos:

**Primera: validación cross-country de los mecanismos clásicos.** La aplicación de un modelo de panel con efectos fijos a cuatro economías avanzadas durante 2000-2024 confirma que el mecanismo de coste de oportunidad de los tipos reales y el rol de refugio del oro ante la volatilidad financiera son propiedades universales del activo, no artefactos de los datos estadounidenses. Esta validación, que actualiza la evidencia de Baur y McDermott (2010) con 15 años adicionales de datos —incluyendo los episodios post-GFC más relevantes— no tiene precedente en trabajos académicos de pregrado de economía en España, hasta donde llega el conocimiento del autor.

**Segunda: cuantificación formal de la inestabilidad estructural.** La combinación de tests de Chow en puntos de quiebre económicamente motivados con el análisis CUSUM proporciona evidencia formal de que las relaciones estimadas cambian en los episodios de crisis. Esta formalización va más allá de la mera descripción de correlaciones rolling, habitual en la literatura, y permite datar con precisión cuándo y en qué magnitud las relaciones econométricas se alteraron.

**Tercera: validación cruzada VECM-SHAP.** La comparación sistemática entre la importancia de las variables medida por la descomposición de varianza del VECM y los valores SHAP del modelo de ML con el mismo conjunto de variables y el mismo período constituye una forma de validación cruzada metodológica que no se encuentra frecuentemente en la literatura aplicada. La convergencia de resultados entre dos metodologías con supuestos tan diferentes tiene valor epistémico propio, independientemente de la precisión absoluta de cada modelo individual.

**Cuarta: análisis integrador del episodio 2022-2024.** La "paradoja" de un oro históricamente alto coexistiendo con tipos reales históricamente altos se analiza de forma integrada con las tres metodologías, conectando la detección econométrica de la ruptura estructural con la explicación económica de la de-dolarización y con el cambio en los pesos SHAP en ese sub-período. Este análisis integrador de un episodio reciente tiene relevancia directa para la gestión de carteras y para el debate de política económica sobre el futuro del sistema monetario internacional.

---

## 9.3 Limitaciones y cautelas

Las conclusiones deben leerse con las siguientes cautelas:

La dimensión cross-sectional del panel (N = 4) es insuficiente para extraer inferencia robusta sobre la heterogeneidad entre países: los resultados son ilustrativos del orden de magnitud de los efectos pero no permiten generalizaciones estadísticas sobre poblaciones de economías. El tamaño de la muestra de ML (271 observaciones, 35 características) implica que los resultados predictivos deben interpretarse como indicativos, no como estimaciones definitivas de la mejora sobre el benchmark. La ausencia de una variable que capture las compras de bancos centrales emergentes a alta frecuencia es la omisión más importante del modelo, especialmente para la interpretación del episodio 2022-2024. Por último, los tests de raíz unitaria y cointegración en panel no se aplican formalmente, lo que constituye una limitación metodológica respecto al rigor exigible a un trabajo de investigación doctoral o de máster.

---

## 9.4 Líneas de investigación futura

Los resultados de este trabajo apuntan a cuatro extensiones naturales que quedan pendientes de desarrollo:

**Ampliar el panel a economías emergentes.** La motivación de Baur y McDermott (2010) para distinguir entre economías avanzadas y BRIC sigue siendo válida. Un panel ampliado que incluya China, India, Brasil y Turquía permitiría contrastar si el mecanismo de coste de oportunidad opera de forma diferente en economías con menor desarrollo de los mercados de capitales locales y con distintos patrones culturales de demanda de oro físico.

**Modelización explícita de la demanda de bancos centrales.** La inclusión de los datos trimestrales de reservas oficiales de oro del FMI-IFS como variable explicativa adicional en el VECM y en el modelo de panel permitiría capturar directamente el canal de de-dolarización que este trabajo solo puede inferir indirectamente de los tests de estabilidad estructural.

**Extensión temporal a frecuencia diaria.** El análisis a frecuencia mensual es apropiado para capturar los determinantes macroeconómicos de largo plazo, pero pierde información sobre la dinámica intradiaria y los efectos de los anuncios de política monetaria sobre el precio del oro. Un modelo de alta frecuencia con datos diarios y variables de texto (NLP sobre actas de la Fed, noticias geopolíticas) podría capturar dimensiones del comportamiento del oro que este trabajo no alcanza.

**Modelos de cambio de régimen.** La evidencia de inestabilidad estructural documentada en este trabajo es consistente con la hipótesis de que el oro opera bajo distintos regímenes macroeconómicos. La estimación de un modelo de Markov Switching VAR —que permite que los parámetros del sistema cambien entre dos o más regímenes de forma endógena— sería la extensión econométrica más directa y podría proporcionar una caracterización formal de los regímenes de "alta sensibilidad a los tipos reales" y "dominancia de la demanda soberana" que este trabajo identifica de forma descriptiva.

---

## 9.5 Reflexión final

El oro es un activo que desafía las categorías convencionales de la teoría financiera. No genera flujos de caja, no tiene valor de uso mayoritario, y sin embargo ha preservado poder adquisitivo durante milenios. Su precio en el periodo 2000-2025 concentra algunos de los episodios macroeconómicos y geopolíticos más extraordinarios de la historia reciente, y su comportamiento en cada uno de ellos —a veces predecible, a veces desconcertante— refleja la complejidad de las fuerzas que lo mueven.

Este trabajo ha demostrado que, a pesar de esa complejidad, existe un núcleo de determinantes que los datos confirman con robustez metodológica notable: los tipos de interés reales y la inflación como catalizadores macroeconómicos fundamentales, la incertidumbre financiera global como activador del rol de refugio, y la inestabilidad estructural como característica permanente de sus relaciones con el entorno. El oro no es, como a veces se sostiene en el debate popular, ni un activo perfectamente predecible por modelos mecánicos ni un misterio económico impenetrable. Es un activo con determinantes identificables cuyas ponderaciones cambian según el régimen de mercado dominante, y cuya comprensión requiere exactamente la combinación de econometría estructural, perspectiva comparada y herramientas adaptativas que este trabajo ha intentado aportar.

En el momento de redactar estas conclusiones, el precio del oro supera los 4.500 dólares por onza y los bancos centrales siguen acumulando reservas a un ritmo sin precedentes. Los modelos estimados en este trabajo no predicen con exactitud qué ocurrirá el mes que viene. Pero sí establecen las condiciones bajo las que el oro tenderá a subir —tipos reales cayendo, incertidumbre financiera elevada, dólar débil, demanda soberana sostenida— y las condiciones bajo las que su coste de oportunidad se hace difícilmente justificable. Esa capacidad de articular condiciones, más que un número concreto, es lo que la econometría rigurosa puede aportar al análisis de los mercados financieros.

---

## Referencias de este capítulo

- Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.
- Chicago Fed. (2021). *What drives gold prices?* Chicago Fed Letter, No. 464.
- Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica, 57*(2), 357–384.
- O'Connor, F. A., Lucey, B. M., Batten, J. A., & Baur, D. G. (2015). The financial economics of gold — a survey. *International Review of Financial Analysis, 41*, 186–205.
- World Gold Council. (2023). *Gold Demand Trends: Full Year 2023.* World Gold Council.
- World Gold Council. (2024). *Gold Demand Trends: Full Year 2024.* World Gold Council.

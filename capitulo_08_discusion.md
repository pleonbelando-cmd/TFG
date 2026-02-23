# Capítulo 8: Discusión integrada

## 8.1 Introducción: tres metodologías, una pregunta

Los tres capítulos analíticos de este trabajo —el VECM del Capítulo 5, el análisis de panel del Capítulo 6 y los modelos de machine learning del Capítulo 7— se diseñaron para responder a las mismas preguntas de investigación desde ángulos complementarios. Cada metodología tiene un conjunto propio de supuestos, fortalezas y limitaciones: el VECM captura relaciones de equilibrio de largo plazo pero asume linealidad y parámetros constantes; el panel cross-country añade la dimensión comparativa internacional a costa de trabajar con retornos trimestrales y cuatro unidades; el ML aprende patrones no lineales y captura regímenes cambiantes, pero a expensas de mayor complejidad y menor interpretabilidad directa. El hecho de que tres enfoques metodológicamente independientes converjan en conclusiones similares sobre los determinantes del oro es, en sí mismo, el hallazgo más robusto de este trabajo.

Este capítulo sintetiza los resultados de los tres pilares, responde explícitamente a las preguntas de investigación planteadas en el §1.3, discute la "paradoja de 2022-2024" como caso de estudio integrador y concluye con una valoración honesta de las limitaciones del análisis.

---

## 8.2 Respuesta a las preguntas de investigación

### 8.2.1 Pregunta 1: ¿Qué variables determinan el precio del oro en el periodo 2000-2025?

La respuesta que emerge del conjunto del análisis es clara y consistente: **los tipos de interés reales y el índice del dólar son los determinantes estructurales dominantes del precio del oro**, en ese orden de importancia.

La evidencia proveniente de las tres metodologías apunta en la misma dirección. En el análisis VECM del Capítulo 5, el tipo de interés real medido por los TIPS a 10 años aparece como la variable con mayor causalidad de Granger (p < 0,001 a todos los horizontes), la mayor magnitud de impulso-respuesta y la mayor contribución a la varianza del error de predicción en horizontes superiores a seis meses. El índice del dólar (DXY) ocupa el segundo lugar en las tres métricas. La renta variable (S&P 500) actúa como determinante de corto y medio plazo —especialmente visible en los episodios de crisis, cuando los inversores mueven capital entre activos de riesgo y refugio—, pero su contribución a la varianza de largo plazo del oro es más modesta.

El análisis de panel del Capítulo 6 confirma que estos mecanismos no son una idiosincrasia del mercado estadounidense. El coeficiente del tipo de interés real local es negativo y estadísticamente significativo en las cuatro economías del panel (EE.UU., Eurozona, Reino Unido y Japón), con errores estándar robustos de Driscoll-Kraay que son válidos bajo dependencia transversal y temporal simultánea. Este resultado establece que el mecanismo de coste de oportunidad —el canal por el que los tipos reales más altos reducen el atractivo relativo del oro, que no genera rendimiento corriente— opera universalmente como propiedad estructural del activo, independientemente de la moneda en que se mide el precio del oro y del banco central cuya política monetaria se considere.

El análisis SHAP del Capítulo 7 cierra el círculo desde la perspectiva del machine learning. En el modelo XGBoost con validación walk-forward, las dos variables con mayor importancia SHAP sobre el período de test son, por ese orden, la inflación CPI con retardo de un mes (|φ̄| = 0,954) y el TIPS a 10 años con retardo de dos meses (|φ̄| = 0,617). Nótese que la inflación lidera en el análisis SHAP de corto plazo mientras que los tipos reales lideran en el FEVD de largo plazo: lejos de ser contradictorios, estos resultados son complementarios. En el horizonte de un mes, la sorpresa en la inflación reciente es la señal más directa del coste de oportunidad del oro; en horizontes de 12 a 24 meses, el nivel de los tipos reales es el anclaje estructural al que el precio del oro tiende a revertir.

La inflación, la volatilidad (VIX), el momentum del propio precio del oro y el petróleo (WTI) completan el conjunto de determinantes relevantes, pero con un papel secundario respecto a los tipos reales y el dólar. La Tabla 8.1 resume la jerarquía de determinantes según las tres metodologías.

**Tabla 8.1 — Jerarquía de determinantes del oro según las tres metodologías**

| Variable | VECM (FEVD 12m) | Panel EF (coef. β) | SHAP (|φ̄| test) |
|---|---|---|---|
| Tipos reales (TIPS / local r) | **#1** contribución varianza | **#1** más significativo | **#2** XGBoost test |
| Inflación (CPI / π local) | Variable exógena (I(0)) | Significativo (β > 0) | **#1** XGBoost test |
| DXY (dólar) | **#2** contribución varianza | — (variable común USD) | **#8** (retardo 3m) |
| VIX (volatilidad global) | Variable exógena | **#3** (coef. β > 0) | Incluido entre top-10 |
| S&P 500 / renta variable | **#3** contribución varianza | Significativo (β < 0) | **#6** (retardo 1m) |

### 8.2.2 Pregunta 2: ¿Han cambiado los determinantes del oro tras los episodios de crisis?

La respuesta es afirmativa, y la evidencia es especialmente pronunciada para el episodio 2022-2024. Los tests de Chow del §5.8.2 rechazan la estabilidad de los parámetros en los puntos de quiebre asociados a los principales episodios de crisis, con el mayor estadístico F correspondiente al inicio del ciclo de subidas de tipos de la Reserva Federal (marzo de 2022). El análisis CUSUM del §5.8.3 refuerza este diagnóstico: la función CUSUM sale de las bandas de confianza al 5% durante el periodo 2022-2024, indicando que los parámetros del modelo cambiaron cualitativamente en ese período.

Los coeficientes rolling de la Figura 5.6 hacen visible la dinámica concreta de ese cambio. El coeficiente del TIPS —habitualmente próximo a -0,7 durante el periodo 2000-2021— se atenuó significativamente durante 2022-2024, periodo en que los tipos reales subieron de forma histórica pero el precio del oro no colapsó como la relación histórica habría predicho. El coeficiente del DXY mostró una evolución similar: la correlación oro-dólar, estructuralmente negativa durante dos décadas, se acercó a cero e incluso fue positiva transitoriamente cuando dólar y oro subieron en paralelo en 2022-2023.

El análisis SHAP del Capítulo 7 añade una perspectiva dinámica a este diagnóstico. En el análisis *waterfall* de episodios representativos, el VIX y el momentum del oro amplificaron su contribución a la predicción en los momentos de mayor tensión geopolítica y financiera (episodio COVID de 2020, rally de 2025), mientras que los TIPS dominaron la señal en los periodos de estabilidad relativa. Esta variación en la importancia relativa de las variables según el régimen de mercado es exactamente lo que los coeficientes constantes del VECM —estimados sobre la muestra completa— no pueden capturar, y es la contribución analítica más específica del componente ML del trabajo.

La interpretación económica de la inestabilidad estructural apunta a un cambio en la composición de la demanda de oro. Las compras masivas de los bancos centrales de economías emergentes —especialmente China, India, Turquía y Polonia, que superaron las 1.000 toneladas netas en 2022 y 2023 según el World Gold Council— introdujeron un flujo de demanda estructural que opera con lógicas diferentes a las del inversor financiero occidental. Para un banco central que busca reducir su exposición al dólar en un contexto de sanciones geopolíticas y de-dolarización, el nivel de los TIPS es un factor secundario. Esta demanda inelástica al tipo real explica por qué la relación econométrica histórica se debilitó sin desaparecer completamente.

### 8.2.3 Pregunta 3: ¿Puede el ML mejorar la predicción sobre el VECM?

La respuesta es afirmativa pero matizada. El mejor modelo ML —la red LSTM— alcanza una precisión direccional (DA) del 61,5% sobre los 109 meses del período de test (octubre 2016 – octubre 2025), frente al 55,9% del benchmark naive (paseo aleatorio). Esta mejora de 5,6 puntos porcentuales en dirección y la reducción del RMSE de 5,054 a 3,815 puntos porcentuales son estadística y económicamente relevantes: para un inversor que toma decisiones mensuales de sobreponderar o subponderar el oro en su cartera, un 61,5% de aciertos direccionales supone una mejora operativa significativa respecto al azar calibrado.

Sin embargo, el ML no "bate" al VECM en el sentido en que los dos modelos responden a preguntas diferentes. El VECM está diseñado para modelar relaciones de equilibrio de largo plazo y cuantificar el ajuste al mismo; su predicción de corto plazo es secundaria a su función de inferencia estructural. El LSTM, en cambio, optimiza explícitamente la predicción de un paso hacia adelante a través de una función de pérdida que penaliza los errores de magnitud, sin imponer ninguna estructura económica. La comparación más honesta no es "¿cuál predice mejor?" sino "¿para qué pregunta es cada modelo más adecuado?": el VECM para cuantificar mecanismos de transmisión y velocidades de ajuste; el LSTM para optimizar señales de predicción de corto plazo.

El resultado del XGBoost, que obtiene una DA inferior al naive (52,3% vs. 55,9%), ilustra un riesgo real de los modelos de boosting en muestras financieras cortas: la optimización del error cuadrático medio puede llevar al modelo a reducir el error de magnitud a costa de introducir ruido en la dirección del movimiento. Este resultado no invalida el XGBoost como herramienta analítica —sus valores SHAP son los más interpretables de los tres modelos— pero pone de manifiesto que la elección de la métrica de evaluación importa tanto como la elección del modelo.

---

## 8.3 Convergencia entre metodologías: qué es robusto

La convergencia entre el análisis econométrico, el panel cross-country y el machine learning en los siguientes puntos permite calificarlos como hallazgos robustos del trabajo:

**Robustez del mecanismo de coste de oportunidad.** La relación negativa entre tipos de interés reales y precio del oro se documenta en el análisis de causalidad de Granger (p < 0,001), en el vector de cointegración (coeficiente TIPS < 0), en los efectos fijos del panel (β₂ < 0 en las cuatro economías), y en el análisis SHAP (TIPS = segunda variable más influyente en el modelo XGBoost). Cuatro metodologías con supuestos completamente distintos convergen en el mismo signo y en la misma jerarquía relativa. Esta convergencia es la evidencia más sólida del trabajo y hace la conclusión prácticamente inmune a los supuestos específicos de cualquier modelo individual.

**Universalidad del safe haven.** El coeficiente positivo del VIX en el análisis de panel (β₃ > 0, estadísticamente significativo) confirma que el oro actúa como refugio no solo para los inversores estadounidenses, sino también para los europeos, británicos y japoneses cuando los mercados financieros globales entran en pánico. Este resultado actualiza y extiende la evidencia de Baur y McDermott (2010) a economías avanzadas durante el periodo post-GFC, incluyendo los episodios COVID-19, el ciclo de tipos de 2022 y el shock arancelario de 2025.

**Papel secundario de la renta variable.** El S&P 500 y sus equivalentes locales en el panel aparecen como determinantes estadísticamente significativos pero con una contribución menor a la varianza de largo plazo del oro que los tipos reales o el dólar. Las IRF muestran que el impacto de un shock en la renta variable sobre el oro es real pero de magnitud inferior y más rápida disipación que el de los tipos reales. El SHAP (posición #6 para S&P 500 con retardo de 1 mes) confirma este papel secundario en el horizonte predictivo de corto plazo.

**Inestabilidad como regla, no como excepción.** Ninguna de las tres metodologías sugiere que las relaciones entre el oro y sus determinantes sean constantes en el tiempo. El CUSUM rechaza la estabilidad global de los parámetros, el test de Chow detecta rupturas en los episodios de crisis, los coeficientes rolling fluctúan de forma sistemática, y el análisis SHAP muestra que el peso relativo de cada variable cambia entre regímenes. Esta inestabilidad no es un defecto del modelo: es una característica del activo. El oro responde a fuerzas distintas según el régimen macroeconómico dominante —crisis de liquidez, crisis de inflación, episodios de apetito de riesgo, shocks geopolíticos— y cualquier modelo que la ignore subestimará la incertidumbre de sus predicciones.

---

## 8.4 La paradoja de 2022-2024: una interpretación unificada

El episodio 2022-2024 merece una atención especial porque es el banco de pruebas más exigente para los modelos de este trabajo y, a la vez, el más informativamente rico. Entre marzo de 2022 y diciembre de 2024, la Reserva Federal elevó los tipos de interés oficiales desde 0-0,25% hasta 5,25-5,50% —el ciclo de subidas más agresivo desde 1980— y los tipos reales TIPS a 10 años pasaron de valores profundamente negativos (-1%) a positivos (1,5-2%). Según la relación histórica estimada por el VECM, esta subida debería haber producido una caída sustancial del precio del oro. Sin embargo, el oro cerró 2024 con un nuevo máximo histórico.

Los tres pilares analíticos del trabajo ofrecen piezas complementarias de la explicación:

**El VECM diagnostica la ruptura pero no la causa.** Los tests de Chow y el CUSUM confirman que los parámetros del modelo cambiaron en torno a 2022. El coeficiente rolling del TIPS se atenúa durante este periodo, lo que indica que la sensibilidad del oro a los tipos reales disminuyó cualitativamente. Desde la perspectiva del VECM, el vector de cointegración seguirá siendo válido en el largo plazo, pero el mecanismo de corrección de errores tardó más de lo habitual en operar porque una fuerza contraria —la demanda de los bancos centrales emergentes— ralentizó la corrección hacia el equilibrio histórico.

**El panel identifica la heterogeneidad geográfica de la demanda.** Los efectos fijos estimados para las cuatro economías del panel revelan diferencias en la demanda estructural de oro que las variables del modelo no capturan plenamente. La demanda de bancos centrales de economías no incluidas en el panel —China, India, Turquía— es inelástica a los tipos de interés reales de los países avanzados y responde a incentivos geopolíticos y de gestión de reservas que están ausentes del modelo. En este sentido, el panel no solo confirma los mecanismos del VECM, sino que señala la dimensión geográfica de su limitación.

**El ML captura el cambio de régimen sin necesidad de especificarlo a priori.** La red LSTM, entrenada con ventana expandible, incorpora gradualmente la nueva información del periodo 2022-2024 en sus predicciones. La variable *dummy* de régimen crisis y el momentum del oro figuran entre las de mayor importancia SHAP en los episodios de máximo diferencial entre los tipos reales y el comportamiento del precio del oro, lo que indica que el modelo aprendió empíricamente que en esos periodos la señal de los TIPS pierde poder predictivo y las variables de momentum toman el relevo. Esta adaptabilidad implícita es la ventaja más clara del ML sobre el VECM en el corto plazo.

La "paradoja de 2022-2024" puede resumirse, pues, como la superposición de dos fuerzas opuestas: el mecanismo de coste de oportunidad —que tiende a deprimir el precio del oro cuando los tipos reales suben— y la demanda estructural de los bancos centrales emergentes en un contexto de de-dolarización —que la sustenta independientemente del nivel de tipos. La resultante de estas dos fuerzas fue un oro que subió menos de lo que su demanda de refugio habría justificado, pero bastante más de lo que la lógica del coste de oportunidad predecía. Los tres pilares analíticos contribuyen a entender cada dimensión de este resultado, ninguno lo captura de forma autónoma y completa.

---

## 8.5 Implicaciones para inversores e instituciones

Los hallazgos del trabajo tienen implicaciones concretas para distintos perfiles de participantes en el mercado del oro:

**Para el inversor minorista y el gestor de carteras**, la conclusión más práctica es que el oro protege mejor en contextos de tipos reales negativos o decrecientes y de alta incertidumbre financiera (VIX elevado), mientras que tiende a ser un activo con coste de oportunidad elevado en contextos de tipos reales positivos y estables. La DA del 61,5% de la LSTM sugiere que es posible mejorar la temporización de las posiciones en oro con señales cuantitativas, aunque el margen de mejora sobre el azar calibrado es modesto y debe contextualizarse contra los costes de transacción.

**Para el banco central y el gestor de reservas soberanas**, el análisis de panel aporta evidencia de que el oro cumple una función de cobertura contra la inflación estadísticamente significativa en todas las economías avanzadas analizadas. La universalidad del mecanismo de coste de oportunidad implica también que, desde la perspectiva de un banco central que gestiona su balance, el coste implícito de mantener oro en reservas varía con las condiciones de tipos de interés de su propia moneda de referencia —no solo con los tipos de EE.UU.

**Para el investigador económico**, la convergencia entre el análisis SHAP y el VECM en los determinantes más importantes —tipos reales e inflación, en ese orden— es una validación cruzada metodológica valiosa. En la práctica, cuando dos enfoques tan diferentes en supuestos e implementación producen jerarquías de variables tan similares, la evidencia de causalidad económica se fortalece más allá de lo que cada metodología podría establecer por separado.

---

## 8.6 Limitaciones del análisis

La honestidad académica exige reconocer las limitaciones que condicionan el alcance de las conclusiones presentadas.

**Primera limitación: tamaño de la muestra.** El análisis de machine learning trabaja con 271 observaciones efectivas y 35 características, lo que implica una ratio observaciones/características de solo 7,7. En este régimen de escasez relativa, los modelos de árboles tienden a sobreajustar la muestra de entrenamiento y a ser inestables en el test. Los resultados del Capítulo 7 deben interpretarse como indicativos de la dirección del efecto, no como estimaciones precisas de la mejora predictiva absoluta.

**Segunda limitación: la dimensión N del panel.** Con cuatro unidades cross-seccionales, el panel del Capítulo 6 tiene una dimensión N muy reducida. El test de Hausman y los errores de Driscoll-Kraay son asintóticos en T, lo que los hace razonablemente válidos con T = 96, pero la inferencia sobre la heterogeneidad entre países —y en particular la interpretación de los efectos fijos— está basada en solo cuatro puntos de datos transversales. La extensión del panel a un conjunto más amplio de economías sería la mejora más directa del análisis cross-country.

**Tercera limitación: ausencia de variables inobservables clave.** El modelo no incluye las compras de bancos centrales emergentes como variable explicativa porque no existe una serie de alta frecuencia y amplia cobertura geográfica disponible para todo el periodo 2000-2025. Esta omisión es especialmente relevante para el episodio 2022-2024, donde la evidencia anecdótica y los datos del World Gold Council sugieren que esa demanda fue el factor dominante. La inclusión de un proxy de compras soberanas —por ejemplo, las reservas oficiales de oro del FMI-IFS a frecuencia trimestral— es una extensión natural del trabajo.

**Cuarta limitación: estacionariedad en el panel.** El modelo de panel del Capítulo 6 se estima sobre retornos trimestrales, que son aproximadamente estacionarios. Sin embargo, no se aplican tests formales de raíz unitaria en panel (Im-Pesaran-Shin o Levin-Lin-Chu) ni tests de cointegración en panel (Pedroni o Westerlund). La metodología de series temporales del Capítulo 5 muestra que los niveles de las variables son I(1), pero la transición al análisis de panel en diferencias no se formaliza con la misma rigurosidad que el análisis univariante del Capítulo 4. Esta es una limitación metodológica que un trabajo de investigación más extenso debería subsanar.

**Quinta limitación: el sesgo de selección del período.** El periodo de análisis 2000-2025 no es aleatorio: se selecciona porque concentra los episodios más interesantes de la historia reciente del oro. Esta selección maximiza la varianza explicada disponible, pero puede inflar la importancia aparente de los determinantes que dominaron precisamente en esos episodios (tipos reales post-GFC, demanda de refugio en COVID). Los resultados pueden no generalizarse directamente a periodos de mayor estabilidad macroeconómica.

---

## 8.7 Síntesis final: ¿qué sabemos sobre el oro que no sabíamos?

Este trabajo parte de tres preguntas de investigación y, al término del análisis, puede ofrecer respuestas informadas a las tres. El precio del oro en el periodo 2000-2025 está determinado, de forma primordial, por los tipos de interés reales y el índice del dólar —resultado que replica la literatura fundacional con datos actualizados y con una validación cross-country que los trabajos anteriores no ofrecen. El mecanismo de coste de oportunidad no es una particularidad del mercado estadounidense: opera en las cuatro economías avanzadas analizadas, con el mismo signo y con significatividad estadística robusta.

Las relaciones no son estables en el tiempo. Los grandes episodios de crisis —especialmente el de 2022-2024— producen rupturas estructurales formalmente detectables que tienen una explicación económica coherente: cambios en la composición de la demanda de oro que alteran temporalmente el peso relativo de los determinantes macroeconómicos clásicos. Esta inestabilidad no niega la vigencia del marco teórico, sino que lo enriquece: el oro responde a distintas lógicas según el régimen de mercado dominante, y cualquier análisis que ignore esa heterogeneidad dinámica tendrá un poder explicativo y predictivo limitado.

El machine learning, lejos de ser una caja negra que mejora la predicción sin decir nada sobre los mecanismos, confirma a través del análisis SHAP las mismas jerarquías que la econometría. Esta convergencia metodológica —el hallazgo quizás más satisfactorio del trabajo— sugiere que las conclusiones sobre los determinantes del oro son sólidas frente a la elección del enfoque analítico. Y esa robustez, en un activo tan sometido a narrativas y ciclos de opinión como el oro, tiene un valor propio.

---

## Referencias de este capítulo

- Baur, D. G., & Lucey, B. M. (2010). Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold. *Financial Review, 45*(2), 217–229.
- Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.
- Erb, C. B., & Harvey, C. R. (2013). The golden dilemma. *Financial Analysts Journal, 69*(4), 10–42.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica, 46*(6), 1251–1271.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67.
- O'Connor, F. A., Lucey, B. M., Batten, J. A., & Baur, D. G. (2015). The financial economics of gold — a survey. *International Review of Financial Analysis, 41*, 186–205.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica, 48*(1), 1–48.
- World Gold Council. (2023). *Gold Demand Trends: Full Year 2023.* World Gold Council.
- World Gold Council. (2024). *Gold Demand Trends: Full Year 2024.* World Gold Council.

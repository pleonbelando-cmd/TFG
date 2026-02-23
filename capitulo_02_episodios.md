# Capítulo 2 (Ampliación) — Los cinco episodios históricos: cómo se comportaron los catalizadores

> Esta sección amplía el apartado 2.5 del Capítulo 2 y constituye el hilo narrativo que recorrerá todo el TFG. Cada episodio se analiza con los mismos catalizadores identificados en el Capítulo 3, creando un puente explícito entre la teoría y el análisis empírico de los Capítulos 5 y 6.

---

## Episodio I — La Crisis Financiera Global (2007-2009): el safe haven que primero falló y luego triunfó

### El contexto

La crisis financiera global de 2008 nació en el mercado hipotecario subprime estadounidense y se convirtió en la mayor perturbación del sistema financiero desde la Gran Depresión. La quiebra de Lehman Brothers el 15 de septiembre de 2008 es el momento simbólico de la crisis, pero el deterioro comenzó meses antes y la recuperación del oro tardó más de lo que la narrativa convencional sugiere.

### La cronología del precio y el papel de los catalizadores

**Fase 1 — El primer máximo y la corrección (marzo-octubre 2008).** El oro llegó a superar los 1.000 dólares por onza en marzo de 2008, impulsado por el colapso de Bear Stearns, la debilidad del dólar (DXY en mínimos de varios años) y las expectativas de inflación crecientes. Fue el primer contacto del metal con el nivel psicológico de cuatro dígitos desde que el mercado era libre.

Lo que ocurrió a continuación contradijo la narrativa del oro como refugio perfecto. Desde el máximo de marzo hasta el mínimo de octubre de 2008, el oro perdió un **30%**, tocando fondo en 692,50 dólares el 24 de octubre. El S&P 500 cayó más del 40% en ese mismo periodo, pero el oro no fue el refugio que los manuales prometían. La explicación: en la fase más aguda del pánico de liquidez, los inversores institucionales —fondos de cobertura, bancos de inversión— se vieron obligados a liquidar todo lo que pudiera convertirse en efectivo, incluido el oro, para atender los márgenes de sus posiciones. No fue un fallo del oro como activo; fue una crisis de liquidez en la que el único activo aceptable era el dólar en efectivo. El VIX, que llegó a superar los 80 puntos en octubre de 2008 —el máximo histórico registrado—, ilustra la magnitud del pánico.

**Tabla de catalizadores en la fase aguda (septiembre-octubre 2008):**

| Catalizador | Comportamiento | Efecto sobre el oro |
|---|---|---|
| DXY | Fuerte apreciación (+15%) | Negativo (presión bajista) |
| TIPS 10Y | Caída en rendimientos reales | Positivo, pero superado por liquidez |
| VIX | Subida a 80+ (máximo histórico) | Positivo en teoría, negativo por ventas forzadas |
| S&P 500 | Colapso -40% en semanas | Negativo (correlación positiva en el pánico) |
| WTI | Caída de $147 a $30 en meses | Negativo |

**Fase 2 — La recuperación explosiva (noviembre 2008 - agosto 2011).** Una vez pasado el pánico de liquidez, los catalizadores estructurales tomaron el mando. La Reserva Federal comenzó el primer programa de relajación cuantitativa (QE1) en noviembre de 2008, inyectando 1,75 billones de dólares en el sistema. Los tipos nominales se redujeron a cero y los tipos reales cayeron a territorio negativo. El dólar, que se había apreciado por la demanda de liquidez, inició un nuevo ciclo bajista. El oro inició su gran recuperación: desde los 692,50 dólares del mínimo de octubre de 2008 hasta los 1.921 dólares del máximo de septiembre de 2011, acumuló una ganancia del **177% en menos de tres años**. Esta es la mayor recuperación documentada del metal en el periodo de mercado libre.

### Lección econométrica del episodio

Este episodio ilustra dos fenómenos que el modelo econométrico del Capítulo 5 debe ser capaz de capturar:

Primero, la **no linealidad de la relación VIX-oro**: a niveles moderados del VIX (15-25), la correlación VIX-oro es positiva moderada; a niveles extremos (>40), la demanda de liquidez en dólares puede dominar transitoriamente y la correlación se vuelve negativa. Esta asimetría es precisamente el tipo de relación que el modelo GJR-GARCH captura explícitamente.

Segundo, el **retardo en la transmisión del estímulo monetario**: la relación TIPS-oro no es instantánea. El mercado tardó varios meses en incorporar las implicaciones de las tasas reales negativas en el precio del oro. Los modelos VAR con retardos múltiples están diseñados para capturar exactamente este tipo de dinámica de transmisión diferida.

---

## Episodio II — El pico post-QE (2011): cuando todos los catalizadores se alinearon

### El contexto

El 5 de septiembre de 2011, el precio del oro alcanzó 1.921,17 dólares por onza, el máximo nominal que se mantendría vigente durante nueve años. Este no fue un pico especulativo aislado: fue la culminación de una confluencia extraordinaria de todos los catalizadores identificados en el Capítulo 3 apuntando simultáneamente en la misma dirección alcista.

### Los catalizadores en su punto de máxima alineación

**Tipos reales en mínimos históricos.** El QE2 (noviembre 2010) y la operación Twist (septiembre 2011) redujeron el rendimiento real del TIPS a 10 años hasta valores próximos a cero e incluso negativos. El coste de oportunidad del oro era prácticamente inexistente.

**Dólar en mínimos multianuales.** El índice DXY cayó hasta aproximadamente 73 puntos en 2011, el nivel más bajo desde su creación. Cada unidad de depreciación del dólar era gasolina para el precio del oro.

**Inflación en alza.** El IPC estadounidense superó el 3% en 2011, y los breakevens de inflación a 10 años se situaban en torno al 2,5%. La narrativa del oro como cobertura inflacionaria —aunque Erb y Harvey demostrarían más tarde que es poco fiable— tenía en ese momento toda la apariencia de ser verdad.

**Tormenta geopolítica perfecta.** El año 2011 acumuló una concentración de riesgos geopolíticos sin precedentes: la Primavera Árabe en Túnez, Egipto, Libia y Siria; la intervención militar de la OTAN en Libia; las huelgas generales en Grecia ante los recortes de la troika; los disturbios en el Reino Unido; y la crisis de la deuda soberana de la eurozona con Italia y España bajo presión de los mercados. El VIX se mantuvo elevado durante meses, reflejando una ansiedad de mercado sostenida, no episódica.

**Compras de bancos centrales como nuevo actor.** Los bancos centrales de economías emergentes —especialmente China, India, Rusia y México— se convirtieron por primera vez desde los años setenta en compradores netos de oro. En 2011, el sector oficial compró 457 toneladas netas, el mayor registro en décadas hasta ese momento. Este factor estructural nuevo dio al mercado un suelo de demanda que sustentó precios que de otra manera podrían haber parecido especulativos.

### La corrección de 2013: el episodio dentro del episodio

La caída que siguió al pico de 2011 es casi tan instructiva como la subida. El 15 de abril de 2013, el oro perdió un 13% en dos jornadas —la mayor caída en treinta años— empujado por rumores de que Chipre podría vender sus reservas de oro para financiar el rescate bancario, y por la publicación de actas de la Fed que sugerían una posible reducción del QE. El precio cayó de 1.600 a 1.321 dólares en 48 horas.

Este episodio —el llamado *taper tantrum* del oro— demuestra la importancia de las expectativas sobre los tipos futuros, no solo de los tipos presentes. El oro no cayó porque los tipos reales hubieran subido ya: cayó porque el mercado anticipó que subirían. Esta distinción entre niveles actuales y expectativas futuras es econométricamente relevante: los modelos que solo incluyen el nivel actual del TIPS pueden estar subestimando el poder explicativo de los cambios en las expectativas sobre tipos, un argumento adicional para incluir los breakevens de inflación como variable separada.

---

## Episodio III — COVID-19 (2020): el crash más rápido y la recuperación más potente

### El contexto

La pandemia de COVID-19 generó el desplome bursátil más rápido de la historia moderna: el S&P 500 perdió el 34% en apenas 33 días de cotización entre el 19 de febrero y el 23 de marzo de 2020, batiendo el récord de velocidad de caída que había establecido el crash de 1929.

### La paradoja de marzo: el oro también cayó

En los primeros días del pánico de marzo de 2020, el oro replicó el comportamiento de 2008: también cayó. El 16 de marzo, cuando los mercados de acciones derrumbaban con pérdidas diarias del 10-12%, el oro perdió un 4%. El mínimo del año se alcanzó el 17 de marzo con 1.472,35 dólares por onza.

El mecanismo fue idéntico al de 2008: ventas forzadas para obtener liquidez en dólares. Los fondos de cobertura, que mantenían posiciones largas en oro como cobertura de cartera, las liquidaron para satisfacer márgenes en sus posiciones de renta variable. El VIX cerró el 18 de marzo en 85,47 puntos, superando el máximo de 2008 y convirtiéndose en el dato de volatilidad más alto jamás registrado.

### La respuesta sin precedentes y la gran subida

La Reserva Federal actuó con una velocidad e intensidad que no tiene precedentes históricos. En menos de dos semanas:
- Redujo el tipo de los fondos federales a cero en dos reuniones de emergencia (3 y 15 de marzo)
- Anunció compras ilimitadas de bonos del Tesoro y titulizaciones hipotecarias
- Inyectó más de 3 billones de dólares en el sistema entre marzo y junio de 2020

El efecto sobre los tipos reales fue inmediato y devastador para el coste de oportunidad del oro: el rendimiento del TIPS a 10 años pasó de +0,1% en febrero a -1,0% en agosto de 2020. Nunca antes habían estado tan bajos.

El resultado fue la mayor entrada de capital en ETFs de oro de la historia: 734 toneladas en la primera mitad de 2020, equivalentes a 39.500 millones de dólares. El 6 de agosto de 2020, el oro estableció su nuevo récord histórico: **2.067,15 dólares por onza**, superando el pico de 2011 por primera vez en nueve años.

### Los catalizadores en agosto de 2020: el cuadro completo

| Catalizador | Nivel aproximado | Contribución al precio del oro |
|---|---|---|
| TIPS 10Y | -1,0% (mínimo histórico) | Máxima (coste de oportunidad cero) |
| DXY | ~93 (depreciado) | Alta (dólar débil) |
| VIX | ~24 (moderado post-pánico) | Moderada (incertidumbre persistente) |
| Inflación esperada (BE) | ~1,8% (en alza) | Moderada |
| ETF flows | +734 ton en H1 | Alta (demanda directa) |
| Google Trends "gold" | Pico histórico en marzo | Precede la subida |

### Lección econométrica del episodio

El episodio COVID ilustra dos aspectos metodológicos importantes. Primero, la **asimetría temporal de los catalizadores**: Google Trends y la búsqueda masiva de información sobre el oro precedieron en semanas a los flujos reales hacia los ETFs, que a su vez precedieron en meses al nuevo máximo del precio. Este escalonamiento temporal es relevante para la especificación de retardos en el modelo VAR. Segundo, la **capacidad del modelo LSTM** para aprender secuencias: la red neuronal recurrente del Capítulo 6 está específicamente diseñada para capturar este tipo de dependencias donde la señal de hoy (búsquedas de Google) precede al efecto económico de mañana (precio del oro).

---

## Episodio IV — El ciclo de subidas de tipos (2022-2024): la paradoja que rompió el modelo estándar

### El contexto

Entre marzo de 2022 y julio de 2023, la Reserva Federal ejecutó la serie de subidas de tipos más agresiva desde los años ochenta: de 0,00-0,25% hasta 5,25-5,50% en apenas 16 meses. El rendimiento real del TIPS a 10 años pasó de -1,0% en enero de 2022 a +2,5% en octubre de 2023.

Desde la perspectiva del modelo estándar de determinantes del oro, este movimiento debería haber sido devastador para el metal. Un aumento de 350 puntos básicos en los tipos reales equivale, por la relación histórica documentada en el Capítulo 3, a una caída potencial del precio del oro de más del 50%.

No ocurrió nada parecido.

### Qué pasó realmente

El oro cerró 2022 con una caída del 0,3%: prácticamente plano pese al shock de tipos más severo en cuatro décadas. En 2023, subió un 13%. En 2024, superó los 2.400 dólares por onza, alcanzando nuevos máximos históricos por encima del pico de 2020.

¿Por qué falló el modelo estándar? La respuesta está en el factor que la ecuación clásica no incluye: **la demanda soberana de los bancos centrales**. El 24 de febrero de 2022, Rusia invadió Ucrania. En los días siguientes, Estados Unidos y sus aliados congelaron aproximadamente 300.000 millones de dólares en reservas del Banco Central de Rusia —activos denominados en dólares, euros, yenes y libras esterlinas depositados en bancos occidentales—. Fue la mayor confiscación de reservas soberanas de la historia.

El mensaje para el resto del mundo fue inequívoco: las reservas en moneda extranjera no son propiedad soberana incondicional; pueden ser congeladas por una decisión política de la potencia emisora. Los bancos centrales de decenas de países —especialmente los que mantienen relaciones ambiguas con Occidente— comenzaron a reconsiderar su composición de reservas. El oro, que no es deuda de nadie, no puede ser congelado, no tiene riesgo de contraparte y ha funcionado como reserva de valor durante cinco mil años, se convirtió en la alternativa obvia.

El resultado fue la mayor oleada de compras de bancos centrales en décadas: **1.082 toneladas netas en 2022** y **1.037 toneladas netas en 2023** —ambas superando el récord anterior de 625 toneladas de 2010 por un margen enorme—. Este factor de demanda estructural compensó con creces el efecto negativo de los tipos reales altos, produciendo la "paradoja de 2022-2024" que ningún modelo unidimensional podía predecir.

### Lección econométrica del episodio

Este episodio justifica de forma poderosa el análisis de **ruptura estructural** que se realizará en el Capítulo 5. Si se estima el modelo VAR con datos hasta 2021 y se usa para predecir el periodo 2022-2024, el error de predicción será grande —el modelo no puede anticipar la emergencia del factor de reservas soberanas como determinante dominante—. El test de Chow y el análisis CUSUM detectarán este punto de quiebre y motivarán la estimación de un modelo separado para el subperiodo 2022-2025, donde el peso del catalizador "reservas de bancos centrales" habrá de ser mucho mayor.

Además, el episodio ilustra una **limitación fundamental de todos los modelos**: ningún modelo puede capturar correctamente factores que están ausentes de su especificación. La de-dolarización geopolítica es un fenómeno estructural nuevo que no existía en el periodo 2000-2021 con la misma intensidad, y su incorporación al modelo (mediante las reservas de bancos centrales con retardo) es necesaria pero imperfecta.

---

## Episodio V — 2025: el año en que convergieron todas las fuerzas

### El contexto

Si el episodio de 2022-2024 fue la "paradoja que rompió el modelo", el de 2025 fue la confirmación de que ese modelo estaba siendo sustituido por uno nuevo. La convergencia de factores que se produjo en 2025 es, desde una perspectiva de análisis de serie temporal, el equivalente a un experimento natural de máxima intensidad.

### La cronología y los datos

El año comenzó con el oro ya en máximos históricos por encima de 2.600 dólares, heredando el impulso de 2024. En enero, Donald Trump tomó posesión de la presidencia de Estados Unidos y anunció inmediatamente una política arancelaria agresiva: aranceles del 25% a Canadá y México, del 10% a China, y amenazas generalizadas al resto de socios comerciales.

Los mercados reaccionaron con pánico selectivo: el S&P 500 entró en corrección, el dólar fluctuó violentamente y el oro comenzó su escalada. En **marzo de 2025**, el oro superó los 3.000 dólares por primera vez en la historia, un nivel que analistas como el propio Goldman Sachs habían predicho para "algún momento de la próxima década". Llegó en tres meses.

La subida no se detuvo. El metal estableció **53 nuevos máximos históricos** a lo largo del año —una frecuencia de más de uno por semana—, cerró en **4.549 dólares por onza en diciembre** y registró una media anual de **3.431 dólares**, un 44% por encima de la media de 2024. La subida anual del **+65%** fue la mayor desde 1979.

### Por qué 2025 fue diferente: la triple confluencia

**Primera fuerza — Safe haven de corto plazo.** La guerra arancelaria de Trump generó la mayor incertidumbre sobre el comercio global desde los años treinta. Cada anuncio de nuevos aranceles —y cada respuesta de represalia de China, la UE o Canadá— disparaba el VIX y empujaba capital hacia activos de refugio. El oro no era solo una cobertura contra la inflación esperada por los aranceles; era también una cobertura contra la incertidumbre sobre el propio sistema de comercio internacional.

**Segunda fuerza — Entorno financiero favorable.** Simultáneamente, los bancos centrales reanudaron los recortes de tipos. El BCE bajó hasta el 2,25% —su nivel más bajo desde finales de 2022— y la Fed comenzó su propio ciclo de recortes. Los tipos reales cayeron de nuevo, reduciendo el coste de oportunidad del oro. Los breakevens de inflación, al alza por el efecto de los aranceles sobre los precios domésticos, añadían otro argumento para el metal.

**Tercera fuerza — Demanda soberana acelerada.** El proceso de de-dolarización se aceleró: según el World Gold Council (2025), el **73% de los bancos centrales** declaró su intención de reducir sus reservas en dólares en los próximos años, el porcentaje más alto jamás registrado. Las compras netas de oro por parte del sector oficial se mantuvieron en niveles récord por tercer año consecutivo.

### La tabla de catalizadores en 2025: todos en verde

| Catalizador | Situación en 2025 | Dirección para el oro |
|---|---|---|
| DXY | Depreciación por incertidumbre arancelaria | Alcista |
| TIPS 10Y | A la baja (recortes de tipos) | Alcista |
| VIX | Elevado y volátil (guerra comercial) | Alcista |
| Inflación esperada | Al alza (aranceles = inflación importada) | Alcista |
| S&P 500 | En corrección (uncertainty) | Alcista |
| WTI | Volátil al alza | Alcista |
| Reservas bancos centrales | Compras récord (tercer año) | Alcista |
| ETF flows | Máximos históricos ($26bn en Q3) | Alcista |
| Google Trends "gold" | Picos históricos | Alcista |

Por primera vez en el periodo de análisis, **todos los catalizadores del modelo apuntaron simultáneamente en la dirección alcista** durante un periodo prolongado. El resultado —una subida del 65%— es consistente con la predicción del modelo: cuando todas las variables ejercen presión en la misma dirección, el efecto sobre el precio es amplificado y sostenido.

### Lección econométrica del episodio

El episodio 2025 proporciona el banco de pruebas *out-of-sample* más exigente que podría diseñarse para los modelos del Capítulo 6. Los modelos de machine learning entrenados sobre el periodo 2000-2024 serán evaluados sobre 2025, donde la señal de los catalizadores es excepcionalmente clara. Un modelo bien especificado debería haber detectado la convergencia alcista y generado predicciones ajustadas. Un modelo mal especificado —por ejemplo, uno que no incluya el factor de reservas de bancos centrales— habrá subestimado sistemáticamente el precio durante todo el año.

Esta evaluación *out-of-sample* en el episodio más extremo del siglo es, en sí misma, una contribución metodológica: permite distinguir qué variables fueron informativas y cuáles resultaron redundantes o irrelevantes en el entorno más exigente.

---

## Síntesis: lo que los cinco episodios enseñan al modelo

Los cinco episodios documentados en este capítulo permiten extraer cinco conclusiones metodológicas que guiarán el análisis empírico de los Capítulos 5 y 6:

**1. Los catalizadores no operan de forma independiente.** En todos los episodios, el precio del oro fue determinado por la interacción entre múltiples fuerzas, algunas reforzándose y otras contrarrestándose. El modelo multivariante (VAR) está mejor equipado que cualquier regresión bivariante para capturar estas interacciones.

**2. La relación entre catalizadores y precio es no lineal y régimen-dependiente.** El VIX tiene un efecto alcista moderado en condiciones normales y un efecto ambiguo en pánico extremo. Los TIPS tienen un efecto negativo sobre el oro que se amplifica cuando los tipos reales son negativos y se comprime cuando otros factores dominan. Los modelos de machine learning tienen ventaja sobre la econometría lineal clásica para capturar estas no linealidades.

**3. Los puntos de ruptura estructural son reales y relevantes.** La crisis de 2008, el pico de 2011, el COVID y la invasión de Ucrania son potenciales puntos de ruptura en las relaciones entre variables. La homogeneidad del modelo a lo largo del periodo completo es una hipótesis fuerte que debe testarse, no asumirse.

**4. Los retardos importan.** En todos los episodios, los catalizadores de sentimiento (Google Trends) precedieron a los flujos de ETF, que a su vez precedieron al precio. La especificación de los retardos óptimos en el VAR no es un detalle técnico: es metodológicamente crucial.

**5. Ningún modelo es completo.** La emergencia de la de-dolarización soberana como factor dominante en 2022-2025 no era predecible con los modelos estimados sobre el periodo anterior. Las limitaciones del modelo —que se abordarán honestamente en las Conclusiones— son en parte inherentes a cualquier enfoque que use historia para predecir el futuro.

---

## Referencias de esta sección

- Baur, D. G., & McDermott, T. K. (2010). Is gold a safe haven? International evidence. *Journal of Banking & Finance, 34*(8), 1886–1898.
- Bureau of Labor Statistics. (2013). Gold prices during and after the Great Recession. *Beyond the Numbers, 2*(14).
- Gainesville Coins. (2024). Historical gold prices: 50 years of market lessons.
- World Economic Forum. (2020). Why has the price of gold reached an all-time high?
- World Gold Council. (2019). The gold perspective: 10 years after Lehman Brothers failed. *Gold Investor.*
- World Gold Council. (2023). *Gold Demand Trends: Full Year 2023.*
- World Gold Council. (2025). *Gold Demand Trends: Full Year 2025.*
- World Gold Council. (2025). *Gold ETF: Holdings and Flows — Q3 2025.*

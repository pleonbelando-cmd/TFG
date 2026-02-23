# Capítulo 7: Extensión predictiva con Machine Learning

> *Nota metodológica:* Este capítulo implementa modelos de machine learning como extensión complementaria al análisis econométrico de los Capítulos 5 y 6. Las técnicas aquí aplicadas —gradient boosting, random forests y redes neuronales recurrentes— van más allá del temario del curso de Econometría III, pero se incluyen porque aportan una perspectiva predictiva que contrasta directamente con la econometría clásica y enriquece la respuesta a la tercera pregunta de investigación del trabajo.

## 7.1 Motivación y enfoque

Los modelos econométricos del Capítulo 5 identifican relaciones estructurales de largo plazo entre el oro y sus determinantes. Sin embargo, los modelos lineales presentan limitaciones cuando las relaciones son no lineales, cuando los efectos de régimen son importantes, o cuando el interés es la predicción de corto plazo. El machine learning ofrece una perspectiva complementaria: en lugar de partir de relaciones teóricas *a priori*, los algoritmos aprenden patrones directamente de los datos, capturando no linealidades sin especificación manual.

Este capítulo implementa tres arquitecturas: **XGBoost** (gradient boosting sobre árboles), **Random Forest** (conjunto de árboles independientes) y **LSTM** (red neuronal recurrente con memoria de largo-corto plazo). Todos se evalúan con el mismo protocolo de validación temporal *walk-forward*, que replica la situación real de un inversor que solo dispone de información pasada en cada momento de predicción.

---

## 7.2 Datos y características

La matriz de características se construye a partir de las 312 observaciones mensuales del dataset maestro (enero 2000 — diciembre 2025), con las siguientes transformaciones:

- **Retornos logarítmicos** para variables de precio no estacionarias (DXY, WTI, S&P 500)
- **Niveles** para variables ya estacionarias (TIPS, VIX, CPI, Breakeven)
- **Retardos 1, 2, 3** de cada variable para capturar dependencias temporales (sin look-ahead bias)
- **Momentum del oro**: media móvil de retornos (3 y 6 meses) y volatilidad realizada (3 meses)
- **Dummy de régimen**: variable binaria (`is_crisis`) para los cinco episodios históricos

La matriz resultante tiene **35 características** y **271 observaciones efectivas** (tras eliminar los NaN iniciales por los retardos). El reparto de la muestra se resume en la Tabla 7.1.

**Tabla 7.1 — Diseño de la evaluación walk-forward**

| Concepto | Valor |
|---|---|
| Período total efectivo | Abril 2003 — Octubre 2025 |
| Muestra de entrenamiento inicial | 162 obs. (Abril 2003 — Sept. 2016) |
| Muestra de test (walk-forward) | 109 obs. (Oct. 2016 — Oct. 2025) |
| Variable objetivo | Retorno logarítmico mensual del oro (pp) |
| Número de características ($p$) | 35 |

---

## 7.3 Metodología: validación walk-forward

La validación cruzada estándar (*k-fold*) introduce *look-ahead bias* en series temporales al barajar aleatoriamente las observaciones: el modelo podría entrenarse con datos de 2022 para predecir 2015. La **walk-forward con ventana expandible** elimina este problema: el modelo se entrena en $[1, t-1]$ y predice $t$; a continuación, el entrenamiento se amplía a $[1, t]$ y se predice $t+1$, nunca incluyendo información posterior al instante de predicción (López de Prado, 2018).

Para acotar el coste computacional, el reentrenamiento completo se realiza cada 3 pasos para XGBoost y cada 6 para Random Forest y LSTM.

**Métricas de evaluación:**

| Métrica | Descripción | Por qué importa |
|---|---|---|
| RMSE | Raíz del error cuadrático medio (pp) | Penaliza errores grandes |
| MAE | Error absoluto medio (pp) | Robusta a outliers |
| MAPE | Error porcentual medio absoluto | Comparabilidad relativa |
| **DA** | % meses con dirección acertada | **Más relevante operativamente** |

El benchmark de referencia es el **modelo naive** (paseo aleatorio): predice que el retorno del mes siguiente es igual al del mes anterior. Un modelo útil debe superar al naive en al menos una métrica clave.

---

## 7.4 Los tres modelos

**XGBoost** construye árboles de decisión secuencialmente: cada árbol nuevo corrige los errores del anterior. La configuración es conservadora (profundidad máxima 3, tasa de aprendizaje 0,05, regularización L1 y L2) para evitar sobreajuste con solo 162 observaciones de entrenamiento inicial.

**Random Forest** construye 300 árboles en paralelo, cada uno sobre un subconjunto aleatorio de observaciones y variables. La decorrelación entre árboles reduce la varianza del conjunto y lo hace especialmente robusto en muestras pequeñas.

**LSTM** (*Long Short-Term Memory*) es una red neuronal recurrente que procesa secuencias temporales de 6 meses de historia y aprende qué información retener y cuál olvidar mediante compuertas internas. La arquitectura es intencionadamente simple (32 unidades, 1 capa) para evitar sobreajuste. Se implementa *early stopping* con paciencia de 20 épocas sobre el 15% final de cada conjunto de entrenamiento.

---

## 7.5 Resultados comparativos

La Tabla 7.2 presenta las cuatro métricas de evaluación sobre los 109 meses del período de test.

**Tabla 7.2 — Comparativa de modelos predictivos (walk-forward, oct. 2016 — oct. 2025)**

| Modelo | RMSE (pp) | MAE (pp) | MAPE (%) | DA (%) |
|---|---|---|---|---|
| Naive (random walk) | 5,054 | 4,043 | 244,9 | 55,9 |
| XGBoost | 4,340 | 3,476 | 308,0 | 52,3 |
| Random Forest | 3,882 | 3,181 | 226,5 | 58,7 |
| **LSTM** | **3,815** | **3,142** | 278,8 | **61,5** |

*Nota: La desviación típica incondicional del retorno del oro en la muestra completa es 4,65 pp.*

Tres conclusiones destacan. Primero, la LSTM obtiene el mejor rendimiento en todas las métricas clave (RMSE y DA), lo que indica que la estructura recurrente capta dependencias temporales que los modelos de árboles no explotan. Segundo, el Random Forest supera al XGBoost en todas las métricas, resultado frecuente en series financieras cortas (n < 500) donde el *bagging* es más robusto que el *boosting* secuencial. Tercero, XGBoost obtiene una DA inferior al benchmark naive (52,3% vs. 55,9%), lo que indica que reduce el error de magnitud pero introduce ruido en la dirección del movimiento —el aspecto más relevante para decisiones de inversión.

---

## 7.6 Interpretabilidad: análisis SHAP

Los modelos de árboles permiten calcular los **valores SHAP** (*SHapley Additive exPlanations*), que descomponen cada predicción en la contribución marginal de cada variable. Para XGBoost se utilizan valores TreeSHAP exactos (Lundberg et al., 2020).

La Tabla 7.3 presenta las 8 variables más influyentes, medidas por la importancia media de sus valores SHAP en absoluto sobre el período de test.

**Tabla 7.3 — Top 8 variables por importancia SHAP (XGBoost, período de test)**

| Rango | Variable | SHAP $|\bar{\phi}|$ | Interpretación |
|---|---|---|---|
| 1 | CPI YoY (t-1) | 0,954 | Inflación pasada: predictor más potente del retorno del oro |
| 2 | TIPS 10Y (t-2) | 0,617 | Tipos reales retardados 2 meses (consistente con causalidad Granger) |
| 3 | Ret. oro (t-1) | 0,526 | Momentum de 1 mes del propio oro |
| 4 | Breakeven (t-3) | 0,485 | Expectativas inflacionarias anticipadas 3 meses |
| 5 | WTI (t-2) | 0,423 | Petróleo como proxy de presiones inflacionarias globales |
| 6 | S&P 500 (t-1) | 0,397 | Sustitución renta variable-oro rezagada |
| 7 | Vol3 oro | 0,379 | Volatilidad realizada reciente del oro |
| 8 | DXY (t-3) | 0,329 | Inercia del ciclo del dólar |

El *summary plot* de SHAP confirma las direcciones esperadas: inflación alta → SHAP positivo (predicción alcista del oro), tipos reales altos → SHAP negativo (coste de oportunidad), S&P 500 alto → SHAP negativo (sustitución). Estos resultados son plenamente coherentes con los hallazgos del análisis econométrico del Capítulo 5 y del panel cross-country del Capítulo 6.

El análisis *waterfall* de tres episodios representativos (GFC 2008, COVID 2020, diciembre 2025) muestra que el peso relativo de cada variable cambia entre regímenes: en episodios de crisis, el VIX y el momentum amplifican la señal; en entornos de tipo de interés excepcionales, los TIPS dominan; en diciembre 2025, la confluencia de todos los catalizadores genera la señal SHAP más intensa del período analizado.

---

## 7.7 Conclusiones del capítulo

1. La **LSTM es el modelo más preciso** (DA = 61,5%), mejorando en 5,6 puntos porcentuales al benchmark naive, gracias a su capacidad de capturar dependencias temporales de medio plazo.

2. El **análisis SHAP converge con la econometría**: los determinantes más importantes del oro —tanto en el VECM como en ML— son los tipos de interés reales y la inflación, lo que valida la especificación del modelo a través de dos metodologías independientes.

3. Los modelos de ML **complementan, no sustituyen**, al análisis econométrico: el VECM identifica relaciones estructurales de largo plazo y cuantifica el ajuste al equilibrio; el ML mejora la predicción de corto plazo y cuantifica el peso relativo de las variables en cada régimen de mercado.

4. La **limitación principal** es el tamaño de la muestra (271 observaciones, 35 features): los resultados deben interpretarse con cautela y se beneficiarían de replicación con muestras más largas o a mayor frecuencia.

---

## Referencias de este capítulo

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Liang, C., Li, Y., et al. (2023). Forecasting gold price using machine learning methodologies. *Chaos, Solitons & Fractals, 173.*
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30.*
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*(1), 56–67.
- Plakandaras, V., et al. (2022). Forecasting the price of gold using machine learning methodologies. *Applied Economics, 54*(33), 3768–3783.

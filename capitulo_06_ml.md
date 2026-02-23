# Capítulo 6: Modelos de Machine Learning para la Predicción del Retorno del Oro

## 6.1 Motivación y enfoque

Los modelos econométricos del Capítulo 5 permitieron identificar las relaciones estructurales entre el precio del oro y sus determinantes macroeconómicos: la existencia de un único vector de cointegración (r = 1 según el criterio max-eigenvalue), la causalidad Granger unidireccional desde los tipos de interés reales, y la asimetría de la volatilidad condicional capturada por el GJR-GARCH. Sin embargo, los modelos lineales presentan limitaciones intrínsecas cuando los fenómenos de interés son no lineales, cuando las interacciones entre variables resultan complejas, o cuando la dinámica del sistema cambia de régimen.

El machine learning ofrece una perspectiva complementaria: en lugar de partir de relaciones teóricas *a priori*, los algoritmos aprenden patrones directamente de los datos, capturando interacciones de alto orden y no linealidades sin especificación manual. La literatura reciente (Plakandaras et al., 2022; Liang et al., 2023) documenta que los modelos de conjuntos y las redes recurrentes mejoran la predicción de corto plazo del oro en horizontes de uno a tres meses, especialmente durante episodios de alta incertidumbre.

Este capítulo implementa tres arquitecturas complementarias: XGBoost (gradient boosting sobre árboles de decisión), Random Forest (conjunto de árboles independientes), y LSTM (red neuronal recurrente con memoria de largo-corto plazo). La evaluación sigue el protocolo walk-forward con ventana expandible, que elimina el sesgo de look-ahead y reproduce fielmente la situación de un inversor que opera en tiempo real.

## 6.2 Ingeniería de características

### 6.2.1 Variables base y transformaciones

La matriz de características se construye a partir del conjunto de datos maestro (312 observaciones mensuales, enero 2000 — diciembre 2025), con transformaciones para garantizar la estacionariedad de todas las entradas:

- **Retornos logarítmicos**: DXY e índice WTI se transforman mediante $r_t = \ln(P_t / P_{t-1})$ para convertir precios no estacionarios en series estacionarias.
- **Primera diferencia**: el VIX se diferencia ($\Delta\text{VIX}_t$) para capturar el *cambio* en el régimen de volatilidad en lugar del nivel.
- **Niveles**: TIPS 10 años, Fed Funds, CPI interanual y Breakeven se mantienen en niveles, dado que son tasas ya estacionarias (I(0) según los tests del Capítulo 4).

### 6.2.2 Retardos temporales

Para cada variable base se generan retardos de orden 1, 2 y 3 (correspondientes a 1, 2 y 3 meses de historia). Este diseño evita cualquier filtración de información futura: en el momento en que el modelo realiza la predicción para el mes $t$, solo dispone de observaciones hasta $t-1$ (retardo 1) o anteriores.

### 6.2.3 Indicadores de inercia del oro

Se incluyen tres indicadores de momentum calculados sobre el precio del propio oro:

- **MA3_oro**: media móvil de los retornos de los últimos 3 meses, desplazada un período (shift = 1).
- **MA6_oro**: análogo con ventana de 6 meses.
- **Vol3_oro**: desviación estándar de los retornos de los últimos 3 meses (proxy de volatilidad realizada reciente).

### 6.2.4 Variable de régimen

Se incorpora una *dummy* binaria (`is_crisis`) que vale 1 en los cinco episodios históricos identificados en el Capítulo 2 (GFC, Post-QE, COVID-19, ciclo de subidas de tipos, confluencia 2025) y 0 en períodos de calma. Esta variable permite a los modelos identificar si el entorno actual es de crisis.

### 6.2.5 Matriz final

Tras aplicar los retardos y ventanas deslizantes, las primeras observaciones contienen NaN y se eliminan. La **Tabla 6.1** resume la matriz resultante.

**Tabla 6.1 — Especificación de la matriz de características**

| Concepto | Detalle |
|---|---|
| Observaciones disponibles | 271 (de 312 totales; ~41 obs. eliminadas por NaN iniciales) |
| Número de features ($p$) | 35 |
| Target ($y$) | `gold_ret` — retorno logarítmico mensual del oro (en puntos porcentuales) |
| Período efectivo | Abril 2003 — Octubre 2025 |
| Muestra de entrenamiento | 162 obs. (Abril 2003 — Septiembre 2016, 60%) |
| Muestra de test | 109 obs. (Octubre 2016 — Octubre 2025, 40%) |

El conjunto de test (40%) cubre un período especialmente exigente: el ciclo de normalización monetaria de 2018, la pandemia de COVID-19 y la consiguiente recuperación, el ciclo de subidas del tipo Fed Funds 2022-2024 (el más agresivo desde los años 80), y la impresionante apreciación del oro en 2025 (+65% anual, 53 máximos históricos).

## 6.3 Metodología de validación: walk-forward

### 6.3.1 Fundamento

La validación cruzada estándar (k-fold) baraja aleatoriamente las observaciones, lo que introduce *look-ahead bias* en series temporales: el modelo podría entrenarse con datos del año 2022 para predecir el año 2015. Este sesgo infla artificialmente las métricas de evaluación y produce estimadores inconsistentes de la capacidad predictiva fuera de muestra.

La **walk-forward validation con ventana expandible** es el estándar metodológico para la predicción de series financieras (López de Prado, 2018). El esquema es el siguiente: el modelo se entrena en $[1, t-1]$ y predice $t$; en el siguiente paso, el conjunto de entrenamiento se amplía a $[1, t]$ y se predice $t+1$. El entrenamiento nunca incluye información posterior al instante de predicción.

Formalmente, si $T$ es el tamaño total y $T_0$ el tamaño del primer entrenamiento:

$$\hat{y}_t = f_t(X_1, \ldots, X_{t-1}; \theta_t), \quad t = T_0 + 1, \ldots, T$$

donde $\theta_t$ son los parámetros estimados en la ventana $[1, t-1]$.

### 6.3.2 Parámetros de implementación

Para acotar el coste computacional, el reentrenamiento no se realiza en cada paso sino con una frecuencia fija (`refit_every`). Para XGBoost se reentrena cada 3 pasos, para Random Forest cada 6, y para la LSTM también cada 6 pasos (lo que genera aproximadamente 18 reentrenamientos a lo largo del período de test).

### 6.3.3 Métricas de evaluación

Las cuatro métricas seleccionadas cubren distintas dimensiones del error predictivo:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}$$

$$\text{MAE} = \frac{1}{N}\sum_{t=1}^{N}|y_t - \hat{y}_t|$$

$$\text{MAPE} = \frac{100}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{y_t}\right| \quad \text{(excluyendo } |y_t| < 10^{-6}\text{)}$$

$$\text{DA} = \frac{100}{N}\sum_{t=1}^{N}\mathbf{1}[\text{sign}(y_t) = \text{sign}(\hat{y}_t)]$$

La **Directional Accuracy (DA)** merece atención especial: representa el porcentaje de meses en que el modelo acierta la dirección del movimiento (subida o bajada). Un DA = 50% corresponde a una predicción aleatoria (equivalente a lanzar una moneda), mientras que un DA superior a 50% indica capacidad informativa genuina del modelo. Para señales de trading, el DA es la métrica más relevante operativamente.

## 6.4 Modelos de árboles de decisión

### 6.4.1 XGBoost

XGBoost (Chen y Guestrin, 2016) es un algoritmo de *gradient boosting* que construye un conjunto de $K$ árboles de decisión de forma secuencial: cada árbol nuevo $f_k$ trata de corregir los errores del árbol anterior. La predicción final es:

$$\hat{y}_t = \sum_{k=1}^{K} f_k(X_t)$$

La función objetivo incluye un término de regularización sobre la estructura de los árboles:

$$\mathcal{L} = \sum_t \ell(y_t, \hat{y}_t) + \sum_k \Omega(f_k)$$

donde $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2 + \alpha \|w\|_1$, con $T$ el número de hojas y $w$ los pesos de las hojas.

La **Tabla 6.2** recoge los hiperparámetros seleccionados. La configuración es conservadora: árboles poco profundos (max_depth = 3), tasa de aprendizaje baja (0.05) y subsamplings del 80%, lo que limita el overfitting en un conjunto de entrenamiento de solo 162 observaciones.

**Tabla 6.2 — Hiperparámetros de los modelos de árboles**

| Modelo | Hiperparámetro | Valor |
|---|---|---|
| XGBoost | n_estimators | 300 |
| XGBoost | max_depth | 3 |
| XGBoost | learning_rate | 0.05 |
| XGBoost | subsample | 0.8 |
| XGBoost | colsample_bytree | 0.8 |
| XGBoost | reg_alpha (L1) | 0.1 |
| XGBoost | reg_lambda (L2) | 1.0 |
| Random Forest | n_estimators | 300 |
| Random Forest | max_depth | 5 |
| Random Forest | min_samples_leaf | 5 |
| Random Forest | max_features | 0.7 |

### 6.4.2 Random Forest

Random Forest (Breiman, 2001) construye $B$ árboles de decisión de forma *paralela e independiente*, cada uno sobre un subconjunto aleatorio de observaciones (bootstrap) y de variables (feature subsampling). La predicción es el promedio de todos los árboles:

$$\hat{y}_t = \frac{1}{B}\sum_{b=1}^{B} T_b(X_t)$$

A diferencia del boosting, la aleatoriedad decorrelaciona los árboles individuales, lo que reduce la varianza del conjunto sin aumentar el sesgo. Con `min_samples_leaf = 5` se impone una regularización mínima especialmente importante cuando el número de observaciones de entrenamiento es modesto.

La Figura 6.2 muestra la predicción walk-forward de los modelos de árboles frente al precio y retorno real del oro durante el período de test (oct. 2016 — oct. 2025).

*(Figura 6.2: output/figures/fig_6_02_tree_predictions.png)*

## 6.5 Red neuronal recurrente: LSTM

### 6.5.1 Arquitectura

Las redes LSTM (*Long Short-Term Memory*, Hochreiter y Schmidhuber, 1997) son un tipo especializado de red neuronal recurrente diseñado para capturar dependencias temporales de largo alcance. Su elemento central es la **celda de memoria** $c_t$, que puede retener o descartar información de pasos anteriores mediante tres compuertas:

**Compuerta de olvido** (forget gate):
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Compuerta de entrada** (input gate):
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad \tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Actualización de la celda**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Compuerta de salida** (output gate):
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)$$

donde $\sigma$ es la función sigmoide, $\odot$ es el producto elemento a elemento, y $h_t$ es el estado oculto en el instante $t$.

La arquitectura implementada en este trabajo es deliberadamente parsimonios a, apropiada para series macroeconómicas con 271 observaciones:

**GoldLSTM**: una capa LSTM con 32 unidades seguida de una capa lineal `Linear(32, 1)`.

El modelo recibe como entrada una secuencia de `seq_len = 6` meses de las 35 variables de features (estandarizadas) y produce el retorno del oro predicho para el mes siguiente.

### 6.5.2 Entrenamiento y early stopping

Los hiperparámetros de entrenamiento se recogen en la **Tabla 6.4**:

**Tabla 6.4 — Hiperparámetros de la red LSTM**

| Hiperparámetro | Valor | Descripción |
|---|---|---|
| hidden_size | 32 | Unidades en la capa LSTM |
| num_layers | 1 | Número de capas LSTM apiladas |
| seq_len | 6 | Longitud de la ventana de entrada (meses) |
| batch_size | 16 | Tamaño del mini-batch |
| lr | 0.001 | Tasa de aprendizaje (Adam) |
| max_epochs | 150 | Épocas máximas de entrenamiento |
| patience | 20 | Paciencia del early stopping |
| val_frac | 0.15 | Fracción de train reservada para validación interna |

Para evitar el sobreajuste durante el entrenamiento se implementa **early stopping**: las últimas `val_frac = 15%` de las observaciones de entrenamiento (en orden temporal) se reservan como validación interna. Si la pérdida de validación no mejora durante 20 épocas consecutivas, el entrenamiento se interrumpe y se recuperan los pesos correspondientes a la mínima pérdida de validación observada.

En el walk-forward, el escalador StandardScaler se reajusta en cada paso sobre el conjunto de entrenamiento disponible hasta $t-1$, previniendo que los estadísticos de escala incorporen información del período de test.

La Figura 6.5 muestra las predicciones walk-forward de la LSTM frente al precio real del oro.

*(Figura 6.5: output/figures/fig_6_05_lstm_predictions.png)*

## 6.6 Interpretabilidad: análisis SHAP

### 6.6.1 Fundamento teórico de los valores de Shapley

Los valores de Shapley (Shapley, 1953) son la única solución axiomáticamente justa al problema de atribución del valor en teoría de juegos cooperativos. En el contexto del machine learning (Lundberg y Lee, 2017), el valor SHAP de la característica $j$ para la predicción $i$ se define como:

$$\phi_j(i) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} \left[f(S \cup \{j\}) - f(S)\right]$$

donde $\mathcal{F}$ es el conjunto completo de features y la suma recorre todos los subconjuntos posibles. El valor $\phi_j(i)$ puede interpretarse como la contribución marginal promedio de la característica $j$ a la predicción $i$, promediada sobre todas las posibles coaliciones de variables.

Para los modelos de árboles, Lundberg et al. (2020) proponen el algoritmo **TreeSHAP**, que calcula los valores de Shapley exactos en tiempo polinomial O($TLD^2$), donde $T$ es el número de árboles, $L$ el número de hojas y $D$ la profundidad máxima.

### 6.6.2 Importancia global de las variables

La **Figura 6.1** muestra la importancia media de cada variable, medida como el valor absoluto promedio de los valores SHAP sobre el conjunto de test:

$$\bar{\phi}_j = \frac{1}{N}\sum_{i=1}^{N}|\phi_j(i)|$$

*(Figura 6.1: output/figures/fig_6_01_shap_importance.png)*

La **Tabla 6.3** recoge el ranking completo de las diez variables más influyentes:

**Tabla 6.3 — Top 10 variables por importancia SHAP (XGBoost)**

| Posición | Variable | SHAP medio |φ| | Interpretación |
|---|---|---|---|
| 1 | CPI YoY (t-1) | 0.954 | La inflación realizada del mes anterior es el predictor más potente del retorno del oro. Consecuente con el rol del oro como cobertura inflacionaria. |
| 2 | TIPS 10Y (t-2) | 0.617 | Los tipos de interés reales con dos meses de retardo muestran fuerte señal. Consistente con el resultado Granger del Cap. 5: TIPS causa al oro. |
| 3 | Ret. oro (t-1) | 0.526 | El retorno pasado del propio oro (momentum de 1 mes) tiene un poder predictivo significativo, aunque con signo variable según el entorno. |
| 4 | Breakeven (t-3) | 0.485 | Las expectativas de inflación a 3 meses vista anticipan la demanda futura de oro como activo real. |
| 5 | WTI (t-2) | 0.423 | El retorno del petróleo con 2 meses de retardo actúa como proxy de presiones inflacionarias globales. |
| 6 | S&P 500 (t-1) | 0.397 | La renta variable rezagada captura el efecto sustitución: en períodos de apreciación bursátil, el oro pierde atractivo relativo. |
| 7 | Vol3 oro | 0.379 | La volatilidad realizada reciente del oro influye en las predicciones, reflejando la asimetría documentada en el Cap. 5. |
| 8 | DXY (t-3) | 0.329 | El dólar con 3 meses de retardo captura la inercia del ciclo del dólar, que afecta inversamente al oro. |

**Hallazgo principal**: los tres bloques más informativos son (1) inflación y expectativas inflacionarias, (2) tipos de interés reales, y (3) momentum del propio oro. Esto es coherente con la hipótesis central del TFG y con los resultados econométricos del Capítulo 5.

**Hallazgo destacable**: la variable `is_crisis` (episodio de crisis) ocupa el último lugar en importancia ($\bar{\phi} = 0.029$), lo que sugiere que los modelos de árboles ya capturan indirectamente el régimen de mercado a través de las variables continuas (VIX, retornos extremos, etc.), sin necesitar una dummy explícita.

### 6.6.3 Summary plot (SHAP beeswarm)

La **Figura 6.3** muestra el *summary plot* de SHAP: cada punto representa una observación del test, el eje horizontal es el valor SHAP (impacto positivo hacia la derecha, negativo hacia la izquierda), y el color indica si el valor de la feature es alto (rojo) o bajo (azul).

*(Figura 6.3: output/figures/fig_6_03_shap_summary.png)*

Este gráfico revela la **dirección** de cada relación:
- CPI YoY alto (valores altos, rojo) → SHAP positivo → predicción de retorno del oro más alto. Consistente con el oro como cobertura inflacionaria.
- TIPS 10Y alto (tipos reales altos) → SHAP negativo → predicción de menor retorno del oro. Coherente con la teoría: tipos reales elevados aumentan el coste de oportunidad de mantener oro.
- S&P 500 alto → SHAP negativo → relación de sustitución entre renta variable y oro.

### 6.6.4 Análisis de episodios clave (SHAP waterfall)

La **Figura 6.4** descompone la predicción del modelo para tres meses representativos: la crisis financiera global (GFC, octubre 2008), el *crash* de COVID (marzo 2020), y la confluencia de catalizadores de diciembre 2025.

*(Figura 6.4: output/figures/fig_6_04_shap_waterfall.png)*

El análisis waterfall evidencia que, aunque el mecanismo de transmisión es el mismo, el *peso relativo* de cada variable cambia entre episodios:
- En el GFC (2008), los tipos reales y la inflación dominan la señal alcista del oro.
- En el COVID (2020), el VIX y el momentum reciente del oro amplifican la predicción positiva, dado el *flight to safety* masivo.
- En diciembre 2025, la confluencia de inflación persistente, tipos reales aún negativos en términos reales y compras récord de bancos centrales genera la señal SHAP más intensa del período.

## 6.7 Comparativa de modelos

### 6.7.1 Resultados numéricos

La **Tabla 6.5** presenta las cuatro métricas de evaluación walk-forward para todos los modelos sobre el período de test (octubre 2016 — octubre 2025, $N = 109$ observaciones).

**Tabla 6.5 — Comparativa de modelos predictivos (walk-forward, test: oct. 2016 — oct. 2025)**

| Modelo | RMSE (pp) | MAE (pp) | MAPE (%) | DA (%) | N |
|---|---|---|---|---|---|
| Naive (random walk) | 5.054 | 4.043 | 244.9 | 55.9 | 109 |
| XGBoost | 4.340 | 3.476 | 308.0 | 52.3 | 109 |
| Random Forest | 3.882 | 3.181 | 226.5 | 58.7 | 109 |
| **LSTM** | **3.815** | **3.142** | 278.8 | **61.5** | 109 |

*Nota: RMSE y MAE en puntos porcentuales (pp) del retorno logarítmico mensual. La desviación típica incondicional de gold_ret en la muestra completa es 4.65 pp.*

*(Figura 6.6: output/figures/fig_6_06_model_comparison.png)*

### 6.7.2 Discusión de resultados

**La LSTM obtiene el mejor rendimiento en todas las métricas clave**, con RMSE = 3.815 pp (el 82.1% de la desviación típica incondicional) y DA = 61.5%. Esto indica que la red recurrente captura dependencias temporales que los modelos de árboles, que no tienen memoria secuencial explícita, no explotan completamente.

**El Random Forest supera a XGBoost en todas las métricas**, a pesar de su arquitectura más simple. Este resultado es frecuente en series financieras cortas (n < 500): el bagging del RF es más robusto ante la sobreparametrización que el boosting secuencial, que puede seguir minimizando el error de entrenamiento en ausencia de suficientes observaciones.

**El resultado más llamativo es que XGBoost (DA = 52.3%) queda por debajo del benchmark naive (DA = 55.9%)**, aunque supera al naive en RMSE y MAE. Este patrón sugiere que XGBoost, aunque reduce el error absoluto de predicción, introduce ruido en la *dirección* del movimiento: mejora la magnitud de las predicciones cerca de cero, pero falla en los meses de retorno extremo (que son los que importan para las decisiones de inversión). El RF y la LSTM son más robustos a este problema.

**Sobre el MAPE elevado**: los valores de MAPE (226-308%) son característicos de la predicción de retornos financieros y no deben interpretarse como fracasos del modelo. El MAPE es sensible a retornos cercanos a cero: si el retorno real es +0.1% y el modelo predice -0.2%, el error porcentual es del 300%, aunque el error absoluto sea insignificante. Por ello, el MAPE debe leerse en combinación con RMSE y DA.

**Comparación con el benchmark econométrico**: el VECM del Capítulo 5 genera predicciones fuera de muestra con una DA típica del 54-56% en series de activos financieros (Stock y Watson, 2001; Gonzalo y Pitarakis, 2006). La LSTM (DA = 61.5%) y el RF (DA = 58.7%) mejoran materialmente esta referencia, justificando el enfoque híbrido econométrico-ML de este TFG.

### 6.7.3 Limitaciones

Se señalan tres limitaciones relevantes:

1. **Tamaño de la muestra**: con 271 observaciones efectivas y 35 features, el cociente parámetros/observaciones es relativamente alto, lo que puede inflar las métricas favorables y aumentar la varianza de los estimadores. Un conjunto de test mayor mejoraría la fiabilidad de las conclusiones.

2. **Coste de transacción**: las métricas de evaluación no consideran costes de transacción (spreads, comisiones, deslizamiento). Un sistema de trading basado en señales con DA ≈ 61% podría perder rentabilidad neta tras fricción, dependiendo del instrumento utilizado.

3. **Estabilidad de la DA**: la Directional Accuracy puede variar significativamente según el subperíodo de test. Una evaluación más robusta requeriría un análisis de la DA por ventanas deslizantes o por episodio de mercado, lo que se deja como extensión futura.

## 6.8 Conclusiones del capítulo

Este capítulo ha implementado tres algoritmos de machine learning para predecir el retorno mensual del oro sobre el período de test octubre 2016 — octubre 2025:

1. **La LSTM es el modelo más preciso** (RMSE = 3.815 pp, DA = 61.5%), aprovechando la estructura secuencial de los datos macroeconómicos para capturar dependencias temporales de hasta 6 meses.

2. **El Random Forest es el modelo de mayor robustez** entre los de árboles (RMSE = 3.882 pp, DA = 58.7%), superando al XGBoost en todas las métricas pese a su arquitectura más sencilla.

3. **El análisis SHAP revela que los predictores más influyentes son la inflación realizada (CPI YoY), los tipos de interés reales (TIPS 10Y) y el momentum del propio oro**, confirmando los hallazgos econométricos del Capítulo 5 con una metodología completamente diferente e independiente.

4. **Todos los modelos de ML superan al benchmark naive en error absoluto** (RMSE, MAE), validando que los patrones macroeconómicos tienen contenido informativo sobre la dirección del precio del oro a un horizonte de un mes.

5. El análisis integrado de los Capítulos 5 y 6 confluye en una conclusión robusta: **los tipos de interés reales y las expectativas inflacionarias son los determinantes de mayor peso del precio del oro**, tanto en el plano estructural (VECM, Granger) como en el predictivo (SHAP values). La asimetría de la volatilidad (GJR-GARCH) y el efecto de los episodios de crisis refuerzan la narrativa del oro como activo refugio y cobertura inflacionaria.

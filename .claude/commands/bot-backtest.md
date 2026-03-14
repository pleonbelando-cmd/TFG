---
description: Ejecuta el backtest completo del bot de trading de oro y presenta un resumen estructurado de métricas (DA, CAGR, Sharpe, Max Drawdown) por estrategia.
argument-hint: "[sin argumentos]"
allowed-tools: Bash, Read
---

Ejecuta el backtest completo del bot de trading semanal de oro y analiza los resultados.

## Pasos a seguir

### 1. Ejecutar el backtest

Ejecuta desde `C:\TFG`:

```bash
python -X utf8 bot_trading/run.py --mode backtest
```

Captura toda la salida de consola.

### 2. Parsear y presentar métricas

A partir del output, extrae las métricas de cada estrategia y preséntalo en una tabla clara:

| Estrategia | DA (%) | CAGR (%) | Sharpe | Max Drawdown (%) |
|------------|--------|----------|--------|------------------|
| ...        | ...    | ...      | ...    | ...              |

**Indicadores clave a destacar:**
- **DA (Directional Accuracy)**: porcentaje de aciertos de dirección. Objetivo: ≥ 54%
- **CAGR**: rentabilidad anualizada compuesta
- **Sharpe**: ratio rentabilidad/riesgo ajustado (>1 = aceptable, >1.5 = bueno)
- **Max Drawdown**: caída máxima desde pico (cuanto menor, mejor)

### 3. Verificar figuras generadas

Comprueba que existen las figuras PNG en `bot_trading/output/`. Lista los archivos encontrados y confirma cuáles se han generado correctamente.

### 4. Conclusión

Indica claramente:
- Si alguna estrategia supera el objetivo de DA ≥ 54%
- Cuál es la estrategia con mejor Sharpe
- Si hay alguna advertencia o error en el output
- Recomendación breve sobre el estado del modelo (apto para señal / requiere ajuste)

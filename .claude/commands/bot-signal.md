---
description: Obtiene la señal de trading de la próxima semana (BUY / FLAT / CASH) con probabilidades de los modelos XGBoost, LightGBM y Ensemble. Sin ejecutar ninguna orden.
argument-hint: "[sin argumentos]"
allowed-tools: Bash
---

Obtén la señal de trading semanal del bot de oro y preséntala de forma clara.

## Pasos a seguir

### 1. Ejecutar el modo señal

Ejecuta desde `C:\TFG`:

```bash
python -X utf8 bot_trading/run.py --mode signal
```

### 2. Presentar la señal

A partir del output, presenta la información de forma estructurada:

**Fecha de referencia:** [última fecha de datos disponibles]

| Modelo    | Probabilidad BUY | Señal individual |
|-----------|-----------------|-----------------|
| XGBoost   | X.XX%           | BUY / FLAT      |
| LightGBM  | X.XX%           | BUY / FLAT      |
| **Ensemble** | **X.XX%**    | **BUY / FLAT / CASH** |

**SEÑAL FINAL: [BUY / FLAT / CASH]**

### 3. Interpretación

Explica brevemente:
- **Umbral de decisión**: qué probabilidad mínima se requiere para BUY vs FLAT vs CASH
- **Qué implica la señal**: si es BUY → acumular exposición larga al oro esa semana; si es FLAT → mantener posición neutral; si es CASH → salir o no entrar
- **Nivel de confianza**: si los dos modelos coinciden (señal robusta) o divergen (señal menos fiable)
- **Contexto de mercado** si hay información disponible en el output (precio actual del oro, tendencia reciente)

---
description: Ejecuta la señal del bot en MetaTrader 5. Sin argumento = modo demo (imprime la orden sin enviarla). Con --live = modo real (requiere confirmación explícita).
argument-hint: "[--live para trading real, vacío = demo]"
allowed-tools: Bash
---

Ejecuta el bot de trading de oro en modo live (MetaTrader 5).

El argumento recibido es: $ARGUMENTS

## Pasos a seguir

### 1. Determinar el modo

Comprueba si `$ARGUMENTS` contiene `--live`:

**Si NO contiene `--live` (modo demo):**

Ejecuta desde `C:\TFG`:
```bash
python -X utf8 bot_trading/run.py --mode live
```
Indica claramente al usuario: "Modo DEMO — la orden se imprime pero NO se envía a MetaTrader 5."

**Si SÍ contiene `--live` (modo real):**

⚠️ **ADVERTENCIA: Modo LIVE activado — esto enviará órdenes reales a MetaTrader 5.**

Antes de ejecutar, confirma con el usuario que desea proceder con dinero real. Luego ejecuta:
```bash
python -X utf8 bot_trading/run.py --mode live --live
```

### 2. Presentar el resultado

A partir del output, muestra:

| Campo       | Valor                  |
|-------------|------------------------|
| Modo        | DEMO / REAL            |
| Señal       | BUY / FLAT / CASH      |
| Símbolo     | XAUUSD (u otro)        |
| Precio      | X.XX                   |
| Stop Loss   | X.XX                   |
| Take Profit | X.XX                   |
| Volumen     | X.XX lotes             |
| Estado      | Enviada / Impresa / Error |

### 3. Advertencias

- Si la señal es CASH o FLAT, indica que no se abre ninguna posición nueva
- Si hay error de conexión con MetaTrader 5, muestra el mensaje de error completo y sugiere verificar que MT5 está abierto y las credenciales son correctas
- Recuerda siempre al usuario que el modo real opera con capital real

# Datos manuales — World Gold Council

Este directorio contiene datos que deben descargarse manualmente del World Gold Council (GoldHub).

## Archivos requeridos

### 1. Reservas de bancos centrales (`cb_gold_reserves.csv`)
- **Fuente:** https://www.gold.org/goldhub/data/gold-reserves-by-country
- **Instrucciones:**
  1. Acceder a GoldHub (registro gratuito requerido)
  2. Seleccionar "Gold reserves by country" → "World" (total agregado)
  3. Periodo: 2000-01 a 2025-12
  4. Exportar como CSV
  5. Renombrar a `cb_gold_reserves.csv`
- **Formato esperado:** Columnas `Date`, `Tonnes` (compras netas o reservas totales)
- **Frecuencia:** Trimestral o anual (el pipeline interpola a mensual)

### 2. Flujos de ETFs de oro (`etf_gold_flows.csv`)
- **Fuente:** https://www.gold.org/goldhub/data/global-gold-backed-etf-holdings-and-flows
- **Instrucciones:**
  1. Acceder a GoldHub
  2. Seleccionar "Global gold-backed ETF holdings and flows"
  3. Periodo: 2004-01 a 2025-12 (ETFs de oro no existían antes de 2003)
  4. Exportar como CSV
  5. Renombrar a `etf_gold_flows.csv`
- **Formato esperado:** Columnas `Date`, `Tonnes` (flujos netos mensuales) y/o `Total_Holdings`
- **Frecuencia:** Mensual

## Notas
- Si los datos de GoldHub no están disponibles, el pipeline asignará NaN a estas columnas.
- Las reservas de BC pre-2000 no son necesarias para el periodo de análisis.
- Los ETF flows pre-2004 se codifican como NaN (los ETFs de oro comenzaron con GLD en Nov 2004).

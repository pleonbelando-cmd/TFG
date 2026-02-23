"""
clean.py — Limpieza, resampleo e imputación de series individuales.

Cada función toma una serie cruda (diaria o irregular) y devuelve una serie
mensual limpia, guardándola en data/processed/.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_MANUAL, DATA_PROCESSED, DATA_RAW, END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _monthly_index() -> pd.DatetimeIndex:
    """Índice mensual completo del periodo de análisis (fin de mes)."""
    return pd.date_range(START_DATE, END_DATE, freq="ME")


def _read_fred(name: str) -> pd.Series:
    """Lee un CSV de FRED desde data/raw/."""
    path = DATA_RAW / f"fred_{name}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s.index = pd.DatetimeIndex(s.index)
    return s.dropna()


# ── Oro ──────────────────────────────────────────────────────────────────────
def clean_gold() -> pd.Series:
    """Oro: resampleo a fin de mes (último precio disponible). Fuente: Yahoo Finance (GC=F)."""
    path = DATA_RAW / "yahoo_gold.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    close.index = pd.DatetimeIndex(close.index)
    monthly = close.resample("ME").last()
    monthly.name = "gold"
    return monthly


# ── DXY ──────────────────────────────────────────────────────────────────────
def clean_dxy() -> pd.Series:
    """
    DXY: media mensual.

    Combina DTWEXB (2000-2019) y DTWEXBGS (2006-presente) usando el periodo
    de solapamiento (2006-2019) para rescalar la serie antigua.
    """
    s_old = _read_fred("dxy_old")   # DTWEXB: 2000–2019
    s_new = _read_fred("dxy_new")   # DTWEXBGS: 2006–presente

    # Resampleo mensual
    m_old = s_old.resample("ME").mean()
    m_new = s_new.resample("ME").mean()

    # Rescalar serie antigua al nivel de la nueva usando solapamiento
    overlap = m_old.index.intersection(m_new.index)
    if len(overlap) > 12:
        scale = m_new.loc[overlap].mean() / m_old.loc[overlap].mean()
        m_old = m_old * scale

    # Combinar: preferir nueva donde existe, rellenar con antigua
    combined = m_new.reindex(m_old.index.union(m_new.index))
    missing = combined.isna()
    combined.loc[missing] = m_old.reindex(combined.index).loc[missing]

    combined.name = "dxy"
    return combined


# ── TIPS 10Y ─────────────────────────────────────────────────────────────────
def clean_tips(cpi_yoy: pd.Series | None = None) -> pd.Series:
    """
    TIPS 10Y: fin de mes.

    Proxy 2000-2002: DGS10 − CPI_YoY (tipo real ex-post), ya que TIPS
    data de FRED comienza en enero 2003.
    """
    s = _read_fred("tips_10y")
    monthly = s.resample("ME").last()

    # Proxy para 2000-2002
    if monthly.first_valid_index() > pd.Timestamp("2000-06-01"):
        logger.info("Construyendo proxy TIPS 2000-2002: DGS10 − CPI_YoY")
        dgs10 = _read_fred("dgs10").resample("ME").last()
        if cpi_yoy is not None:
            proxy = dgs10 - cpi_yoy
            proxy_period = proxy.loc[proxy.index < monthly.first_valid_index()]
            monthly = pd.concat([proxy_period, monthly]).sort_index()
            monthly = monthly[~monthly.index.duplicated(keep="last")]

    monthly.name = "tips_10y"
    return monthly


# ── CPI → tasa interanual ───────────────────────────────────────────────────
def clean_cpi() -> pd.Series:
    """CPI: tasa interanual = (CPI_t / CPI_{t-12} − 1) × 100."""
    s = _read_fred("cpi")
    # CPI ya es mensual, tomar último del mes por si hay duplicados
    monthly = s.resample("ME").last()
    yoy = (monthly / monthly.shift(12) - 1) * 100
    yoy.name = "cpi_yoy"
    return yoy


# ── Breakeven 10Y ───────────────────────────────────────────────────────────
def clean_breakeven() -> pd.Series:
    """Breakeven 10Y: fin de mes. NaN pre-2003."""
    s = _read_fred("breakeven")
    monthly = s.resample("ME").last()
    monthly.name = "breakeven"
    return monthly


# ── VIX ──────────────────────────────────────────────────────────────────────
def clean_vix() -> pd.Series:
    """VIX: media mensual (mean-reverting, no usar fin de mes)."""
    s = _read_fred("vix")
    monthly = s.resample("ME").mean()
    monthly.name = "vix"
    return monthly


# ── S&P 500 ──────────────────────────────────────────────────────────────────
def clean_sp500() -> tuple[pd.Series, pd.Series]:
    """
    S&P 500: cierre ajustado fin de mes + retorno logarítmico mensual.

    Returns:
        (sp500_level, sp500_log_return)
    """
    path = DATA_RAW / "yahoo_sp500.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    close.index = pd.DatetimeIndex(close.index)

    monthly = close.resample("ME").last()
    monthly.name = "sp500"

    log_ret = np.log(monthly / monthly.shift(1)) * 100
    log_ret.name = "sp500_ret"

    return monthly, log_ret


# ── WTI ──────────────────────────────────────────────────────────────────────
def clean_wti() -> pd.Series:
    """WTI: fin de mes."""
    s = _read_fred("wti")
    monthly = s.resample("ME").last()
    monthly.name = "wti"
    return monthly


# ── Fed Funds Rate ───────────────────────────────────────────────────────────
def clean_fedfunds() -> pd.Series:
    """Fed Funds Rate: fin de mes."""
    s = _read_fred("fedfunds")
    monthly = s.resample("ME").last()
    monthly.name = "fedfunds"
    return monthly


# ── Reservas de bancos centrales (manual) ────────────────────────────────────
def clean_cb_reserves() -> pd.Series:
    """
    Reservas BC: primera diferencia + lag 2 meses.

    Lee de data/manual/cb_gold_reserves.csv.
    Si los datos son anuales o trimestrales, interpola a mensual.
    """
    path = DATA_MANUAL / "cb_gold_reserves.csv"
    if not path.exists():
        logger.warning(
            f"Archivo no encontrado: {path}. "
            "Columna cb_reserves será NaN. Ver data/manual/README.md."
        )
        idx = _monthly_index()
        return pd.Series(np.nan, index=idx, name="cb_reserves")

    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]

    # Intentar parsear columnas Date y Tonnes/Value
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    val_col = [c for c in df.columns if c.lower() != date_col.lower()][0]
    s = pd.Series(df[val_col].values, index=pd.DatetimeIndex(df[date_col]))
    s = s.sort_index()

    # Interpolar a mensual si es necesario
    monthly = s.resample("ME").interpolate(method="linear")

    # Primera diferencia (cambio neto mensual)
    diff = monthly.diff()
    # Lag 2 meses (los datos de BC se publican con retraso)
    lagged = diff.shift(2)
    lagged.name = "cb_reserves"
    return lagged


# ── Google Trends ────────────────────────────────────────────────────────────
def clean_google_trends() -> pd.Series:
    """Google Trends: ya mensual tras descarga. NaN pre-2004."""
    path = DATA_RAW / "google_trends.csv"
    if not path.exists():
        logger.warning("Google Trends no descargado. Columna será NaN.")
        idx = _monthly_index()
        return pd.Series(np.nan, index=idx, name="google_trends")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = df.iloc[:, 0]
    s.index = pd.DatetimeIndex(s.index)
    # Re-indexar a fin de mes para coherencia
    monthly = s.resample("ME").last()
    monthly.name = "google_trends"
    return monthly


# ── ETF Flows (manual) ──────────────────────────────────────────────────────
def clean_etf_flows() -> pd.Series:
    """ETF flows: datos WGC manuales. NaN pre-2004."""
    path = DATA_MANUAL / "etf_gold_flows.csv"
    if not path.exists():
        logger.warning(
            f"Archivo no encontrado: {path}. "
            "Columna etf_flows será NaN. Ver data/manual/README.md."
        )
        idx = _monthly_index()
        return pd.Series(np.nan, index=idx, name="etf_flows")

    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]

    date_col = [c for c in df.columns if "date" in c.lower()][0]
    # Buscar columna de flujos (Tonnes, Flows, etc.)
    val_candidates = [c for c in df.columns if c.lower() != date_col.lower()]
    val_col = val_candidates[0]

    s = pd.Series(df[val_col].values, index=pd.DatetimeIndex(df[date_col]))
    s = s.sort_index()
    monthly = s.resample("ME").last()
    monthly.name = "etf_flows"
    return monthly


# ── Pendiente de la curva de tipos (10Y − 2Y) ────────────────────────────────
def clean_yield_curve() -> pd.Series:
    """
    Pendiente curva: spread 10Y − 2Y en puntos porcentuales.

    Proxy del ciclo económico y de las expectativas de la Fed.
    Invertida (< 0) históricamente precede a recesiones con ~12 meses.
    Fuente: FRED T10Y2Y (ya es un spread directo).
    """
    s = _read_fred("yield_curve")
    monthly = s.resample("ME").last()
    monthly.name = "yield_curve"
    return monthly


# ── High Yield spread (proxy de apetito al riesgo) ───────────────────────────
def clean_hy_spread() -> pd.Series:
    """
    High Yield OAS spread (ICE BofA, BAMLH0A0HYM2) en puntos porcentuales.

    Cuando el spread sube → los inversores exigen más prima de riesgo →
    entorno de risk-off → demanda de oro como activo refugio.
    Serie diaria. Se toma el último valor del mes.
    Disponible desde 1997-12.
    """
    s = _read_fred("hy_spread")
    monthly = s.resample("ME").last()
    monthly.name = "hy_spread"
    return monthly


# ── Orquestador ──────────────────────────────────────────────────────────────
def clean_all() -> dict[str, pd.Series]:
    """Limpia todas las series y guarda en data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    results = {}

    # Primero CPI (necesario para proxy TIPS)
    cpi_yoy = clean_cpi()
    results["cpi_yoy"] = cpi_yoy

    results["gold"] = clean_gold()
    results["dxy"] = clean_dxy()
    results["tips_10y"] = clean_tips(cpi_yoy=cpi_yoy)
    results["breakeven"] = clean_breakeven()
    results["vix"] = clean_vix()

    sp500, sp500_ret = clean_sp500()
    results["sp500"] = sp500
    results["sp500_ret"] = sp500_ret

    results["wti"] = clean_wti()
    results["fedfunds"] = clean_fedfunds()
    results["cb_reserves"] = clean_cb_reserves()
    results["google_trends"] = clean_google_trends()
    results["etf_flows"] = clean_etf_flows()
    results["yield_curve"] = clean_yield_curve()
    results["hy_spread"] = clean_hy_spread()

    # Guardar cada serie procesada
    for name, s in results.items():
        filepath = DATA_PROCESSED / f"{name}.csv"
        s.to_csv(filepath, header=True)
        n_valid = s.notna().sum()
        logger.info(f"  {name}: {n_valid} obs válidas → {filepath.name}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_all()

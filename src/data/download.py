"""
download.py — Descarga de datos de FRED, Yahoo Finance y Google Trends.

Cada función guarda CSVs en data/raw/ y devuelve DataFrames.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    DATA_RAW,
    END_DATE,
    FRED_SERIES,
    GOOGLE_TRENDS_GEO,
    GOOGLE_TRENDS_KEYWORD,
    PROJECT_ROOT,
    START_DATE,
    YAHOO_TICKERS,
)

logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env")


# ── FRED ─────────────────────────────────────────────────────────────────────
def download_fred_series() -> dict[str, pd.Series]:
    """Descarga todas las series FRED definidas en config."""
    from fredapi import Fred

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY no encontrada. Crea un archivo .env con tu clave "
            "(ver .env.example)."
        )
    fred = Fred(api_key=api_key)
    series_dict = {}

    for name, code in FRED_SERIES.items():
        try:
            logger.info(f"Descargando FRED: {name} ({code})")
            s = fred.get_series(code, START_DATE, END_DATE)
            s.name = name
            s.index.name = "date"
            # Guardar CSV
            filepath = DATA_RAW / f"fred_{name}.csv"
            s.to_csv(filepath, header=True)
            series_dict[name] = s
            logger.info(f"  → {len(s)} observaciones guardadas en {filepath.name}")
        except Exception as e:
            logger.error(f"Error descargando {name} ({code}): {e}")

    return series_dict


# ── Yahoo Finance ────────────────────────────────────────────────────────────
def download_yahoo_series() -> dict[str, pd.DataFrame]:
    """Descarga tickers de Yahoo Finance."""
    import yfinance as yf

    series_dict = {}

    for name, ticker in YAHOO_TICKERS.items():
        try:
            logger.info(f"Descargando Yahoo: {name} ({ticker})")
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                logger.warning(f"  → Sin datos para {ticker}")
                continue

            # Flatten multi-level columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = "date"
            filepath = DATA_RAW / f"yahoo_{name}.csv"
            df.to_csv(filepath)
            series_dict[name] = df
            logger.info(f"  → {len(df)} observaciones guardadas en {filepath.name}")
        except Exception as e:
            logger.error(f"Error descargando {name} ({ticker}): {e}")

    return series_dict


# ── Google Trends ────────────────────────────────────────────────────────────
def download_google_trends() -> pd.DataFrame | None:
    """
    Descarga Google Trends con chunks solapados de 5 años y rescalado.

    Google Trends normaliza cada consulta a 0-100 dentro del periodo solicitado,
    así que para periodos largos hay que descargar en chunks y re-escalar usando
    los meses de solapamiento.
    """
    from pytrends.request import TrendReq

    logger.info(f"Descargando Google Trends: '{GOOGLE_TRENDS_KEYWORD}'")

    try:
        pytrends = TrendReq(hl="en-US", tz=0)

        # Definir chunks de 5 años con 6 meses de solapamiento
        # Google Trends da datos mensuales para periodos > 5 años
        chunk_starts = pd.date_range("2004-01-01", END_DATE, freq="4YS")
        chunks = []

        for start in chunk_starts:
            end = min(start + pd.DateOffset(years=5) - pd.DateOffset(days=1),
                      pd.Timestamp(END_DATE))
            timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
            logger.info(f"  Chunk: {timeframe}")

            pytrends.build_payload(
                [GOOGLE_TRENDS_KEYWORD],
                cat=0,
                timeframe=timeframe,
                geo=GOOGLE_TRENDS_GEO,
            )
            df_chunk = pytrends.interest_over_time()
            if df_chunk.empty:
                continue
            df_chunk = df_chunk[[GOOGLE_TRENDS_KEYWORD]].rename(
                columns={GOOGLE_TRENDS_KEYWORD: "google_trends"}
            )
            chunks.append(df_chunk)
            time.sleep(2)  # Rate limiting

        if not chunks:
            logger.warning("Sin datos de Google Trends")
            return None

        # Rescalar chunks usando periodos de solapamiento
        merged = chunks[0].copy()
        for i in range(1, len(chunks)):
            overlap = merged.index.intersection(chunks[i].index)
            if len(overlap) > 0:
                scale = merged.loc[overlap, "google_trends"].mean() / \
                        max(chunks[i].loc[overlap, "google_trends"].mean(), 1)
                chunks[i]["google_trends"] = chunks[i]["google_trends"] * scale
            # Añadir solo las fechas nuevas
            new_dates = chunks[i].index.difference(merged.index)
            merged = pd.concat([merged, chunks[i].loc[new_dates]])

        merged = merged.sort_index()
        merged.index.name = "date"

        filepath = DATA_RAW / "google_trends.csv"
        merged.to_csv(filepath)
        logger.info(f"  → {len(merged)} observaciones guardadas en {filepath.name}")
        return merged

    except Exception as e:
        logger.error(f"Error descargando Google Trends: {e}")
        return None


# ── Orquestador ──────────────────────────────────────────────────────────────
def download_all() -> dict:
    """Descarga todas las fuentes de datos y guarda en data/raw/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    results = {}
    results["fred"] = download_fred_series()
    results["yahoo"] = download_yahoo_series()
    results["google_trends"] = download_google_trends()

    logger.info("Descarga completa. Archivos en data/raw/:")
    for f in sorted(DATA_RAW.glob("*.csv")):
        logger.info(f"  {f.name}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all()

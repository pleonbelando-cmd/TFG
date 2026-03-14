# -*- coding: utf-8 -*-
"""
data.py
Descarga datos semanales de Yahoo Finance para el bot de oro.

Activos: GC=F (oro), DX-Y.NYB (DXY), ^GSPC (S&P 500),
         ^VIX (VIX), CL=F (WTI), SI=F (plata)

Frecuencia: semanal (interval="1wk") → ~1.300 observaciones 2000-2026
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKERS, START, END


def load_prices() -> pd.DataFrame:
    """
    Descarga precios de cierre semanales para todos los tickers.

    Devuelve un DataFrame con columnas:
        gold, dxy, sp500, vix, wti, silver
    más retornos logarítmicos:
        gold_ret, dxy_ret, sp500_ret, wti_ret
    y el nivel de VIX (no retorno, es mean-reverting).
    """
    print("Descargando datos semanales de Yahoo Finance...")

    frames = {}
    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(
                ticker,
                start=START,
                end=END,
                interval="1wk",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"  AVISO: {ticker} devolvió datos vacíos")
                continue

            # Compatibilidad con yfinance >= 0.2 (MultiIndex)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            close = raw["Close"].copy()
            close.index = pd.to_datetime(close.index)
            close.name  = name
            frames[name] = close
            print(f"  {ticker}: {len(close)} semanas ({close.index[0].date()} -> {close.index[-1].date()})")
        except Exception as exc:
            print(f"  ERROR descargando {ticker}: {exc}")

    if "gold" not in frames:
        raise RuntimeError("No se pudo descargar el precio del oro (GC=F). Comprueba la conexión.")

    # Unir en un único DataFrame, frecuencia semanal
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)

    # Asegurar frecuencia semanal sin huecos (forward-fill máx 2 semanas)
    df = df.resample("W-MON").last()
    df = df.ffill(limit=2)

    # Retornos logarítmicos (sin mirar el futuro: ret en t = log(P_t/P_{t-1}))
    for col in ["gold", "dxy", "sp500", "wti"]:
        if col in df.columns:
            df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    # VIX: usar el nivel directamente (es mean-reverting, no log-retorno)
    # ya está en df["vix"]

    # Eliminar primera fila (NaN por el retorno)
    df = df.dropna(subset=["gold_ret"])

    print(f"\nDataFrame final: {len(df)} semanas × {len(df.columns)} columnas")
    print(f"Periodo: {df.index[0].date()} -> {df.index[-1].date()}")

    return df


if __name__ == "__main__":
    df = load_prices()
    print(df.tail())
    print(df.describe())

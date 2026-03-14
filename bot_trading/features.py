# -*- coding: utf-8 -*-
"""
features.py
15 features técnicas semanales para el bot de oro.

Principio de diseño:
  - Todos los features usan lag ≥ 1 (nunca datos del periodo actual)
  - Sin look-ahead bias: la feature del lunes t usa datos del lunes t-1
  - Target: 1 si el retorno logarítmico del oro de la semana siguiente > 0

Features:
  1-4  gold_ret_w1..w4  : retorno log oro, lags 1-4
  5-6  dxy_ret_w1,w2   : retorno log DXY, lags 1-2
  7    sp500_ret_w1     : retorno log S&P 500, lag 1
  8    vix_w1           : nivel VIX, lag 1
  9    vix_chg_w1       : cambio VIX semana anterior
  10   wti_ret_w1       : retorno log WTI, lag 1
  11   gold_rsi14       : RSI(14) semanal del oro, lag 1
  12   gold_ma4         : media móvil 4 semanas (≈1 mes), lag 1
  13   gold_ma13        : media móvil 13 semanas (≈trimestre), lag 1
  14   gold_vol4        : volatilidad 4 semanas, lag 1
  15   gs_ratio_chg     : cambio % ratio Gold/Silver, lag 1
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder (copia de features_v2.py del modelo_prediccion)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> tuple:
    """
    Construye las 15 features semanales a partir del DataFrame de precios.

    Parámetros:
        df : DataFrame con columnas gold, gold_ret, dxy_ret, sp500_ret,
             vix, wti_ret, silver (opcionales excepto gold y gold_ret)

    Devuelve:
        data        : pd.DataFrame limpio (sin NaN, con target)
        feature_cols: list[str]  nombres de las 15 features
        TARGET      : str        nombre de la variable objetivo
    """
    data = df.copy()
    feature_cols = []

    # ─── 1-4. Lags del retorno del oro ────────────────────────────────────────
    for lag in range(1, 5):
        name = f"gold_ret_w{lag}"
        data[name] = data["gold_ret"].shift(lag)
        feature_cols.append(name)

    # ─── 5-6. DXY lags 1-2 ───────────────────────────────────────────────────
    if "dxy_ret" in data.columns:
        for lag, suffix in [(1, "w1"), (2, "w2")]:
            name = f"dxy_ret_{suffix}"
            data[name] = data["dxy_ret"].shift(lag)
            feature_cols.append(name)

    # ─── 7. S&P 500 lag 1 ────────────────────────────────────────────────────
    if "sp500_ret" in data.columns:
        data["sp500_ret_w1"] = data["sp500_ret"].shift(1)
        feature_cols.append("sp500_ret_w1")

    # ─── 8. Nivel VIX lag 1 ───────────────────────────────────────────────────
    if "vix" in data.columns:
        data["vix_w1"] = data["vix"].shift(1)
        feature_cols.append("vix_w1")

    # ─── 9. Cambio VIX (aceleración del miedo) ────────────────────────────────
    if "vix" in data.columns:
        data["vix_chg_w1"] = data["vix"].diff().shift(1)
        feature_cols.append("vix_chg_w1")

    # ─── 10. WTI lag 1 ────────────────────────────────────────────────────────
    if "wti_ret" in data.columns:
        data["wti_ret_w1"] = data["wti_ret"].shift(1)
        feature_cols.append("wti_ret_w1")

    # ─── 11. RSI(14) del oro, lag 1 ───────────────────────────────────────────
    if "gold" in data.columns:
        data["gold_rsi14"] = _rsi(data["gold"], period=14).shift(1)
        feature_cols.append("gold_rsi14")

    # ─── 12. Media móvil 4 semanas (≈1 mes), lag 1 ───────────────────────────
    data["gold_ma4"] = data["gold_ret"].rolling(4).mean().shift(1)
    feature_cols.append("gold_ma4")

    # ─── 13. Media móvil 13 semanas (≈trimestre), lag 1 ──────────────────────
    data["gold_ma13"] = data["gold_ret"].rolling(13).mean().shift(1)
    feature_cols.append("gold_ma13")

    # ─── 14. Volatilidad 4 semanas, lag 1 ────────────────────────────────────
    data["gold_vol4"] = data["gold_ret"].rolling(4).std().shift(1)
    feature_cols.append("gold_vol4")

    # ─── 15. Gold/Silver ratio cambio %, lag 1 ────────────────────────────────
    if "gold" in data.columns and "silver" in data.columns:
        ratio = data["gold"] / data["silver"].replace(0, np.nan)
        data["gs_ratio_chg"] = ratio.pct_change().shift(1)
        feature_cols.append("gs_ratio_chg")

    # ─── Target binario ───────────────────────────────────────────────────────
    # y = 1 si el retorno de la semana actual > 0
    # (la feature de lag 1 predice el retorno t usando info de t-1)
    TARGET = "target_up"
    data[TARGET] = (data["gold_ret"] > 0).astype(int)

    # ─── Mantener solo columnas relevantes ───────────────────────────────────
    feature_cols = [c for c in feature_cols if c in data.columns]

    keep_cols = ["gold", "gold_ret"] + feature_cols + [TARGET]
    keep_cols = [c for c in keep_cols if c in data.columns]
    # Eliminar duplicados preservando orden
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    result = data[keep_cols].dropna()

    print(f"  Features construidas: {len(feature_cols)}")
    print(f"  Observaciones tras dropna: {len(result)}")
    pct_up = result[TARGET].mean()
    print(f"  Distribución target: UP={pct_up*100:.1f}% / DOWN={(1-pct_up)*100:.1f}%")

    return result, feature_cols, TARGET


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data import load_prices

    df_raw = load_prices()
    data, feat, tgt = build_features(df_raw)
    print(f"\n{len(feat)} features: {feat}")
    print(data.tail())

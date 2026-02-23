"""
features.py — Feature engineering para los modelos de ML del Capítulo 6.

Transforma el dataset maestro en una matriz de características que los modelos
de árboles (XGBoost, Random Forest) y la LSTM pueden consumir directamente.

Criterios de diseño:
- Todas las variables entran como retornos o z-scores, no en niveles,
  para evitar la no-estacionariedad y hacer comparable la escala entre features.
- Se incluyen retardos de orden 1, 2 y 3 para capturar inercia de corto plazo.
- El target es el retorno logarítmico del oro a 1 mes vista (gold_ret),
  que equivale a predecir la dirección y magnitud del movimiento mensual.
- No se incluyen variables futuras (no look-ahead bias).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columnas base del dataset maestro que usaremos como features
BASE_FEATURES = [
    "dxy",         # Índice del dólar
    "tips_10y",    # Tipo de interés real
    "cpi_yoy",     # Inflación realizada
    "vix",         # Volatilidad implícita
    "sp500_ret",   # Retorno mensual S&P 500
    "wti",         # Precio petróleo WTI
    "fedfunds",    # Fed Funds Rate
    "breakeven",   # Expectativas de inflación
    "yield_curve", # Pendiente curva 10Y-2Y
    "hy_spread",   # High Yield spread (apetito al riesgo)
]

# Variables que se transformarán a retornos logarítmicos (precio → cambio %)
PRICE_TO_RETURN = ["dxy", "wti"]

# Target
TARGET = "gold_ret"


def build_feature_matrix(
    df: pd.DataFrame,
    lags: int = 3,
    add_momentum: bool = True,
    add_regime: bool = True,
) -> pd.DataFrame:
    """
    Construye la matriz de características para los modelos ML.

    Args:
        df: Dataset maestro (gold_macro_monthly.csv).
        lags: Número de retardos de cada variable a incluir.
        add_momentum: Si incluir medias móviles del oro (3 y 6 meses).
        add_regime: Si incluir variable dummy de episodio de crisis.

    Returns:
        DataFrame con features, target y sin NaN (observaciones completas).
    """
    data = df.copy()

    # ── 1. Transformaciones base ─────────────────────────────────────────────
    # Retornos logarítmicos para variables de precio
    for col in PRICE_TO_RETURN:
        if col in data.columns and data[col].min() > 0:
            data[f"{col}_ret"] = np.log(data[col]).diff()

    # VIX: primera diferencia (en niveles ya es estacionario, pero la diff
    # captura mejor el cambio en el régimen de miedo)
    if "vix" in data.columns:
        data["vix_chg"] = data["vix"].diff()

    # TIPS: nivel (puede ser negativo, no se puede tomar log)
    # fedfunds: nivel
    # cpi_yoy: nivel (ya es tasa de variación)
    # breakeven: nivel

    # ── 2. Lags (1, 2, 3 meses) ─────────────────────────────────────────────
    # Features "base" que se retardan: incluyen las versiones transformadas
    lag_candidates = [
        "gold_ret",     # retorno pasado del oro (momentum)
        "dxy_ret",      # cambio del dólar
        "tips_10y",     # nivel del tipo real
        "vix",          # nivel del VIX
        "vix_chg",      # cambio en el VIX
        "sp500_ret",    # retorno S&P 500
        "cpi_yoy",      # inflación realizada
        "fedfunds",     # tipo de política monetaria
        "breakeven",    # inflación esperada
        "wti_ret",      # cambio precio petróleo
        "yield_curve",  # pendiente curva 10Y-2Y (nueva)
        "hy_spread",    # High Yield spread (nueva)
    ]

    feature_cols = []
    for col in lag_candidates:
        if col not in data.columns:
            continue
        for lag in range(1, lags + 1):
            new_col = f"{col}_lag{lag}"
            data[new_col] = data[col].shift(lag)
            feature_cols.append(new_col)

    # ── 3. Momentum del oro ───────────────────────────────────────────────────
    if add_momentum and "gold_ret" in data.columns:
        data["gold_ma3"] = data["gold_ret"].rolling(3).mean().shift(1)
        data["gold_ma6"] = data["gold_ret"].rolling(6).mean().shift(1)
        data["gold_vol3"] = data["gold_ret"].rolling(3).std().shift(1)
        feature_cols += ["gold_ma3", "gold_ma6", "gold_vol3"]

    # ── 4. Spread tipos: real vs nominal ─────────────────────────────────────
    if "fedfunds" in data.columns and "cpi_yoy" in data.columns:
        data["real_rate_proxy"] = data["fedfunds"] - data["cpi_yoy"]
        data["real_rate_proxy_lag1"] = data["real_rate_proxy"].shift(1)
        feature_cols.append("real_rate_proxy_lag1")

    # ── 4b. Variables computadas adicionales ─────────────────────────────────
    # Momentum de tipos reales (aceleración del ciclo de tipos)
    if "tips_10y" in data.columns:
        data["real_rate_chg3"] = data["tips_10y"].diff(3).shift(1)
        feature_cols.append("real_rate_chg3")

    # Volatilidad del dólar (incertidumbre cambiaria)
    if "dxy_ret" in data.columns:
        data["dxy_vol3"] = data["dxy_ret"].rolling(3).std().shift(1)
        feature_cols.append("dxy_vol3")

    # Diferencial tipo real − tipo a corto (posición de política monetaria real)
    if "tips_10y" in data.columns and "fedfunds" in data.columns:
        data["tips_fedfunds_spread"] = (data["tips_10y"] - data["fedfunds"]).shift(1)
        feature_cols.append("tips_fedfunds_spread")

    # ── 5. Régimen de mercado ─────────────────────────────────────────────────
    if add_regime and "episode" in data.columns:
        data["is_crisis"] = (data["episode"] != "calma").astype(int)
        feature_cols.append("is_crisis")

    # ── 6. Selección final y eliminación de NaN ───────────────────────────────
    # Nos quedamos solo con las columnas que realmente existen
    feature_cols = [c for c in feature_cols if c in data.columns]
    # Eliminar duplicados manteniendo orden
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    cols_needed = feature_cols + [TARGET]
    result = data[cols_needed].dropna()

    return result, feature_cols


def split_train_test(
    data: pd.DataFrame,
    feature_cols: list,
    train_frac: float = 0.60,
) -> tuple:
    """
    División temporal train/test (sin mezcla aleatoria).

    El 60% inicial es la muestra de entrenamiento del primer fold.
    El 40% restante es la ventana walk-forward de evaluación.
    """
    n = len(data)
    split = int(n * train_frac)

    X = data[feature_cols].values
    y = data[TARGET].values
    dates = data.index

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_train, dates_test = dates[:split], dates[split:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test, feature_cols


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """
    Estandarización z-score: fit solo en train, apply en train y test.
    Fundamental para la LSTM; opcional para XGBoost/RF.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler

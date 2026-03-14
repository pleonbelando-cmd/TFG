# -*- coding: utf-8 -*-
"""
model.py
Walk-forward validation con XGBoost + LightGBM (ensemble 50/50).
Guarda/carga modelos con joblib.

Sin LSTM: demasiado ruidoso con datos tabulares semanales y añade complejidad
sin beneficio de DA demostrado.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    print("AVISO: LightGBM no disponible. Usando solo XGBoost.")

from config import (
    XGB_PARAMS, LGBM_PARAMS, ENSEMBLE_WEIGHTS,
    THRESH_UP, THRESH_DOWN, MIN_TRAIN_OBS,
    TEST_START, OUTPUT_DIR, SEED,
)


# ─── Entrenamiento de un solo modelo ──────────────────────────────────────────

def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    params = {k: v for k, v in XGB_PARAMS.items() if k != "refit_every"}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def _train_lgbm(X_train: pd.DataFrame, y_train: pd.Series):
    if not _HAS_LGBM:
        return None
    params = {k: v for k, v in LGBM_PARAMS.items() if k != "refit_every"}
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def _ensemble_prob(prob_xgb: np.ndarray, prob_lgbm: np.ndarray | None) -> np.ndarray:
    """Combina probabilidades XGB + LGBM con pesos del config."""
    w_xgb, w_lgbm = ENSEMBLE_WEIGHTS
    if prob_lgbm is None or not _HAS_LGBM:
        return prob_xgb
    return w_xgb * prob_xgb + w_lgbm * prob_lgbm


def _signal_from_prob(prob: float) -> int:
    """Convierte probabilidad en señal: +1 (BUY), -1 (CASH), 0 (neutral→cash)."""
    if prob > THRESH_UP:
        return 1
    if prob < THRESH_DOWN:
        return -1
    return 0


# ─── Walk-forward validation ──────────────────────────────────────────────────

def walk_forward(
    data: pd.DataFrame,
    feature_cols: list,
    target: str,
) -> pd.DataFrame:
    """
    Walk-forward validation expansivo.

    Train inicial: TRAIN_START → TEST_START (~900 semanas)
    Test:          TEST_START  → fin (~420 semanas)
    Refit:         cada refit_every semanas (configurable en XGB_PARAMS)

    Devuelve DataFrame con columnas:
        gold_ret, prob_xgb, prob_lgbm, prob_ensemble, signal, y_true
    """
    print("\n=== Walk-forward validation ===")

    X = data[feature_cols].copy()
    y = data[target].copy()

    test_mask = data.index >= TEST_START
    test_idx  = data.index[test_mask]

    if len(test_idx) == 0:
        raise ValueError(f"No hay datos de test desde {TEST_START}.")

    refit_every = XGB_PARAMS.get("refit_every", 52)

    records = []
    xgb_model  = None
    lgbm_model = None

    for i, date in enumerate(test_idx):
        # Ventana de entrenamiento: todos los datos anteriores a date
        train_mask = data.index < date
        X_tr = X[train_mask]
        y_tr = y[train_mask]

        if len(X_tr) < MIN_TRAIN_OBS:
            continue

        # Re-entrenar cada refit_every pasos o en el primer paso
        if i % refit_every == 0:
            xgb_model  = _train_xgb(X_tr, y_tr)
            lgbm_model = _train_lgbm(X_tr, y_tr)
            if i == 0:
                print(f"  Entrenamiento inicial: {len(X_tr)} obs -> {date.date()}")
            else:
                print(f"  Re-entrenamiento: {len(X_tr)} obs -> {date.date()}")

        if xgb_model is None:
            continue

        X_te = X.loc[[date]]

        prob_xgb  = float(xgb_model.predict_proba(X_te)[0, 1])
        prob_lgbm = (
            float(lgbm_model.predict_proba(X_te)[0, 1])
            if lgbm_model is not None else None
        )
        prob_ens = float(_ensemble_prob(
            np.array([prob_xgb]),
            np.array([prob_lgbm]) if prob_lgbm is not None else None,
        )[0])

        records.append({
            "date":          date,
            "gold_ret":      data.loc[date, "gold_ret"],
            "prob_xgb":      prob_xgb,
            "prob_lgbm":     prob_lgbm if prob_lgbm is not None else np.nan,
            "prob_ensemble": prob_ens,
            "signal":        _signal_from_prob(prob_ens),
            "y_true":        int(y.loc[date]),
        })

    results = pd.DataFrame(records).set_index("date")
    print(f"\n  Walk-forward completo: {len(results)} semanas de test")
    return results


# ─── Directional Accuracy ─────────────────────────────────────────────────────

def compute_da_per_model(results: pd.DataFrame) -> dict:
    """Calcula DA para XGB, LGBM y Ensemble."""
    das = {}
    for col in ["prob_xgb", "prob_lgbm", "prob_ensemble"]:
        if col not in results.columns:
            continue
        valid = results.dropna(subset=[col])
        pred  = (valid[col] >= 0.5).astype(int)
        das[col] = float((pred == valid["y_true"]).mean())

    print("\n=== Directional Accuracy (DA) ===")
    labels = {"prob_xgb": "XGBoost", "prob_lgbm": "LightGBM", "prob_ensemble": "Ensemble"}
    for k, v in das.items():
        marker = " ✓" if v >= 0.54 else ""
        print(f"  {labels.get(k, k):15s}: {v*100:.2f}%{marker}")

    return das


# ─── Guardar / Cargar modelos ─────────────────────────────────────────────────

def save_models(xgb_model, lgbm_model) -> None:
    xgb_path = OUTPUT_DIR / "xgb_model.pkl"
    joblib.dump(xgb_model, xgb_path)
    print(f"  XGBoost guardado: {xgb_path}")

    if lgbm_model is not None:
        lgbm_path = OUTPUT_DIR / "lgbm_model.pkl"
        joblib.dump(lgbm_model, lgbm_path)
        print(f"  LightGBM guardado: {lgbm_path}")


def load_models() -> tuple:
    """Carga los modelos guardados. Devuelve (xgb, lgbm)."""
    xgb_path  = OUTPUT_DIR / "xgb_model.pkl"
    lgbm_path = OUTPUT_DIR / "lgbm_model.pkl"

    xgb_model  = joblib.load(xgb_path)  if xgb_path.exists()  else None
    lgbm_model = joblib.load(lgbm_path) if lgbm_path.exists() else None

    if xgb_model is None:
        raise FileNotFoundError(
            f"Modelo XGBoost no encontrado en {xgb_path}. "
            "Ejecuta primero: python bot_trading/run.py --mode backtest"
        )

    return xgb_model, lgbm_model


def retrain_full(
    data: pd.DataFrame,
    feature_cols: list,
    target: str,
) -> tuple:
    """
    Re-entrena ambos modelos con todos los datos disponibles.
    Guarda los modelos en output/.
    Devuelve (xgb_model, lgbm_model).
    """
    print("\n=== Re-entrenamiento completo ===")
    X = data[feature_cols]
    y = data[target]
    print(f"  Entrenando con {len(X)} semanas...")

    xgb_model  = _train_xgb(X, y)
    lgbm_model = _train_lgbm(X, y)

    save_models(xgb_model, lgbm_model)
    return xgb_model, lgbm_model


def predict_latest(
    data: pd.DataFrame,
    feature_cols: list,
    xgb_model,
    lgbm_model,
) -> dict:
    """
    Genera la señal para la próxima semana usando la última fila de features.

    Devuelve dict con:
        prob_xgb, prob_lgbm, prob_ensemble, signal, signal_str
    """
    latest = data[feature_cols].iloc[[-1]]

    prob_xgb  = float(xgb_model.predict_proba(latest)[0, 1])
    prob_lgbm = (
        float(lgbm_model.predict_proba(latest)[0, 1])
        if lgbm_model is not None else None
    )
    prob_ens = float(_ensemble_prob(
        np.array([prob_xgb]),
        np.array([prob_lgbm]) if prob_lgbm is not None else None,
    )[0])

    sig = _signal_from_prob(prob_ens)
    sig_str = {1: "BUY", -1: "CASH", 0: "FLAT"}.get(sig, "FLAT")

    return {
        "prob_xgb":      prob_xgb,
        "prob_lgbm":     prob_lgbm,
        "prob_ensemble": prob_ens,
        "signal":        sig,
        "signal_str":    sig_str,
        "date":          data.index[-1],
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data import load_prices
    from features import build_features

    df_raw = load_prices()
    data, feat, tgt = build_features(df_raw)
    results = walk_forward(data, feat, tgt)
    das     = compute_da_per_model(results)

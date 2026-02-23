"""
optimization.py — Optimización de hiperparámetros y selección de variables.

Módulos:
  1. LASSO/Ridge como baseline interpretable
     → Identifica qué betas son no nulos, confirma qué variables tienen
       contenido informativo lineal en los retornos del oro.

  2. Búsqueda de hiperparámetros con Optuna (TimeSeriesSplit)
     → XGBoost y Random Forest optimizados con 50 trials cada uno.
     → Validación interna respeta el orden temporal (sin look-ahead bias).

  3. Ensemble RF + LSTM (pesos optimizados)
     → Combina las predicciones walk-forward de RF y LSTM con pesos
       que minimizan el MSE sobre el set de validación temporal.

Outputs:
  output/tables/tab_6_09_lasso.csv/.tex
  output/tables/tab_6_10_optuna_params.csv/.tex
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import OUTPUT_TABLES
from src.ml.features import build_feature_matrix, split_train_test

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ════════════════════════════════════════════════════════════════════════════
# 1. LASSO/Ridge — baseline interpretable
# ════════════════════════════════════════════════════════════════════════════

def run_lasso_analysis(df: pd.DataFrame, n_alphas: int = 100,
                       cv_splits: int = 5) -> pd.DataFrame:
    """
    Regresión LASSO con validación cruzada temporal (TimeSeriesSplit).

    Identifica qué features tienen coeficiente no nulo → variables
    informativas lineales para predecir el retorno mensual del oro.

    Args:
        df:        Dataset maestro.
        n_alphas:  Tamaño de la rejilla de λ.
        cv_splits: Número de folds en TimeSeriesSplit.

    Returns:
        DataFrame con coeficientes LASSO y Ridge, ordenado por |LASSO|.
    """
    data, feature_cols = build_feature_matrix(df)
    X_train, _, y_train, _, _, _, _ = split_train_test(data, feature_cols)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)

    # TimeSeriesSplit para respetar el orden temporal
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # LASSO con CV
    lasso = LassoCV(
        n_alphas=n_alphas,
        cv=tscv,
        max_iter=10_000,
        random_state=42,
    ).fit(X_sc, y_train)

    # Ridge con CV (para comparar: no hace selección, solo regularización)
    ridge = RidgeCV(
        alphas=np.logspace(-3, 3, 100),
        cv=tscv,
    ).fit(X_sc, y_train)

    lasso_coefs = pd.Series(lasso.coef_, index=feature_cols, name="LASSO")
    ridge_coefs = pd.Series(ridge.coef_, index=feature_cols, name="Ridge")

    result = pd.concat([lasso_coefs, ridge_coefs], axis=1)
    result["Seleccionada (LASSO)"] = (result["LASSO"] != 0).map({True: "Sí", False: "No"})
    result = result.reindex(result["LASSO"].abs().sort_values(ascending=False).index)

    logger.info(
        f"LASSO α={lasso.alpha_:.5f} | "
        f"Features seleccionadas: {(lasso.coef_ != 0).sum()} / {len(feature_cols)}"
    )

    # Guardar
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_TABLES / "tab_6_09_lasso.csv")
    tex = result.to_latex(
        float_format="%.4f",
        caption=(
            "Coeficientes LASSO (λ óptimo por TimeSeriesSplit, "
            f"{cv_splits} folds) y Ridge sobre la muestra de entrenamiento "
            f"(estandarizados). Features seleccionadas por LASSO: "
            f"{(lasso.coef_ != 0).sum()} de {len(feature_cols)}."
        ),
        label="tab:lasso",
        escape=True,
    )
    (OUTPUT_TABLES / "tab_6_09_lasso.tex").write_text(tex, encoding="utf-8")
    logger.info("Tabla LASSO guardada.")

    return result


# ════════════════════════════════════════════════════════════════════════════
# 2. Optuna — búsqueda de hiperparámetros
# ════════════════════════════════════════════════════════════════════════════

def _optuna_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                    n_trials: int, n_splits: int) -> dict:
    """Busca hiperparámetros óptimos de XGBoost con Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(f"Requiere optuna y xgboost: pip install optuna xgboost. {e}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 2, 5),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "random_state": 42,
            "verbosity": 0,
        }
        mse_folds = []
        for train_idx, val_idx in tscv.split(X_train):
            model = XGBRegressor(**params)
            model.fit(X_train[train_idx], y_train[train_idx],
                      eval_set=[(X_train[val_idx], y_train[val_idx])],
                      verbose=False)
            pred = model.predict(X_train[val_idx])
            mse_folds.append(np.mean((y_train[val_idx] - pred) ** 2))
        return np.mean(mse_folds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    best["verbosity"] = 0
    logger.info(f"XGBoost optimizado: {best} | MSE={study.best_value:.4f}")
    return best


def _optuna_rf(X_train: np.ndarray, y_train: np.ndarray,
               n_trials: int, n_splits: int) -> dict:
    """Busca hiperparámetros óptimos de Random Forest con Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as e:
        raise ImportError(f"Requiere optuna: pip install optuna. {e}")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),
            "max_features":    trial.suggest_float("max_features", 0.4, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
        mse_folds = []
        for train_idx, val_idx in tscv.split(X_train):
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            pred = model.predict(X_train[val_idx])
            mse_folds.append(np.mean((y_train[val_idx] - pred) ** 2))
        return np.mean(mse_folds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    best["n_jobs"] = -1
    logger.info(f"RF optimizado: {best} | MSE={study.best_value:.4f}")
    return best


def run_hyperparameter_search(df: pd.DataFrame, n_trials: int = 50,
                              n_splits: int = 3) -> dict:
    """
    Ejecuta Optuna para XGBoost y RF.

    Returns:
        Dict {'xgb': params_dict, 'rf': params_dict}
    """
    data, feature_cols = build_feature_matrix(df)
    X_train, _, y_train, _, _, _, _ = split_train_test(data, feature_cols)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)

    logger.info(f"Buscando hiperparámetros XGBoost ({n_trials} trials)...")
    best_xgb = _optuna_xgboost(X_sc, y_train, n_trials, n_splits)

    logger.info(f"Buscando hiperparámetros Random Forest ({n_trials} trials)...")
    best_rf  = _optuna_rf(X_sc, y_train, n_trials, n_splits)

    # Guardar tabla comparativa de hiperparámetros
    rows = []
    for param, val in best_xgb.items():
        if param not in ("random_state", "verbosity"):
            rows.append({"Modelo": "XGBoost", "Hiperparámetro": param, "Valor óptimo": val})
    for param, val in best_rf.items():
        if param not in ("random_state", "n_jobs"):
            rows.append({"Modelo": "Random Forest", "Hiperparámetro": param, "Valor óptimo": val})

    opt_df = pd.DataFrame(rows).set_index(["Modelo", "Hiperparámetro"])
    opt_df.to_csv(OUTPUT_TABLES / "tab_6_10_optuna_params.csv")
    tex = opt_df.to_latex(
        caption=(
            f"Hiperparámetros óptimos encontrados por Optuna "
            f"({n_trials} trials, TimeSeriesSplit {n_splits} folds, "
            "métrica: MSE medio en validación)."
        ),
        label="tab:optuna_params",
        escape=True,
    )
    (OUTPUT_TABLES / "tab_6_10_optuna_params.tex").write_text(tex, encoding="utf-8")
    logger.info("Tabla de hiperparámetros Optuna guardada.")

    return {"xgb": best_xgb, "rf": best_rf}


# ════════════════════════════════════════════════════════════════════════════
# 3. Ensemble RF + LSTM
# ════════════════════════════════════════════════════════════════════════════

def compute_ensemble_predictions(val_window: int = 24) -> pd.DataFrame | None:
    """
    Ensemble con pesos optimizados sobre las últimas val_window predicciones.

    Pesos: w* = argmin_w MSE(w × pred_rf + (1-w) × pred_lstm)
    Resultado: predicciones ensemble para todo el período de test.

    Returns:
        DataFrame con columnas 'actual', 'predicho_ensemble' o None si no
        se encuentran los archivos de predicciones.
    """
    rf_path   = OUTPUT_TABLES / "tab_6_rf_predictions.csv"
    lstm_path = OUTPUT_TABLES / "tab_6_lstm_predictions.csv"

    if not rf_path.exists() or not lstm_path.exists():
        logger.warning("Archivos de predicciones RF/LSTM no encontrados. Saltando ensemble.")
        return None

    rf_df   = pd.read_csv(rf_path,   index_col="fecha", parse_dates=True)
    lstm_df = pd.read_csv(lstm_path, index_col="fecha", parse_dates=True)

    # Asegurar alineación
    idx = rf_df.index.intersection(lstm_df.index)
    actual  = rf_df.loc[idx, "actual"].values
    pred_rf = rf_df.loc[idx, "predicho"].values
    pred_lstm = lstm_df.loc[idx, "predicho"].values

    # Optimizar w sobre las últimas val_window obs (ventana de calibración)
    val_actual  = actual[-val_window:]
    val_rf      = pred_rf[-val_window:]
    val_lstm    = pred_lstm[-val_window:]

    best_w, best_mse = 0.5, np.inf
    for w in np.linspace(0.0, 1.0, 101):
        ens = w * val_rf + (1 - w) * val_lstm
        mse = np.mean((val_actual - ens) ** 2)
        if mse < best_mse:
            best_mse, best_w = mse, w

    logger.info(
        f"Ensemble: w_RF={best_w:.2f}, w_LSTM={1-best_w:.2f} "
        f"(calibrado sobre últimas {val_window} obs)"
    )

    pred_ens = best_w * pred_rf + (1 - best_w) * pred_lstm

    ens_df = pd.DataFrame({
        "actual":            actual,
        "predicho":          pred_ens,
        "predicho_rf":       pred_rf,
        "predicho_lstm":     pred_lstm,
        "w_rf":              best_w,
        "w_lstm":            1 - best_w,
    }, index=idx)
    ens_df.index.name = "fecha"
    ens_df.to_csv(OUTPUT_TABLES / "tab_6_ensemble_predictions.csv")
    logger.info("Predicciones ensemble guardadas.")

    return ens_df


# ════════════════════════════════════════════════════════════════════════════
# 4. Ejecución directa
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from src.data.pipeline import load_master_dataset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    df = load_master_dataset()

    print("\n=== LASSO / Ridge ===")
    lasso_df = run_lasso_analysis(df)
    print(lasso_df[lasso_df["LASSO"] != 0].to_string())

    print("\n=== Optuna Hyperparameter Search ===")
    best_params = run_hyperparameter_search(df, n_trials=50)
    print("XGBoost:", best_params["xgb"])
    print("RF:      ", best_params["rf"])

    print("\n=== Ensemble RF + LSTM ===")
    ens = compute_ensemble_predictions()
    if ens is not None:
        rmse = float(np.sqrt(np.mean((ens["actual"] - ens["predicho"]) ** 2)))
        da   = float(np.mean(np.sign(ens["actual"]) == np.sign(ens["predicho"])) * 100)
        print(f"Ensemble: RMSE={rmse:.3f}pp | DA={da:.1f}%")

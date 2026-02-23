"""
walk_forward.py — Validación temporal walk-forward y métricas de evaluación.

La validación walk-forward (expanding window) es la metodología correcta para
series financieras: en cada paso t, el modelo se entrena con todas las
observaciones hasta t-1 y predice t. Nunca ve datos futuros durante el
entrenamiento. Esto contrasta con la cross-validation estándar (k-fold),
que introduciría look-ahead bias al mezclar observaciones de diferentes épocas.

Métricas reportadas:
- RMSE: penaliza errores grandes (relevante para el riesgo de predicción)
- MAE: error absoluto medio (interpretación directa)
- MAPE: error porcentual (permite comparar entre activos y periodos)
- Directional Accuracy (DA): % de meses donde el modelo acierta la dirección
  del movimiento (sube/baja). Es la métrica más relevante para señales de trading.
"""

import numpy as np
import pandas as pd


def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    model_class,
    model_params: dict,
    initial_train_size: int,
    step: int = 1,
    refit_every: int = 1,
    scaler=None,
) -> pd.DataFrame:
    """
    Ejecuta walk-forward validation con ventana expandible.

    Args:
        X: Matriz de features (n_obs, n_features).
        y: Vector target (n_obs,).
        dates: Índice temporal.
        model_class: Clase del modelo (XGBRegressor, RandomForestRegressor, etc.)
        model_params: Hiperparámetros del modelo.
        initial_train_size: Número de observaciones del primer entrenamiento.
        step: Observaciones entre reentrenamientos (1 = mensual).
        refit_every: Cada cuántos pasos se reentrena el modelo.
        scaler: Si no None, StandardScaler que se reajusta en cada ventana.

    Returns:
        DataFrame con columnas: fecha, actual, predicho, error.
    """
    n = len(y)
    predictions, actuals, pred_dates = [], [], []

    model = None
    last_fit = -refit_every  # fuerza reentrenamiento en el primer paso

    for t in range(initial_train_size, n, step):
        X_train_t = X[:t]
        y_train_t = y[:t]

        # Reescalar si hay scaler
        if scaler is not None:
            from sklearn.preprocessing import StandardScaler
            sc_t = StandardScaler()
            X_train_sc = sc_t.fit_transform(X_train_t)
            X_pred_sc = sc_t.transform(X[t:t+step])
        else:
            X_train_sc = X_train_t
            X_pred_sc = X[t:t+step]

        # Reentrenar según la frecuencia especificada
        if (t - initial_train_size) % (refit_every * step) == 0:
            model = model_class(**model_params)
            model.fit(X_train_sc, y_train_t)
            last_fit = t

        if model is None:
            continue

        pred = model.predict(X_pred_sc)
        predictions.extend(pred.tolist())
        actuals.extend(y[t:t+step].tolist())
        pred_dates.extend(dates[t:t+step].tolist())

    return pd.DataFrame({
        "fecha": pred_dates,
        "actual": actuals,
        "predicho": predictions,
        "error": np.array(actuals) - np.array(predictions),
    }).set_index("fecha")


def compute_metrics(results: pd.DataFrame, model_name: str) -> dict:
    """
    Calcula las cuatro métricas de evaluación a partir de los resultados
    walk-forward.

    Args:
        results: DataFrame con columnas 'actual' y 'predicho'.
        model_name: Nombre del modelo para el reporte.

    Returns:
        Dict con RMSE, MAE, MAPE, DA y nombre del modelo.
    """
    actual = results["actual"].values
    pred = results["predicho"].values

    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))

    # MAPE: excluir meses donde actual ≈ 0 (retornos exactamente nulos)
    mask = np.abs(actual) > 1e-6
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

    # Directional Accuracy: acierta la dirección del movimiento
    dir_actual = np.sign(actual)
    dir_pred = np.sign(pred)
    da = np.mean(dir_actual == dir_pred) * 100

    return {
        "Modelo": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "Direc. Accuracy (%)": da,
        "N predicciones": len(actual),
    }


def naive_benchmark(y: np.ndarray, dates: pd.DatetimeIndex, initial_size: int) -> pd.DataFrame:
    """
    Benchmark naive: predice que el retorno del próximo mes = retorno de este mes
    (random walk con drift = 0, el peor benchmark a batir).
    """
    test_actual = y[initial_size:]
    test_pred = y[initial_size - 1: -1]  # lag 1
    test_dates = dates[initial_size:]

    return pd.DataFrame({
        "fecha": test_dates,
        "actual": test_actual,
        "predicho": test_pred,
        "error": test_actual - test_pred,
    }).set_index("fecha")

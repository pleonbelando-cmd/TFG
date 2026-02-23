"""
tree_models.py — XGBoost, Random Forest y análisis SHAP.

Genera para el Capítulo 6:
- Tab 6.2: Hiperparámetros seleccionados (XGBoost y RF)
- Fig 6.1: Importancia de features (XGBoost, SHAP mean |φ|)
- Fig 6.2: Predicción walk-forward vs precio real del oro (mejor modelo árboles)
- Fig 6.3: SHAP summary plot (beeswarm)
- Fig 6.4: SHAP waterfall — episodios clave (GFC 2008, COVID 2020, 2025)
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.config import OUTPUT_FIGURES, OUTPUT_TABLES, VARIABLE_LABELS
from src.utils.latex_tables import save_table
from src.utils.plotting import set_academic_style
from src.ml.walk_forward import walk_forward_predict, compute_metrics

logger = logging.getLogger(__name__)

# ── Hiperparámetros ──────────────────────────────────────────────────────────
XGB_PARAMS = {
    # Hiperparámetros optimizados por Optuna (50 trials, TimeSeriesSplit 3 folds)
    "n_estimators":     117,
    "max_depth":        4,
    "learning_rate":    0.0101,
    "subsample":        0.9553,
    "colsample_bytree": 0.6214,
    "reg_alpha":        0.1905,  # L1
    "reg_lambda":       0.0669,  # L2
    "random_state":     42,
    "verbosity":        0,
}

RF_PARAMS = {
    # Hiperparámetros optimizados por Optuna (50 trials, TimeSeriesSplit 3 folds)
    "n_estimators":     403,
    "max_depth":        5,
    "min_samples_leaf": 13,  # fuerte regularización, apropiada para n<300
    "max_features":     0.4056,
    "random_state":     42,
    "n_jobs":           -1,
}

PALETTE = ["#1565C0", "#D32F2F", "#2E7D32", "#F57C00", "#6A1B9A"]


def save_hyperparams_table(feature_cols: list) -> None:
    """Tabla 6.2: Hiperparámetros usados en XGBoost y RF."""
    rows = [
        {"Modelo": "XGBoost", "Hiperparámetro": k, "Valor": str(v)}
        for k, v in XGB_PARAMS.items() if k != "verbosity"
    ] + [
        {"Modelo": "Random Forest", "Hiperparámetro": k, "Valor": str(v)}
        for k, v in RF_PARAMS.items() if k != "n_jobs"
    ]
    df = pd.DataFrame(rows).set_index(["Modelo", "Hiperparámetro"])
    save_table(df, "tab_6_02_hyperparams",
               caption="Hiperparámetros de los modelos de árboles (XGBoost y Random Forest)",
               label="tab:hyperparams", float_format="%.4f")


def run_xgboost_walkforward(
    X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
    initial_train_size: int, feature_cols: list,
) -> tuple:
    """Ejecuta XGBoost con walk-forward validation."""
    logger.info("  Ejecutando XGBoost walk-forward...")
    results = walk_forward_predict(
        X, y, dates, XGBRegressor, XGB_PARAMS,
        initial_train_size=initial_train_size, step=1, refit_every=3,
    )
    metrics = compute_metrics(results, "XGBoost")
    logger.info(f"    RMSE={metrics['RMSE']:.4f}, DA={metrics['Direc. Accuracy (%)']:.1f}%")
    return results, metrics


def run_rf_walkforward(
    X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
    initial_train_size: int, feature_cols: list,
) -> tuple:
    """Ejecuta Random Forest con walk-forward validation."""
    logger.info("  Ejecutando Random Forest walk-forward...")
    results = walk_forward_predict(
        X, y, dates, RandomForestRegressor, RF_PARAMS,
        initial_train_size=initial_train_size, step=1, refit_every=6,
    )
    metrics = compute_metrics(results, "Random Forest")
    logger.info(f"    RMSE={metrics['RMSE']:.4f}, DA={metrics['Direc. Accuracy (%)']:.1f}%")
    return results, metrics


def plot_predictions(
    results_xgb: pd.DataFrame, results_rf: pd.DataFrame,
    df_original: pd.DataFrame,
) -> None:
    """
    Figura 6.2: Predicción walk-forward de retornos del oro vs real.
    Panel superior: precio del oro real + precio reconstruido con predicciones.
    Panel inferior: retorno predicho vs real.
    """
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # ── Panel inferior: retornos predichos vs reales ──────────────────────────
    actual = results_xgb["actual"]
    ax2.plot(actual.index, actual * 100, color="black", linewidth=1.0,
             label="Real", alpha=0.8)
    ax2.plot(results_xgb.index, results_xgb["predicho"] * 100,
             color=PALETTE[0], linewidth=0.9, alpha=0.75, label="XGBoost")
    ax2.plot(results_rf.index, results_rf["predicho"] * 100,
             color=PALETTE[1], linewidth=0.9, alpha=0.75, linestyle="--",
             label="Random Forest")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("Retorno mensual del oro (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel superior: precio reconstruido ──────────────────────────────────
    # Reconstruir desde ln_gold con los retornos predichos
    test_start = results_xgb.index[0]
    gold_prices = df_original["gold"].dropna()
    # Precio base (último precio conocido antes del test)
    p0_xgb = gold_prices[gold_prices.index < test_start].iloc[-1]
    gold_actual_test = gold_prices[gold_prices.index >= test_start]

    # Precio reconstruido XGBoost
    pred_xgb = results_xgb["predicho"].values
    price_xgb = [p0_xgb]
    for r in pred_xgb:
        price_xgb.append(price_xgb[-1] * np.exp(r))
    price_xgb = pd.Series(price_xgb[1:], index=results_xgb.index)

    # Precio reconstruido RF
    pred_rf = results_rf["predicho"].values
    p0_rf = p0_xgb
    price_rf = [p0_rf]
    for r in pred_rf:
        price_rf.append(price_rf[-1] * np.exp(r))
    price_rf = pd.Series(price_rf[1:], index=results_rf.index)

    ax1.plot(gold_actual_test.index, gold_actual_test.values,
             color="black", linewidth=1.2, label="Precio real (USD/oz)")
    ax1.plot(price_xgb.index, price_xgb.values,
             color=PALETTE[0], linewidth=1.0, alpha=0.8, label="XGBoost (reconstruido)")
    ax1.plot(price_rf.index, price_rf.values,
             color=PALETTE[1], linewidth=1.0, alpha=0.8, linestyle="--",
             label="Random Forest (reconstruido)")
    ax1.set_ylabel("Precio del oro (USD/oz)")
    ax1.set_title("Figura 6.2: Predicción Walk-Forward — Modelos de Árboles vs Precio Real",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_FIGURES / f"fig_6_02_tree_predictions.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura 6.2 guardada.")


def compute_and_plot_shap(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, feature_cols: list,
    df_data: pd.DataFrame,
) -> np.ndarray:
    """
    Calcula SHAP values con XGBoost entrenado en toda la muestra de train.
    Genera Fig 6.1 (importancia SHAP), Fig 6.3 (summary beeswarm)
    y Fig 6.4 (waterfall de episodios clave).
    """
    set_academic_style()

    # Entrenar modelo final sobre todo el train
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # Calcular SHAP values
    logger.info("  Calculando SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # (n_test, n_features)

    # Etiquetas legibles para las features
    def clean_label(col):
        base = col.replace("_lag1", "").replace("_lag2", "").replace("_lag3", "")
        lag = ""
        if "_lag" in col:
            lag = f" (t-{col.split('_lag')[1]})"
        base_map = {
            "gold_ret": "Ret. oro",
            "dxy_ret": "DXY",
            "tips_10y": "TIPS 10Y",
            "vix": "VIX",
            "vix_chg": "ΔVIX",
            "sp500_ret": "S&P 500",
            "cpi_yoy": "CPI YoY",
            "fedfunds": "Fed Funds",
            "breakeven": "Breakeven",
            "wti_ret": "WTI",
            "gold_ma3": "MA3 oro",
            "gold_ma6": "MA6 oro",
            "gold_vol3": "Vol3 oro",
            "real_rate_proxy_lag1": "Tipo real proxy",
            "is_crisis": "Episodio crisis",
        }
        return base_map.get(base, base) + lag

    labels = [clean_label(c) for c in feature_cols]

    # ── Fig 6.1: Importancia media |SHAP| ────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": labels,
        "SHAP mean |φ|": mean_abs_shap,
    }).sort_values("SHAP mean |φ|", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(6, len(labels) * 0.35)))
    bars = ax.barh(importance_df["Feature"], importance_df["SHAP mean |φ|"],
                   color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Importancia SHAP media (|φ|)")
    ax.set_title("Figura 6.1: Importancia de Variables — XGBoost (SHAP mean |φ|)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_FIGURES / f"fig_6_01_shap_importance.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura 6.1 guardada.")

    # Guardar tabla de importancias
    save_table(
        importance_df.sort_values("SHAP mean |φ|", ascending=False).set_index("Feature"),
        "tab_6_03_shap_importance",
        caption="Importancia de variables por SHAP (XGBoost) — media del valor absoluto de Shapley",
        label="tab:shap_importance", float_format="%.5f",
    )

    # ── Fig 6.3: Summary plot (beeswarm manual) ───────────────────────────────
    # Ordenar por importancia media
    order = np.argsort(mean_abs_shap)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    # Scatter: eje x = SHAP value, eje y = feature (ranking), color = valor feature
    for rank, feat_idx in enumerate(order[::-1]):  # bottom-up
        sv = shap_values[:, feat_idx]
        fv = X_test[:, feat_idx]
        # Normalizar valores de feature para color
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        sc = ax.scatter(sv, [rank] * len(sv), c=fv_norm, cmap="RdBu_r",
                        alpha=0.6, s=15, vmin=0, vmax=1)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[i] for i in order[::-1]], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impacto sobre la predicción del retorno del oro)")
    ax.set_title("Figura 6.3: SHAP Summary Plot — XGBoost",
                 fontsize=12, fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Valor de la feature (azul=bajo, rojo=alto)", fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_FIGURES / f"fig_6_03_shap_summary.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura 6.3 guardada.")

    # ── Fig 6.4: Waterfall de episodios clave ────────────────────────────────
    _plot_shap_waterfalls(shap_values, X_test, df_data, labels, explainer.expected_value)

    return shap_values


def _plot_shap_waterfalls(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    df_data: pd.DataFrame,
    labels: list,
    base_value: float,
) -> None:
    """
    Figura 6.4: Waterfall SHAP para 3 observaciones de episodios clave.
    Muestra qué variables dominan la predicción en meses específicos.
    """
    set_academic_style()

    # Fechas de interés: GFC peak (oct 2008), COVID crash (mar 2020), máximo 2025
    target_dates = {
        "GFC — oct. 2008": "2008-10",
        "COVID — mar. 2020": "2020-03",
        "Triple confluencia — dic. 2025": "2025-12",
    }

    # Buscar índices en df_data que correspondan al período de test
    test_idx = df_data.index[df_data.index >= df_data.index[int(len(df_data) * 0.6)]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    for ax, (label, date_str) in zip(axes, target_dates.items()):
        # Encontrar el índice más cercano en el test set
        target_ts = pd.Timestamp(date_str)
        if len(test_idx) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        closest_idx = np.argmin(np.abs(test_idx - target_ts))
        if closest_idx >= len(shap_values):
            closest_idx = len(shap_values) - 1

        sv = shap_values[closest_idx]
        actual_date = test_idx[closest_idx].strftime("%b %Y")

        # Top 8 features por |SHAP|
        top_k = 8
        order = np.argsort(np.abs(sv))[::-1][:top_k]
        top_labels = [labels[i] for i in order]
        top_vals = sv[order]

        # Waterfall manual
        colors = [PALETTE[1] if v < 0 else PALETTE[2] for v in top_vals]
        y_pos = range(top_k)

        ax.barh(list(y_pos), top_vals, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_labels, fontsize=8)
        ax.set_xlabel("SHAP value (φ)")
        ax.set_title(f"{label}\n({actual_date})", fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Figura 6.4: SHAP Waterfall — Contribución de Variables en Episodios Clave",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_FIGURES / f"fig_6_04_shap_waterfall.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura 6.4 guardada.")


def generate_tree_models(
    X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex,
    initial_train_size: int, feature_cols: list,
    df_original: pd.DataFrame,
) -> dict:
    """Función principal: ejecuta XGBoost, RF y SHAP."""
    logger.info("=== Modelos de árboles (Capítulo 6) ===")

    save_hyperparams_table(feature_cols)

    results_xgb, metrics_xgb = run_xgboost_walkforward(X, y, dates, initial_train_size, feature_cols)
    results_rf, metrics_rf = run_rf_walkforward(X, y, dates, initial_train_size, feature_cols)

    # Guardar predicciones como CSV para el texto del capítulo
    results_xgb.to_csv(OUTPUT_TABLES / "tab_6_xgb_predictions.csv")
    results_rf.to_csv(OUTPUT_TABLES / "tab_6_rf_predictions.csv")

    # Figuras de predicción
    plot_predictions(results_xgb, results_rf, df_original)

    # SHAP — entrenar sobre muestra train completa
    X_train = X[:initial_train_size]
    X_test = X[initial_train_size:]
    y_train = y[:initial_train_size]
    df_test = df_original.iloc[initial_train_size:]

    shap_values = compute_and_plot_shap(X_train, X_test, y_train, feature_cols, df_test)

    logger.info("=== Modelos de árboles completados ===")
    return {
        "xgb": {"results": results_xgb, "metrics": metrics_xgb},
        "rf": {"results": results_rf, "metrics": metrics_rf},
        "shap_values": shap_values,
    }

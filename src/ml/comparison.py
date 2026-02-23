"""
comparison.py — Tabla de comparación de todos los modelos del Capítulo 6.

Consolida los resultados walk-forward de:
  - Naive benchmark (random walk con drift = 0)
  - XGBoost
  - Random Forest
  - LSTM (PyTorch)

Genera la Tabla 6.5 (métricas comparativas) y la Figura 6.6 (comparativa
visual de RMSE, MAE y Directional Accuracy).

Uso:
    from src.ml.comparison import run_all_models
    comparison_df = run_all_models(df)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import OUTPUT_FIGURES, OUTPUT_TABLES
from src.utils.latex_tables import save_table
from src.utils.plotting import set_academic_style
from src.ml.walk_forward import compute_metrics, naive_benchmark
from src.ml.features import build_feature_matrix, split_train_test
from src.ml.tree_models import generate_tree_models
from src.ml.lstm_model import generate_lstm_model

logger = logging.getLogger(__name__)

PALETTE = ["#718096", "#1565C0", "#D32F2F", "#6A1B9A"]  # naive, xgb, rf, lstm


# ─────────────────────────────────────────────────────────────────────────────
#  Función principal
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models(
    df: pd.DataFrame,
    train_frac: float = 0.60,
) -> pd.DataFrame:
    """
    Ejecuta todos los modelos sobre el dataset maestro y devuelve la tabla
    comparativa de métricas walk-forward.

    Args:
        df: Dataset maestro (gold_macro_monthly.csv) con al menos las columnas
            de BASE_FEATURES + 'gold_ret' + 'episode'.
        train_frac: Fracción temporal usada como conjunto de entrenamiento inicial.

    Returns:
        DataFrame con una fila por modelo y columnas RMSE, MAE, MAPE, DA, N.
    """
    logger.info("=== Comparativa de modelos — Capítulo 6 ===")

    # ── Preparar features ────────────────────────────────────────────────────
    data, feature_cols = build_feature_matrix(df)
    n = len(data)
    initial_size = int(n * train_frac)

    X = data[feature_cols].values
    y = data["gold_ret"].values
    dates = data.index

    logger.info(
        f"Dataset listo: {n} obs, {len(feature_cols)} features, "
        f"train={initial_size} ({initial_size/n:.0%}), "
        f"test={n - initial_size} ({(n-initial_size)/n:.0%})"
    )

    metrics_list = []

    # ── 1. Naive benchmark ───────────────────────────────────────────────────
    logger.info("Ejecutando Naive benchmark...")
    naive_res = naive_benchmark(y, dates, initial_size)
    metrics_list.append(compute_metrics(naive_res, "Naive (random walk)"))

    # ── 2. XGBoost + Random Forest ───────────────────────────────────────────
    logger.info("Ejecutando XGBoost y Random Forest...")
    tree_out = generate_tree_models(
        X, y, dates, initial_size, feature_cols, df_original=df,
    )
    metrics_list.append(tree_out["xgb"]["metrics"])
    metrics_list.append(tree_out["rf"]["metrics"])

    # ── 3. LSTM ──────────────────────────────────────────────────────────────
    logger.info("Ejecutando LSTM (esto puede tardar varios minutos)...")
    lstm_out = generate_lstm_model(
        X, y, dates, initial_size, feature_cols, df_original=df,
    )
    metrics_list.append(lstm_out["metrics"])

    # ── Tabla comparativa (Tab 6.5) ──────────────────────────────────────────
    comparison_df = pd.DataFrame(metrics_list).set_index("Modelo")

    save_table(
        comparison_df,
        "tab_6_05_model_comparison",
        caption=(
            "Comparativa de modelos predictivos — Walk-forward validation "
            "(muestra de test: 40\\% final, 2015--2025)"
        ),
        label="tab:model_comparison",
        float_format="%.4f",
    )
    logger.info("Tabla 6.5 guardada.")
    logger.info("\n" + comparison_df.to_string())

    # ── Figura comparativa (Fig 6.6) ─────────────────────────────────────────
    _plot_comparison(comparison_df)

    logger.info("=== Comparativa completada ===")
    return comparison_df


# ─────────────────────────────────────────────────────────────────────────────
#  Figura 6.6
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Figura 6.6: Comparativa de modelos en dos paneles.
    Panel izquierdo: RMSE y MAE (error absoluto, menor es mejor).
    Panel derecho: Directional Accuracy (mayor es mejor, línea de referencia 50%).
    """
    set_academic_style()

    models = comparison_df.index.tolist()
    x = np.arange(len(models))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel izquierdo: RMSE y MAE ─────────────────────────────────────────
    bars_rmse = ax1.bar(
        x - width / 2, comparison_df["RMSE"], width,
        label="RMSE", color=[PALETTE[i] for i in range(len(models))],
        alpha=0.85,
    )
    bars_mae = ax1.bar(
        x + width / 2, comparison_df["MAE"], width,
        label="MAE", color=[PALETTE[i] for i in range(len(models))],
        alpha=0.55, hatch="//",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax1.set_title("Error de predicción (menor es mejor)")
    ax1.set_ylabel("Magnitud del error (log-retorno)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Anotar valores encima de las barras RMSE
    for bar in bars_rmse:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 0.0003,
            f"{h:.4f}", ha="center", va="bottom", fontsize=7,
        )

    # ── Panel derecho: Directional Accuracy ──────────────────────────────────
    da_values = comparison_df["Direc. Accuracy (%)"]
    bars_da = ax2.bar(
        models, da_values,
        color=[PALETTE[i] for i in range(len(models))],
        alpha=0.85,
    )
    ax2.axhline(50, color="black", lw=1.0, linestyle="--",
                label="50% (predicción aleatoria)", zorder=3)
    ax2.set_ylim(0, min(85, da_values.max() + 15))
    ax2.set_title("Directional Accuracy (mayor es mejor)")
    ax2.set_ylabel("% aciertos de dirección del movimiento")
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars_da:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, h + 0.5,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle(
        "Figura 6.6: Comparativa de Modelos Predictivos — Walk-Forward Validation",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_6_06_model_comparison.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 6.6 guardada.")

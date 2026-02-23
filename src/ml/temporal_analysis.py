"""
temporal_analysis.py — Análisis de estabilidad temporal de las predicciones ML.

Módulos:
  1. Rolling Directional Accuracy (ventana deslizante 24 meses)
     → Muestra cuándo cada modelo funciona bien y cuándo falla.

  2. Métricas por subperíodo
     → RMSE y DA desglosados por fases del mercado (Pre-COVID, COVID,
       ciclo de tipos, confluencia 2025).

  3. Error condicional
     → Segmenta errores según magnitud y dirección del retorno real:
       meses de movimiento grande vs. pequeño; subidas vs. bajadas.

Outputs:
  output/figures/fig_6_07_rolling_da.png/.pdf
  output/tables/tab_6_07_subperiod.csv/.tex
  output/tables/tab_6_08_conditional_error.csv/.tex
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import OUTPUT_FIGURES, OUTPUT_TABLES
from src.utils.plotting import set_academic_style as apply_academic_style

logger = logging.getLogger(__name__)

# ── Subperíodos del conjunto de test (Oct 2016 – Oct 2025) ──────────────────
SUBPERIODS = {
    "Pre-COVID\n(Oct 2016–Ene 2020)":        ("2016-10", "2020-01"),
    "COVID y recuperación\n(Feb 2020–Dic 2021)": ("2020-02", "2021-12"),
    "Ciclo subidas tipos\n(Ene 2022–Dic 2024)":  ("2022-01", "2024-12"),
    "Confluencia 2025\n(Ene 2025–Oct 2025)":     ("2025-01", "2025-10"),
}

PRED_FILES = {
    "XGBoost":       OUTPUT_TABLES / "tab_6_xgb_predictions.csv",
    "Random Forest": OUTPUT_TABLES / "tab_6_rf_predictions.csv",
    "LSTM":          OUTPUT_TABLES / "tab_6_lstm_predictions.csv",
}

MODEL_COLORS = {
    "XGBoost":       "#E07B39",
    "Random Forest": "#2A6496",
    "LSTM":          "#2E7D32",
}


def _load_predictions() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in PRED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {path}. "
                "Ejecuta primero run_chapter6.py."
            )
        dfs[name] = pd.read_csv(path, index_col="fecha", parse_dates=True)
    return dfs


# ════════════════════════════════════════════════════════════════════════════
# 1. Rolling Directional Accuracy
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_da(dfs: dict, window: int = 24) -> pd.DataFrame:
    """
    Calcula la Directional Accuracy con ventana deslizante.

    Args:
        dfs:    Dict {nombre: DataFrame con 'actual', 'predicho'}.
        window: Tamaño de la ventana en meses.

    Returns:
        DataFrame con columnas = modelos, índice = fechas.
    """
    result = {}
    for name, df in dfs.items():
        correct = (np.sign(df["actual"]) == np.sign(df["predicho"])).astype(float)
        rolling_da = correct.rolling(window).mean() * 100
        result[name] = rolling_da

    return pd.DataFrame(result)


def plot_rolling_da(rolling_da: pd.DataFrame, window: int = 24) -> None:
    """Figura 6.7 — Rolling DA con ventana de 24 meses."""
    apply_academic_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))

    for name, color in MODEL_COLORS.items():
        if name in rolling_da.columns:
            ax.plot(rolling_da.index, rolling_da[name],
                    label=name, color=color, linewidth=1.8)

    # Línea de referencia al 50%
    ax.axhline(50, color="black", linewidth=1.0, linestyle="--",
               label="DA = 50% (sin poder predictivo)")

    ax.set_title(
        f"Directional Accuracy con ventana deslizante ({window} meses)",
        fontsize=11, pad=8
    )
    ax.set_ylabel("DA (%)", fontsize=10)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(20, 95)

    # Sombrear subperíodos del test
    shade_colors = ["#D32F2F", "#7B1FA2", "#1565C0", "#2E7D32"]
    shade_labels = ["Pre-COVID", "COVID", "Subidas tipos", "2025"]
    for (_, (s, e)), color, label in zip(SUBPERIODS.items(), shade_colors, shade_labels):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.06, color=color, label=label)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"fig_6_07_rolling_da.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        logger.info(f"Figura guardada: {path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 2. Métricas por subperíodo
# ════════════════════════════════════════════════════════════════════════════

def compute_subperiod_metrics(dfs: dict) -> pd.DataFrame:
    """
    RMSE y DA por subperíodo para cada modelo.

    Returns:
        DataFrame MultiIndex (modelo, subperíodo) × (RMSE, DA, N).
    """
    rows = []
    for name, df in dfs.items():
        for label, (start, end) in SUBPERIODS.items():
            mask = (df.index >= start) & (df.index <= end)
            sub = df.loc[mask]
            if len(sub) < 3:
                continue

            actual = sub["actual"].values
            pred   = sub["predicho"].values

            rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
            da   = float(np.mean(np.sign(actual) == np.sign(pred)) * 100)

            # Etiqueta corta para la tabla
            label_short = label.replace("\n", " ")
            rows.append({
                "Modelo":       name,
                "Subperíodo":   label_short,
                "N":            len(sub),
                "RMSE (pp)":    round(rmse, 3),
                "DA (%)":       round(da, 1),
            })

    return pd.DataFrame(rows).set_index(["Modelo", "Subperíodo"])


# ════════════════════════════════════════════════════════════════════════════
# 3. Error condicional
# ════════════════════════════════════════════════════════════════════════════

def compute_conditional_errors(dfs: dict, big_move_threshold: float = 3.0) -> pd.DataFrame:
    """
    Segmenta los errores según:
      - Magnitud del retorno real: |retorno| > umbral (grande) vs. ≤ umbral (pequeño)
      - Dirección del retorno real: subida (>0) vs. bajada (<0)

    Args:
        big_move_threshold: Umbral en pp para definir movimiento grande (default 3pp).

    Returns:
        DataFrame con RMSE y DA condicionales.
    """
    rows = []
    for name, df in dfs.items():
        actual = df["actual"].values
        pred   = df["predicho"].values

        for segment_name, mask in [
            (f"|retorno| > {big_move_threshold}pp (grande)",
             np.abs(actual) > big_move_threshold),
            (f"|retorno| ≤ {big_move_threshold}pp (pequeño)",
             np.abs(actual) <= big_move_threshold),
            ("Retorno positivo (subida)",  actual > 0),
            ("Retorno negativo (bajada)",  actual < 0),
        ]:
            if mask.sum() < 5:
                continue
            a_seg = actual[mask]
            p_seg = pred[mask]
            rows.append({
                "Modelo":    name,
                "Segmento":  segment_name,
                "N":         int(mask.sum()),
                "RMSE (pp)": round(float(np.sqrt(np.mean((a_seg - p_seg) ** 2))), 3),
                "DA (%)":    round(float(np.mean(np.sign(a_seg) == np.sign(p_seg)) * 100), 1),
            })

    return pd.DataFrame(rows).set_index(["Modelo", "Segmento"])


# ════════════════════════════════════════════════════════════════════════════
# 4. Función principal
# ════════════════════════════════════════════════════════════════════════════

def run_temporal_analysis(window: int = 24) -> dict:
    """
    Ejecuta el análisis temporal completo y guarda outputs.

    Returns:
        Dict con DataFrames: 'rolling_da', 'subperiod', 'conditional'.
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    dfs = _load_predictions()

    # ── Rolling DA ─────────────────────────────────────────────────────────
    rolling_da = compute_rolling_da(dfs, window=window)
    plot_rolling_da(rolling_da, window=window)

    # ── Subperíodos ─────────────────────────────────────────────────────────
    subperiod_df = compute_subperiod_metrics(dfs)
    subperiod_df.to_csv(OUTPUT_TABLES / "tab_6_07_subperiod.csv")
    tex = subperiod_df.to_latex(
        caption=(
            "Métricas de predicción por subperíodo (walk-forward, muestra de test). "
            "RMSE en puntos porcentuales del retorno logarítmico mensual del oro."
        ),
        label="tab:subperiod",
        escape=True,
    )
    (OUTPUT_TABLES / "tab_6_07_subperiod.tex").write_text(tex, encoding="utf-8")
    logger.info("Tabla de subperíodos guardada.")

    # ── Error condicional ───────────────────────────────────────────────────
    conditional_df = compute_conditional_errors(dfs)
    conditional_df.to_csv(OUTPUT_TABLES / "tab_6_08_conditional_error.csv")
    tex2 = conditional_df.to_latex(
        caption=(
            "Error condicional de predicción según magnitud y dirección del retorno real. "
            "Umbral movimiento grande: |retorno| > 3 pp."
        ),
        label="tab:conditional_error",
        escape=True,
    )
    (OUTPUT_TABLES / "tab_6_08_conditional_error.tex").write_text(tex2, encoding="utf-8")
    logger.info("Tabla de error condicional guardada.")

    return {
        "rolling_da":  rolling_da,
        "subperiod":   subperiod_df,
        "conditional": conditional_df,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. Ejecución directa
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    results = run_temporal_analysis()
    print("\n=== Métricas por subperíodo ===\n")
    print(results["subperiod"].to_string())
    print("\n=== Error condicional ===\n")
    print(results["conditional"].to_string())

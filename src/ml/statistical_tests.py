"""
statistical_tests.py — Validación formal del poder predictivo de los modelos ML.

Tests implementados:
  1. Diebold-Mariano (DM) con corrección Harvey-Leybourne-Newbold (HLN, 1997)
     H₀: el modelo no mejora al benchmark naive en MSE
     Estadístico: t_{n-1} con corrección para muestras pequeñas

  2. Out-of-sample R² (Campbell & Thompson, 2008)
     OOS-R² = 1 − MSE_modelo / MSE_naive
     > 0 → mejora sobre random walk; < 0 → peor que no predecir

  3. Test binomial para Directional Accuracy
     H₀: DA = 50% (sin contenido informativo sobre la dirección)
     Estadístico: z ~ N(0,1)

  4. Intervalos de confianza bootstrap por bloques (circular)
     Preserva la autocorrelación de la serie de errores.
     block_size = 12 meses, 1 000 réplicas, α = 5%.

Output: output/tables/tab_6_06_dm_tests.csv/.tex
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import OUTPUT_TABLES

logger = logging.getLogger(__name__)


# ── Rutas de predicciones existentes ────────────────────────────────────────
PRED_FILES = {
    "XGBoost":      OUTPUT_TABLES / "tab_6_xgb_predictions.csv",
    "Random Forest": OUTPUT_TABLES / "tab_6_rf_predictions.csv",
    "LSTM":          OUTPUT_TABLES / "tab_6_lstm_predictions.csv",
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Funciones de test
# ════════════════════════════════════════════════════════════════════════════

def diebold_mariano_hln(actual: np.ndarray, pred_model: np.ndarray,
                        pred_naive: np.ndarray, h: int = 1) -> dict:
    """
    Test de Diebold-Mariano con corrección Harvey-Leybourne-Newbold (1997).

    Args:
        actual:     Valores reales (n,).
        pred_model: Predicciones del modelo a evaluar (n,).
        pred_naive: Predicciones del benchmark naive (n,).
        h:          Horizonte de predicción (1 = un paso).

    Returns:
        Dict con 'dm_stat', 'p_value', 'sig'.
        p_value bilateral; H₀ rechazada si p < 0.05.
    """
    n = len(actual)
    e_model = actual - pred_model
    e_naive = actual - pred_naive

    # Pérdida cuadrática diferencial: d_t = e_naive² - e_model²
    # (positivo → modelo mejor que naive)
    d = e_naive ** 2 - e_model ** 2
    d_bar = np.mean(d)

    # Varianza de la pérdida diferencial (Newey-West, h-1 lags)
    var_d = np.var(d, ddof=0)
    for tau in range(1, h):
        autocov = np.mean((d[tau:] - d_bar) * (d[:-tau] - d_bar))
        var_d += 2 * autocov

    # Estadístico DM original
    dm_stat_raw = d_bar / np.sqrt(max(var_d, 1e-12) / n)

    # Factor de corrección HLN para muestras pequeñas
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = dm_stat_raw * hln_factor

    # p-valor bilateral ~ t_{n-1}
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    sig = ""
    if p_value < 0.01:
        sig = "***"
    elif p_value < 0.05:
        sig = "**"
    elif p_value < 0.10:
        sig = "*"

    return {"dm_stat": dm_stat, "p_value": p_value, "sig": sig}


def oos_r2(actual: np.ndarray, pred_model: np.ndarray,
           pred_naive: np.ndarray) -> float:
    """
    Out-of-sample R² de Campbell & Thompson (2008).

    OOS-R² = 1 − MSE_modelo / MSE_naive

    Interpretación:
        > 0 → el modelo reduce el error cuadrático respecto al random walk.
        = 0 → empata con el naive.
        < 0 → peor que no predecir.
    """
    mse_model = np.mean((actual - pred_model) ** 2)
    mse_naive = np.mean((actual - pred_naive) ** 2)
    return float(1.0 - mse_model / mse_naive)


def da_binomial_test(actual: np.ndarray, pred: np.ndarray) -> dict:
    """
    Test binomial para Directional Accuracy.

    H₀: p = 0.5 (el modelo no tiene contenido informativo sobre la dirección).
    Estadístico: z = (DA_obs − 0.5) / √(0.25/n) ~ N(0,1) bajo H₀.
    p-valor unilateral superior (HA: DA > 50%).
    """
    n = len(actual)
    correct = np.sum(np.sign(actual) == np.sign(pred))
    da_obs = correct / n

    z = (da_obs - 0.5) / np.sqrt(0.25 / n)
    p_value = stats.norm.sf(z)   # unilateral: P(Z > z)

    sig = ""
    if p_value < 0.01:
        sig = "***"
    elif p_value < 0.05:
        sig = "**"
    elif p_value < 0.10:
        sig = "*"

    return {"da": da_obs * 100, "z_stat": z, "p_value_da": p_value, "sig_da": sig}


def block_bootstrap_ci(actual: np.ndarray, pred: np.ndarray,
                       n_boot: int = 1000, block_size: int = 12,
                       alpha: float = 0.05, seed: int = 42) -> dict:
    """
    Intervalos de confianza (1 − α) para RMSE y DA mediante bootstrap circular
    por bloques.  Preserva la autocorrelación de la serie de errores.

    Args:
        actual:     Valores reales.
        pred:       Predicciones del modelo.
        n_boot:     Réplicas bootstrap.
        block_size: Tamaño del bloque (12 = 1 año).
        alpha:      Nivel de significancia (IC al 95% si alpha=0.05).
        seed:       Semilla para reproducibilidad.

    Returns:
        Dict con 'rmse_lo', 'rmse_hi', 'da_lo', 'da_hi'.
    """
    rng = np.random.default_rng(seed)
    n = len(actual)
    rmse_boots, da_boots = [], []

    n_blocks_needed = int(np.ceil(n / block_size))

    for _ in range(n_boot):
        # Bloques circulares: el índice se toma módulo n para "circular"
        starts = rng.integers(0, n, size=n_blocks_needed)
        idx = np.concatenate(
            [np.arange(s, s + block_size) % n for s in starts]
        )[:n]

        act_b  = actual[idx]
        pred_b = pred[idx]

        rmse_b = np.sqrt(np.mean((act_b - pred_b) ** 2))
        da_b   = np.mean(np.sign(act_b) == np.sign(pred_b)) * 100

        rmse_boots.append(rmse_b)
        da_boots.append(da_b)

    return {
        "rmse_lo": float(np.percentile(rmse_boots, 100 * alpha / 2)),
        "rmse_hi": float(np.percentile(rmse_boots, 100 * (1 - alpha / 2))),
        "da_lo":   float(np.percentile(da_boots,   100 * alpha / 2)),
        "da_hi":   float(np.percentile(da_boots,   100 * (1 - alpha / 2))),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Función principal
# ════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(n_boot: int = 1_000, block_size: int = 12) -> pd.DataFrame:
    """
    Ejecuta todos los tests de poder predictivo sobre las predicciones walk-forward
    guardadas en output/tables/tab_6_*_predictions.csv.

    Returns:
        DataFrame con una fila por modelo y columnas de tests.
        También guarda tab_6_06_dm_tests.csv/.tex en output/tables/.
    """
    # ── Cargar predicciones ─────────────────────────────────────────────────
    dfs = {}
    for name, path in PRED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de predicciones: {path}\n"
                "Ejecuta primero run_chapter6.py para generar los modelos."
            )
        dfs[name] = pd.read_csv(path, index_col="fecha", parse_dates=True)

    # Verificar que todas tienen el mismo índice
    base_idx = dfs["XGBoost"].index
    for name, df in dfs.items():
        if not df.index.equals(base_idx):
            raise ValueError(f"Índice temporal inconsistente en {name}.")

    actual = dfs["XGBoost"]["actual"].values   # idéntico en los tres archivos
    n = len(actual)

    # ── Reconstruir predicciones naive (lag-1 del actual) ───────────────────
    # La primera predicción naive se construye desde el último dato de train,
    # que no está en los CSVs. Usamos el lag-1 interno del test: actual[t] → actual[t+1]
    naive_pred = np.empty(n)
    naive_pred[0] = 0.0              # primer mes: sin información previa → predice 0
    naive_pred[1:] = actual[:-1]     # resto: retorno del mes anterior

    # ── Ejecutar tests por modelo ────────────────────────────────────────────
    rows = []

    for name, df in dfs.items():
        pred = df["predicho"].values

        dm   = diebold_mariano_hln(actual, pred, naive_pred)
        r2   = oos_r2(actual, pred, naive_pred)
        da_t = da_binomial_test(actual, pred)
        ci   = block_bootstrap_ci(actual, pred, n_boot=n_boot,
                                  block_size=block_size)

        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        da   = float(np.mean(np.sign(actual) == np.sign(pred)) * 100)

        rows.append({
            "Modelo":             name,
            "RMSE (pp)":          round(rmse, 3),
            "IC 95% RMSE":        f"[{ci['rmse_lo']:.3f}, {ci['rmse_hi']:.3f}]",
            "OOS-R² (%)":         round(r2 * 100, 2),
            "DA (%)":             round(da, 1),
            "IC 95% DA":          f"[{ci['da_lo']:.1f}, {ci['da_hi']:.1f}]",
            "DM stat (HLN)":      round(dm["dm_stat"], 3),
            "p-valor DM":         round(dm["p_value"], 4),
            "Sig. DM":            dm["sig"],
            "z DA":               round(da_t["z_stat"], 3),
            "p-valor DA":         round(da_t["p_value_da"], 4),
            "Sig. DA":            da_t["sig_da"],
        })

        logger.info(
            f"{name}: RMSE={rmse:.3f} | OOS-R²={r2*100:.1f}% | "
            f"DA={da:.1f}% | DM p={dm['p_value']:.4f}{dm['sig']} | "
            f"DA p={da_t['p_value_da']:.4f}{da_t['sig_da']}"
        )

    result_df = pd.DataFrame(rows).set_index("Modelo")

    # ── Guardar ─────────────────────────────────────────────────────────────
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_TABLES / "tab_6_06_dm_tests.csv"
    result_df.to_csv(csv_path)
    logger.info(f"Tabla guardada: {csv_path}")

    tex_path = OUTPUT_TABLES / "tab_6_06_dm_tests.tex"
    latex_str = result_df.to_latex(
        caption=(
            "Tests estadísticos de poder predictivo. "
            "DM: test Diebold-Mariano (corrección HLN) vs. benchmark naive, "
            "H\\textsubscript{0}: igual precisión; "
            "OOS-R\\textsuperscript{2}: Campbell \\& Thompson (2008); "
            "DA: test binomial (H\\textsubscript{0}: DA = 50\\%). "
            "IC 95\\% obtenidos mediante bootstrap circular por bloques "
            f"(block\\_size=12, B=1\\,000). N={n}."
        ),
        label="tab:dm_tests",
        escape=True,
    )
    tex_path.write_text(latex_str, encoding="utf-8")
    logger.info(f"Tabla LaTeX guardada: {tex_path}")

    return result_df


# ════════════════════════════════════════════════════════════════════════════
# 3. Ejecución directa
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    df = run_statistical_tests(n_boot=1_000, block_size=12)
    print("\n=== RESULTADOS — Tests estadísticos de poder predictivo ===\n")
    print(df.to_string())

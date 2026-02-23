"""
lstm_model.py — Red LSTM (PyTorch) para predicción del retorno mensual del oro.

Arquitectura:
  GoldLSTM: LSTM(input_size → hidden_size) + Linear(hidden_size → 1)
  Entrada: secuencia de seq_len meses de features macroeconómicos
  Salida: retorno logarítmico del oro en t+1

Entrenamiento:
  - Optimizador Adam con tasa de aprendizaje 1e-3
  - Pérdida MSE, mini-batch de 16 observaciones
  - Early stopping sobre validación interna (últimos val_frac del train)

Walk-forward:
  - En cada paso t el escalador se reajusta solo sobre [0:t] (sin look-ahead bias)
  - Reentrenamiento cada refit_every=6 pasos (compromise entre adaptación y coste)
  - Con 125 pasos de test y refit_every=6, hay ~21 reentrenamientos totales
"""

import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUT_FIGURES, OUTPUT_TABLES, VARIABLE_LABELS
from src.utils.latex_tables import save_table
from src.utils.plotting import set_academic_style
from src.ml.walk_forward import compute_metrics

logger = logging.getLogger(__name__)

# ── Hiperparámetros por defecto ───────────────────────────────────────────────
LSTM_PARAMS = {
    "hidden_size": 32,      # unidades ocultas (parsimonioso con 312 obs)
    "num_layers": 1,         # una capa LSTM es suficiente con series cortas
    "seq_len": 6,            # ventana de 6 meses como contexto temporal
    "batch_size": 16,
    "lr": 1e-3,
    "max_epochs": 150,
    "patience": 20,          # paciencia del early stopping
    "val_frac": 0.15,        # fracción de train para validación interna
    "dropout": 0.0,          # dropout solo activo si num_layers > 1
    "seed": 42,
}

PALETTE = ["#1565C0", "#D32F2F", "#2E7D32", "#F57C00", "#6A1B9A"]


# ─────────────────────────────────────────────────────────────────────────────
#  Arquitectura del modelo
# ─────────────────────────────────────────────────────────────────────────────

class GoldLSTM(nn.Module):
    """
    Red recurrente LSTM para regresión de series temporales financieras.

    Acepta lotes de secuencias (batch, seq_len, input_size) y devuelve
    el retorno predicho para el mes siguiente (batch,).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: (batch, seq_len, hidden_size); solo el último paso temporal
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (batch,)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades de secuencias
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple:
    """
    Transforma (T, F) + (T,) en secuencias overlapping.

    La convención es: la secuencia X[t-seq_len:t] predice y[t],
    de modo que no hay look-ahead bias (y[t] no está en la secuencia).

    Returns:
        X_seq: (T-seq_len, seq_len, F) float32
        y_seq: (T-seq_len,) float32
    """
    Xs, ys = [], []
    for t in range(seq_len, len(y)):
        Xs.append(X[t - seq_len: t])
        ys.append(y[t])
    return (
        np.array(Xs, dtype=np.float32),
        np.array(ys, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Entrenamiento con early stopping
# ─────────────────────────────────────────────────────────────────────────────

def _train_lstm(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    input_size: int,
    params: dict,
    device: torch.device,
) -> GoldLSTM:
    """
    Entrena la LSTM y devuelve los mejores pesos según validación interna.

    El conjunto de validación son las últimas val_frac observaciones del
    entrenamiento actual — temporalmente ordenadas, sin mezcla aleatoria.
    """
    torch.manual_seed(params["seed"])

    val_frac = params["val_frac"]
    n = len(y_seq)
    n_val = max(2, int(n * val_frac))
    n_train = n - n_val

    X_tr = torch.from_numpy(X_seq[:n_train])
    y_tr = torch.from_numpy(y_seq[:n_train])
    X_val = torch.from_numpy(X_seq[n_train:]).to(device)
    y_val = torch.from_numpy(y_seq[n_train:]).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=False
    )

    model = GoldLSTM(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_count = 0

    for _ in range(params["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Evaluación sobre validación
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= params["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Walk-forward validation LSTM
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_lstm(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_train_size: int,
    params: dict = None,
    refit_every: int = 6,
) -> pd.DataFrame:
    """
    Walk-forward validation con ventana expandible para la LSTM.

    En cada paso t:
      1. Estandariza los features sobre la ventana [0:t] (sin look-ahead).
      2. Construye secuencias de longitud seq_len con la historia escalada.
      3. Cada refit_every pasos, reentrena la LSTM desde cero.
      4. Predice y[t] usando la secuencia X_scaled[t-seq_len:t].

    Args:
        X: Matriz de features (n, f).
        y: Vector target (n,).
        dates: Índice temporal de longitud n.
        initial_train_size: Observaciones del primer entrenamiento.
        params: Hiperparámetros (usa LSTM_PARAMS si None).
        refit_every: Cada cuántos pasos reentrenar.

    Returns:
        DataFrame con columnas actual, predicho, error indexado por fecha.
    """
    if params is None:
        params = LSTM_PARAMS.copy()

    seq_len = params["seq_len"]
    device = torch.device("cpu")   # instalación CPU-only (torch 2.10.0+cpu)

    predictions, actuals, pred_dates = [], [], []
    model = None
    steps_since_refit = refit_every  # fuerza refit en el primer paso

    n = len(y)
    total_test = n - initial_train_size
    logger.info(f"[LSTM] Walk-forward: {total_test} predicciones, refit c/{refit_every} pasos")

    for t in range(initial_train_size, n):
        # ── 1. Escalar sobre [0:t] ──────────────────────────────────────────
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X[:t])  # (t, f)

        # ── 2. Construir secuencias de entrenamiento ────────────────────────
        X_seq, y_seq = make_sequences(X_scaled, y[:t], seq_len)

        # Necesitamos al menos 2 observaciones para entrenar
        if len(X_seq) < max(2, params["batch_size"] // 2):
            continue

        # ── 3. Reentrenar si corresponde ────────────────────────────────────
        if model is None or steps_since_refit >= refit_every:
            model = _train_lstm(X_seq, y_seq, X.shape[1], params, device)
            steps_since_refit = 0

        steps_since_refit += 1

        # ── 4. Predicción ───────────────────────────────────────────────────
        if len(X_scaled) < seq_len:
            continue  # historia insuficiente para la ventana

        seq_input = X_scaled[-seq_len:].astype(np.float32)  # (seq_len, f)
        x_t = torch.from_numpy(seq_input).unsqueeze(0).to(device)  # (1, seq_len, f)

        model.eval()
        with torch.no_grad():
            pred_val = model(x_t).item()

        predictions.append(pred_val)
        actuals.append(y[t])
        pred_dates.append(dates[t])

    return pd.DataFrame({
        "fecha": pred_dates,
        "actual": actuals,
        "predicho": predictions,
        "error": np.array(actuals) - np.array(predictions),
    }).set_index("fecha")


# ─────────────────────────────────────────────────────────────────────────────
#  Figuras del Capítulo 6
# ─────────────────────────────────────────────────────────────────────────────

def plot_lstm_predictions(
    results_lstm: pd.DataFrame,
    df_original: pd.DataFrame,
) -> None:
    """
    Figura 6.5: Predicción walk-forward LSTM vs retorno real.
    Dos paneles: precio reconstruido (arriba) y retornos (abajo).
    """
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    actual = results_lstm["actual"]
    pred = results_lstm["predicho"]

    # ── Panel inferior: retornos ─────────────────────────────────────────────
    ax2.plot(actual.index, actual * 100, color="black", lw=1.0,
             label="Real", alpha=0.9)
    ax2.plot(pred.index, pred * 100, color=PALETTE[4], lw=0.9,
             linestyle="--", alpha=0.8, label="LSTM")
    ax2.axhline(0, color="gray", lw=0.5, linestyle=":")
    ax2.set_ylabel("Retorno mensual del oro (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel superior: precio reconstruido ──────────────────────────────────
    test_start = results_lstm.index[0]
    gold_prices = df_original["gold"].dropna()
    p0 = gold_prices[gold_prices.index < test_start].iloc[-1]
    gold_actual_test = gold_prices[gold_prices.index >= test_start]

    price_lstm = [p0]
    for r in pred.values:
        price_lstm.append(price_lstm[-1] * np.exp(r))
    price_lstm = pd.Series(price_lstm[1:], index=pred.index)

    ax1.plot(gold_actual_test.index, gold_actual_test.values,
             color="black", lw=1.2, label="Precio real (USD/oz)")
    ax1.plot(price_lstm.index, price_lstm.values,
             color=PALETTE[4], lw=1.0, linestyle="--", alpha=0.85,
             label="LSTM (reconstruido)")
    ax1.set_ylabel("Precio del oro (USD/oz)")
    ax1.set_title(
        "Figura 6.5: Predicción Walk-Forward — LSTM vs Precio Real",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            OUTPUT_FIGURES / f"fig_6_05_lstm_predictions.{ext}",
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    logger.info("Figura 6.5 guardada.")


def save_lstm_hyperparams_table() -> None:
    """Tabla 6.4: Hiperparámetros de la LSTM."""
    display_params = {
        k: v for k, v in LSTM_PARAMS.items()
        if k not in ("seed",)
    }
    rows = [
        {"Hiperparámetro": k, "Valor": str(v), "Descripción": _param_desc(k)}
        for k, v in display_params.items()
    ]
    df = pd.DataFrame(rows).set_index("Hiperparámetro")
    save_table(
        df, "tab_6_04_lstm_hyperparams",
        caption="Hiperparámetros de la red LSTM (Gold LSTM, PyTorch 2.10)",
        label="tab:lstm_params", float_format="%.4f",
    )


def _param_desc(k: str) -> str:
    desc = {
        "hidden_size": "Unidades en la capa LSTM",
        "num_layers": "Número de capas LSTM apiladas",
        "seq_len": "Longitud de la ventana de entrada (meses)",
        "batch_size": "Tamaño del mini-batch",
        "lr": "Tasa de aprendizaje (Adam)",
        "max_epochs": "Épocas máximas de entrenamiento",
        "patience": "Paciencia del early stopping (épocas sin mejora)",
        "val_frac": "Fracción de train reservada para validación interna",
        "dropout": "Dropout (activo solo con num_layers > 1)",
    }
    return desc.get(k, "")


# ─────────────────────────────────────────────────────────────────────────────
#  Función principal
# ─────────────────────────────────────────────────────────────────────────────

def generate_lstm_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_train_size: int,
    feature_cols: list,
    df_original: pd.DataFrame,
    params: dict = None,
    refit_every: int = 6,
) -> dict:
    """
    Función principal del Capítulo 6 para la LSTM.

    Ejecuta el walk-forward, calcula métricas, guarda predicciones y figuras.

    Returns:
        dict con claves 'results' (DataFrame), 'metrics' (dict), 'params'.
    """
    logger.info("=== LSTM — Capítulo 6 ===")

    if params is None:
        params = LSTM_PARAMS.copy()

    save_lstm_hyperparams_table()

    results = walk_forward_lstm(
        X, y, dates, initial_train_size, params, refit_every,
    )

    metrics = compute_metrics(results, "LSTM")
    logger.info(
        f"[LSTM] RMSE={metrics['RMSE']:.4f}  "
        f"MAE={metrics['MAE']:.4f}  "
        f"DA={metrics['Direc. Accuracy (%)']:.1f}%"
    )

    # Guardar predicciones
    results.to_csv(OUTPUT_TABLES / "tab_6_lstm_predictions.csv")

    # Figuras
    plot_lstm_predictions(results, df_original)

    logger.info("=== LSTM completado ===")
    return {"results": results, "metrics": metrics, "params": params}

"""
plotting.py — Estilo académico para matplotlib.

Configura fuentes, colores, grid y exportación en PDF + PNG.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.config import OUTPUT_FIGURES

# ── Paleta sobria ────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#1B3A5C",   # azul oscuro
    "secondary":  "#C62828",   # rojo
    "tertiary":   "#2E7D32",   # verde
    "quaternary": "#616161",   # gris
    "gold":       "#D4A017",   # dorado (para el oro)
    "light_gray": "#E0E0E0",   # grid
    "bg":         "#FFFFFF",   # fondo
}

PALETTE = [
    COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
    COLORS["quaternary"], COLORS["gold"], "#6A1B9A", "#00838F", "#E65100",
]


def set_academic_style():
    """Aplica estilo académico global a matplotlib."""
    plt.rcParams.update({
        # Fuentes
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # Figura
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # Ejes
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": COLORS["light_gray"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,

        # Color cycle
        "axes.prop_cycle": mpl.cycler(color=PALETTE),

        # Leyenda
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": COLORS["light_gray"],

        # Líneas
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


def save_figure(fig: plt.Figure, name: str, formats: tuple = ("png", "pdf")):
    """
    Guarda una figura en output/figures/ en los formatos indicados.

    Args:
        fig: Figura matplotlib.
        name: Nombre base (sin extensión), e.g. "fig_4_01_gold_price".
        formats: Tupla de formatos ('png', 'pdf').
    """
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = OUTPUT_FIGURES / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
    plt.close(fig)


def create_figure(nrows=1, ncols=1, figsize=None, **kwargs) -> tuple:
    """Crea figura con estilo académico aplicado."""
    set_academic_style()
    if figsize is None:
        figsize = (10, 6) if nrows == 1 else (10, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes

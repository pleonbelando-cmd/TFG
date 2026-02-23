"""
episode_markers.py — Sombreado de episodios históricos en gráficos temporales.

Añade bandas verticales semitransparentes con los 5 episodios del Capítulo 2.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import EPISODES


def add_episode_shading(
    ax: plt.Axes,
    episodes: dict | None = None,
    add_labels: bool = True,
    label_y: float = 0.97,
    fontsize: int = 6,
) -> None:
    """
    Añade bandas verticales sombreadas para cada episodio histórico.

    Args:
        ax: Eje matplotlib.
        episodes: Dict de episodios (por defecto usa EPISODES de config).
        add_labels: Si True, añade etiqueta de texto encima de cada banda.
        label_y: Posición vertical de la etiqueta (0-1, coordenadas del eje).
        fontsize: Tamaño de fuente de las etiquetas.
    """
    if episodes is None:
        episodes = EPISODES

    for key, ep in episodes.items():
        start = pd.Timestamp(ep["start"])
        end = pd.Timestamp(ep["end"])

        ax.axvspan(
            start, end,
            alpha=ep["alpha"],
            color=ep["color"],
            zorder=0,
            label=None,
        )

        if add_labels:
            mid = start + (end - start) / 2
            ax.text(
                mid, label_y,
                ep["name"],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=fontsize,
                color=ep["color"],
                alpha=0.8,
                rotation=90,
            )


def add_episode_legend(ax: plt.Axes, loc: str = "lower right") -> None:
    """Añade una leyenda compacta con los colores de los episodios."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=ep["color"], alpha=0.3, label=ep["name"])
        for ep in EPISODES.values()
    ]
    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=6,
        framealpha=0.9,
        ncol=2,
    )

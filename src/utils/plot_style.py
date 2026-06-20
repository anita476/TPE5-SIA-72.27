"""Estilo de gráficos compartido para el TP5.

Replica la paleta y tipografía usadas en el TP4 (fondo crema ``#fff5ec``,
texto ``#343434``, grilla ``#e8dcd0`` punteada y fuente Segoe UI) para que
todos los gráficos del trabajo tengan una apariencia consistente.

Importar este módulo aplica el estilo automáticamente vía ``rcParams``, por lo
que alcanza con ``import`` para que las figuras creadas a continuación lo usen.
También se expone :func:`apply_style` por si hace falta reaplicarlo.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# ── Paleta (idéntica a la del TP4) ──────────────────────────────────────────
BG_COLOR = "#fff5ec"
TEXT_COLOR = "#343434"
GRID_COLOR = "#e8dcd0"
GRID_MINOR = "#d4c8bc"
STATS_TEXT = "#555555"

STYLE = {
    "figure_bg": BG_COLOR,
    "axes_bg": BG_COLOR,
    "text_title": TEXT_COLOR,
    "text_axis": TEXT_COLOR,
    "grid": GRID_COLOR,
    "grid_minor": GRID_MINOR,
    "stats_text": STATS_TEXT,
}

PLOT_RC = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "text.color": TEXT_COLOR,
    "axes.titlesize": 13,
    "axes.titlecolor": TEXT_COLOR,
    "axes.labelsize": 11,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
}


def apply_style() -> None:
    """Aplica el estilo del TP a los ``rcParams`` globales de matplotlib."""
    plt.rcParams.update(PLOT_RC)


# Se aplica al importar para que cualquier figura posterior herede el estilo.
apply_style()

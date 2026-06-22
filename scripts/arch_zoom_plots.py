"""arch_zoom_plots.py
Re-dibuja las comparaciones de arquitectura del denoiser (latente / capas
ocultas / profundidad / activación) agregando un INSET con zoom a la zona de
ruido bajo (0–0.2), que es donde las curvas se separan; a ruido alto convergen
y no aportan. Lee los CSV ya generados (no reentrena).

Uso:
    python scripts/arch_zoom_plots.py            # los 4 estudios
    python scripts/arch_zoom_plots.py --zoom 0.25
"""

import _bootstrap
from _bootstrap import resolve

import argparse
import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_style  # aplica el estilo del TP al importar

# (ruta relativa a output/denoising, titulo) de los estudios del denoiser
STUDIES = [
    ("arch_compare/denoising_arch_comparison.csv",       "Denoising por tamaño de latente"),
    ("arch_compare/denoising_hidden_comparison.csv",     "Denoising por ancho de capas ocultas"),
    ("arch_compare/denoising_depth_comparison.csv",      "Denoising por profundidad"),
    ("arch_compare/denoising_activation_comparison.csv", "Denoising por función de activación"),
    ("arch_compare/denoising_batch_comparison.csv",      "Denoising por batch size"),
    ("arch_compare/denoising_lr_comparison.csv",         "Denoising por learning rate"),
    ("trainlevel_compare/trainlevel_salt_pepper.csv",    "Denoising por ruido de entrenamiento"),
]


def read_csv(path):
    """Return OrderedDict {label: (levels, mean, std)} preserving file order."""
    rows = OrderedDict()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        label_col = reader.fieldnames[0]
        for r in reader:
            lab = r[label_col]
            rows.setdefault(lab, ([], [], []))
            rows[lab][0].append(float(r["noise_level"]))
            rows[lab][1].append(float(r["recon_err_mean"]))
            rows[lab][2].append(float(r["recon_err_std"]))
    return {k: (np.array(a), np.array(b), np.array(c))
            for k, (a, b, c) in rows.items()}, label_col


def plot_with_inset(data, title, out_path, zoom_x=0.2):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5.2))

    # inset en la zona superior-izquierda (vacía en estos gráficos)
    axins = ax.inset_axes([0.07, 0.46, 0.46, 0.5])

    ymax_zoom = 0.0
    for i, (label, (x, m, s)) in enumerate(data.items()):
        c = cmap(i % 10)
        for a in (ax, axins):
            a.plot(x, m, "o-", color=c, markersize=4, linewidth=1.3, label=label)
            a.fill_between(x, m - s, m + s, color=c, alpha=0.18)
        mask = x <= zoom_x + 1e-9
        if mask.any():
            ymax_zoom = max(ymax_zoom, float((m[mask] + s[mask]).max()))

    ax.set_xlabel("Noise level")
    ax.set_ylabel("Mean recon pixel error (vs clean)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # configurar el zoom
    axins.set_xlim(-0.01, zoom_x + 0.01)
    axins.set_ylim(-0.1, ymax_zoom * 1.1 + 0.1)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=7)

    # líneas indicadoras conectando el recuadro del zoom con la región real
    ax.indicate_inset_zoom(axins, edgecolor="#888888", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zoom", type=float, default=0.2,
                   help="Límite superior de noise level para el inset.")
    args = p.parse_args()

    in_root = resolve("output/denoising")
    out_dir = resolve("output/denoising/arch_compare/zoom")
    os.makedirs(out_dir, exist_ok=True)

    for relpath, title in STUDIES:
        path = os.path.join(in_root, relpath)
        if not os.path.exists(path):
            print(f"  (falta {relpath}, salteo)")
            continue
        data, _ = read_csv(path)
        fname = os.path.basename(relpath)
        out = os.path.join(out_dir, fname.replace(".csv", "_zoom.png"))
        plot_with_inset(data, title, out, zoom_x=args.zoom)
        print(f"-> {out}")


if __name__ == "__main__":
    main()

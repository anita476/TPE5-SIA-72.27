from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.noise import add_noise

ROWS, COLS = 7, 5


def _binarise(output: np.ndarray, threshold: float) -> np.ndarray:
    return (output >= threshold).astype(np.float32)


def evaluate_at_level(
    ae,
    X: np.ndarray,
    level: float,
    noise_type: str,
    rng: np.random.Generator,
    threshold: float,
    max_errors: int,
):
    noisy = add_noise(X, level, noise_type, rng)

    recon_raw = ae.decode(ae.encode(noisy))
    recon_bin = _binarise(recon_raw, threshold)

    errors = np.sum(recon_bin != X, axis=1)
    noisy_err = np.sum(_binarise(noisy, threshold) != X, axis=1)

    return {
        "level": level,
        "noisy": noisy,
        "recon_bin": recon_bin,
        "errors": errors,
        "avg_err": float(errors.mean()),
        "max_err": float(errors.max()),
        "passed": int(np.sum(errors <= max_errors)),
        "avg_noisy_err": float(noisy_err.mean()),
    }


def _grid(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(ROWS, COLS)


def _plot_examples(X, labels, res, out_dir, n_examples=8):
    level = res["level"]
    n = min(n_examples, len(X))
    fig, axes = plt.subplots(3, n, figsize=(1.4 * n, 4.6))
    if n == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["Clean", f"Noisy (l={level})", "Reconstructed"]
    sources = [X, res["noisy"], res["recon_bin"]]

    for r in range(3):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(_grid(sources[r][c]), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(str(labels[c]), fontsize=9)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9)

    fig.suptitle(f"Denoising - noise level = {level}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"denoising_examples_{level}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_curve(results, out_dir):
    levels = [r["level"] for r in results]
    avg_err = [r["avg_err"] for r in results]
    noisy_err = [r["avg_noisy_err"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(levels, noisy_err, "o--", color="gray", label="Noisy input")
    ax.plot(levels, avg_err, "o-", color="steelblue", label="DAE output")
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Mean pixel error (vs clean)")
    ax.set_title("Denoising capability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "denoising_vs_noise.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_csv(results, out_dir, total):
    path = os.path.join(out_dir, "denoising_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "noise_level", "avg_noisy_pixel_err", "avg_recon_pixel_err",
            "max_recon_pixel_err", "passed", "total",
        ])
        for r in results:
            writer.writerow([
                r["level"], round(r["avg_noisy_err"], 4), round(r["avg_err"], 4),
                r["max_err"], r["passed"], total,
            ])
    return path


def run_denoising_study(
    ae,
    X: np.ndarray,
    labels: list[str],
    noise_levels: list[float],
    noise_type: str,
    out_dir: str,
    seed: int = 0,
    threshold: float = 0.5,
    max_errors: int = 1,
    n_examples: int = 8,
) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    results = []
    print(f"\n{'Level':>8}  {'NoisyErr':>9}  {'ReconErr':>9}  {'Passed':>8}")
    print("-" * 40)
    for level in noise_levels:
        res = evaluate_at_level(ae, X, level, noise_type, rng, threshold, max_errors)
        results.append(res)
        _plot_examples(X, labels, res, out_dir, n_examples)
        print(f"{level:>8}  {res['avg_noisy_err']:>9.2f}  "
              f"{res['avg_err']:>9.2f}  {res['passed']:>4}/{len(X)}")

    curve_path = _plot_curve(results, out_dir)
    csv_path = _write_csv(results, out_dir, len(X))
    print(f"\nCurve -> {curve_path}")
    print(f"CSV   -> {csv_path}")
    return results

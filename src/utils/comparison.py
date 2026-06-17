"""Compare several autoencoder variants under the same budget.

Each variant is a full hyperparameter dict (built by merging per-variant
overrides onto a common base). Reuses ``single_run.execute_run`` to train and
evaluate, and produces overlaid loss curves, a metric bar chart and a CSV.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.single_run import execute_run


def run_variants(variants, base, X, labels, out_dir):
    """Train/evaluate each variant. Returns a list of (name, RunResult)."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, var in enumerate(variants, start=1):
        name = var.get("name", f"variant_{i}")
        params = dict(base)
        params.update({k: v for k, v in var.items() if k != "name"})
        print(f"\n=== variant {i}/{len(variants)}: {name} ===")
        res = execute_run(i, params, X, labels, out_dir)
        results.append((name, res))
    return results


def plot_loss_curves(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results:
        ax.plot(r.epoch_losses, linewidth=0.9, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Training loss comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "compare_loss.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_bars(results, out_dir, total):
    names = [n for n, _ in results]
    passed = [r.passed for _, r in results]
    avg_err = [r.avg_pixel_errors for _, r in results]
    x = range(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(x, passed, color="seagreen", edgecolor="black")
    ax1.axhline(total, color="gray", linestyle="--", label=f"total ({total})")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Passed (<= max_errors)")
    ax1.set_title("Characters learned")
    ax1.legend(fontsize=8)

    ax2.bar(x, avg_err, color="indianred", edgecolor="black")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Avg pixel error")
    ax2.set_title("Average reconstruction error")

    plt.tight_layout()
    path = os.path.join(out_dir, "compare_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_csv(results, out_dir):
    path = os.path.join(out_dir, "compare_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "variant", "layer_dims", "optimizer", "lr", "epochs_run",
            "passed", "failed", "avg_pixel_errors", "final_mse",
        ])
        for name, r in results:
            writer.writerow([
                name, r.params.get("layer_dims"), r.params.get("optimizer"),
                r.params.get("lr"), len(r.epoch_losses), r.passed, r.failed,
                round(r.avg_pixel_errors, 4),
                round(r.epoch_losses[-1], 6) if r.epoch_losses else "",
            ])
    return path


def run_comparison(variants, base, X, labels, out_dir):
    """End-to-end: train variants and write all comparison artefacts."""
    results = run_variants(variants, base, X, labels, out_dir)
    loss_path = plot_loss_curves(results, out_dir)
    bars_path = plot_metric_bars(results, out_dir, total=len(X))
    csv_path = write_csv(results, out_dir)

    print(f"\n{'variant':<16} {'passed':>7} {'avg_err':>8} {'final_mse':>11}")
    print("-" * 46)
    for name, r in results:
        final = r.epoch_losses[-1] if r.epoch_losses else float("nan")
        print(f"{name:<16} {r.passed:>4}/{len(X)} "
              f"{r.avg_pixel_errors:>8.2f} {final:>11.6f}")
    print(f"\nLoss   -> {loss_path}")
    print(f"Bars   -> {bars_path}")
    print(f"CSV    -> {csv_path}")
    return results

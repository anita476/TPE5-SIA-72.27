"""Compare several autoencoder variants under the same budget, aggregated over
several seeds (mean ± std).

Each variant is a full hyperparameter dict (built by merging per-variant
overrides onto a common base). Delegates the per-(variant, seed) training and
the aggregated plots to :mod:`utils.multiseed`.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from utils import multiseed


def _write_multiseed_csv(grouped, out_dir):
    path = os.path.join(out_dir, "compare_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "variant", "n_seeds", "passed_mean", "passed_std",
            "avg_err_mean", "avg_err_std", "final_loss_mean",
        ])
        for name, results in grouped.items():
            passed = [r.passed for r in results]
            avg_err = [r.avg_pixel_errors for r in results]
            final = [r.epoch_losses[-1] for r in results if r.epoch_losses]
            writer.writerow([
                name, len(results),
                round(float(np.mean(passed)), 3), round(float(np.std(passed)), 3),
                round(float(np.mean(avg_err)), 4), round(float(np.std(avg_err)), 4),
                round(float(np.mean(final)), 6) if final else "",
            ])
    return path


def run_comparison(variants, base, X, labels, out_dir, seeds, workers=1,max_errors=1):
    """End-to-end: train each variant over several seeds and write all
    comparison artefacts with mean ± std. ``seeds`` is an explicit list."""
    seed_list = list(seeds)
    print(f"Comparison over seeds={seed_list} | workers={workers}")

    grouped = multiseed.run_variant_seeds(
        variants, base, X, list(labels), seed_list, out_dir, workers=workers)
    # Preserve the variants' declared order.
    ordered = {v.get("name", f"variant_{i}"): grouped[v.get("name", f"variant_{i}")]
               for i, v in enumerate(variants, start=1)}

    loss_path = multiseed.plot_compare_loss_bands(
        ordered, os.path.join(out_dir, "compare_loss.png"))
    bars_path = multiseed.plot_compare_metric_bars(
        ordered, len(X), os.path.join(out_dir, "compare_metrics.png"),max_errors)
    csv_path = _write_multiseed_csv(ordered, out_dir)

    print(f"\n{'variant':<16} {'passed (mean±std)':>20} {'avg_err (mean±std)':>22}")
    print("-" * 60)
    for name, results in ordered.items():
        passed = [r.passed for r in results]
        avg_err = [r.avg_pixel_errors for r in results]
        print(f"{name:<16} {np.mean(passed):>8.1f} ± {np.std(passed):<5.1f}/{len(X)}"
              f"   {np.mean(avg_err):>8.2f} ± {np.std(avg_err):<5.2f}")
    print(f"\nLoss   -> {loss_path}")
    print(f"Bars   -> {bars_path}")
    print(f"CSV    -> {csv_path}")
    return ordered

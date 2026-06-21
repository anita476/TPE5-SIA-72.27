"""Compare several autoencoder variants under the same budget, aggregated over
several seeds (mean ± std).

Each variant is a full hyperparameter dict (built by merging per-variant
overrides onto a common base). Delegates the per-(variant, seed) training and
the aggregated plots to :mod:`utils.multiseed`.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np

from utils import multiseed


def _safe_name(name: str) -> str:
    """Return a filesystem-friendly variant directory name."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return clean.strip("_") or "variant"


def _write_multiseed_csv(grouped, out_dir):
    path = os.path.join(out_dir, "compare_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "variant", "weight_init", "n_seeds", "passed_mean", "passed_std",
            "avg_err_mean", "avg_err_std", "final_loss_mean",
        ])
        for name, results in grouped.items():
            passed = [r.passed for r in results]
            avg_err = [r.avg_pixel_errors for r in results]
            final = [r.epoch_losses[-1] for r in results if r.epoch_losses]
            init = results[0].params.get("weight_init", results[0].params.get("init", "he"))
            writer.writerow([
                name, init, len(results),
                round(float(np.mean(passed)), 3), round(float(np.std(passed)), 3),
                round(float(np.mean(avg_err)), 4), round(float(np.std(avg_err)), 4),
                round(float(np.mean(final)), 6) if final else "",
            ])
    return path


def _write_variant_summary_csv(name, results, out_dir):
    summary = multiseed.summarise(results)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["variant", name])
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
        writer.writerow([])
        writer.writerow([
            "seed", "passed", "avg_pixel_errors", "max_pixel_errors",
            "final_loss", "weight_init",
        ])
        for r in results:
            final_loss = r.epoch_losses[-1] if r.epoch_losses else ""
            init = r.params.get("weight_init", r.params.get("init", "he"))
            writer.writerow([
                r.params.get("seed"), r.passed,
                round(r.avg_pixel_errors, 4), r.max_pixel_errors,
                round(float(final_loss), 8) if final_loss != "" else "",
                init,
            ])
    return summary, path


def _write_test_autoencoder_outputs(ordered, out_dir, max_errors, latent_show):
    """Write test_autoencoder-style artefacts for each comparison variant."""
    artefacts = {}
    for name, results in ordered.items():
        variant_dir = os.path.join(out_dir, _safe_name(name))
        os.makedirs(variant_dir, exist_ok=True)
        n = len(results)

        loss_path = multiseed.plot_loss_band(
            results,
            os.path.join(variant_dir, "loss.png"),
            title=f"Training Loss - media +/- desv. ({n} seeds) - {name}",
        )
        err_path = multiseed.plot_per_char_band(
            results,
            max_errors,
            os.path.join(variant_dir, "errors.png"),
            title=f"Errores por caracter - media +/- desv. ({n} seeds) - {name}",
        )
        latent_path = multiseed.plot_latent_seeds(
            results,
            os.path.join(variant_dir, "latent_seeds.png"),
            n_show=latent_show,
            title=f"Espacio latente por seed - {name}",
        )
        summary, csv_path = _write_variant_summary_csv(name, results, variant_dir)
        artefacts[name] = {
            "dir": variant_dir,
            "loss": loss_path,
            "errors": err_path,
            "latent": latent_path,
            "csv": csv_path,
            "summary": summary,
        }
    return artefacts


def run_comparison(
    variants, base, X, labels, out_dir, seeds, workers=1, max_errors=1,
    latent_show=6,
):
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
    variant_outputs = _write_test_autoencoder_outputs(
        ordered, out_dir, max_errors, latent_show)

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
    print("Per-variant outputs:")
    for name, paths in variant_outputs.items():
        print(f"  {name:<16} -> {paths['dir']}")
    return ordered
